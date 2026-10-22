"""
ParsedDocument → список чанков с расширенными метаданными.

Правила:
- header приклеивается к следующему тексту (для контекста)
- accent (z1qcye) — отдельный чанк
- text разбивается на ~1000 символов с overlap 100 (увеличено для кода/JSON)
- таблицы НЕ разрезаем; каждая строка = чанк с заголовками колонок
- list — объединяется в один чанк
- Склейка JSON-структур: если скобки не сбалансированы, склеиваем со следующими чанками.
"""
import hashlib
import re
import unicodedata
from typing import Any

from loguru import logger


CHUNK_SIZE = 1000      # Увеличено с 500 для сохранения JSON
CHUNK_OVERLAP = 100    # Увеличено с 50


def _make_chunk_id(document_id: str, index: int) -> str:
    raw = f"{document_id}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()


def _split_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[tuple[str, int, int]]:
    """Разбивает текст на куски по ~size символов с overlap."""
    text = text.strip()
    if not text:
        return []

    if len(text) <= size:
        return [(text, 0, len(text))]

    chunks = []
    start = 0
    min_step = max(1, size - overlap)

    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]

        if end < len(text):
            last_dot = max(
                chunk.rfind('. '),
                chunk.rfind('!\n'),
                chunk.rfind('?\n'),
                chunk.rfind('.\n'),
                chunk.rfind('. \xa0'),
                chunk.rfind('! '),
                chunk.rfind('? '),
                chunk.rfind(':\n'),
                chunk.rfind('},'),  # Разрыв по границе JSON-объекта
            )
            if last_dot > size * 0.6:
                chunk = chunk[: last_dot + 1]
                end = start + len(chunk)

        chunks.append((chunk.strip(), start, end))

        next_start = end - overlap
        if next_start <= start:
            next_start = start + min_step
        start = next_start

        if end >= len(text):
            break

    return [(c, s, e) for (c, s, e) in chunks if c]


def _table_to_chunks(table: dict[str, Any]) -> list[tuple[str, int]]:
    rows = table.get("rows", [])
    if not rows:
        return []

    headers = rows[0] if rows else []
    data_rows = rows[1:] if len(rows) > 1 else []

    chunks = []
    for row_idx, row in enumerate(data_rows, start=1):
        parts = []
        for h, v in zip(headers, row):
            h = (h or "").strip()
            v = (v or "").strip()
            if h and v:
                parts.append(f"{h}: {v}")
            elif v:
                parts.append(v)
        if parts:
            chunks.append((" | ".join(parts), row_idx))

    if not chunks and headers:
        chunks.append((" | ".join(str(c) for c in headers if c), 0))

    return chunks


def _make_section_slug(text: str) -> str:
    text = unicodedata.normalize('NFC', text.strip().lower())
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    return text[:50].strip('_')


def _build_chunk(
    doc_id: str,
    index: int,
    text: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": _make_chunk_id(doc_id, index),
        "text": text,
        "payload": payload,
    }


def chunk_document(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    document_id = parsed["document_id"]
    source_file = parsed.get("source_file", "")
    source_type = parsed.get("source_type", "unknown")
    parser_name = parsed.get("parser", "unknown")
    pages = parsed.get("pages", [])

    all_chunks: list[dict[str, Any]] = []
    chunk_counter = 0
    global_char_pos = 0

    for page in pages:
        page_num = page.get("page")
        page_label = page.get("page_label", "")
        elements = page.get("elements", [])
        current_section = ""
        current_section_slug = ""
        pending_header = ""

        for el in elements:
            etype = el.get("type", "text")

            if el.get("section"):
                current_section = el["section"]
                current_section_slug = _make_section_slug(current_section)
            elif etype == "header":
                current_section = el.get("text", "")
                current_section_slug = _make_section_slug(current_section)

            # Определяем, является ли секция методом API
            is_api_method = bool(re.search(r"(POST|GET)\s+/SUPPLIERPORTAL", current_section, re.IGNORECASE))
            api_method_name = current_section if is_api_method else None

            if etype == "header":
                pending_header = el.get("text", "").strip()
                continue

            if etype == "accent":
                text = el.get("text", "").strip()
                if not text:
                    continue
                if pending_header and pending_header not in text:
                    text = f"{pending_header}\n{text}"
                    pending_header = ""

                c_start = global_char_pos
                c_end = c_start + len(text)

                all_chunks.append(_build_chunk(
                    doc_id=document_id,
                    index=chunk_counter,
                    text=text,
                    payload={
                        "document_id": document_id,
                        "source_file": source_file,
                        "source_type": source_type,
                        "parser": parser_name,
                        "page": page_num,
                        "page_label": page_label,
                        "section": current_section,
                        "section_slug": current_section_slug,
                        "api_method": api_method_name,
                        "element_type": "accent",
                        "chunk_index": chunk_counter,
                        "chunk_char_start": c_start,
                        "chunk_char_end": c_end,
                        "is_table_row": False,
                    },
                ))
                chunk_counter += 1
                global_char_pos = c_end
                continue

            if etype == "list":
                text = el.get("text", "").strip()
                if not text:
                    continue
                if pending_header:
                    text = f"{pending_header}\n{text}"
                    pending_header = ""

                c_start = global_char_pos
                c_end = c_start + len(text)

                all_chunks.append(_build_chunk(
                    doc_id=document_id,
                    index=chunk_counter,
                    text=text,
                    payload={
                        "document_id": document_id,
                        "source_file": source_file,
                        "source_type": source_type,
                        "parser": parser_name,
                        "page": page_num,
                        "page_label": page_label,
                        "section": current_section,
                        "section_slug": current_section_slug,
                        "api_method": api_method_name,
                        "element_type": "list",
                        "chunk_index": chunk_counter,
                        "chunk_char_start": c_start,
                        "chunk_char_end": c_end,
                        "is_table_row": False,
                    },
                ))
                chunk_counter += 1
                global_char_pos = c_end
                continue

            if etype == "table":
                caption = el.get("caption", "").strip()
                table_chunks = _table_to_chunks(el)
                for t_chunk, row_idx in table_chunks:
                    text = t_chunk
                    if caption:
                        text = f"{caption}\n{text}"
                    elif pending_header:
                        text = f"{pending_header}\n{text}"
                    if pending_header:
                        pending_header = ""

                    c_start = global_char_pos
                    c_end = c_start + len(text)

                    all_chunks.append(_build_chunk(
                        doc_id=document_id,
                        index=chunk_counter,
                        text=text,
                        payload={
                            "document_id": document_id,
                            "source_file": source_file,
                            "source_type": source_type,
                            "parser": parser_name,
                            "page": page_num,
                            "page_label": page_label,
                            "section": current_section,
                            "section_slug": current_section_slug,
                            "api_method": api_method_name,
                            "element_type": "table",
                            "chunk_index": chunk_counter,
                            "chunk_char_start": c_start,
                            "chunk_char_end": c_end,
                            "caption": caption,
                            "is_table_row": True,
                            "table_row_index": row_idx,
                        },
                    ))
                    chunk_counter += 1
                    global_char_pos = c_end
                continue

            if etype == "text":
                text = el.get("text", "").strip()
                if not text:
                    continue

                if pending_header:
                    text = f"{pending_header}\n{text}"
                    pending_header = ""

                parts = _split_text(text)
                for part, p_start, p_end in parts:
                    c_start = global_char_pos + p_start
                    c_end = global_char_pos + p_end

                    all_chunks.append(_build_chunk(
                        doc_id=document_id,
                        index=chunk_counter,
                        text=part,
                        payload={
                            "document_id": document_id,
                            "source_file": source_file,
                            "source_type": source_type,
                            "parser": parser_name,
                            "page": page_num,
                            "page_label": page_label,
                            "section": current_section,
                            "section_slug": current_section_slug,
                            "api_method": api_method_name,
                            "element_type": "text",
                            "chunk_index": chunk_counter,
                            "chunk_char_start": c_start,
                            "chunk_char_end": c_end,
                            "is_table_row": False,
                        },
                    ))
                    chunk_counter += 1
                global_char_pos += len(text)

    # === УЛУЧШЕННАЯ ЗАЩИТА ОТ РАЗРЫВА JSON/КОДА ===
    # Если чанк содержит открывающие скобки и не закрывает их, 
    # склеиваем со следующими чанками, пока скобки не сбалансируются.
    merged_chunks: list[dict[str, Any]] = []
    i = 0
    while i < len(all_chunks):
        current = all_chunks[i]
        text = current["text"]
        
        # Проверяем баланс фигурных и квадратных скобок
        open_braces = text.count('{') + text.count('[')
        close_braces = text.count('}') + text.count(']')
        
        if open_braces > close_braces:
            j = i + 1
            while j < len(all_chunks) and all_chunks[j]["payload"]["document_id"] == current["payload"]["document_id"]:
                text += "\n" + all_chunks[j]["text"]
                open_braces = text.count('{') + text.count('[')
                close_braces = text.count('}') + text.count(']')
                if open_braces <= close_braces:
                    break
                j += 1
            current["text"] = text
            merged_chunks.append(current)
            i = j + 1
        else:
            merged_chunks.append(current)
            i += 1

    all_chunks = merged_chunks

    # Проставляем prev/next ссылки
    for i, ch in enumerate(all_chunks):
        if i > 0:
            ch["payload"]["prev_chunk_id"] = all_chunks[i - 1]["id"]
        if i < len(all_chunks) - 1:
            ch["payload"]["next_chunk_id"] = all_chunks[i + 1]["id"]

    logger.info(
        f"[chunker] {document_id}: {len(all_chunks)} чанков "
        f"из {sum(len(p.get('elements', [])) for p in pages)} элементов"
    )
    return all_chunks