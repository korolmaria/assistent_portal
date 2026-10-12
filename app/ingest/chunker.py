"""
ParsedDocument → список чанков с расширенными метаданными.

Правила:
- header приклеивается к следующему тексту (для контекста)
- accent (z1qcye) — отдельный чанк
- text разбивается на ~500 символов с overlap 50
- таблицы НЕ разрезаем; каждая строка = чанк с заголовками колонок
- list — объединяется в один чанк

Метаданные включают:
- идентификацию документа (document_id, source_file, source_type, parser)
- позицию в документе (page, page_label, section, section_slug)
- тип и индекс (element_type, chunk_index)
- позицию в тексте (chunk_char_start, chunk_char_end)
- связи с соседями (prev_chunk_id, next_chunk_id)
- доп. инфо для таблиц (caption, is_table_row, table_row_index)
"""
import hashlib
import re
import unicodedata
from typing import Any

from loguru import logger


# ==========================================================================
# Настройки чанкинга
# ==========================================================================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ==========================================================================
# Хелперы
# ==========================================================================
def _make_chunk_id(document_id: str, index: int) -> str:
    """Детерминированный ID чанка."""
    raw = f"{document_id}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()


def _split_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[tuple[str, int, int]]:
    """
    Разбивает текст на куски по ~size символов с overlap.
    Возвращает список (chunk_text, char_start, char_end).

    ГАРАНТИРУЕТ продвижение вперёд — нет бесконечных циклов.
    Учитывает \\xa0 (неразрывные пробелы) в JSON-коде.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= size:
        return [(text, 0, len(text))]

    chunks = []
    start = 0
    min_step = max(1, size - overlap)  # минимальный шаг вперёд

    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]

        # Ищем границы предложений (включая \xa0)
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
            )
            if last_dot > size * 0.6:
                chunk = chunk[: last_dot + 1]
                end = start + len(chunk)

        chunks.append((chunk.strip(), start, end))

        # === ЗАЩИТА ОТ БЕСКОНЕЧНОГО ЦИКЛА ===
        next_start = end - overlap
        if next_start <= start:
            next_start = start + min_step
        start = next_start

        # Ранний выход
        if end >= len(text):
            break

    return [(c, s, e) for (c, s, e) in chunks if c]


def _table_to_chunks(table: dict[str, Any]) -> list[tuple[str, int]]:
    """
    Таблица → список чанков.
    Возвращает список (chunk_text, row_index).
    """
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
    """Простой slug из заголовка (латиница + цифры)."""
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
    """Собирает чанк."""
    return {
        "id": _make_chunk_id(doc_id, index),
        "text": text,
        "payload": payload,
    }


# ==========================================================================
# Главная функция
# ==========================================================================
def chunk_document(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """
    ParsedDocument → список чанков с расширенными метаданными.
    """
    document_id = parsed["document_id"]
    source_file = parsed.get("source_file", "")
    source_type = parsed.get("source_type", "unknown")
    parser_name = parsed.get("parser", "unknown")
    pages = parsed.get("pages", [])

    all_chunks: list[dict[str, Any]] = []
    chunk_counter = 0
    global_char_pos = 0

    for page in pages:
        page_num = page.get("page", 1)
        elements = page.get("elements", [])
        current_section = ""
        current_section_slug = ""
        pending_header = ""

        for el in elements:
            etype = el.get("type", "text")

            # Обновляем раздел
            if el.get("section"):
                current_section = el["section"]
                current_section_slug = _make_section_slug(current_section)
            elif etype == "header":
                current_section = el.get("text", "")
                current_section_slug = _make_section_slug(current_section)

            # === header → запоминаем ===
            if etype == "header":
                pending_header = el.get("text", "").strip()
                continue

            # === accent ===
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
                        "page_label": f"стр. {page_num}",
                        "section": current_section,
                        "section_slug": current_section_slug,
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

            # === list ===
            if etype == "list":
                items = el.get("items", [])
                if not items:
                    continue
                text = "\n".join(f"• {item}" for item in items)
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
                        "page_label": f"стр. {page_num}",
                        "section": current_section,
                        "section_slug": current_section_slug,
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

            # === table ===
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
                            "page_label": f"стр. {page_num}",
                            "section": current_section,
                            "section_slug": current_section_slug,
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

            # === text ===
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
                            "page_label": f"стр. {page_num}",
                            "section": current_section,
                            "section_slug": current_section_slug,
                            "element_type": "text",
                            "chunk_index": chunk_counter,
                            "chunk_char_start": c_start,
                            "chunk_char_end": c_end,
                            "is_table_row": False,
                        },
                    ))
                    chunk_counter += 1
                global_char_pos += len(text)

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
