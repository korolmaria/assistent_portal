"""
DOCX → ParsedDocument.

Порядок элементов: параграф → таблица → параграф.
Умеет извлекать реальные номера страниц из:
1. Колонтитулов (например, "Страница 56 из 68").
2. Жестких разрывов страниц Word (w:br w:type="page").
3. Эвристики по количеству символов (если нет ни того, ни другого).
"""
import re
import time
from pathlib import Path
from typing import Any

from docx import Document
from docx.document import Document as _Doc
from docx.oxml.ns import qn
from loguru import logger

from app.ingest.document_id import make_document_id, make_section_slug


# ==========================================================================
# Эвристики для классификации параграфов
# ==========================================================================
def _is_all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3: return False
    non_alpha = sum(1 for c in text if not (c.isalpha() or c.isspace() or c.isdigit()))
    if len(text) > 0 and non_alpha / len(text) > 0.2: return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) > 0.8

def _is_section_number(text: str) -> bool:
    return bool(re.match(r"^\d+(\.\d+)*\.?\s+\S", text))

def _is_figure_caption(text: str) -> bool:
    return bool(re.match(r"^Рисунок\s+\d+", text, re.IGNORECASE))

def _is_list_item(text: str) -> bool:
    return bool(re.match(r"^[•\-\*→]\s+\S", text))

def _is_code_like(text: str) -> bool:
    stripped = text.lstrip()
    if stripped and stripped[0] in '{[}"\'':
        return True
    if len(text) > 0:
        non_alpha = sum(1 for c in text if not (c.isalpha() or c.isspace() or c.isdigit()))
        if non_alpha / len(text) > 0.25: return True
    return False

def _is_garbage_numbers(text: str) -> bool:
    """Проверяет, является ли строка мусором из цифр (например, '1. 2. 3. 4.' или '1 1 1 1')."""
    cleaned = text.replace(" ", "").replace(".", "")
    return cleaned.isdigit() and len(cleaned) > 3

def _classify_paragraph(text: str, style_name: str) -> str:
    text = text.strip()
    if style_name == 'z1qcye': return 'accent'
    if _is_all_caps(text): return 'header'
    if _is_section_number(text): return 'header'
    if _is_figure_caption(text): return 'text'
    if _is_list_item(text): return 'list'
    if _is_code_like(text): return 'text'
    if (
        len(text) <= 80
        and len(text.split()) <= 8
        and text[0:1].isalpha()
        and not text.endswith(('.', '!', '?', ':', ';', ',', '}', ']'))
        and any(c.isalpha() for c in text)
    ):
        return 'header'
    return 'text'


# ==========================================================================
# Парсер
# ==========================================================================
def _extract_table(table) -> dict[str, Any]:
    rows = []
    for row in table.rows:
        cells = [c.text.strip() for c in row.cells]
        rows.append(cells)
    rows = [r for r in rows if any(c for c in r)]
    return {
        "type": "table",
        "rows": rows,
        "num_rows": len(rows),
        "num_cols": len(rows[0]) if rows else 0,
    }

def _iter_block_items(doc: _Doc):
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    parent = doc.element.body
    for child in parent.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, doc)
        elif child.tag == qn('w:tbl'):
            yield Table(child, doc)

def _has_page_break(paragraph) -> bool:
    """Проверяет, содержит ли параграф жесткий разрыв страницы Word."""
    for run in paragraph.runs:
        for br in run._element.findall(qn('w:br')):
            if br.get(qn('w:type')) == 'page':
                return True
    return False

def parse_docx(docx_path: str | Path) -> dict[str, Any]:
    docx_path = Path(docx_path)
    document_id = make_document_id(docx_path)
    start_time = time.time()

    logger.info(f"[docx_parser] начинаю: {docx_path.name}")

    doc = Document(str(docx_path))
    
    pages_dict: dict[int, list[dict[str, Any]]] = {}
    current_section = ""
    current_page = 1
    heuristic_char_count = 0
    has_footer = bool(doc.sections and doc.sections[0].footer and doc.sections[0].footer.paragraphs)

    # Регулярка для поиска страницы в колонтитулах
    PAGE_FOOTER_RE = re.compile(r"Страница\s+(\d+)", re.IGNORECASE)

    for block in _iter_block_items(doc):
        if block.__class__.__name__ == 'Paragraph':
            text = block.text.strip()
            if not text: continue
            
            # === ОЧИСТКА ОТ МУСОРА ===
            if _is_garbage_numbers(text):
                logger.debug(f"[docx_parser] пропущен мусор: {text[:30]}...")
                continue
            
            # === ПОИСК КОЛОНТИТУЛА ===
            match = PAGE_FOOTER_RE.search(text)
            if match:
                current_page = int(match.group(1))
                heuristic_char_count = 0
                logger.debug(f"[docx_parser] найден колонтитул: стр. {current_page}")
                continue

            # === ЖЕСТКИЙ РАЗРЫВ СТРАНИЦЫ ===
            # Если в параграфе есть разрыв, и на текущей странице уже что-то есть,
            # переходим на следующую страницу.
            if _has_page_break(block):
                if pages_dict.get(current_page):
                    current_page += 1
                    heuristic_char_count = 0
                    logger.debug(f"[docx_parser] найден разрыв страницы: стр. {current_page}")

            # === ЭВРИСТИКА (если нет колонтитулов и разрывов) ===
            heuristic_char_count += len(text)
            if heuristic_char_count > 2500 and not has_footer:
                current_page += 1
                heuristic_char_count = len(text)

            style_name = block.style.name if block.style else 'Normal'
            etype = _classify_paragraph(text, style_name)

            el: dict[str, Any] = {
                "type": etype,
                "text": text,
                "style": style_name,
            }

            if etype == 'header':
                current_section = text
                el["section"] = current_section
            else:
                el["section"] = current_section
            
            if current_page not in pages_dict:
                pages_dict[current_page] = []
            pages_dict[current_page].append(el)

        elif block.__class__.__name__ == 'Table':
            # Таблицы тоже могут содержать разрывы (например, если таблица длинная),
            # но для простоты мы просто добавляем к текущей странице.
            heuristic_char_count += 500
            if heuristic_char_count > 2500 and not has_footer:
                current_page += 1
                heuristic_char_count = 500

            table_el = _extract_table(block)
            table_el["section"] = current_section
            
            if current_page not in pages_dict:
                pages_dict[current_page] = []
            pages_dict[current_page].append(table_el)

    parse_time = time.time() - start_time

    pages_list = []
    total_elements = 0
    stats = {}
    for p_num in sorted(pages_dict.keys()):
        els = pages_dict[p_num]
        total_elements += len(els)
        for el in els:
            stats[el["type"]] = stats.get(el["type"], 0) + 1
        
        pages_list.append({
            "page": p_num,
            "page_label": f"стр. {p_num}",
            "elements": els
        })

    result = {
        "document_id": document_id,
        "source_file": docx_path.name,
        "source_type": "docx",
        "parser": "python-docx",
        "pages": pages_list,
        "metadata": {
            "total_pages": len(pages_list),
            "parse_time_sec": round(parse_time, 2),
            "elements_count": stats,
        },
    }

    logger.info(f"[docx_parser] ✅ {docx_path.name}: {stats} за {parse_time:.1f} сек, страниц: {len(pages_list)}")
    return result