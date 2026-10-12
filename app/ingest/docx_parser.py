"""
DOCX → ParsedDocument.

Порядок элементов: параграф → таблица → параграф.
Типы элементов:
- header — заголовки (z1qcye, ALL CAPS, N. Текст, короткие без точки)
- accent — жирные абзацы (z1qcye), отдельная мысль
- text — обычный текст (включая JSON-код из API-инструкций)
- list — параграф-список (начинается с •, -, *)
- table — таблица (структурная, из python-docx)
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
    """
    ВСЁ ЗАГЛАВНЫМИ (заголовок раздела).
    Только для буквенных строк, исключая JSON-код и мусор.
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3:
        return False
    # Если >20% символов не буквы/пробелы/цифры — это код, не заголовок
    non_alpha = sum(
        1 for c in text
        if not (c.isalpha() or c.isspace() or c.isdigit())
    )
    if len(text) > 0 and non_alpha / len(text) > 0.2:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) > 0.8


def _is_section_number(text: str) -> bool:
    """Начинается с 'N.' или 'N.N.' (нумерованный раздел)."""
    return bool(re.match(r"^\d+(\.\d+)*\.?\s+\S", text))


def _is_figure_caption(text: str) -> bool:
    """Подпись рисунка: 'Рисунок 1 — ...'."""
    return bool(re.match(r"^Рисунок\s+\d+", text, re.IGNORECASE))


def _is_list_item(text: str) -> bool:
    """Начинается с •, -, *, → — элемент списка."""
    return bool(re.match(r"^[•\-\*→]\s+\S", text))


def _is_code_like(text: str) -> bool:
    """
    Похоже на JSON-код, а не на заголовок.
    Признаки:
    - начинается с {, }, [, ], ", '
    - содержит >25% не-буквенных символов
    """
    stripped = text.lstrip()
    if stripped and stripped[0] in '{[}"\'':
        return True
    if len(text) > 0:
        non_alpha = sum(
            1 for c in text
            if not (c.isalpha() or c.isspace() or c.isdigit())
        )
        if non_alpha / len(text) > 0.25:
            return True
    return False


def _classify_paragraph(text: str, style_name: str) -> str:
    """
    Возвращает тип элемента: header | accent | list | text.
    Ужесточённая эвристика — исключаем JSON-код и мусор.
    """
    text = text.strip()

    # z1qcye — жирный акцент
    if style_name == 'z1qcye':
        return 'accent'

    # ALL CAPS → header (только буквенные строки)
    if _is_all_caps(text):
        return 'header'

    # Нумерованный раздел → header
    if _is_section_number(text):
        return 'header'

    # Подпись рисунка → text (не заголовок)
    if _is_figure_caption(text):
        return 'text'

    # Список
    if _is_list_item(text):
        return 'list'

    # JSON-код и мусор → text
    if _is_code_like(text):
        return 'text'

    # Очень короткая строка без точки в конце → header
    # УЖЕСТОЧЕНО: ≤ 8 слов, ≤ 80 символов, начинается с буквы
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
    """python-docx Table → dict с rows."""
    rows = []
    for row in table.rows:
        cells = [c.text.strip() for c in row.cells]
        rows.append(cells)
    # Убираем полностью пустые строки
    rows = [r for r in rows if any(c for c in r)]
    return {
        "type": "table",
        "rows": rows,
        "num_rows": len(rows),
        "num_cols": len(rows[0]) if rows else 0,
    }


def _iter_block_items(doc: _Doc):
    """
    Итерирует параграфы и таблицы в исходном порядке документа.
    Стандартный python-docx не сохраняет порядок — это обходной путь.
    """
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    parent = doc.element.body
    for child in parent.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, doc)
        elif child.tag == qn('w:tbl'):
            yield Table(child, doc)


def parse_docx(docx_path: str | Path) -> dict[str, Any]:
    """
    DOCX → ParsedDocument.
    Сохраняет порядок элементов: параграф → таблица → параграф.
    """
    docx_path = Path(docx_path)
    document_id = make_document_id(docx_path)
    start_time = time.time()

    logger.info(f"[docx_parser] начинаю: {docx_path.name}")

    doc = Document(str(docx_path))
    elements: list[dict[str, Any]] = []

    current_section = ""
    current_section_slug = ""

    # Проходим параграфы и таблицы в порядке
    for block in _iter_block_items(doc):
        if block.__class__.__name__ == 'Paragraph':
            text = block.text.strip()
            if not text:
                continue

            style_name = block.style.name if block.style else 'Normal'
            etype = _classify_paragraph(text, style_name)

            el: dict[str, Any] = {
                "type": etype,
                "text": text,
                "style": style_name,
            }

            # Для header — обновляем текущий раздел
            if etype == 'header':
                current_section = text
                current_section_slug = make_section_slug(text)
                el["section"] = current_section
            else:
                el["section"] = current_section

            elements.append(el)

        elif block.__class__.__name__ == 'Table':
            table_el = _extract_table(block)
            table_el["section"] = current_section
            elements.append(table_el)
            logger.debug(
                f"[docx_parser] таблица "
                f"{table_el['num_rows']}x{table_el['num_cols']}"
            )

    parse_time = time.time() - start_time

    # Статистика
    stats: dict[str, int] = {}
    for el in elements:
        stats[el["type"]] = stats.get(el["type"], 0) + 1

    result = {
        "document_id": document_id,
        "source_file": docx_path.name,
        "source_type": "docx",
        "parser": "python-docx",
        "pages": [
            {
                "page": 1,  # DOCX не имеет чётких страниц
                "elements": elements,
            }
        ],
        "metadata": {
            "total_pages": 1,
            "parse_time_sec": round(parse_time, 2),
            "elements_count": stats,
        },
    }

    logger.info(
        f"[docx_parser] ✅ {docx_path.name}: "
        f"{stats} за {parse_time:.1f} сек"
    )
    return result


# ==========================================================================
# CLI
# ==========================================================================
if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m app.ingest.docx_parser <path.docx>")
        sys.exit(1)

    result = parse_docx(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2)[:4000])
