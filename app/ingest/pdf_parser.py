"""
PDF → ParsedDocument.

Гибрид:
- pdfplumber — текст + таблицы
- Qwen2.5-VL — скриншоты и таблицы (для точного извлечения)

Колонтитулы вырезаются, но номер страницы берется ИЗ НИХ.
"""
import json
import re
import time
from pathlib import Path
from typing import Any

import pdfplumber
from loguru import logger

from app.ingest.document_id import make_document_id
from app.llm.vllm_client import vllm_client


# ==========================================================================
# Колонтитулы и мусор
# ==========================================================================
# Регулярка для поиска номера страницы в колонтитуле (например, "Страница 21 из 29")
PAGE_FOOTER_RE = re.compile(r"Страница\s+(\d+)\s+из\s+(\d+)", re.IGNORECASE)

# Регулярки для вырезания колонтитулов из текста
HEADER_PATTERNS = [
    re.compile(r"Инструкция\s*по\s*работе\s*с\s*Web-сервисом", re.IGNORECASE),
    re.compile(r"«Личный\s*кабинет\s*поставщика»", re.IGNORECASE),
    re.compile(r"\d{1,2}\s+[а-я]+\s+\d{4}\s*г\.", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"Страница\s+\d+\s+из\s+\d+", re.IGNORECASE),  # <-- ВЫРЕЗАЕМ КОЛОНТИТУЛ ИЗ ТЕКСТА
]


def clean_page_text(text: str) -> str:
    """Убирает колонтитулы и пустые строки."""
    if not text:
        return ""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(p.search(line) for p in HEADER_PATTERNS):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _is_garbage_text(text: str) -> bool:
    """
    Проверяет, является ли текст мусором (только цифры и пробелы).
    Например, страницы, где написано "1 1 1 1 1 1..." или "1 2 3 4 5...".
    """
    if not text:
        return True
    # Убираем все пробелы, переносы, точки
    cleaned = text.replace(" ", "").replace("\n", "").replace("\t", "").replace(".", "")
    # Если остались только цифры и их много — это мусор
    return cleaned.isdigit() and len(cleaned) > 20


# ==========================================================================
# VL-промпты
# ==========================================================================
VL_SCREENSHOT_PROMPT = """Это скриншот из инструкции по работе с Web-сервисом «Личный кабинет поставщика».

Опиши что на нём. Верни JSON строго в формате:
{
  "title": "название раздела/экрана (если видно)",
  "description": "подробное описание что изображено, пошагово",
  "ui_elements": ["кнопка ...", "поле ...", "меню ..."],
  "actions": ["что можно сделать на этом экране"],
  "visible_text": "дословный текст с экрана (надписи, кнопки, поля)"
}

Правила:
1. ui_elements — названия кнопок/полей/меню С КАРТИНКИ (на русском)
2. visible_text — весь видимый текст БЕЗ описания
3. description — что делает пользователь
4. Только JSON, без markdown-обёрток

Опиши скриншот:"""


VL_TABLE_PROMPT = """Извлеки таблицу со страницы документа.

Верни JSON строго:
{
  "tables": [
    {
      "caption": "подпись таблицы (если есть)",
      "headers": ["Колонка 1", "Колонка 2"],
      "rows": [["значение", "значение"], ...]
    }
  ]
}

Правила:
1. Текст извлекай ДОСЛОВНО.
2. Все числа, даты, единицы измерения — как на картинке.
3. Если несколько таблиц — все в массив.
4. Только JSON.

Извлеки таблицы:"""


# ==========================================================================
# Хелперы
# ==========================================================================
def _extract_json(text: str) -> dict[str, Any]:
    """Извлекает JSON из ответа LLM с попыткой починки."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Не найден JSON: {text[:200]}")
    
    json_str = text[start : end + 1]
    
    # Попытка 1: обычный json.loads
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"[pdf_parser] json.loads упал: {e}. Пытаюсь починить...")
    
    # Попытка 2: фикс забытых запятых
    fixed = re.sub(r'(["\d\]])\s*\n\s*(["\d\[])', r'\1,\n\2', json_str)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Попытка 3: json_repair
    try:
        from json_repair import repair_json
        return json.loads(repair_json(json_str))
    except ImportError:
        pass
    except Exception:
        pass
    
    logger.error(f"[pdf_parser] не удалось распарсить JSON: {json_str[:300]}")
    return {}


async def _vl_parse_screenshot(image_path: Path) -> list[dict[str, Any]]:
    try:
        response = await vllm_client.generate_with_image(
            image_path=image_path,
            prompt=VL_SCREENSHOT_PROMPT,
            max_tokens=2000,
            temperature=0.1,
        )
        parsed = _extract_json(response)
        if not parsed:
            return []
        
        elements = []
        if parsed.get("title"):
            elements.append({"type": "header", "level": 2, "text": parsed["title"]})
        if parsed.get("description"):
            elements.append({"type": "text", "text": parsed["description"]})
        if parsed.get("ui_elements"):
            elements.append({"type": "list", "items": parsed["ui_elements"]})
        if parsed.get("actions"):
            elements.append({
                "type": "text",
                "text": "Возможные действия: " + "; ".join(parsed["actions"]),
            })
        if parsed.get("visible_text"):
            elements.append({
                "type": "text",
                "text": f"Текст на экране: {parsed['visible_text']}",
            })
        return elements
    except Exception as e:
        logger.error(f"[pdf_parser] VL-ошибка (скриншот): {e}")
        return []


async def _vl_parse_tables(image_path: Path) -> list[dict[str, Any]]:
    try:
        response = await vllm_client.generate_with_image(
            image_path=image_path,
            prompt=VL_TABLE_PROMPT,
            max_tokens=3000,
            temperature=0.1,
        )
        parsed = _extract_json(response)
        if not parsed:
            return []
        
        elements = []
        for tbl in parsed.get("tables", []):
            rows = tbl.get("rows", [])
            if tbl.get("headers"):
                rows = [tbl["headers"]] + rows
            elements.append({
                "type": "table",
                "rows": rows,
                "caption": tbl.get("caption", ""),
            })
        return elements
    except Exception as e:
        logger.error(f"[pdf_parser] VL-ошибка (таблица): {e}")
        return []


# ==========================================================================
# Главный парсер
# ==========================================================================
async def parse_pdf(
    pdf_path: str | Path,
    dpi: int = 150,
    max_pages: int | None = None,
) -> dict[str, Any]:
    pdf_path = Path(pdf_path)
    document_id = make_document_id(pdf_path)
    start_time = time.time()

    logger.info(f"[pdf_parser] начинаю: {pdf_path.name}")

    pages: list[dict[str, Any]] = []
    vl_calls = 0

    tmp_dir = Path("data/processed") / document_id / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)
        if max_pages:
            total_pages = min(total_pages, max_pages)

        for i in range(total_pages):
            page = pdf.pages[i]
            physical_page_num = i + 1

            # 1. Извлекаем сырой текст (без очистки)
            raw_text = page.extract_text() or ""
            
            # 2. Ищем номер страницы в колонтитуле
            footer_match = PAGE_FOOTER_RE.search(raw_text)
            if footer_match:
                page_label_num = int(footer_match.group(1))
                page_num = page_label_num  # <-- ИСПОЛЬЗУЕМ НОМЕР ИЗ КОЛОНТИТУЛА
            else:
                page_num = physical_page_num

            # 3. Очищаем текст от колонтитулов
            text = clean_page_text(raw_text)

            # 4. Проверяем на мусор (только цифры)
            if _is_garbage_text(text):
                logger.debug(f"[pdf_parser] стр. {physical_page_num} (page={page_num}): мусор, пропускаем")
                continue

            tables = page.extract_tables()
            has_tables = bool(tables)
            text_len = len(text)
            used_vl = False

            elements: list[dict[str, Any]] = []

            if has_tables or text_len < 300:
                used_vl = True
                vl_calls += 1
                try:
                    pil_image = page.to_image(resolution=dpi).original
                    img_path = tmp_dir / f"page_{physical_page_num:03d}.png"
                    pil_image.save(img_path, "PNG")

                    if has_tables:
                        elements = await _vl_parse_tables(img_path)
                    else:
                        elements = await _vl_parse_screenshot(img_path)

                    if text and "Рисунок" in text:
                        for line in text.split("\n"):
                            if "Рисунок" in line:
                                elements.insert(0, {"type": "text", "text": line})
                                break
                    if text:
                        for line in text.split("\n"):
                            if re.match(r"^\d+(\.\d+)*\.?\s+[А-Я]", line):
                                elements.insert(0, {"type": "header", "level": 2, "text": line})
                                break
                except Exception as e:
                    logger.error(f"[pdf_parser] стр. {physical_page_num}: VL-ошибка: {e}")
                    if text:
                        elements = [{"type": "text", "text": text}]
            else:
                elements = [{"type": "text", "text": text}]

            pages.append({
                "page": page_num,  # <-- ИСПОЛЬЗУЕМ НОМЕР ИЗ КОЛОНТИТУЛА
                "page_label": f"стр. {page_num}",
                "elements": elements,
                "source": "vl" if used_vl else "pdfplumber",
            })

            if physical_page_num % 10 == 0 or physical_page_num == total_pages:
                logger.info(f"[pdf_parser] {physical_page_num}/{total_pages} (VL: {vl_calls}, page={page_num})")

    # Чистка временных файлов
    for f in tmp_dir.glob("*.png"):
        f.unlink()
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    parse_time = time.time() - start_time

    result = {
        "document_id": document_id,
        "source_file": pdf_path.name,
        "source_type": "pdf",
        "parser": "pdfplumber+vl",
        "pages": pages,
        "metadata": {
            "total_pages": len(pages),
            "dpi": dpi,
            "parse_time_sec": round(parse_time, 2),
            "vl_calls": vl_calls,
        },
    }

    logger.info(
        f"[pdf_parser] ✅ {pdf_path.name}: {len(pages)} стр. "
        f"(VL: {vl_calls}) за {parse_time:.1f} сек"
    )
    return result


# ==========================================================================
# CLI
# ==========================================================================
if __name__ == "__main__":
    import asyncio
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m app.ingest.pdf_parser <path.pdf>")
        sys.exit(1)

    async def main():
        result = await parse_pdf(sys.argv[1])
        print(json.dumps(result, ensure_ascii=False, indent=2)[:4000])

    asyncio.run(main())