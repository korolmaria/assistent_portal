"""
Synthesis-агент: собирает финальный ответ через Qwen3-8B.

Поддерживает:
- только RAG (документы)
- только Tool (данные из API)
- комбинированный (RAG + Tool)

Особенности:
- Нормализация цитат [doc_id:pN]
- Приоритет технических деталей (методы API, URL, JSON-поля)
- Разные инструкции для rag/tool/rag+tool
"""
import json
import re
from typing import Any

from loguru import logger

from app.llm.vllm_client import vllm_client
from app.schemas import Source


# ==========================================================================
# Системный промпт
# ==========================================================================
SYSTEM_PROMPT = """Ты — ИИ-помощник для поставщиков торговой сети «Евроторг».
Твоя задача — отвечать на вопросы пользователя, используя предоставленный контекст.

ПРАВИЛА ОТВЕТА:
1. Отвечай ТОЛЬКО на основе контекста. Не выдумывай.
2. Если в контексте нет ответа — так и скажи: «В предоставленных данных нет ответа».
3. Пиши кратко, по делу, на русском языке.
4. Если ответ — процедура/шаги, оформляй списком.
5. Если ответ — таблица или числовые данные, приводи точно.

6. ССЫЛКИ НА ИСТОЧНИКИ (критически важно):
   - Для фрагментов документов используй ТОЛЬКО формат [doc_id:pN]
   - doc_id — идентификатор (без слова "doc_id")
   - N — номер страницы (без слова "стр.")
   - НЕ пиши "| стр. N", "стр. N", "doc_id:"
   - Правильно: [trebovaniya_10fc:p1]
   - Неправильно: [doc_id:trebovaniya_10fc | стр. 1]
   - Для данных из API: без ссылок вообще.

7. Если контекст содержит И правила из документов, И данные из API:
   - Сначала объясни правило/регламент со ссылкой
   - Потом покажи конкретные данные
   - Сделай вывод

8. ТЕХНИЧЕСКИЕ ДЕТАЛИ (ВАЖНО!):
   Если в контексте упоминаются технические термины — ОБЯЗАТЕЛЬНО приводи их дословно:
     • названия методов API (например, SupplierPortal.ConfirmOrderBuyGroup)
     • URL эндпоинты (POST https://...)
     • названия полей JSON (OrderBuyGroupId, ConfirmType, Checkx)
     • HTTP-методы (POST, GET)
     • статусы (10, 50, 60, 90)
     • названия кнопок интерфейса («Подтвердить поставку»)

   НЕ заменяй технические термины на общие слова («метод», «запрос», «API»).
   Если пользователь спрашивает «как технически» / «каким методом» / «как через API» —
   он ЖДЁТ:
     - название метода
     - URL
     - параметры запроса (JSON)

   Пример правильного ответа:
   «Для подтверждения заказа используется метод
   SupplierPortal.ConfirmOrderBuyGroup. POST-запрос отправляется на
   https://[ip]:[port]/SupplierPortal.ConfirmOrderBuyGroup с параметрами:
   OrderBuyGroupId: ["75215840"], ConfirmType: "1" [api_instruktsiya_ea8f:p1].»

9. ПРИОРИТЕТ ИСТОЧНИКОВ:
   - Если запрос про «как технически», «какой метод», «как через API» —
     В ПЕРВУЮ ОЧЕРЕДЬ используй чанки из API_инструкции / технической документации.
   - Если запрос про «как в интерфейсе», «какие кнопки» —
     используй инструкции по интерфейсу.
   - Если есть И то, И другое — приведи ОБА варианта:
     сначала технический (метод, URL, JSON), затем интерфейсный (кнопки, шаги).

10. НЕ ОБРЫВАЙ ОТВЕТ НА ПОЛУСЛОВЕ. Если не хватает места —
    сократи детали, но заверши мысль.

СТИЛЬ:
- Профессиональный, деловой.
- Конкретика: числа, названия, статусы, методы, URL.
- Без воды.
"""


# ==========================================================================
# Нормализация цитат
# ==========================================================================
def normalize_citations(text: str) -> str:
    """
    Приводит все форматы цитат к [doc_id:pN].

    Преобразует:
      [doc_id:xxx | стр. 1]  → [xxx:p1]
      [doc_id:xxx:стр.1]     → [xxx:p1]
      [xxx:стр.1]            → [xxx:p1]
      [xxx | стр. 1]         → [xxx:p1]
      [doc_id:xxx:p1]        → [xxx:p1]
      [doc_id:xxx]           → [xxx]
      [xxx:стр. N]           → [xxx:pN]
    """
    text = re.sub(
        r"\[doc_id:([a-zA-Z0-9_\-]+)\s*[|:]\s*стр\.\s*(\d+)\]",
        r"[\1:p\2]",
        text,
    )
    text = re.sub(
        r"\[([a-zA-Z0-9_\-]+):стр\.\s*(\d+)\]",
        r"[\1:p\2]",
        text,
    )
    text = re.sub(
        r"\[([a-zA-Z0-9_\-]+)\s*\|\s*стр\.\s*(\d+)\]",
        r"[\1:p\2]",
        text,
    )
    text = re.sub(
        r"\[doc_id:([a-zA-Z0-9_\-]+):p(\d+)\]",
        r"[\1:p\2]",
        text,
    )
    text = re.sub(
        r"\[doc_id:([a-zA-Z0-9_\-]+)\]",
        r"[\1]",
        text,
    )
    text = re.sub(
        r"\[([a-zA-Z0-9_\-]+):\s*стр\.\s*(\d+)\]",
        r"[\1:p\2]",
        text,
    )
    return text


def extract_citations(text: str) -> list[str]:
    """Вытаскивает цитаты [doc_id:pN] из ответа."""
    pattern = r"\[([a-zA-Z0-9_\-]+):p(\d+)\]"
    return [m.group(0) for m in re.finditer(pattern, text)]


def clean_citations_for_display(text: str) -> str:
    """Убирает все цитаты из текста (для отображения без них)."""
    return re.sub(r"\s*\[[a-zA-Z0-9_\-]+(?::p\d+)?\]", "", text).strip()


# ==========================================================================
# Главная функция
# ==========================================================================
async def synthesize(
    query: str,
    rag_chunks: list[dict[str, Any]] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    extra_text: str | None = None,
    max_tokens: int = 2500,
) -> tuple[str, list[Source], list[str]]:
    """
    Собирает финальный ответ.

    Args:
        query: вопрос пользователя
        rag_chunks: чанки от RAG-агента
        tool_results: [{tool_name, text, data}, ...] от Tool-агента
        extra_text: доп. текст
        max_tokens: лимит ответа

    Returns:
        (answer, sources, tools_used)
    """
    rag_chunks = rag_chunks or []
    tool_results = tool_results or []

    if not rag_chunks and not tool_results and not extra_text:
        return (
            "К сожалению, у меня нет информации по этому вопросу.",
            [],
            [],
        )

    # === Собираем контекст ===
    context_parts: list[str] = []
    sources: list[Source] = []
    tools_used: list[str] = []

    # 1. RAG-фрагменты
    if rag_chunks:
        context_parts.append("=== ФРАГМЕНТЫ ИЗ ДОКУМЕНТОВ ===\n")
        for i, ch in enumerate(rag_chunks, 1):
            header = f"[{i}] doc_id={ch.get('document_id', '')}"
            if ch.get("page"):
                header += f" | стр. {ch['page']}"
            if ch.get("section"):
                header += f" | раздел: {ch['section']}"
            context_parts.append(f"{header}\n{ch['text']}\n")

            sources.append(Source(
                doc_id=ch.get("document_id", ""),
                doc_title=ch.get("source_file", ""),
                page=ch.get("page"),
                page_label=ch.get("page_label"),
                section=ch.get("section", ""),
                content_type=ch.get("element_type", "text"),
                snippet=ch["text"][:200],
                grade_score=ch.get("score"),
            ))

    # 2. Tool-результаты
    if tool_results:
        context_parts.append("\n=== ДАННЫЕ ИЗ API ===\n")
        for tr in tool_results:
            tool_name = tr.get("tool_name", "unknown")
            tools_used.append(tool_name)
            text = tr.get("text", "")
            data = tr.get("data")
            context_parts.append(f"Tool: {tool_name}")
            if text:
                context_parts.append(text)
            if data:
                context_parts.append(json.dumps(data, ensure_ascii=False)[:2000])
            context_parts.append("")

    # 3. Extra text
    if extra_text:
        context_parts.append("\n=== ДОПОЛНИТЕЛЬНО ===\n")
        context_parts.append(extra_text)

    context = "\n".join(context_parts)

    # === Определяем инструкцию ===
    has_rag = bool(rag_chunks)
    has_tool = bool(tool_results)

    # Технический ли запрос?
    is_technical = any(kw in query.lower() for kw in [
        "технически", "метод", "api", "апи", "запрос", "endpoint",
        "url", "json", "параметр",
    ])

    if has_rag and has_tool:
        instruction = (
            "В контексте есть И правила из документов, И данные из API. "
            "Сначала объясни правило со ссылкой [doc_id:pN], "
            "потом приведи данные из API, в конце сделай вывод."
        )
    elif has_rag and is_technical:
        instruction = (
            "⚠️ ЗАПРОС ТЕХНИЧЕСКИЙ! Пользователь спрашивает про метод/API/технику. "
            "В контексте есть техническая документация (API_инструкция). "
            "ОБЯЗАТЕЛЬНО назови КОНКРЕТНЫЙ метод API (например, "
            "SupplierPortal.ConfirmOrderBuyGroup), URL и параметры запроса. "
            "Ссылки в формате [doc_id:pN]."
        )
    elif has_rag:
        instruction = (
            "В контексте есть фрагменты из документов. "
            "Отвечай на их основе. Ссылки ТОЛЬКО в формате [doc_id:pN]."
        )
    else:
        instruction = (
            "В контексте есть только данные из API. "
            "Ответь на их основе, без ссылок на документы."
        )

    # === Промпт ===
    prompt = f"""ВОПРОС: {query}

{instruction}

ВАЖНО: ссылки на документы только в формате [doc_id:pN].
Например: [trebovaniya_10fc:p1]

Если в контексте есть технические детали (методы API, URL, JSON-поля) —
приводи их ДОСЛОВНО, не заменяй общими словами.

КОНТЕКСТ:
{context}

Ответ:"""

    logger.info(
        f"[synthesis] Qwen3-8B: {len(rag_chunks)} rag, "
        f"{len(tool_results)} tool, {len(context)} симв., "
        f"technical={is_technical}"
    )

    answer = await vllm_client.generate_text(
        prompt=prompt,
        system=SYSTEM_PROMPT,
        max_tokens=max_tokens,
        temperature=0.1,
    )

    # === Постпроцессинг ===
    answer = normalize_citations(answer)

    logger.info(f"[synthesis] ответ: {len(answer)} симв.")
    return answer.strip(), sources, tools_used
