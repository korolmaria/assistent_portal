"""
Synthesis-агент: собирает финальный ответ через Qwen3-8B.
"""
import json
import re
from typing import Any

from loguru import logger

from app.agents.memory import format_history, format_summary
from app.llm.vllm_client import vllm_client
from app.schemas import ChatHistoryItem, Source


SYSTEM_PROMPT = """Ты — ИИ-помощник для поставщиков торговой сети «Евроторг».
Твоя задача — отвечать на вопросы пользователя, используя предоставленный контекст и историю диалога.

ПРАВИЛА ОТВЕТА:
1. Отвечай на основе КОНТЕКСТА или ПАМЯТИ ДИАЛОГА (истории). Не выдумывай факты, которых нет ни там, ни там.
2. Если ответа нет ни в контексте, ни в истории диалога — так и скажи: «В предоставленных данных нет ответа».
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
   - Для данных из API и для ответов из истории диалога: без ссылок вообще.

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

9. ДАННЫЕ ИЗ API (tool-запросы):
   Если тебе даны данные о заказах (в карточке/таблице) — ОТВЕЧАЙ НА ВОПРОС,
   используя эти данные. 
   
   ⚠️ ГЛАВНОЕ ПРАВИЛО: Сначала пойми, ЧТО СПРАШИВАЕТ пользователь, и ответь ИМЕННО НА ЭТО.
   • Если спрашивают «могу ли подтвердить / отменить / изменить?» — 
     ответь «Да, можете» или «Нет, нельзя» (на основе статуса из данных).
     НЕ пиши про сумму, если спросили про возможность.
   • Если спрашивают «какая сумма / статус / дата?» — 
     ответь конкретным значением. Начни с ответа, не пиши «Какая сумма?».
   • Если спрашивают «есть ли товар?» — 
     ответь «Да, есть...» или «Нет, не найдено».
   • Если вопрос общий («расскажи», «покажи») — дай краткую сводку.
   
   • НЕ пересказывай ВСЕ поля таблицы. Скажи главное.
   • НИКОГДА не дублируй вопрос пользователя в ответе (не пиши «Какая сумма?» перед суммой).
   • Максимум 2-3 предложения.

10. ПАМЯТЬ ДИАЛОГА (ОЧЕНЬ ВАЖНО):
    В начале промпта есть блоки «КРАТКОЕ СОДЕРЖАНИЕ» и/или «ПОСЛЕДНИЕ СООБЩЕНИЯ».
    Если пользователь задаёт вопрос о фактах ИЗ ДИАЛОГА (например, «как меня зовут?», 
    «что я спрашивал ранее?»), ОТВЕЧАЙ ИЗ ИСТОРИИ. 
    Если вопрос про инструкции/заказы — ищи факты в блоке КОНТЕКСТ.

СТИЛЬ:
- Профессиональный, деловой.
- Конкретика: числа, названия, статусы, методы, URL.
- Без воды.
"""


def normalize_citations(text: str) -> str:
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
    pattern = r"\[([a-zA-Z0-9_\-]+):p(\d+)\]"
    return [m.group(0) for m in re.finditer(pattern, text)]


def clean_citations_for_display(text: str) -> str:
    return re.sub(r"\s*\[[a-zA-Z0-9_\-]+(?::p\d+)?\]", "", text).strip()


async def synthesize(
    query: str,
    rag_chunks: list[dict[str, Any]] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    extra_text: str | None = None,
    max_tokens: int = 2500,
    history: list[ChatHistoryItem] | None = None,
    summary: str | None = None,
) -> tuple[str, list[Source], list[str]]:
    rag_chunks = rag_chunks or []
    tool_results = tool_results or []
    q_lower = query.lower()

    # === 1. Мета-вопросы о прошлом диалоге ===
    if any(kw in q_lower for kw in ["про какой заказ", "о чем я", "о чём я", "что я спрашивал", "что ты написал"]):
        logger.info("[synthesis] перехвачен мета-вопрос о диалоге")
        history_text = format_history(history or [])
        prompt = f"""=== ПОСЛЕДНИЕ СООБЩЕНИЯ ===
{history_text}

=== ТЕКУЩИЙ ВОПРОС ===
{query}

Ответь на вопрос пользователя, опираясь ТОЛЬКО на историю сообщений выше. 
Если в истории есть номер заказа, назови его.
Не используй ссылки на документы.
Не пиши слово "Ответ:", просто дай текст.

Ответ:"""
        answer = await vllm_client.generate_text(
            prompt=prompt, 
            system="Ты — внимательный ассистент.", 
            max_tokens=300
        )
        return answer.strip(), [], []

    # === 2. Социальные запросы (имя) ===
    is_social = any(kw in q_lower for kw in ["меня зовут", "как меня зовут", "моё имя", "мое имя"])
    
    if is_social and not tool_results:
        logger.info("[synthesis] обнаружен социальный запрос, пропускаем RAG-контекст")
        history_text = format_history(history or [])
        summary_text = format_summary(summary)
        memory_block = ""
        if summary_text:
            memory_block += f"=== КРАТКОЕ СОДЕРЖАНИЕ ПРЕДЫДУЩЕГО ДИАЛОГА ===\n{summary_text}\n\n"
        if history_text:
            memory_block += f"=== ПОСЛЕДНИЕ СООБЩЕНИЯ ===\n{history_text}\n\n"

        prompt = f"""{memory_block}=== ТЕКУЩИЙ ВОПРОС ===
{query}

Ответь вежливо, используя историю диалога.
Если пользователь сообщает имя — запомни его и поприветствуй.
Если спрашивает имя, а его нет в истории — скажи, что не знаешь.
НЕ ссылайся на документы. Это просто светская беседа.

Ответ:"""

        answer = await vllm_client.generate_text(
            prompt=prompt,
            system="Ты — вежливый ИИ-ассистент.",
            max_tokens=200,
            temperature=0.1,
        )
        return answer.strip(), [], []

    if not rag_chunks and not tool_results and not extra_text:
        return ("К сожалению, у меня нет информации по этому вопросу.", [], [])

    # === 3. Собираем контекст ===
    context_parts: list[str] = []
    sources: list[Source] = []
    tools_used: list[str] = []

    # 3.1 RAG
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

    # 3.2 Tool
    if tool_results:
        context_parts.append("\n=== ДАННЫЕ ИЗ API ===\n")
        for tr in tool_results:
            tool_name = tr.get("tool_name", "unknown")
            tools_used.append(tool_name)
            text = tr.get("text", "")
            data = tr.get("data")
            
            if tool_name == "get_order_items" and data:
                items_count = len(data.get("data", []))
                context_parts.append(f"Tool: {tool_name} (найдено {items_count} товаров)")
            else:
                context_parts.append(f"Tool: {tool_name}")
                
            if text:
                context_parts.append(text)
            if data:
                context_parts.append(json.dumps(data, ensure_ascii=False)[:2000])
            context_parts.append("")

    if extra_text:
        context_parts.append("\n=== ДОПОЛНИТЕЛЬНО ===\n")
        context_parts.append(extra_text)

    context = "\n".join(context_parts)

    # === 4. Инструкция ===
    has_rag = bool(rag_chunks)
    has_tool = bool(tool_results)
    has_display = any(tr.get("data") for tr in tool_results)

    is_technical = any(kw in q_lower for kw in [
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
            "ОБЯЗАТЕЛЬНО назови КОНКРЕТНЫЙ метод API, URL и параметры запроса. "
            "Ссылки в формате [doc_id:pN]."
        )
    elif has_rag:
        instruction = (
            "В контексте есть фрагменты из документов. "
            "Отвечай на их основе. Ссылки ТОЛЬКО в формате [doc_id:pN]."
        )
    elif has_tool and has_display:
        instruction = (
            "⚠️ ВАЖНО: данные о заказах УЖЕ ПОКАЗАНЫ пользователю "
            "в структурированном виде (карточка или таблица) на фронте. "
            "Твоя задача — ОТВЕТИТЬ НА ВОПРОС ПОЛЬЗОВАТЕЛЯ кратко и по делу.\n\n"
            "ПРАВИЛА:\n"
            "• Если вопрос про конкретное поле (статус, сумма, дата, склад, "
            "бренд, контрагент, адрес) — ответь ТОЛЬКО на этот вопрос одной фразой.\n"
            "• Если вопрос про возможность («могу ли», «можно ли») — "
            "ответь «Да, можете» или «Нет, нельзя» на основе статуса.\n"
            "• Если вопрос общий («расскажи», «покажи») — дай КРАТКУЮ сводку "
            "(1-2 предложения).\n"
            "• НЕ пересказывай ВСЕ поля карточки — только релевантные вопросу.\n"
            "• Максимум 2-3 предложения."
        )
    elif has_tool and not has_display:
        instruction = (
            "В контексте есть данные из API. "
            "Опиши кратко, что найдено. Без ссылок на документы."
        )
    else:
        instruction = (
            "В контексте есть только данные из API. "
            "Ответь на их основе, без ссылок на документы."
        )

    # === 5. Память ===
    history_text = format_history(history or [])
    summary_text = format_summary(summary)

    memory_block = ""
    if summary_text:
        memory_block += (
            "=== КРАТКОЕ СОДЕРЖАНИЕ ПРЕДЫДУЩЕГО ДИАЛОГА ===\n"
            f"{summary_text}\n\n"
        )
    if history_text:
        memory_block += (
            "=== ПОСЛЕДНИЕ СООБЩЕНИЯ ===\n"
            f"{history_text}\n\n"
        )

    # === 6. Промпт ===
    prompt = f"""{memory_block}
=== ТЕКУЩИЙ ВОПРОС ПОЛЬЗОВАТЕЛЯ ===
{query}

{instruction}

ВАЖНО: ссылки на документы только в формате [doc_id:pN].
Если в контексте есть технические детали (методы API, URL, JSON-поля) —
приводи их ДОСЛОВНО, не заменяй общими словами.

КОНТЕКСТ:
{context}

Ответ:"""

    # === 7. max_tokens ===
    actual_max_tokens = max_tokens
    if has_tool and not has_rag and has_display:
        actual_max_tokens = 1000

    logger.info(
        f"[synthesis] Qwen3-8B: {len(rag_chunks)} rag, "
        f"{len(tool_results)} tool, {len(context)} симв., "
        f"memory={len(history or [])} msgs + "
        f"{'summary' if summary else 'no-summary'}, "
        f"technical={is_technical}, has_display={has_display}, "
        f"max_tokens={actual_max_tokens}"
    )

    answer = await vllm_client.generate_text(
        prompt=prompt,
        system=SYSTEM_PROMPT,
        max_tokens=actual_max_tokens,
        temperature=0.1,
    )

    # === 8. Постпроцессинг ===
    answer = normalize_citations(answer)

    logger.info(f"[synthesis] ответ: {len(answer)} симв.")
    return answer.strip(), sources, tools_used