"""
Synthesis-агент: собирает финальный ответ через Qwen3-8B.
"""
import json
import re
from typing import Any, Literal

from loguru import logger

from app.agents.memory import format_history, format_summary
from app.config import settings
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
   
   ⚠️ КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО:
     • Придумывать URL-адреса, которых нет в контексте.
     • Придумывать названия методов API, которых нет в контексте.
     • Придумывать значения параметров (например, "Full", "true"), которых нет в контексте.
     • Если в контексте для параметра написано `"[ConfirmType]"`, то так и пиши, НЕ заменяй на "Full" или "Partial".

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

11. ОТВЕТЫ НА ВОПРОСЫ ПО ТАБЛИЦАМ (ЗАКАЗЫ, ТОВАРЫ, ПРЕТЕНЗИИ):
    В контексте могут быть таблицы с данными (заказы, товары, претензии).
    Если пользователь задаёт КОНКРЕТНЫЙ вопрос (когда, какой, сколько, почему, есть ли):
    • СНАЧАЛА ответь на вопрос ОДНИМ предложением, найдя нужное значение в таблице.
    • НЕ нужно пересказывать всю таблицу или писать «Найдено 7 позиций».
    • Пользователь сам увидит таблицу под твоим ответом.
    
    Пример: «Когда создана претензия?» → «Претензия создана 29.06.2026.»
    Пример: «Какой товар самый дорогой?» → «Самый дорогой товар — Хлеб "ФИТНЕС" (1.43 BYN).»
    Пример: «Сколько всего недопоставили?» → «Всего недопоставлено 7 единиц товара.»

12. АНАЛИТИЧЕСКИЙ БЛОК:
    Если в контексте есть блок «=== АНАЛИЗ ===», используй его как ОСНОВУ ответа.
    Не пересказывай данные таблицы заново — опирайся на выводы аналитика.
    Если аналитик указал числа/проценты — приведи их в ответе.

13. ПРИОРИТЕТ АНАЛИЗА (КРИТИЧЕСКИ ВАЖНО):
    Если в контексте присутствует блок «=== АНАЛИЗ ===», это означает, что другой агент
    УЖЕ изучил данные и сделал выводы. Твоя задача — ПЕРЕСКАЗАТЬ ЭТИ ВЫВОДЫ пользователю.
    Даже если в таблице данных нет прямого ответа, но в блоке «=== АНАЛИЗ ===» написано,
    что данных недостаточно — ты ОБЯЗАН сообщить пользователю именно это («Аналитик сообщает, что...»).
    НИКОГДА не пиши «В предоставленных данных нет ответа», если блок АНАЛИЗ не пуст.

14. ЕСЛИ НЕ УВЕРЕН В НОМЕРЕ СТРАНИЦЫ:
    Если ты не можешь точно определить номер страницы (page) из контекста — 
    НЕ пиши ссылку [doc_id:pN] вообще. Лучше ответ без ссылки, чем с неправильной.

15. URL В КОНТЕКСТЕ:
    Если в контексте указан URL (например, `https://rest.eurotorg.by/10195/Json` или `https://rest.eurotorg.by/10165/Json`), 
    приведи его ДОСЛОВНО, без изменений.

СТИЛЬ:
- Профессиональный, деловой.
- Конкретика: числа, названия, статусы, методы, URL.
- Без воды.
"""


def normalize_citations(text: str) -> str:
    text = re.sub(r"\[doc_id:([a-zA-Z0-9_\-]+)\s*[|:]\s*стр\.\s*(\d+)\]", r"[\1:p\2]", text)
    text = re.sub(r"\[([a-zA-Z0-9_\-]+):стр\.\s*(\d+)\]", r"[\1:p\2]", text)
    text = re.sub(r"\[([a-zA-Z0-9_\-]+)\s*\|\s*стр\.\s*(\d+)\]", r"[\1:p\2]", text)
    text = re.sub(r"\[doc_id:([a-zA-Z0-9_\-]+):p(\d+)\]", r"[\1:p\2]", text)
    text = re.sub(r"\[doc_id:([a-zA-Z0-9_\-]+)\]", r"[\1]", text)
    text = re.sub(r"\[([a-zA-Z0-9_\-]+):\s*стр\.\s*(\d+)\]", r"[\1:p\2]", text)
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
    mode: Literal["verbatim", "explain", "analysis"] = "explain",
    analysis: str | None = None,
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
            prompt=prompt, system="Ты — внимательный ассистент.", max_tokens=300,
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
            prompt=prompt, system="Ты — вежливый ИИ-ассистент.", max_tokens=200, temperature=0.1,
        )
        return answer.strip(), [], []

    if not rag_chunks and not tool_results and not extra_text:
        return ("К сожалению, у меня нет информации по этому вопросу.", [], [])

    # === 3. Контекст (с ограничением по символам и заменой URL) ===
    context_parts: list[str] = []
    sources: list[Source] = []
    tools_used: list[str] = []

    MAX_CONTEXT_CHARS = 20000
    current_context_len = 0
    api_url = settings.supplier_portal_url  # <-- URL из config.py

    if rag_chunks:
        context_parts.append("=== ФРАГМЕНТЫ ИЗ ДОКУМЕНТОВ ===\n")
        for i, ch in enumerate(rag_chunks, 1):
            chunk_text = ch.get("text", "")
            if current_context_len + len(chunk_text) > MAX_CONTEXT_CHARS:
                logger.warning(f"[synthesis] контекст обрезан на чанке {i} (лимит {MAX_CONTEXT_CHARS} симв.)")
                break

            # ⚠️ ЗАМЕНА [ip]:[port] НА URL ИЗ ОКРУЖЕНИЯ
            chunk_text = chunk_text.replace("[ip]:[port]", api_url)
            chunk_text = chunk_text.replace("[ip]", api_url)
            chunk_text = chunk_text.replace("[port]", "")

            header = f"[{i}] doc_id={ch.get('document_id', '')}"
            if ch.get("page"):
                header += f" | стр. {ch['page']}"
            if ch.get("section"):
                header += f" | раздел: {ch['section']}"
            block = f"{header}\n{chunk_text}\n"
            context_parts.append(block)
            current_context_len += len(block)

            sources.append(Source(
                doc_id=ch.get("document_id", ""),
                doc_title=ch.get("source_file", ""),
                page=ch.get("page"),
                page_label=ch.get("page_label"),
                section=ch.get("section", ""),
                content_type=ch.get("element_type", "text"),
                snippet=chunk_text[:200],
                grade_score=ch.get("score"),
            ))

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
                context_parts.append(json.dumps(data, ensure_ascii=False)[:8000])
            context_parts.append("")

    if analysis:
        context_parts.append("\n=== АНАЛИЗ ===\n")
        context_parts.append(analysis)
        context_parts.append("")

    if extra_text:
        context_parts.append("\n=== ДОПОЛНИТЕЛЬНО ===\n")
        context_parts.append(extra_text)

    context = "\n".join(context_parts)

    # === 4. Инструкция (учитывает mode) ===
    has_rag = bool(rag_chunks)
    has_tool = bool(tool_results)
    has_display = any(tr.get("data") for tr in tool_results)

    is_technical = any(kw in q_lower for kw in [
        "технически", "метод", "api", "апи", "запрос", "endpoint",
        "url", "json", "параметр",
    ])

    if mode == "analysis" and analysis:
        instruction = (
            "⚠️ КРИТИЧЕСКИ ВАЖНО: в контексте есть блок «=== АНАЛИЗ ===» — это уже готовый "
            "аналитический вывод по данным, сделанный отдельным агентом.\n\n"
            "ТВОЯ ЗАДАЧА: пересказать этот АНАЛИЗ в ответ пользователю. "
            "Опирайся на него как на ГЛАВНЫЙ источник. Таблица данных — вспомогательная.\n\n"
            "ПРАВИЛА:\n"
            "• Если в АНАЛИЗЕ есть числа, проценты, сравнения — ОБЯЗАТЕЛЬНО приведи их.\n"
            "• Если в АНАЛИЗЕ написано, что данных недостаточно (например, «за сентябрь данных нет») — "
            "так и напиши пользователю: «Аналитик сообщает, что...».\n"
            "• НИКОГДА не пиши «В предоставленных данных нет ответа», если в блоке АНАЛИЗ "
            "есть хоть какой-то текст. Анализ — это и есть ответ.\n"
            "• НЕ пересказывай таблицу заново — она уже показана пользователю на фронте.\n"
            "• Ответ — 2-4 предложения."
        )
    elif mode == "verbatim" and has_display and not has_rag:
        instruction = (
            "Данные уже показаны пользователю в структурированном виде (карточка/таблица). "
            "Дай КРАТКУЮ сводку (1-2 предложения) без пересказа всех полей."
        )
    elif has_rag and has_tool:
        instruction = (
            "В контексте есть И правила из документов, И данные из API. "
            "Сначала объясни правило со ссылкой [doc_id:pN], "
            "потом приведи данные из API, в конце сделай вывод."
        )
    elif has_rag and is_technical:
        instruction = (
            "⚠️ ЗАПРОС ТЕХНИЧЕСКИЙ! Пользователь спрашивает про метод/API/технику. "
            "ОБЯЗАТЕЛЬНО назови КОНКРЕТНЫЙ метод API и его параметры, ЕСЛИ ОНИ ЕСТЬ В КОНТЕКСТЕ.\n\n"
            "⚠️ ЖЕЛЕЗНОЕ ПРАВИЛО:\n"
            "• Если в контексте указан URL (например, `https://rest.eurotorg.by/10195/Json` или `https://rest.eurotorg.by/10165/Json`), "
            "приведи ЕГО ДОСЛОВНО.\n"
            "• Если в контексте для параметра написано `[ConfirmType]`, так и пиши `[ConfirmType]`, "
            "НЕ ПРИДУМЫВАЙ значения вроде 'Full' или 'true'.\n"
            "• КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО придумывать URL или параметры, которых нет в тексте.\n"
            "• Ссылки на документы в формате [doc_id:pN]."
        )
    elif has_rag:
        instruction = (
            "В контексте есть фрагменты из документов. "
            "Отвечай на их основе. Ссылки ТОЛЬКО в формате [doc_id:pN]."
        )
    elif has_tool and has_display:
        instruction = (
            "⚠️ ВАЖНО: данные (заказы, товары или претензии) УЖЕ ПОКАЗАНЫ пользователю "
            "в структурированном виде (карточка или таблица) на фронте. "
            "Твоя задача — ОТВЕТИТЬ НА ВОПРОС ПОЛЬЗОВАТЕЛЯ, найдя нужную информацию в таблице.\n\n"
            "ПРАВИЛА:\n"
            "• Если спрашивают про конкретное поле (когда создан, какая сумма, какой статус, "
            "какой товар самый дорогой, сколько недопоставлено) — найди это в таблице и ответь одной фразой.\n"
            "• Если спрашивают про возможность («могу ли», «можно ли») — "
            "ответь «Да, можете» или «Нет, нельзя» на основе статуса.\n"
            "• Если вопрос общий («расскажи», «покажи») — дай КРАТКУЮ сводку (1-2 предложения).\n"
            "• НЕ пересказывай ВСЕ поля таблицы. Скажи только то, что спросили.\n"
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
        memory_block += f"=== КРАТКОЕ СОДЕРЖАНИЕ ПРЕДЫДУЩЕГО ДИАЛОГА ===\n{summary_text}\n\n"
    if history_text:
        memory_block += f"=== ПОСЛЕДНИЕ СООБЩЕНИЯ ===\n{history_text}\n\n"

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

    # === 7. Динамический max_tokens с учетом размера контекста ===
    estimated_input_tokens = len(prompt) // 2.5
    MAX_MODEL_TOKENS = 16384
    available_output_tokens = MAX_MODEL_TOKENS - estimated_input_tokens - 500
    actual_max_tokens = max(300, min(max_tokens, int(available_output_tokens)))
    
    if actual_max_tokens < max_tokens:
        logger.warning(
            f"[synthesis] уменьшаю max_tokens: {max_tokens} -> {actual_max_tokens} "
            f"(контекст ~{estimated_input_tokens:.0f} токенов)"
        )
    
    if mode == "analysis":
        actual_max_tokens = min(actual_max_tokens, 1200)
    elif has_tool and not has_rag and has_display:
        actual_max_tokens = min(actual_max_tokens, 1000)

    logger.info(
        f"[synthesis] mode={mode}, {len(rag_chunks)} rag, "
        f"{len(tool_results)} tool, {len(context)} симв., "
        f"analysis={'yes' if analysis else 'no'}, "
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

    answer = normalize_citations(answer)
    logger.info(f"[synthesis] ответ: {len(answer)} симв.")
    return answer.strip(), sources, tools_used