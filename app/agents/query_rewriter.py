"""
Query rewriter: переписывает запрос пользователя с учётом истории.

Зачем: пользователь пишет «а какой у него статус?» — без истории непонятно,
о чём речь. Rewriter подставит конкретный ID из предыдущего сообщения.

Работает с ЛЮБЫМИ сущностями: заказы (980...), претензии (122...), товары и т.д.
"""
import re

from loguru import logger

from app.llm.vllm_client import vllm_client
from app.schemas import ChatHistoryItem


REWRITE_SYSTEM_PROMPT = """Ты — помощник, который переписывает вопрос пользователя для поиска по базе.

ИСТОРИЯ ДИАЛОГА:
{history}

ТЕКУЩИЙ ВОПРОС:
{query}

ЗАДАЧА:
Если текущий вопрос ссылается на предыдущий контекст (местоимения «он», «этот», «там», «у него», «в ней», «про него», «а какой...», «а можно ли...»), перепиши его, подставив КОНКРЕТНЫЕ ДАННЫЕ (номера заказов, претензий, названия товаров) из истории.

Если текущий вопрос САМОДОСТАТОЧЕН (в нём уже есть все данные) — верни его БЕЗ ИЗМЕНЕНИЙ.

⚠️ НЕ ПОДСТАВЛЯЙ НОМЕРА, если в текущем вопросе:
- есть слова «сравни», «тренд», «динамика», «все», «за период»
- есть новый фильтр: «за сентябрь», «за август», «за прошлую неделю»
- спрашивается про МНОЖЕСТВЕННОЕ число («заказы», «претензии»)

Примеры:
- История: «покажи заказ 98010205». Вопрос: «какой у него статус?» → «какой статус у заказа 98010205?»
- История: «покажи претензию 122363». Вопрос: «а какой склад в ней?» → «какой склад в претензии 122363?»
- История: «покажи товар 1251142». Вопрос: «а сколько он стоит?» → «сколько стоит товар 1251142?»
- История: «покажи заказ 98010205». Вопрос: «а можно ли его отменить?» → «можно ли отменить заказ 98010205?»

Верни ТОЛЬКО переписанный вопрос, без объяснений и кавычек.
"""


# Маркеры самодостаточного запроса — не трогаем
_SELF_SUFFICIENT_RE = re.compile(
    r"\b(?:сравн\w*|тренд\w*|динамик\w*|проанализ\w*|"
    r"статистик\w*|все\s+заказ\w*|все\s+претензи\w*|"
    r"за\s+(?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|"
    r"сентябр|октябр|ноябр|декабр)\w*|"
    r"за\s+(?:последн|прошл)\w+\s+(?:недел|месяц|год)\w*|"
    r"за\s+\d{4}\s*год)\b",
    re.IGNORECASE,
)

# Признаки того, что запрос ссылается на контекст (требует подстановки)
_CONTEXT_RE = re.compile(
    r"\b(?:он|она|оно|они|его|её|ее|их|ему|ей|им|ними|"
    r"этот|эта|это|эти|этому|этой|этом|данном|данной|"
    r"там|тут|него|неё|нее|них|нему|ней|ним|ними)\b",
    re.IGNORECASE,
)


def _is_self_sufficient(query: str) -> bool:
    """Запрос не нуждается в контексте — rewriter его не трогает."""
    return bool(_SELF_SUFFICIENT_RE.search(query))


def _needs_context(query: str) -> bool:
    """Запрос явно ссылается на предыдущий контекст."""
    q_lower = query.lower()
    # Местоимения
    if _CONTEXT_RE.search(query):
        return True
    # Начало с «а» или «и» (продолжение мысли)
    if re.match(r"^(?:а|и)\s+", q_lower):
        return True
    # Короткие запросы (меньше 5 слов) без чисел — скорее всего, это продолжение
    words = query.split()
    if len(words) < 5 and not re.search(r"\d{4,}", query):
        return True
    return False


def _extract_history_text(history: list[ChatHistoryItem], limit: int = 6) -> str:
    if not history:
        return "(пусто)"
    recent = history[-limit:]
    lines = []
    for m in recent:
        prefix = "Пользователь" if m.role == "user" else "Ассистент"
        content = m.content if len(m.content) <= 300 else m.content[:300] + "…"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


async def rewrite_query(
    query: str,
    history: list[ChatHistoryItem] | None = None,
) -> str:
    """
    Переписывает запрос с учётом истории. Возвращает оригинал, если:
    - истории нет,
    - запрос самодостаточен (явные фильтры),
    - LLM упала.
    """
    history = history or []

    if not history:
        return query

    # 1. Не трогаем самодостаточные запросы
    if _is_self_sufficient(query):
        logger.info(f"[rewriter] запрос самодостаточен, не трогаю: {query!r}")
        return query

    # 2. Если запрос не ссылается на контекст — тоже не трогаем
    #    (иначе LLM может выдумать лишние детали)
    if not _needs_context(query):
        logger.info(f"[rewriter] запрос не ссылается на контекст: {query!r}")
        return query

    history_text = _extract_history_text(history)
    prompt = REWRITE_SYSTEM_PROMPT.format(history=history_text, query=query)

    try:
        rewritten = await vllm_client.generate_text(
            prompt=prompt,
            system="Ты — точный переформулировщик. Отвечай только текстом вопроса.",
            max_tokens=200,
            temperature=0.0,
        )
        rewritten = rewritten.strip().strip('"').strip("'")
    except Exception as e:
        logger.warning(f"[rewriter] LLM упала: {e}, возвращаю оригинал")
        return query

    if not rewritten:
        return query

    # 3. Если запрос требовал контекста, но LLM не добавила чисел — 
    #    возможно, она не нашла ID в истории. Это нормально (может, там их нет).
    #    Но если LLM СЛОМАЛА запрос (сделала короче) — откатываем.
    if len(rewritten) < len(query) * 0.5:
        logger.warning(f"[rewriter] rewritten слишком короткий, откат: {rewritten!r}")
        return query

    if rewritten != query:
        logger.info(f"[rewriter] было: {query!r} → стало: {rewritten!r}")
    return rewritten