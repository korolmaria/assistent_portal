"""
Агент-аналитик.

Получает УЖЕ ДОБЫТЫЕ данные (ctx.tool_results, ctx.display_block, ctx.rag_chunks)
и вопрос пользователя. Возвращает текстовый анализ (числа, сравнения, выводы),
который затем synthesize встраивает в финальный ответ.

Аналитик НЕ ходит в API и НЕ ищет в Qdrant. Он только интерпретирует данные.
"""
import json
from typing import TYPE_CHECKING, Any

from loguru import logger

from app.agents.memory import format_history
from app.llm.vllm_client import vllm_client
from app.schemas import ChatHistoryItem

if TYPE_CHECKING:
    from app.agents.executor import ExecutionContext


ANALYST_SYSTEM_PROMPT = """Ты — аналитик данных для поставщиков торговой сети «Евроторг».

Тебе дают:
- ВОПРОС пользователя,
- ДАННЫЕ из API (в виде текста и структурированных таблиц/карточек),
- возможно, фрагменты документации.

Твоя задача — ПРОАНАЛИЗИРОВАТЬ данные и дать краткий аналитический вывод.

ПРАВИЛА:
1. Опирайся ТОЛЬКО на переданные данные. Не выдумывай числа.
2. Если данных недостаточно для ответа — прямо скажи, чего не хватает.
3. Если вопрос на сравнение — приведи конкретные числа и разницу (абсолютную и в %).
4. Если вопрос «почему» — перечисли ВОЗМОЖНЫЕ причины на основе данных
   (например, «сумма упала, потому что заказов стало меньше на 30%»).
   Не выдумывай внешние причины, которых нет в данных.
5. Если вопрос про тренд — оцени направление (рост/падение/стабильно) и
   приведи числа по периодам.
6. Пиши кратко: 3-6 предложений. Числа — с разделителями (1 234,56).
7. НЕ форматируй ответ как финальный ответ пользователю — это сделает другой агент.
   Твой вывод — промежуточный аналитический блок.
8. Не ссылайся на документы в формате [doc_id:pN] — это не твоя задача.

ФОРМАТ ВЫВОДА:
- «Итог: ...» — одно предложение с главным выводом
- «Детали: ...» — числа, сравнения, проценты
- «Причины: ...» — если вопрос «почему»
"""


# ==========================================================================
# Помощники
# ==========================================================================
def _serialize_display(display: dict[str, Any] | None, max_chars: int = 8000) -> str:
    """Компактно сериализует display-блок для промпта аналитика."""
    if not display:
        return ""
    try:
        return json.dumps(display, ensure_ascii=False)[:max_chars]
    except Exception as e:
        logger.warning(f"[analyst] не удалось сериализовать display: {e}")
        return ""


def _build_data_block(ctx: "ExecutionContext") -> str:
    """Собирает блок данных из контекста исполнения."""
    parts: list[str] = []

    if ctx.tool_results:
        parts.append("=== ДАННЫЕ ИЗ API ===")
        for tr in ctx.tool_results:
            tool_name = tr.get("tool_name", "unknown")
            text = tr.get("text", "")
            data = tr.get("data")
            parts.append(f"[tool: {tool_name}]")
            if text:
                parts.append(text)
            if data:
                parts.append(_serialize_display(data))
            parts.append("")

    if ctx.display_block is not None:
        parts.append("=== СТРУКТУРИРОВАННЫЙ БЛОК (display) ===")
        parts.append(_serialize_display(ctx.display_block))
        parts.append("")

    if ctx.rag_chunks:
        parts.append("=== ФРАГМЕНТЫ ДОКУМЕНТАЦИИ (для контекста) ===")
        for i, ch in enumerate(ctx.rag_chunks[:5], 1):
            header = f"[{i}] {ch.get('source_file', '')}"
            if ch.get("page"):
                header += f", стр. {ch['page']}"
            parts.append(f"{header}\n{ch.get('text', '')[:1000]}")
            parts.append("")

    return "\n".join(parts).strip()


# ==========================================================================
# Главная функция
# ==========================================================================
async def run_analyst(
    query: str,
    ctx: "ExecutionContext",
    history: list[ChatHistoryItem] | None = None,
) -> str | None:
    """
    Возвращает текстовый аналитический вывод или None, если данных нет.
    """
    history = history or []

    data_block = _build_data_block(ctx)
    if not data_block:
        logger.warning("[analyst] нет данных для анализа — возвращаю None")
        return None

    history_text = format_history(history)

    prompt = f"""=== ВОПРОС ПОЛЬЗОВАТЕЛЯ ===
{query}

=== ДОСТУПНЫЕ ДАННЫЕ ===
{data_block}

=== ИСТОРИЯ ДИАЛОГА (для контекста) ===
{history_text or "(пусто)"}

Проанализируй данные и дай краткий аналитический вывод по правилам из системного промпта.

АНАЛИЗ:"""

    try:
        analysis = await vllm_client.generate_text(
            prompt=prompt,
            system=ANALYST_SYSTEM_PROMPT,
            max_tokens=800,
            temperature=0.1,
        )
        analysis = (analysis or "").strip()
        if not analysis:
            logger.warning("[analyst] пустой ответ модели")
            return None
        logger.info(f"[analyst] анализ готов: {len(analysis)} симв.")
        return analysis

    except Exception as e:
        logger.exception(f"[analyst] ошибка генерации: {e}")
        return None