"""
Память диалога (stateless).

Бэк НЕ хранит историю. Фронт присылает:
  - history: последние N сообщений (дословно)
  - summary: сжатое резюме более ранних сообщений
  - summarized_up_to_id: id последнего сообщения, уже вошедшего в summary

Здесь:
  - format_history()  — формирует блок "последние сообщения" для промпта
  - format_summary()  — формирует блок "краткое содержание" для промпта
  - maybe_summarize() — при необходимости сворачивает старые сообщения
                        в обновлённое резюме (один вызов qwen-text)
"""
from __future__ import annotations

from loguru import logger

from app.llm.vllm_client import vllm_client
from app.schemas import ChatHistoryItem


# ==========================================================================
# Константы (можно менять)
# ==========================================================================
RAW_WINDOW_MESSAGES = 6          # последние 6 сообщений — дословно всегда
SUMMARY_TRIGGER = 8              # при истории > 8 запускаем суммирование
SUMMARY_MAX_CHARS = 1500         # лимит резюме (~400 токенов)
SUMMARY_INPUT_MAX_CHARS = 6000   # что скармливаем суммаризатору за раз
PER_MESSAGE_MAX_CHARS = 800      # обрезка одного сообщения в промпте


SUMMARY_SYSTEM_PROMPT = """Ты сжимаешь диалог пользователя с ИИ-ассистентом.
Сохрани ВСЁ существенное:
- факты, числа, ID (заказов, документов), имена, даты;
- принятые решения и выводы;
- открытые вопросы и незавершённые задачи.

Не добавляй ничего от себя. Не пересказывай дословно.
Пиши кратко, 3-6 предложений, на русском языке.
"""


# ==========================================================================
# Форматирование для промпта
# ==========================================================================
def _truncate(text: str, limit: int = PER_MESSAGE_MAX_CHARS) -> str:
    """Обрезает длинный текст, добавляя многоточие."""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def format_history(history: list[ChatHistoryItem]) -> str:
    """
    Форматирует последние RAW_WINDOW_MESSAGES сообщений для промпта.

    Returns:
        Строка вида:
            Пользователь: ...
            Ассистент: ...
        или пустая строка, если истории нет.
    """
    if not history:
        return ""

    recent = history[-RAW_WINDOW_MESSAGES:]
    lines: list[str] = []
    for m in recent:
        prefix = "Пользователь" if m.role == "user" else "Ассистент"
        lines.append(f"{prefix}: {_truncate(m.content)}")
    return "\n".join(lines)


def format_summary(summary: str | None) -> str:
    """Форматирует резюме для промпта (или пустую строку)."""
    if not summary:
        return ""
    return summary.strip()


# ==========================================================================
# Суммирование
# ==========================================================================
def _messages_to_summarize(
    history: list[ChatHistoryItem],
    summarized_up_to_id: str | None,
) -> list[ChatHistoryItem]:
    """
    Возвращает сообщения, которые надо свернуть:
      - всё, что СТАРШЕ последних RAW_WINDOW_MESSAGES
      - и НОВЕЕ summarized_up_to_id (если он задан)
    """
    if len(history) <= RAW_WINDOW_MESSAGES:
        return []

    older = history[:-RAW_WINDOW_MESSAGES]  # всё, что не попало в окно

    if not summarized_up_to_id:
        return older

    # отрезаем всё, что уже вошло в резюме (до summarized_up_to_id включительно)
    cutoff = None
    for i, m in enumerate(older):
        if m.id == summarized_up_to_id:
            cutoff = i
            break

    if cutoff is None:
        # summarized_up_to_id не найден — считаем, что ещё ничего не сворачивали
        return older

    return older[cutoff + 1:]


async def maybe_summarize(
    history: list[ChatHistoryItem],
    summary: str | None,
    summarized_up_to_id: str | None,
) -> tuple[str | None, str | None]:
    """
    Обновляет резюме, если истории стало больше SUMMARY_TRIGGER.

    Логика:
      - len(history) <= SUMMARY_TRIGGER  → ничего не делаем
      - иначе сворачиваем всё, что старше последних RAW_WINDOW_MESSAGES
        и новее summarized_up_to_id

    Returns:
        (new_summary, new_summarized_up_to_id)
        Если суммирование не нужно — возвращает входные значения без изменений.
    """
    if len(history) <= SUMMARY_TRIGGER:
        return summary, summarized_up_to_id

    to_fold = _messages_to_summarize(history, summarized_up_to_id)
    if not to_fold:
        logger.debug("[memory] суммировать нечего (всё уже в резюме)")
        return summary, summarized_up_to_id

    # Собираем вход для суммаризатора
    parts: list[str] = []
    if summary:
        parts.append(f"СТАРОЕ РЕЗЮМЕ:\n{summary}\n")

    fold_lines: list[str] = []
    total = 0
    for m in to_fold:
        prefix = "Пользователь" if m.role == "user" else "Ассистент"
        line = f"{prefix}: {_truncate(m.content, 1000)}"
        if total + len(line) > SUMMARY_INPUT_MAX_CHARS:
            break
        fold_lines.append(line)
        total += len(line)

    parts.append("НОВЫЕ СООБЩЕНИЯ ДЛЯ СВОРАЧИВАНИЯ:\n" + "\n".join(fold_lines))
    prompt = "\n\n".join(parts) + "\n\nОБНОВЛЁННОЕ РЕЗЮМЕ:"

    logger.info(
        f"[memory] суммирую: {len(to_fold)} сообщений, "
        f"старое резюме={'да' if summary else 'нет'}"
    )

    try:
        new_summary = await vllm_client.generate_text(
            prompt=prompt,
            system=SUMMARY_SYSTEM_PROMPT,
            max_tokens=500,
            temperature=0.0,
        )
        new_summary = new_summary.strip()[:SUMMARY_MAX_CHARS]
        new_up_to = to_fold[-1].id
        logger.info(
            f"[memory] новое резюме: {len(new_summary)} симв., "
            f"up_to_id={new_up_to}"
        )
        return new_summary, new_up_to

    except Exception as e:
        logger.warning(f"[memory] суммирование не удалось: {e}")
        # не ломаем основной ответ — возвращаем как было
        return summary, summarized_up_to_id