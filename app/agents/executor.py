"""
Исполнитель плана: проходит по шагам Plan и собирает контекст.
Не решает ЧТО делать (это planner) — только ВЫПОЛНЯЕТ.
"""
from typing import Any, AsyncGenerator

from loguru import logger

from app.agents.analyst import run_analyst
from app.agents.planner import Plan
from app.agents.rag_agent import search as rag_search
from app.agents.tool_agent import run_tool_agent
from app.schemas import ChatHistoryItem


class ExecutionContext:
    """Что накопили за время исполнения плана."""
    def __init__(self) -> None:
        self.rag_chunks: list[dict[str, Any]] = []
        self.rag_sources: list = []
        self.tool_results: list[dict[str, Any]] = []
        self.display_block: dict[str, Any] | None = None
        self.tools_used: list[str] = []
        self.analysis: str | None = None


async def run_plan(
    plan: Plan,
    query: str,
    jwt: str,
    history: list[ChatHistoryItem],
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Генератор SSE-событий + финальное событие {"type":"_ctx","ctx":ExecutionContext}.
    Supervisor отфильтрует события с ключом "_".
    """
    ctx = ExecutionContext()
    total = len(plan.steps)

    for i, step in enumerate(plan.steps, 1):
        logger.info(f"[executor] шаг {i}/{total}: {step.agent} — {step.purpose}")

        if step.agent == "rag":
            yield {"type": "step", "message": f"🔍 [{i}/{total}] Поиск по документам..."}
            yield {"type": "thought", "text": "Сканирую базу знаний и инструкции на предмет подходящих регламентов..."} # <--- ДОБАВЛЕНО
            try:
                chunks, sources = rag_search(query, top_k=5)
                ctx.rag_chunks.extend(chunks)
                ctx.rag_sources.extend(sources)
                yield {"type": "step", "message": f"   найдено {len(chunks)} фрагментов"}
            except Exception as e:
                logger.error(f"[executor] RAG-ошибка: {e}")
                yield {"type": "step", "message": f"   ⚠️ ошибка поиска: {e}"}

        elif step.agent == "tool":
            yield {"type": "step", "message": f"📊 [{i}/{total}] Запрос к SupplierPortal..."}
            yield {"type": "thought", "text": "Устанавливаю соединение с базой данных SupplierPortal и формирую запрос..."} # <--- ДОБАВЛЕНО
            try:
                text, display, meta = await run_tool_agent(
                    query=query, jwt=jwt, history=history,
                )
                tools_used = meta.get("tools_used") or ["unknown"]
                ctx.tool_results.append({
                    "tool_name": tools_used[0],
                    "text": text,
                    "data": display,
                })
                ctx.tools_used.extend(tools_used)
                if display is not None:
                    ctx.display_block = display
                yield {"type": "step", "message": f"   {(text or '')[:100]}"}
            except Exception as e:
                logger.exception(f"[executor] Tool-ошибка: {e}")
                yield {"type": "step", "message": f"   ⚠️ ошибка API: {e}"}
                ctx.tool_results.append({
                    "tool_name": "error",
                    "text": f"Ошибка API: {e}",
                    "data": None,
                })

        elif step.agent == "analyst":
            yield {"type": "step", "message": f"🧠 [{i}/{total}] Анализ данных..."}
            yield {"type": "thought", "text": "Сопоставляю полученные данные и вычисляю нужные показатели..."} # <--- ДОБАВЛЕНО
            try:
                ctx.analysis = await run_analyst(query=query, ctx=ctx, history=history)
                if ctx.analysis:
                    yield {"type": "step", "message": "   анализ готов"}
                else:
                    yield {"type": "step", "message": "   ⚠️ аналитик не дал результата"}
            except Exception as e:
                logger.exception(f"[executor] Analyst-ошибка: {e}")
                yield {"type": "step", "message": f"   ⚠️ ошибка аналитика: {e}"}

    yield {"type": "_ctx", "ctx": ctx}