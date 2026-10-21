"""
Supervisor — тонкий фасад: planner → executor → synthesize → memory.
"""
import time
import json
from pathlib import Path
from typing import Any, AsyncGenerator

from loguru import logger

from app.agents.executor import run_plan
from app.agents.memory import maybe_summarize
from app.agents.planner import make_plan
from app.agents.query_rewriter import rewrite_query
from app.agents.router_rules import check_filters
from app.agents.synthesis_agent import synthesize
from app.schemas import ChatHistoryItem


def _log_dataset(entry: dict[str, Any]) -> None:
    """Пишем строку в logs/route_dataset.jsonl для будущего анализа и ML."""
    try:
        path = Path("logs/route_dataset.jsonl")
        path.parent.mkdir(exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"[supervisor] failed to log dataset: {e}")


async def run_supervisor(
    query: str,
    route: str = "auto",
    jwt: str = "",
    history: list[ChatHistoryItem] | None = None,
    summary: str | None = None,
    summarized_up_to_id: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    history = history or []
    original_query = query

    try:
        # 0. Rewrite (с защитой от лишних слов)
        if history:
            rewritten = await rewrite_query(query, history)
            forbidden = [
                "евроопт", "хит", "грошык", "подтверждён", "на сайте",
                "автоподтверждён", "склад", "регион",
            ]
            o_low, r_low = original_query.lower(), rewritten.lower()
            added = [w for w in forbidden if w in r_low and w not in o_low]
            query = original_query if added else rewritten

        # 1. План
        plan = await make_plan(query, forced_route=route)
        route_label = "+".join(s.agent for s in plan.steps)
        logger.info(f"[supervisor] plan={route_label}, source={plan.source}")

        yield {
            "type": "route",
            "route": route_label,
            "reason": plan.reason,
            "plan": [s.model_dump() for s in plan.steps],
        }

        # 2. Исполнение
        ctx = None
        async for event in run_plan(plan, query, jwt, history):
            if event.get("type") == "_ctx":
                ctx = event["ctx"]
            else:
                yield event

        if ctx is None:
            raise RuntimeError("executor не вернул контекст")

        # 3. Synthesis — режим определяется планом
        yield {"type": "step", "message": "✍️ Генерация ответа..."}
        yield {"type": "thought", "text": "Формулирую финальный ответ на основе полученных данных..."}  # <--- ДОБАВЛЕНО

        has_rag = bool(ctx.rag_chunks)
        has_display = ctx.display_block is not None

        if ctx.analysis:
            mode = "analysis"
        elif has_display and not has_rag:
            mode = "verbatim"
        else:
            mode = "explain"

        answer, final_sources, synth_tools = await synthesize(
            query=original_query,
            rag_chunks=ctx.rag_chunks,
            tool_results=ctx.tool_results,
            history=history,
            summary=summary,
            mode=mode,
            analysis=ctx.analysis,
        )

        # 4. Summary
        new_summary, new_summarized_up_to_id = await maybe_summarize(
            history=history,
            summary=summary,
            summarized_up_to_id=summarized_up_to_id,
        )

        # 5. Датасет
        _, flags = check_filters(query)
        _log_dataset({
            "timestamp": time.time(),
            "query": original_query,
            "rewritten": query if query != original_query else None,
            "plan": [s.model_dump() for s in plan.steps],
            "plan_source": plan.source,
            "reason": plan.reason,
            "flags": flags,
            "has_rag": has_rag,
            "has_display": has_display,
            "has_analysis": bool(ctx.analysis),
            "answer_len": len(answer) if answer else 0,
        })

        # 6. Метаданные
        all_sources = final_sources or ctx.rag_sources

        yield {
            "type": "metadata",
            "answer": answer,
            "sources": [s.model_dump() for s in all_sources],
            "route": route_label,
            "route_reason": plan.reason,
            "plan": [s.model_dump() for s in plan.steps],
            "tools_used": ctx.tools_used or synth_tools,
            "display": ctx.display_block,
            "summary": new_summary,
            "summarized_up_to_id": new_summarized_up_to_id,
        }

    except Exception as e:
        logger.exception("[supervisor] критическая ошибка")
        yield {"type": "error", "error": str(e)}