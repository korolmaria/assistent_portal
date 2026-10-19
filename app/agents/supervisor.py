"""
Supervisor — оркестратор мультиагентной системы.
"""
import json
from typing import Any, AsyncGenerator

from loguru import logger

from app.agents.memory import maybe_summarize
from app.agents.query_rewriter import rewrite_query
from app.agents.rag_agent import search as rag_search
from app.agents.synthesis_agent import synthesize
from app.agents.tool_agent import run_tool_agent
from app.llm.vllm_client import vllm_client
from app.schemas import ChatHistoryItem


ROUTER_SYSTEM_PROMPT = """Ты — маршрутизатор запросов пользователя.

Определи, что нужно для ответа:
- rag: информация из документов (инструкции, регламенты, правила, методы API, описания)
- tool: данные из API (список заказов, статусы, суммы, товары в заказе, конкретные значения)
- rag+tool: И правила/инструкции, И данные из API (комбинированный)

ПРАВИЛА:
1. Запрос про **как сделать / как работает / какой метод / инструкция / описание / технически** → **rag**
2. Запрос про **покажи / найди / сколько / какие / есть ли / был ли** (данные из системы) → **tool**
3. **КОМБИНИРОВАННЫЕ ЗАПРОСЫ** — когда нужны И правила, И данные.
   ⚠️ КЛЮЧЕВОЙ КРИТЕРИЙ: есть ли **КОНКРЕТНЫЙ ФИЛЬТР** для данных?

   **КОНКРЕТНЫЕ ФИЛЬТРЫ** (любой из них):
   • Номер заказа: «заказ 98010205», «заказ №56»
   • Период: «за сентябрь», «за последнюю неделю», «за 2026 год»
   • Бренд: «бренда Хит», «Евроопт»
   • Склад: «склад 700», «склад №1»
   • Статус: «в статусе Подтверждён», «подтверждённые»
   • Регион/город: «по Минску», «в Бресте»
   • Тип товара: «молочные», «хлеб»

   → Если **ЕСТЬ фильтр** И вопрос про **правила/возможность** («можно ли», «как») → **rag+tool**
   → Если **НЕТ фильтра** И вопрос про **правила/возможность** → **rag**

4. Если непонятно → **rag**

Формат ответа (строго JSON, без markdown):
{
  "route": "rag" | "tool" | "rag+tool",
  "reason": "краткое объяснение (1 предложение)"
}
"""


async def decide_route(query: str) -> dict[str, str]:
    prompt = f"""Запрос: {query}

Определи маршрут. Верни только JSON."""

    try:
        answer = await vllm_client.generate_text(
            prompt=prompt,
            system=ROUTER_SYSTEM_PROMPT,
            max_tokens=200,
            temperature=0.0,
        )

        start = answer.find("{")
        end = answer.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"Не найден JSON: {answer[:200]}")

        result = json.loads(answer[start:end + 1])
        route = result.get("route", "rag")
        if route not in ("rag", "tool", "rag+tool"):
            route = "rag"

        logger.info(f"[supervisor] route={route}, reason={result.get('reason', '')}")
        return {"route": route, "reason": result.get("reason", "")}

    except Exception as e:
        logger.warning(f"[supervisor] ошибка роутинга: {e}, fallback → rag")
        return {"route": "rag", "reason": "fallback"}


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
        # 0. Rewrite
        if history:
            rewritten = await rewrite_query(query, history)
            
            forbidden_words = [
                "евроопт", "хит", "грошык",           # бренды
                "подтверждён", "на сайте", "автоподтверждён",  # статусы
                "склад", "регион",                     # склад/регион
            ]
            o_low = original_query.lower()
            r_low = rewritten.lower()
            added_forbidden = [w for w in forbidden_words if w in r_low and w not in o_low]
                
            if added_forbidden:
                logger.warning(
                    f"[supervisor] rewriter добавил лишнее ({added_forbidden}), "
                    f"откат к оригиналу: {original_query!r}"
                )
                query = original_query
            else:
                query = rewritten

        # 1. Route
        if route == "auto":
            decision = await decide_route(query)
            chosen_route = decision["route"]
            reason = decision["reason"]
        else:
            chosen_route = route
            reason = f"выбрано пользователем: {route}"

        logger.info(
            f"[supervisor] chosen_route={chosen_route}, "
            f"memory={len(history)} msgs + "
            f"{'summary' if summary else 'no-summary'}, "
            f"query={query!r}"
        )

        yield {"type": "route", "route": chosen_route, "reason": reason}

        # 2. Agents
        rag_chunks: list[dict[str, Any]] = []
        rag_sources = []
        tool_results: list[dict[str, Any]] = []
        display_block = None
        tools_used: list[str] = []

        if chosen_route in ("rag", "rag+tool"):
            yield {"type": "step", "message": "🔍 Поиск по документам..."}
            try:
                rag_chunks, rag_sources = rag_search(query, top_k=5)
                logger.info(f"[supervisor] RAG: {len(rag_chunks)} чанков")
                yield {"type": "step", "message": f"   найдено {len(rag_chunks)} фрагментов"}
            except Exception as e:
                logger.error(f"[supervisor] RAG-ошибка: {e}")
                yield {"type": "step", "message": f"   ⚠️ ошибка поиска: {e}"}

        if chosen_route in ("tool", "rag+tool"):
            yield {"type": "step", "message": "📊 Запрос к SupplierPortal..."}
            try:
                text, display_block, tool_meta = await run_tool_agent(
                    query=query,
                    jwt=jwt,
                    history=history,
                )
                tool_name = "get_supplier_orders"
                if tool_meta.get("tools_used"):
                    tool_name = tool_meta["tools_used"][0]

                tool_results.append({
                    "tool_name": tool_name,
                    "text": text,
                    "data": display_block,
                })
                tools_used.append(tool_name)
                logger.info(f"[supervisor] Tool: {text[:100]}")
                yield {"type": "step", "message": f"   {text[:100]}"}
            except Exception as e:
                logger.exception(f"[supervisor] Tool-ошибка: {e}")
                yield {"type": "step", "message": f"   ⚠️ ошибка API: {e}"}

        # 3. Synthesis
        yield {"type": "step", "message": "✍️ Генерация ответа..."}

        tool_text = ""
        for tr in tool_results:
            if tr.get("text"):
                tool_text = tr["text"]
                break

        has_display = display_block is not None
        has_rag = bool(rag_chunks)

        q_orig = original_query.lower()
        is_specific_question = any(kw in q_orig for kw in [
            "есть", "заказано", "какой", "какая", "какое", "сколько", 
            "сумма", "статус", "подтвердить", "можно", "товары"
        ])

        if has_display and not has_rag and tool_text and not is_specific_question:
            logger.info(f"[supervisor] общий запрос: используем готовый text tool-агента")
            answer = tool_text
            final_sources = []
            synth_tools = tools_used
        else:
            logger.info(f"[supervisor] конкретный вопрос: отдаем в synthesize")
            answer, final_sources, synth_tools = await synthesize(
                query=original_query,
                rag_chunks=rag_chunks,
                tool_results=tool_results,
                history=history,
                summary=summary,
            )

        # Если в ответе нет цитат, значит RAG-источники не использовались — не прикрепляем их
        if answer and "[" not in answer:
            all_sources = final_sources
        else:
            all_sources = rag_sources or final_sources

        # 4. Summary
        new_summary, new_summarized_up_to_id = await maybe_summarize(
            history=history,
            summary=summary,
            summarized_up_to_id=summarized_up_to_id,
        )

        # 5. Metadata
        yield {
            "type": "metadata",
            "answer": answer,
            "sources": [s.model_dump() for s in all_sources],
            "route": chosen_route,
            "route_reason": reason,
            "tools_used": tools_used or synth_tools,
            "display": display_block,
            "summary": new_summary,
            "summarized_up_to_id": new_summarized_up_to_id,
        }

    except Exception as e:
        logger.exception("[supervisor] критическая ошибка")
        yield {"type": "error", "error": str(e)}