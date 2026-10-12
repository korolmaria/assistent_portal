"""
Supervisor — оркестратор мультиагентной системы.

Маршрутизация:
- auto: LLM решает, какие агенты нужны
- rag: только RAG-поиск по документам
- tool: только Tool (SupplierPortal API)
- rag+tool: комбинированный

Возвращает генератор SSE-событий для стриминга на фронт.
"""
import json
from typing import Any, AsyncGenerator

from loguru import logger

from app.agents.rag_agent import search as rag_search
from app.agents.synthesis_agent import synthesize
from app.agents.tool_agent import run_tool_agent
from app.llm.vllm_client import vllm_client


# ==========================================================================
# Определение маршрута через LLM
# ==========================================================================
ROUTER_SYSTEM_PROMPT = """Ты — маршрутизатор запросов пользователя.

Определи, что нужно для ответа:
- rag: информация из документов (инструкции, регламенты, правила, методы API, описания)
- tool: данные из API (список заказов, статусы, суммы, конкретные значения)
- rag+tool: и документы, и данные из API

ПРАВИЛА:
1. Запрос про **как сделать / как работает / какой метод / инструкция / описание / технически** → **rag**
   (даже если упоминается API или метод — сначала ищем документацию)
2. Запрос про **покажи / найди / сколько / какие** (данные из системы) → **tool**
3. Если нужны и правила, и данные → **rag+tool**
4. Запрос про **методы API / технические детали / документацию** → **rag**
5. Если непонятно → **rag** (документы чаще дают ответ)

ПРИМЕРЫ (запрос → route):
- «как подтвердить заказ» → rag (инструкция)
- «каким методом подтвердить заказ» → rag (метод API в документации)
- «как технически подтвердить заказ» → rag (техническая документация)
- «какой API-метод используется для подтверждения» → rag (документация API)
- «покажи мои заказы» → tool (данные из системы)
- «сколько заказов за сентябрь» → tool (данные)
- «покажи заказы и объясни как их подтвердить» → rag+tool (и данные, и правила)

Формат ответа (строго JSON, без markdown):
{
  "route": "rag" | "tool" | "rag+tool",
  "reason": "краткое объяснение (1 предложение)",
  "tool_query": "переформулированный запрос для API (только если route включает tool)"
}
"""


async def decide_route(query: str) -> dict[str, str]:
    """
    LLM решает, какой маршрут использовать.
    """
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

        logger.info(
            f"[supervisor] route={route}, "
            f"reason={result.get('reason', '')}"
        )
        return {
            "route": route,
            "reason": result.get("reason", ""),
            "tool_query": result.get("tool_query", query),
        }

    except Exception as e:
        logger.warning(f"[supervisor] ошибка роутинга: {e}, fallback → rag")
        return {"route": "rag", "reason": "fallback", "tool_query": query}


# ==========================================================================
# Основной запуск
# ==========================================================================
async def run_supervisor(
    query: str,
    route: str = "auto",
    jwt: str = "",
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Главный генератор SSE-событий.

    Yields:
        {"type": "route", "route": ..., "reason": ...}
        {"type": "step", "message": ...}
        {"type": "metadata", "answer": ..., "sources": [...], "display": ..., "tools_used": [...]}
        {"type": "error", "error": ...}
    """
    try:
        # 1. Определяем маршрут
        if route == "auto":
            decision = await decide_route(query)
            chosen_route = decision["route"]
            reason = decision["reason"]
            tool_query = decision["tool_query"]
        else:
            chosen_route = route
            reason = f"выбрано пользователем: {route}"
            tool_query = query

        yield {"type": "route", "route": chosen_route, "reason": reason}

        # 2. Выполняем нужные агенты
        rag_chunks: list[dict[str, Any]] = []
        rag_sources = []
        tool_results: list[dict[str, Any]] = []
        display_block = None
        tools_used: list[str] = []

        # RAG
        if chosen_route in ("rag", "rag+tool"):
            yield {"type": "step", "message": "🔍 Поиск по документам..."}
            try:
                rag_chunks, rag_sources = rag_search(query, top_k=5)
                logger.info(f"[supervisor] RAG: {len(rag_chunks)} чанков")
                yield {
                    "type": "step",
                    "message": f"   найдено {len(rag_chunks)} фрагментов",
                }
            except Exception as e:
                logger.error(f"[supervisor] RAG-ошибка: {e}")
                yield {"type": "step", "message": f"   ⚠️ ошибка поиска: {e}"}

        # Tool
        if chosen_route in ("tool", "rag+tool"):
            yield {"type": "step", "message": "📊 Запрос к SupplierPortal..."}
            try:
                text, display_block, tool_meta = await run_tool_agent(
                    query=tool_query,
                    jwt=jwt,
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
                logger.error(f"[supervisor] Tool-ошибка: {e}")
                yield {"type": "step", "message": f"   ⚠️ ошибка API: {e}"}

        # 3. Synthesis
        yield {"type": "step", "message": "✍️ Генерация ответа..."}

        answer, final_sources, synth_tools = await synthesize(
            query=query,
            rag_chunks=rag_chunks,
            tool_results=tool_results,
        )

        # Объединяем sources: RAG-источники уже дедуплицированы
        all_sources = rag_sources or final_sources

        # 4. Финальные метаданные
        yield {
            "type": "metadata",
            "answer": answer,
            "sources": [s.model_dump() for s in all_sources],
            "route": chosen_route,
            "route_reason": reason,
            "tools_used": tools_used or synth_tools,
            "display": display_block,
        }

    except Exception as e:
        logger.exception("[supervisor] критическая ошибка")
        yield {"type": "error", "error": str(e)}
