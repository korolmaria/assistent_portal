"""
Supervisor — оркестратор мультиагентной системы.
"""
import asyncio
import json
import time
from pathlib import Path
from typing import Any, AsyncGenerator

from loguru import logger

from app.agents.memory import maybe_summarize
from app.agents.query_rewriter import rewrite_query
from app.agents.rag_agent import search as rag_search
from app.agents.router_rules import check_filters, hard_route
from app.agents.synthesis_agent import synthesize
from app.agents.tool_agent import run_tool_agent
from app.llm.vllm_client import vllm_client
from app.schemas import ChatHistoryItem


ROUTER_SYSTEM_PROMPT = """Ты — маршрутизатор запросов пользователя.

Определи, что нужно для ответа:
- rag: информация из документов (инструкции, регламенты, правила, методы API, описания)
- tool: данные из API (список заказов, статусы, суммы, товары в заказе, претензии, конкретные значения)
- rag+tool: И правила/инструкции, И данные из API (комбинированный)

ПРАВИЛА:
1. Запрос про **как сделать / как работает / какой метод / инструкция / описание / технически** → **rag**
2. Запрос про **покажи / найди / сколько / какие / есть ли / был ли / расскажи про претензии** (данные из системы) → **tool**
3. **КОМБИНИРОВАННЫЕ ЗАПРОСЫ** — когда нужны И правила, И данные.
   ⚠️ КЛЮЧЕВОЙ КРИТЕРИЙ: есть ли **КОНКРЕТНЫЙ ФИЛЬТР** для данных?

   **КОНКРЕТНЫЕ ФИЛЬТРЫ**:
   • Номер заказа/претензии: «заказ 98010205», «претензия 122366»
   • Период: «за сентябрь», «за последнюю неделю», «за 2026 год»
   • Бренд: «бренда Хит», «Евроопт»
   • Склад: «склад 700», «склад №1»
   • Статус: «в статусе Подтверждён», «подтверждённые»
   • Регион/город: «по Минску», «в Бресте»

   → Если **ЕСТЬ фильтр** И вопрос про **правила/возможность** («можно ли», «как», «какие шаги») → **rag+tool**
   → Если **НЕТ фильтра** И вопрос про **правила/возможность** → **rag**

4. Если пользователь просит **конкретный экземпляр** (первая, последняя, самая дорогая, заказ №X, претензия №Y) — это **tool**, даже если нет явного номера.
5. Если непонятно → **rag**

⚠️ ВАЖНО ДЛЯ ВОПРОСОВ ПРО ВОЗМОЖНОСТЬ:
Если пользователь спрашивает «МОГУ ЛИ Я...», «МОЖНО ЛИ...», «КАК МНЕ...» — это ВСЕГДА требует обращения к правилам (rag). Даже если в вопросе есть номер заказа. Потому что статус из API не объясняет, какие действия доступны.

ПРИМЕРЫ:
Q: как оформить возврат товара?                 → {"route":"rag"}
Q: покажи заказы за сентябрь                    → {"route":"tool"}
Q: заказ 98010205 — можно ли его отменить?      → {"route":"rag+tool"}
Q: могу ли я подтвердить заказ 98010227?        → {"route":"rag+tool"}
Q: какие методы API для создания заказа?        → {"route":"rag"}
Q: сколько заказов на складе 700?               → {"route":"tool"}
Q: можно ли принять товар на склад 700?         → {"route":"rag+tool"}
Q: как работает автоподтверждение?              → {"route":"rag"}
Q: покажи претензии по бренду Хит               → {"route":"tool"}
Q: расскажи про претензии                       → {"route":"tool"}
Q: как работают претензии?                      → {"route":"rag"}
Q: покажи все заказы                            → {"route":"tool"}
Q: расскажи про самую первую претензию?         → {"route":"tool"}
Q: какая последняя претензия?                   → {"route":"tool"}
Q: что написано в инструкции про претензии?     → {"route":"rag"}

Формат ответа (строго JSON, без markdown):
{
  "route": "rag" | "tool" | "rag+tool",
  "reason": "краткое объяснение (1 предложение)"
}
"""

# JSON-схема для guided JSON (vLLM response_format)
ROUTE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "route_decision",
        "schema": {
            "type": "object",
            "properties": {
                "route": {"type": "string", "enum": ["rag", "tool", "rag+tool"]},
                "reason": {"type": "string"},
            },
            "required": ["route", "reason"],
            "additionalProperties": False,
        },
    },
}


async def decide_route(query: str) -> dict[str, str]:
    prompt = f"""Запрос: {query}

Определи маршрут. Верни только JSON."""

    try:
        answer = await vllm_client.generate_text(
            prompt=prompt,
            system=ROUTER_SYSTEM_PROMPT,
            max_tokens=200,
            temperature=0.0,
            response_format=ROUTE_SCHEMA,
        )

        result = json.loads(answer)
        route = result.get("route", "rag")
        if route not in ("rag", "tool", "rag+tool"):
            route = "rag"

        logger.info(f"[supervisor] LLM route={route}, reason={result.get('reason', '')}")
        return {"route": route, "reason": result.get("reason", "")}

    except Exception as e:
        logger.warning(f"[supervisor] ошибка роутинга LLM: {e}, fallback → rag")
        return {"route": "rag", "reason": "fallback"}


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
        # 0. Rewrite
        if history:
            rewritten = await rewrite_query(query, history)
            
            forbidden_words = [
                "евроопт", "хит", "грошык",
                "подтверждён", "на сайте", "автоподтверждён",
                "склад", "регион",
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
            forced, rule_reason = hard_route(query)
            if forced:
                chosen_route = forced
                reason = rule_reason
                logger.info(f"[supervisor] route={forced} by rule, reason={rule_reason}")
            else:
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
            yield {"type": "thought", "text": "Сканирую базу знаний и инструкции на предмет подходящих регламентов..."}
            await asyncio.sleep(0.8)  # Пауза для чтения
            
            try:
                rag_chunks, rag_sources = rag_search(query, top_k=5)
                logger.info(f"[supervisor] RAG: {len(rag_chunks)} чанков")
                yield {"type": "step", "message": f"   найдено {len(rag_chunks)} фрагментов"}
            except Exception as e:
                logger.error(f"[supervisor] RAG-ошибка: {e}")
                yield {"type": "step", "message": f"   ⚠️ ошибка поиска: {e}"}

        if chosen_route in ("tool", "rag+tool"):
            yield {"type": "step", "message": "📊 Запрос к SupplierPortal..."}
            yield {"type": "thought", "text": "Устанавливаю соединение с базой данных SupplierPortal и формирую запрос..."}
            await asyncio.sleep(0.8)
            
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
                
                yield {"type": "thought", "text": "Данные успешно получены. Проверяю статусы и формирую структурированную таблицу..."}
                await asyncio.sleep(0.8)
            except Exception as e:
                logger.exception(f"[supervisor] Tool-ошибка: {e}")
                yield {"type": "step", "message": f"   ⚠️ ошибка API: {e}"}
                tool_results.append({
                    "tool_name": "error",
                    "text": f"Ошибка API: {e}",
                    "data": None,
                })

        # 3. Synthesis
        yield {"type": "step", "message": "✍️ Генерация ответа..."}

        tool_text = ""
        for tr in tool_results:
            if tr.get("text"):
                tool_text = tr["text"]
                break

        has_display = display_block is not None
        has_rag = bool(rag_chunks)

        # --- ГЕНЕРАЦИЯ ФИНАЛЬНОЙ МЫСЛИ (АНАЛИЗ) ---
        thought_text = ""
        if has_rag and has_display:
            thought_text = "Сопоставляю найденные правила из документации с реальными данными из API..."
        elif has_rag:
            thought_text = "Внимательно изучаю найденные пункты инструкции, чтобы сформулировать точный ответ..."
        elif has_display:
            thought_text = "Анализирую полученную таблицу, чтобы ответить на ваш вопрос максимально точно..."
        elif tool_text and "не найден" not in tool_text.lower():
            thought_text = "Обрабатываю ответ от системы и выделяю главное..."
        else:
            thought_text = "Проверяю, есть ли в системе нужные данные, и готовлю ответ..."
        
        if thought_text:
            yield {"type": "thought", "text": thought_text}
            await asyncio.sleep(1.0)  # Даем время прочитать перед финальным ответом
        # ------------------------------------------

        q_orig = original_query.lower()
        
        is_just_show = any(kw in q_orig for kw in ["покажи", "выведи", "список", "какие заказы", "какие претензии"]) and \
                       not any(kw in q_orig for kw in ["какой", "какая", "когда", "сколько", "почему", "есть ли", "сумма", "статус", "самый", "дорогой", "первая", "первую", "последняя", "последнюю"])

        if has_display and not has_rag and tool_text and is_just_show:
            logger.info("[supervisor] запрос на список: используем готовый text tool-агента")
            answer = tool_text
            final_sources = []
            synth_tools = tools_used
        else:
            logger.info("[supervisor] вопрос по данным: отдаем в synthesize")
            answer, final_sources, synth_tools = await synthesize(
                query=original_query,
                rag_chunks=rag_chunks,
                tool_results=tool_results,
                history=history,
                summary=summary,
            )

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

        # 5. Логирование датасета
        _, flags = check_filters(query)
        _log_dataset({
            "timestamp": time.time(),
            "query": original_query,
            "rewritten": query if query != original_query else None,
            "route": chosen_route,
            "reason": reason,
            "flags": flags,
            "has_rag": has_rag,
            "has_display": has_display,
            "tool_text_len": len(tool_text) if tool_text else 0,
            "answer_len": len(answer) if answer else 0,
        })

        # 6. Metadata
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