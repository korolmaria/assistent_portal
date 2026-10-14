"""
Supervisor — оркестратор мультиагентной системы.

Маршрутизация:
- auto: LLM решает, какие агенты нужны
- rag: только RAG-поиск по документам
- tool: только Tool (SupplierPortal API)
- rag+tool: комбинированный (правила + данные)

Возвращает генератор SSE-событий для стриминга на фронт.

ВАЖНО: LLM-роутер только КЛАССИФИЦИРУЕТ запрос (rag / tool / rag+tool).
Он НЕ переформулирует его — оригинальный query идёт дальше в агенты.
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
     • «Могу ли подтвердить заказы за сентябрь?» → rag+tool (фильтр = период)
     • «Можно ли отменить заказ 98010205?» → rag+tool (фильтр = номер)
     • «Как отменить заказы бренда Хит?» → rag+tool (фильтр = бренд)
     • «Разрешено ли подтверждать заказы по Минску?» → rag+tool (фильтр = регион)

   → Если **НЕТ фильтра** И вопрос про **правила/возможность** → **rag**
     • «Могу ли я массово экспортировать заказы?» → rag (нет фильтра)
     • «Как экспортировать заказы в Excel?» → rag (инструкция)
     • «Какие массовые операции доступны?» → rag (общие правила)
     • «Можно ли отменить заказ?» → rag (без фильтра, общее правило)

4. Если непонятно → **rag**

ПРИМЕРЫ (запрос → route):
# Чистый RAG
- «как подтвердить заказ» → rag
- «каким методом подтвердить заказ» → rag
- «как технически подтвердить заказ» → rag
- «могу ли я массово экспортировать заказы?» → rag (нет фильтра)
- «как экспортировать заказы в Excel?» → rag
- «какие массовые операции доступны?» → rag
- «какие правила отмены заказов?» → rag

# Чистый Tool
- «покажи мои заказы» → tool
- «дай данные по заказу 56» → tool
- «покажи заказ 98010205» → tool
- «сколько заказов за сентябрь» → tool
- «какие товары в заказе 98010225» → tool
- «заказывали ли молоко в заказе 98010225» → tool
- «какой статус у заказа 98010205» → tool

# Комбинированный rag+tool (есть фильтр + вопрос про правила)
- «могу ли отменить заказ 98010205» → rag+tool
- «можно ли подтвердить заказ 98010205» → rag+tool
- «могу ли подтвердить заказы за сентябрь?» → rag+tool
- «разрешено ли отменить заказы бренда Хит?» → rag+tool
- «как отменить заказы по Минску?» → rag+tool
- «покажи заказы за сентябрь и расскажи как их подтвердить» → rag+tool

Формат ответа (строго JSON, без markdown):
{
  "route": "rag" | "tool" | "rag+tool",
  "reason": "краткое объяснение (1 предложение)"
}
"""


async def decide_route(query: str) -> dict[str, str]:
    """
    LLM решает, какой маршрут использовать.

    ВАЖНО: роутер только КЛАССИФИЦИРУЕТ запрос.
    Он НЕ переформулирует его — оригинальный query идёт дальше в агенты.

    Возвращает:
        {"route": "rag" | "tool" | "rag+tool", "reason": "..."}
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
        }

    except Exception as e:
        logger.warning(f"[supervisor] ошибка роутинга: {e}, fallback → rag")
        return {"route": "rag", "reason": "fallback"}


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

    ВАЖНО: query — оригинальный запрос пользователя. Он идёт:
    - в decide_route (для классификации)
    - в rag_search (для поиска по документам)
    - в run_tool_agent (для выбора tool и извлечения параметров)
    - в synthesize (для формирования ответа)
    """
    try:
        # 1. Определяем маршрут
        if route == "auto":
            decision = await decide_route(query)
            chosen_route = decision["route"]
            reason = decision["reason"]
        else:
            chosen_route = route
            reason = f"выбрано пользователем: {route}"

        logger.info(f"[supervisor] chosen_route={chosen_route}, query={query!r}")

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
                # ⚠️ Передаём ОРИГИНАЛЬНЫЙ query, без переформулировок
                text, display_block, tool_meta = await run_tool_agent(
                    query=query,
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
                logger.exception(f"[supervisor] Tool-ошибка: {e}")
                yield {"type": "step", "message": f"   ⚠️ ошибка API: {e}"}

        # 3. Synthesis
        #
        # ВАЖНО: для tool-запросов с display текст УЖЕ готов в tool-агенте
        # (build_items_human_summary / build_human_summary). НЕ перегенерируем
        # его через LLM — иначе модель может галлюцинировать числа и позиции.
        yield {"type": "step", "message": "✍️ Генерация ответа..."}

        tool_text = ""
        for tr in tool_results:
            if tr.get("text"):
                tool_text = tr["text"]
                break

        has_display = display_block is not None
        has_rag = bool(rag_chunks)

        if has_display and not has_rag and tool_text:
            # Используем готовый текст tool-агента как есть
            logger.info(
                f"[supervisor] используем готовый text tool-агента "
                f"({len(tool_text)} симв.), synthesize пропускаем"
            )
            answer = tool_text
            final_sources = []
            synth_tools = tools_used
        else:
            answer, final_sources, synth_tools = await synthesize(
                query=query,
                rag_chunks=rag_chunks,
                tool_results=tool_results,
            )

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
