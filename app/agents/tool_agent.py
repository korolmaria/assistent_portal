"""
Tool-агент — вызывает инструменты SupplierPortal через LLM function calling.
"""
import json
import re
from typing import Any

from loguru import logger

from app.llm.vllm_client import vllm_client
from app.schemas import ChatHistoryItem
from app.tools.supplier_portal import supplier_portal, SupplierPortalError
from app.tools.tool_registry import (
    TOOLS,
    get_all_specs,
    get_tool,
    params_to_api,
)


class ToolAgentError(Exception):
    """Ошибка Tool-агента."""
    pass


def _safe_float(val) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _fmt_qty(val) -> str:
    num = _safe_float(val)
    if num is None:
        return "—"
    return f"{num:g}"


async def extract_params(query: str) -> tuple[str, dict[str, Any]]:
    """
    LLM извлекает имя инструмента и параметры из естественного запроса.
    """
    system_prompt = (
        "Ты — строгий парсер параметров для API. Твоя задача — извлечь из запроса "
        "ТОЛЬКО те фильтры, которые пользователь назвал ЯВНО. Не добавляй ничего от себя.\n"
        "\n"
        "ПРАВИЛА ИЗВЛЕЧЕНИЯ ПАРАМЕТРОВ:\n"
        "1. Номера заказов («заказ 56», «заказы 1 и 2») → order_ids: [56] / [1, 2]\n"
        "2. Бренды: Хит → 2, Грошык → 3, Евроопт → 6 → brand_ids\n"
        "3. Статусы: на сайте → 10, автоподтверждён → 48, "
        "подтверждён → 50, выполнен → 90 → checkx\n"
        "4. Обработка (IsProcessed): ставь ТОЛЬКО если явно сказано «необработанные» (0) или «обработанные» (1).\n"
        "   ⚠️ Слова «неподтвержденные», «подтвержденные», «на сайте», «автоподтверждённые» — это СТАТУС (Checkx), а НЕ обработка (IsProcessed)!\n"
        "5. Области: Брест → 71, Витебск → 73, Гомель → 75, "
        "Гродно → 77, Минск → 78, Могилёв → 81 → region_ids\n"
        "6. ПЕРИОДЫ (ВАЖНО!):\n"
        "   - «за октябрь» / «в октябре» → date_begin: 2026-10-01, date_end: 2026-10-31\n"
        "   - «за сентябрь» → date_begin: 2026-09-01, date_end: 2026-09-30\n"
        "   - Всегда ставь ОБА параметра.\n"
        "7. ЧИСЛОВЫЕ артикулы («артикул 615300») → goods_ids: [615300]\n"
        "8. Названия товаров («молоко», «крем») → goods_name: \"молоко\"\n"
        "\n"
        "🔴 ГЛАВНОЕ ПРАВИЛО (ИЗВЛЕКАЙ ТОЛЬКО ЯВНО УКАЗАННОЕ):\n"
        "Если пользователь НЕ назвал фильтр в тексте — НЕ ЗАПОЛНЯЙ его.\n"
        " - Нет слова «молоко» → не ставь goods_name.\n"
        " - Нет слова «подтверждён» / «на сайте» → не ставь checkx.\n"
        " - Нет слова «обработан» / «не обработан» → не ставь is_processed.\n"
        " - Нет месяца → не ставь date_begin.\n"
        " - Нет бренда → не ставь brand_ids.\n"
        "\n"
        "🔴 АБСОЛЮТНОЕ ПРАВИЛО ДЛЯ НОМЕРА ЗАКАЗА:\n"
        "Если в запросе ЕСТЬ номер заказа (например, «заказ 98010227»), "
        "то ставь ТОЛЬКО order_ids: [98010227]. И БОЛЬШЕ НИЧЕГО.\n"
        "НЕ ДОБАВЛЯЙ checkx, date_begin, brand_ids, даже если в тексте есть слова "
        "«подтверждён», «за октябрь» или «Хит». Номер заказа уже уникален.\n"
        "\n"
        "⚠️ НО ЕСЛИ НОМЕРА ЗАКАЗА НЕТ (например, «покажи неподтверждённые заказы за октябрь»):\n"
        "Тогда извлекай ВСЕ фильтры, которые есть в тексте:\n"
        "  - «неподтверждённые» → checkx: [10]\n"
        "  - «за октябрь» → date_begin: 2026-10-01, date_end: 2026-10-31\n"
        "  - «Хит» → brand_ids: [2]\n"
        "\n"
        "ВАЖНО:\n"
        "- Выбор tool делай ТОЛЬКО по description в списке tools.\n"
        "- Не выдумывай параметры, которых нет в схеме tool.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    message = await vllm_client.chat_with_tools(
        messages=messages,
        tools=get_all_specs(),
        tool_choice="auto",
        temperature=0.1,
    )

    if not message.tool_calls:
        logger.warning(f"[tool_agent] LLM не выбрал инструмент для: {query}")
        raise ToolAgentError(
            "Не удалось определить, какой инструмент нужен для этого запроса."
        )

    tool_call = message.tool_calls[0]
    tool_name = tool_call.function.name

    try:
        params = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError as e:
        logger.error(f"[tool_agent] плохой JSON от LLM: {tool_call.function.arguments}")
        raise ToolAgentError(f"Ошибка парсинга параметров: {e}") from e

    # === ПРЕДОХРАНИТЕЛЬ 1: чистим лишние фильтры, если есть номер заказа ===
    has_order_id = "order_ids" in params and params["order_ids"]
    if has_order_id:
        forbidden = ["checkx", "date_begin", "date_end", "brand_ids", "is_processed", "goods_ids", "goods_name"]
        for f in forbidden:
            if f in params:
                logger.warning(f"[tool_agent] удаляю лишний фильтр '{f}' из запроса (есть order_ids)")
                del params[f]

    # === ПРЕДОХРАНИТЕЛЬ 2: is_processed только если явно сказано "обработан"/"необработан" ===
    query_lower = query.lower()
    if "is_processed" in params:
        has_explicit_processed = any(w in query_lower for w in ["обработан", "необработан", "не обработан"])
        if not has_explicit_processed:
            logger.warning("[tool_agent] удаляю is_processed (не было явного упоминания обработки)")
            del params["is_processed"]
    # =====================================================================

    logger.info(f"[tool_agent] LLM выбрал: {tool_name}({params})")
    return tool_name, params


def build_display_block(orders: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not orders:
        return None
    if len(orders) == 1:
        return {"type": "card", "data": orders[0], "extra": {}}
    total_sum = sum(float(o.get("sum") or 0) for o in orders if o.get("sum") is not None)
    return {"type": "table", "data": orders, "extra": {"count": len(orders), "total_sum": total_sum}}


def build_items_display_block(
    items: list[dict[str, Any]], amount: int, page_qty: int, page_number: int = 1, limit: int = 30
) -> dict[str, Any] | None:
    if not items:
        return None
    total_sum = 0.0
    for i in items:
        v = _safe_float(i.get("order_sum"))
        if v is not None:
            total_sum += v
    return {
        "type": "order_items_table",
        "data": items,
        "extra": {
            "count": len(items), "total_sum": total_sum, "amount": amount,
            "page_qty": page_qty, "page_number": page_number, "limit": limit,
            "has_more": page_number < page_qty,
        },
    }


def build_human_summary(orders: list[dict[str, Any]]) -> str:
    if not orders:
        return "По вашему запросу заказы не найдены."
    count = len(orders)
    total = sum(float(o.get("sum") or 0) for o in orders if o.get("sum") is not None)
    if count == 1:
        o = orders[0]
        order_id = o.get("order_id")
        status = o.get("status", "—")
        contractor = o.get("contractor") or "поставщик"
        return f"Найден 1 заказ №{order_id} от {contractor} на сумму {total:,.2f} руб. Статус: {status}."
    return f"Найдено {count} заказов на общую сумму {total:,.2f} руб. Подробности в таблице ниже."


def build_items_human_summary(items: list[dict[str, Any]], amount: int, page_qty: int) -> str:
    if not items:
        return "В указанных заказах товары не найдены."
    count = len(items)
    order_ids = sorted({i.get("order_id") for i in items if i.get("order_id")})
    orders_part = f"в заказе №{order_ids[0]}" if len(order_ids) == 1 else f"в {len(order_ids)} заказах"
    total = 0.0
    for i in items:
        v = _safe_float(i.get("order_sum"))
        if v is not None:
            total += v
    if count == 1 and page_qty <= 1:
        it = items[0]
        return (
            f"В заказе №{order_ids[0]} найден 1 товар: «{it.get('goods_name') or '—'}» "
            f"(арт. {it.get('goods_id') or '—'}). Заказано {_fmt_qty(it.get('quantity_order'))} шт. "
            f"на сумму {total:,.2f} руб. Статус: {it.get('status') or '—'}."
        )
    if page_qty <= 1:
        return f"Найдено {count} позиций {orders_part} на общую сумму {total:,.2f} руб. Подробности в таблице ниже."
    return f"Найдено {amount} позиций {orders_part}. Показаны первые {count} на сумму {total:,.2f} руб. (стр. 1 из {page_qty})."


async def run_tool_agent(
    query: str,
    jwt: str,
    forced_tool: str | None = None,
    history: list[ChatHistoryItem] | None = None,
) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
    metadata: dict[str, Any] = {"route": "tool", "tools_used": []}

    # Костыль для склейки истории (на случай, если query_rewriter не сработал)
    search_query = query
    if history:
        q_lower = query.lower()
        is_followup = any(pron in q_lower for pron in ["его", "этот", "там", "него", "про него", "у него", "выше", "этом", "данном"]) or len(query) < 20
        if is_followup:
            for msg in reversed(history):
                match = re.search(r'\b(980\d{5})\b', msg.content)
                if match:
                    order_id = match.group(1)
                    if not re.search(r'\b(980\d{5})\b', query):
                        search_query = f"{query} (заказ {order_id})"
                        logger.info(f"[tool_agent] подставили заказ {order_id} из истории")
                    break

    if forced_tool:
        tool_name = forced_tool
        params: dict[str, Any] = {}
    else:
        tool_name, params = await extract_params(search_query)

    tool = get_tool(tool_name)
    if not tool:
        raise ToolAgentError(f"Инструмент '{tool_name}' не зарегистрирован.")

    metadata["tools_used"].append(tool_name)
    api_data = params_to_api(params, tool_name)
    logger.info(f"[tool_agent] вызов {tool['api_method']} с {api_data}")

    try:
        raw = await supplier_portal.call(method=tool["api_method"], data=api_data, jwt=jwt)
    except SupplierPortalError as e:
        logger.error(f"[tool_agent] SupplierPortal error: {e}")
        raise ToolAgentError(f"Ошибка API портала: {e.description}") from e
    except Exception as e:
        logger.exception("[tool_agent] ошибка вызова SupplierPortal")
        raise ToolAgentError(f"Не удалось получить данные: {e}") from e

    parser = tool["parser"]
    parsed = parser(raw)

    if tool_name == "get_order_items":
        items = parsed["items"]
        amount = parsed["amount"]
        page_qty = parsed["page_qty"]
        logger.info(f"[tool_agent] получено {len(items)} позиций, amount={amount}, page_qty={page_qty}")
        text = build_items_human_summary(items, amount, page_qty)
        display = build_items_display_block(
            items, amount, page_qty,
            page_number=api_data.get("PageNumber", 1),
            limit=api_data.get("Limit", 30),
        )
    else:
        orders = parsed
        logger.info(f"[tool_agent] получено {len(orders)} заказов")
        text = build_human_summary(orders)
        display = build_display_block(orders)

    return text, display, metadata