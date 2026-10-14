"""
Tool-агент — вызывает инструменты SupplierPortal через LLM function calling.

Логика:
1. LLM получает запрос пользователя + список инструментов
2. LLM выбирает инструмент и извлекает параметры (function calling)
3. Агент вызывает API
4. Парсит ответ → строит DisplayBlock для фронта
5. Возвращает (текст, DisplayBlock, metadata)
"""
import json
from typing import Any

from loguru import logger

from app.llm.vllm_client import vllm_client
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


# ==========================================================================
# Хелперы для безопасного форматирования
# ==========================================================================
def _safe_float(val) -> float | None:
    """Строка/число → float, или None."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _fmt_qty(val) -> str:
    """Количество: '9' → '9', '9.5' → '9.5', None → '—'."""
    num = _safe_float(val)
    if num is None:
        return "—"
    return f"{num:g}"


# ==========================================================================
# Извлечение параметров через LLM
# ==========================================================================
async def extract_params(query: str) -> tuple[str, dict[str, Any]]:
    """
    LLM извлекает имя инструмента и параметры из естественного запроса.
    Возвращает (tool_name, params).
    """
    system_prompt = (
        "Ты — ассистент, который извлекает параметры из запроса "
        "пользователя.\n"
        "\n"
        "ТЕБЕ ДАЮТ:\n"
        "- запрос пользователя\n"
        "- список доступных tools (описания + JSON-схемы)\n"
        "- ты должен ВЫБРАТЬ tool и ЗАПОЛНИТЬ его параметры\n"
        "\n"
        "ПРАВИЛА ИЗВЛЕЧЕНИЯ ПАРАМЕТРОВ:\n"
        "1. Номера заказов («заказ 56», «заказы 1 и 2») → "
        "order_ids: [56] / [1, 2]\n"
        "2. Бренды: Хит → 2, Грошык → 3, Евроопт → 6 → brand_ids\n"
        "3. Статусы: на сайте → 10, автоподтверждён → 48, "
        "подтверждён → 50, выполнен → 90 → checkx\n"
        "4. Обработка: не обработан → 0, обработан → 1 → is_processed\n"
        "5. Области: Брест → 71, Витебск → 73, Гомель → 75, "
        "Гродно → 77, Минск → 78, Могилёв → 81 → region_ids\n"
        "6. Периоды: «за сентябрь 2026» → date_begin: 2026-09-01, "
        "date_end: 2026-09-30\n"
        "7. Даты — строго в формате YYYY-MM-DD\n"
        "8. ЧИСЛОВЫЕ артикулы («артикул 615300», «товар 1001») "
        "→ goods_ids: [615300]\n"
        "9. Названия товаров («молоко», «крем», «сливки») → "
        "goods_name: \"молоко\" (goods_ids НЕ заполняй)\n"
        "10. Если пользователь не указал фильтры — оставь параметры пустыми\n"
        "\n"
        "🔴 ГЛАВНОЕ ПРАВИЛО — ИЗВЛЕКАЙ ТОЛЬКО ЯВНО УКАЗАННОЕ:\n"
        "- Есть «заказ 98010225» → order_ids: [98010225]\n"
        "- Нет номера заказа в тексте → order_ids НЕ заполняй\n"
        "- Есть «артикул 615300» / «товар 1001» → goods_ids: [615300]\n"
        "- Нет числового артикула → goods_ids НЕ заполняй\n"
        "- Есть «молоко», «крем», «сливки» → goods_name: \"молоко\"\n"
        "- Нет названия товара → goods_name НЕ заполняй\n"
        "\n"
        "❌ НИКОГДА НЕ ДОДУМЫВАЙ: не угадывай артикулы, ID заказов, "
        "бренды, регионы. Если чего-то нет в тексте — оставь поле пустым.\n"
        "\n"
        "ВАЖНО:\n"
        "- Выбор tool делай ТОЛЬКО по description в списке tools.\n"
        "- Не выдумывай параметры, которых нет в схеме tool.\n"
        "- Если не уверен в tool — выбери тот, у которого в description "
        "есть подходящий пример «✅ ИСПОЛЬЗУЙ КОГДА».\n"
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
        logger.error(
            f"[tool_agent] плохой JSON от LLM: {tool_call.function.arguments}"
        )
        raise ToolAgentError(f"Ошибка парсинга параметров: {e}") from e

    logger.info(f"[tool_agent] LLM выбрал: {tool_name}({params})")
    return tool_name, params


# ==========================================================================
# DisplayBlock для заказов (GetOrderBuyGroup)
# ==========================================================================
def build_display_block(orders: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    DisplayBlock для заказов.
    - 0 заказов → None
    - 1 заказ   → card
    - много     → table
    """
    if not orders:
        return None

    if len(orders) == 1:
        return {
            "type": "card",
            "data": orders[0],
            "extra": {},
        }

    total_sum = sum(
        float(o.get("sum") or 0)
        for o in orders
        if o.get("sum") is not None
    )

    return {
        "type": "table",
        "data": orders,
        "extra": {
            "count": len(orders),
            "total_sum": total_sum,
        },
    }


# ==========================================================================
# DisplayBlock для товаров (GetOrderBuy)
# ==========================================================================
def build_items_display_block(
    items: list[dict[str, Any]],
    amount: int,
    page_qty: int,
    page_number: int = 1,
    limit: int = 30,
) -> dict[str, Any] | None:
    """
    DisplayBlock для списка товаров внутри заказа.
    type = "order_items_table"
    """
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
            "count": len(items),
            "total_sum": total_sum,
            "amount": amount,
            "page_qty": page_qty,
            "page_number": page_number,
            "limit": limit,
            "has_more": page_number < page_qty,
        },
    }


# ==========================================================================
# Текстовое резюме для заказов
# ==========================================================================
def build_human_summary(orders: list[dict[str, Any]]) -> str:
    """Короткий текстовый ответ для заказов."""
    if not orders:
        return "По вашему запросу заказы не найдены."

    count = len(orders)
    total = sum(
        float(o.get("sum") or 0)
        for o in orders
        if o.get("sum") is not None
    )

    if count == 1:
        o = orders[0]
        order_id = o.get("order_id")
        status = o.get("status", "—")
        contractor = o.get("contractor") or "поставщик"
        return (
            f"Найден 1 заказ №{order_id} "
            f"от {contractor} на сумму {total:,.2f} руб. "
            f"Статус: {status}."
        )

    return (
        f"Найдено {count} заказов на общую сумму "
        f"{total:,.2f} руб. Подробности в таблице ниже."
    )


# ==========================================================================
# Текстовое резюме для товаров
# ==========================================================================
def build_items_human_summary(
    items: list[dict[str, Any]],
    amount: int,
    page_qty: int,
) -> str:
    """
    Короткий текстовый ответ по списку товаров.
    Структура повторяет логику build_human_summary (для заказов).
    """
    if not items:
        return "В указанных заказах товары не найдены."

    count = len(items)
    order_ids = sorted({
        i.get("order_id") for i in items if i.get("order_id")
    })

    if len(order_ids) == 1:
        orders_part = f"в заказе №{order_ids[0]}"
    else:
        orders_part = f"в {len(order_ids)} заказах"

    total = 0.0
    for i in items:
        v = _safe_float(i.get("order_sum"))
        if v is not None:
            total += v

    # 1 позиция — подробная карточка
    if count == 1 and page_qty <= 1:
        it = items[0]
        goods_name = it.get("goods_name") or "—"
        goods_id = it.get("goods_id") or "—"
        qty_str = _fmt_qty(it.get("quantity_order"))
        status = it.get("status") or "—"
        return (
            f"В заказе №{order_ids[0]} найден 1 товар: "
            f"«{goods_name}» (арт. {goods_id}). "
            f"Заказано {qty_str} шт. на сумму {total:,.2f} руб. "
            f"Статус: {status}."
        )

    # Несколько позиций без пагинации
    if page_qty <= 1:
        return (
            f"Найдено {count} позиций {orders_part} "
            f"на общую сумму {total:,.2f} руб. "
            f"Подробности в таблице ниже."
        )

    # С пагинацией
    return (
        f"Найдено {amount} позиций {orders_part}. "
        f"Показаны первые {count} на сумму {total:,.2f} руб. "
        f"(стр. 1 из {page_qty})."
    )


# ==========================================================================
# Главный вход
# ==========================================================================
async def run_tool_agent(
    query: str,
    jwt: str,
    forced_tool: str | None = None,
) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
    """
    Главный вход Tool-агента.

    Args:
        query: запрос пользователя
        jwt: JWT-токен от фронта (прокидываем в SupplierPortal)
        forced_tool: если нужно вызвать конкретный инструмент (без LLM)

    Returns:
        (текст для пользователя, DisplayBlock, метаданные)
    """
    metadata: dict[str, Any] = {
        "route": "tool",
        "tools_used": [],
    }

    # 1. Определяем инструмент и параметры
    if forced_tool:
        tool_name = forced_tool
        params: dict[str, Any] = {}
    else:
        tool_name, params = await extract_params(query)

    tool = get_tool(tool_name)
    if not tool:
        raise ToolAgentError(f"Инструмент '{tool_name}' не зарегистрирован.")

    metadata["tools_used"].append(tool_name)

    # 2. Вызываем API
    api_data = params_to_api(params, tool_name)
    logger.info(f"[tool_agent] вызов {tool['api_method']} с {api_data}")

    try:
        raw = await supplier_portal.call(
            method=tool["api_method"],
            data=api_data,
            jwt=jwt,
        )
    except SupplierPortalError as e:
        logger.error(f"[tool_agent] SupplierPortal error: {e}")
        raise ToolAgentError(f"Ошибка API портала: {e.description}") from e
    except Exception as e:
        logger.exception("[tool_agent] ошибка вызова SupplierPortal")
        raise ToolAgentError(f"Не удалось получить данные: {e}") from e

    # 3. Парсим ответ
    parser = tool["parser"]
    parsed = parser(raw)

    # 4. Строим ответ — диспетчер по имени tool
    if tool_name == "get_order_items":
        items = parsed["items"]
        amount = parsed["amount"]
        page_qty = parsed["page_qty"]
        logger.info(
            f"[tool_agent] получено {len(items)} позиций, "
            f"amount={amount}, page_qty={page_qty}"
        )
        text = build_items_human_summary(items, amount, page_qty)
        display = build_items_display_block(
            items,
            amount,
            page_qty,
            page_number=api_data.get("PageNumber", 1),
            limit=api_data.get("Limit", 30),
        )
    else:
        orders = parsed
        logger.info(f"[tool_agent] получено {len(orders)} заказов")
        text = build_human_summary(orders)
        display = build_display_block(orders)

    return text, display, metadata
