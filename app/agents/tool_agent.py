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
# Извлечение параметров через LLM
# ==========================================================================
async def extract_params(query: str) -> tuple[str, dict[str, Any]]:
    """
    LLM извлекает имя инструмента и параметры из естественного запроса.
    Возвращает (tool_name, params).
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Ты — ассистент, который преобразует запросы пользователя "
                "в вызовы API SupplierPortal.\n\n"
                "Доступные инструменты:\n"
                "- get_supplier_orders — список заказов поставщика\n\n"
                "ПРАВИЛА ИЗВЛЕЧЕНИЯ ПАРАМЕТРОВ:\n"
                "1. Если пользователь называет конкретные номера заказов "
                "(«заказ 56», «заказы 1 и 2») — используй order_ids: [56] или [1, 2].\n"
                "2. Если упоминается бренд — используй brand_ids:\n"
                "   - Хит → [2]\n"
                "   - Грошык → [3]\n"
                "   - Евроопт → [6]\n"
                "3. Если упоминается статус заказа — используй checkx:\n"
                "   - на сайте → [10]\n"
                "   - автоподтверждён → [48]\n"
                "   - подтверждён → [50]\n"
                "   - выполнен → [90]\n"
                "4. Если упоминается обработка:\n"
                "   - не обработан → is_processed: 0\n"
                "   - обработан → is_processed: 1\n"
                "5. Если упоминается область:\n"
                "   - Брест → region_ids: [71]\n"
                "   - Витебск → region_ids: [73]\n"
                "   - Гомель → region_ids: [75]\n"
                "   - Гродно → region_ids: [77]\n"
                "   - Минск → region_ids: [78]\n"
                "   - Могилёв → region_ids: [81]\n"
                "6. Если упоминается период:\n"
                "   - «за сентябрь» → date_begin: 2026-09-01, date_end: 2026-09-30\n"
                "   - «за последнюю неделю» → вычисли от текущей даты\n"
                "   - «за последние 3 месяца» → вычисли от текущей даты\n"
                "7. Даты передавай в формате YYYY-MM-DD.\n"
                "8. Если фильтры не указаны — оставь пустыми (вернутся все заказы).\n"
            ),
        },
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

    logger.info(f"[tool_agent] LLM выбрал: {tool_name}({params})")
    return tool_name, params


# ==========================================================================
# Построение DisplayBlock для фронта
# ==========================================================================
def build_display_block(orders: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Строит DisplayBlock для фронта.
    - 0 заказов → None
    - 1 заказ → card
    - много → table с extra
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


def build_human_summary(orders: list[dict[str, Any]]) -> str:
    """Короткий текстовый ответ для пользователя."""
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
    api_data = params_to_api(params)
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
    orders = parser(raw)
    logger.info(f"[tool_agent] получено {len(orders)} заказов")

    # 4. Строим ответ
    text = build_human_summary(orders)
    display = build_display_block(orders)

    return text, display, metadata
