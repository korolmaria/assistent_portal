"""
Tool-агент — вызывает инструменты SupplierPortal через LLM function calling.
"""
import json
from typing import Any

from loguru import logger

from app.llm.vllm_client import vllm_client
from app.tools.supplier_portal import supplier_portal, SupplierPortalError
from app.tools.tool_registry import (
    get_all_specs,
    get_tool,
    params_to_api,
)


class ToolAgentError(Exception):
    """Ошибка Tool-агента."""
    pass


async def extract_params(query: str) -> tuple[str, dict[str, Any]]:
    """LLM извлекает имя инструмента и параметры из естественного запроса."""
    messages = [
        {
            "role": "system",
            "content": (
                "Ты — ассистент, который преобразует запросы пользователя "
                "в вызовы API. Если пользователь просит данные о заказах — "
                "вызови функцию get_supplier_orders с подходящими параметрами. "
                "Если параметры не указаны — оставь их пустыми. "
                "Даты указывай в формате YYYY-MM-DD."
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


def build_display_block(orders: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Строит DisplayBlock для фронта."""
    if not orders:
        return None

    if len(orders) == 1:
        return {"type": "card", "data": orders[0], "extra": {}}

    total_sum = sum(
        float(o.get("sum") or 0) for o in orders if o.get("sum") is not None
    )
    return {
        "type": "table",
        "data": orders,
        "extra": {"count": len(orders), "total_sum": total_sum},
    }


def build_human_summary(orders: list[dict[str, Any]]) -> str:
    """Короткий текстовый ответ для пользователя."""
    if not orders:
        return "По вашему запросу заказы не найдены."

    count = len(orders)
    total = sum(float(o.get("sum") or 0) for o in orders if o.get("sum") is not None)

    if count == 1:
        o = orders[0]
        return (
            f"Найден 1 заказ №{o.get('order_id')} "
            f"от {o.get('contractor') or 'поставщика'} "
            f"на сумму {total:,.2f} руб. "
            f"Статус: {o.get('status')}."
        )

    return (
        f"Найдено {count} заказов на общую сумму "
        f"{total:,.2f} руб. Подробности в таблице ниже."
    )


async def run_tool_agent(
    query: str,
    jwt: str,
    forced_tool: str | None = None,
) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
    """Главный вход Tool-агента."""
    metadata: dict[str, Any] = {"route": "tool", "tools_used": []}

    if forced_tool:
        tool_name = forced_tool
        params: dict[str, Any] = {}
    else:
        tool_name, params = await extract_params(query)

    tool = get_tool(tool_name)
    if not tool:
        raise ToolAgentError(f"Инструмент '{tool_name}' не зарегистрирован.")

    metadata["tools_used"].append(tool_name)

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

    parser = tool["parser"]
    orders = parser(raw)
    logger.info(f"[tool_agent] получено {len(orders)} заказов")

    text = build_human_summary(orders)
    display = build_display_block(orders)

    return text, display, metadata
