"""
Tool-агент — вызывает инструменты SupplierPortal через LLM function calling.
Поддерживает цепочки вызовов (Multi-Tool).
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


# Статусы претензий для отображения на фронте
CLAIM_STATUS_MAP = {
    10: ("Ожидает ответа", "warning"),
    20: ("Подтверждена", "success"),
    30: ("Отклонена", "error"),
    40: ("Ожидает ответа", "warning"),
    50: ("Подтверждена", "success"),
    51: ("Отклонена", "error"),
}


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


# ==========================================================================
# Билдеры для заказов
# ==========================================================================
def build_display_block(orders: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not orders:
        return None
    if len(orders) == 1:
        return {"type": "card", "data": orders[0], "extra": {}}
    total_sum = sum(float(o.get("sum") or 0) for o in orders if o.get("sum") is not None)
    return {"type": "table", "data": orders, "extra": {"count": len(orders), "total_sum": total_sum}}


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


# ==========================================================================
# Билдеры для товаров в заказе
# ==========================================================================
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


# ==========================================================================
# Билдеры для претензий
# ==========================================================================
def build_claims_summary(claims: list[dict[str, Any]]) -> str:
    """Короткий текстовый ответ по списку претензий."""
    if not claims:
        return "По вашему запросу претензии не найдены."

    count = len(claims)
    if count == 1:
        c = claims[0]
        return f"Найдена 1 претензия №{c.get('claim_id')} по заказу №{c.get('order_id')}."
    
    return f"Найдено {count} претензий. Подробности в таблице ниже."


def build_claims_display(claims: list[dict[str, Any]]) -> dict[str, Any] | None:
    """DisplayBlock для претензий: обогащает статусы и отдает таблицу или карточку."""
    if not claims:
        return None

    enriched_claims = []
    for c in claims:
        status_code = c.get("status")
        
        # Переводим код в текст и цвет
        if isinstance(status_code, (int, str)) and str(status_code).isdigit():
            status_text, status_type = CLAIM_STATUS_MAP.get(
                int(status_code), (f"Код {status_code}", "default")
            )
        else:
            status_text, status_type = str(status_code or "—"), "default"

        enriched_claims.append({
            "claim_id": c.get("claim_id"),
            "created_at": str(c.get("created_at", "")).split(" ")[0],
            "status": status_text,
            "status_type": status_type,
            "order_id": c.get("order_id"),
            "warehouse": c.get("warehouse"),
            "shipment_date": c.get("shipment_date"),
        })

    if len(enriched_claims) == 1:
        return {"type": "claims_card", "data": enriched_claims[0], "extra": {}}

    return {
        "type": "claims_table",
        "data": enriched_claims,
        "extra": {"count": len(enriched_claims)},
    }


# ==========================================================================
# Билдеры для детализации претензий
# ==========================================================================
def build_claim_details_summary(items: list[dict[str, Any]]) -> str:
    if not items:
        return "По указанным претензиям детали не найдены."
    
    claim_ids = sorted({i.get("claim_id") for i in items if i.get("claim_id")})
    claims_str = ", ".join(str(c) for c in claim_ids)
    
    total_penalty = sum(float(i.get("total_penalty") or 0) for i in items if i.get("total_penalty") is not None)
    
    return (
        f"Найдено {len(items)} позиций в претензии №{claims_str}. "
        f"Общая сумма штрафа: {total_penalty:,.2f} руб."
    )


def build_claim_details_display(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items:
        return None
    
    total_penalty = sum(float(i.get("total_penalty") or 0) for i in items if i.get("total_penalty") is not None)
    
    return {
        "type": "claim_details_table",
        "data": items,
        "extra": {
            "count": len(items),
            "total_penalty": total_penalty
        },
    }


# ==========================================================================
# Форматирование результата для одного инструмента
# ==========================================================================
def _format_tool_result(tool_name: str, parsed: Any, api_data: dict) -> tuple[str, dict | None]:
    if tool_name == "get_order_items":
        items = parsed["items"]
        amount = parsed["amount"]
        page_qty = parsed["page_qty"]
        text = build_items_human_summary(items, amount, page_qty)
        display = build_items_display_block(
            items, amount, page_qty,
            page_number=api_data.get("PageNumber", 1),
            limit=api_data.get("Limit", 30),
        )
        return text, display

    elif tool_name == "get_supplier_claims":
        claims = parsed
        text = build_claims_summary(claims)
        display = build_claims_display(claims)
        return text, display

    elif tool_name == "get_claim_details":
        items = parsed
        text = build_claim_details_summary(items)
        display = build_claim_details_display(items)
        return text, display

    else:
        orders = parsed
        text = build_human_summary(orders)
        display = build_display_block(orders)
        return text, display


# ==========================================================================
# Главный запуск
# ==========================================================================
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

    # Если инструмент задан принудительно — вызываем только его
    if forced_tool:
        tool_name = forced_tool
        tool = get_tool(tool_name)
        if not tool:
            raise ToolAgentError(f"Инструмент '{tool_name}' не зарегистрирован.")
        
        metadata["tools_used"].append(tool_name)
        api_data = params_to_api({}, tool_name)
        logger.info(f"[tool_agent] вызов {tool['api_method']} с {api_data}")
        
        raw = await supplier_portal.call(method=tool["api_method"], data=api_data, jwt=jwt)
        parsed = tool["parser"](raw)
        text, display = _format_tool_result(tool_name, parsed, api_data)
        return text, display, metadata

    # === ЦИКЛ ВЫЗОВА ИНСТРУМЕНТОВ (Multi-Tool) ===
    system_prompt = (
        "Ты — ассистент, который вызывает инструменты API для получения данных.\n"
        "Если для ответа нужно несколько действий (например, сначала найти ID претензии, "
        "а потом посмотреть её детали), вызывай инструменты последовательно.\n"
        "Не выдумывай параметры. Если не хватает данных, вызывай следующий инструмент.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": search_query},
    ]

    last_text = "Не удалось получить данные."
    last_display = None

    # Максимум 3 шага, чтобы избежать бесконечных циклов
    for step in range(3):
        message = await vllm_client.chat_with_tools(
            messages=messages,
            tools=get_all_specs(),
            tool_choice="auto",
            temperature=0.1,
        )

        if not message.tool_calls:
            break  # Инструменты больше не нужны, выходим из цикла

        # Добавляем сообщение от ассистента с tool_calls в историю
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                } for tc in message.tool_calls
            ]
        })

        # Выполняем ВСЕ инструменты, которые запросила модель на этом шаге
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                params = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                params = {}

            # Применяем предохранители (защита от лишних фильтров)
            has_order_id = "order_ids" in params and params["order_ids"]
            if has_order_id and tool_name not in ("get_supplier_claims", "get_claim_details"):
                forbidden = ["checkx", "date_begin", "date_end", "brand_ids", "is_processed", "goods_ids", "goods_name"]
                for f in forbidden:
                    if f in params:
                        del params[f]
            
            query_lower = query.lower()
            if "is_processed" in params:
                if not any(w in query_lower for w in ["обработан", "необработан", "не обработан"]):
                    del params["is_processed"]

            tool = get_tool(tool_name)
            if not tool:
                logger.warning(f"[tool_agent] LLM вызвала неизвестный инструмент: {tool_name}")
                continue

            logger.info(f"[tool_agent] Шаг {step+1}. LLM выбрал: {tool_name}({params})")
            metadata["tools_used"].append(tool_name)

            api_data = params_to_api(params, tool_name)
            logger.info(f"[tool_agent] вызов {tool['api_method']} с {api_data}")

            try:
                raw = await supplier_portal.call(method=tool["api_method"], data=api_data, jwt=jwt)
                parsed = tool["parser"](raw)
                text, display = _format_tool_result(tool_name, parsed, api_data)
                
                # Сохраняем последний результат
                last_text = text
                if display:
                    last_display = display

                # Отправляем результат обратно в LLM, чтобы она могла решить, что делать дальше
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"{text}\nДанные: {json.dumps(display, ensure_ascii=False)[:2000]}" if display else text
                })

            except Exception as e:
                logger.error(f"[tool_agent] Ошибка вызова {tool_name}: {e}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Ошибка API: {e}"
                })

    return last_text, last_display, metadata