"""
Инструменты для претензий (список и детализация).
"""
from typing import Any

from loguru import logger

from app.tools.tool_registry.common import _to_float, _to_int


# ==========================================================================
# Справочники статусов
# ==========================================================================
CLAIM_STATUS_LABELS = {
    "40": ("Ожидает ответа", "warning"),
    "50": ("Подтверждена", "success"),
    "51": ("Отклонена", "error"),
}


# ==========================================================================
# 1. СПИСОК ПРЕТЕНЗИЙ (DeliveryClaimGroupGet)
# ==========================================================================
def parse_claims_response(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Ответ DeliveryClaimGroupGet → список претензий."""
    table = raw.get("Table", [])
    if not isinstance(table, list):
        table = []

    claims: list[dict[str, Any]] = []
    for row in table:
        if not isinstance(row, dict):
            continue
        if "ClaimGroupId" not in row:
            continue

        claims.append({
            "claim_id": row.get("ClaimGroupId"),
            "created_at": row.get("DateCreate"),
            "status": row.get("CheckxSupplier"),
            "order_id": row.get("OrderBuyGroupId"),
            "warehouse": row.get("WarehouseId"),
            "shipment_date": row.get("Datex"),
            "manager": row.get("TimeUser3"),
            "sum": _to_float(row.get("Summ")),
            "amount": row.get("Amount"),
        })

    logger.info(f"[tool_registry] parsed {len(claims)} claims")
    return claims


SPEC_CLAIMS = {
    "type": "function",
    "function": {
        "name": "get_supplier_claims",
        "description": (
            "Получить СПИСОК ПРЕТЕНЗИЙ поставщика (брак, возвраты, недопоставки, рекламации).\n\n"
            "✅ ИСПОЛЬЗУЙ КОГДА пользователь спрашивает:\n"
            "• «покажи мои претензии»\n"
            "• «претензии за октябрь»\n"
            "• «есть ли претензии по заказу X»\n"
            "• «сколько претензий в работе»\n"
            "• «расскажи о претензиях»\n"
            "• «какие у меня есть браки/возвраты»\n\n"
            "❌ НЕ ИСПОЛЬЗУЙ:\n"
            "- Для заказов (даже если в тексте есть слово «заказ») — для этого get_supplier_orders.\n"
            "- Для товаров — для этого get_order_items.\n\n"
            "Если пользователь не указал период или номер заказа — вызывай БЕЗ параметров."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date_begin": {
                    "type": "string",
                    "description": "Дата создания претензии с (YYYY-MM-DD)."
                },
                "date_end": {
                    "type": "string",
                    "description": "Дата создания претензии по (YYYY-MM-DD)."
                },
                "order_ids": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Номера заказов, по которым есть претензии."
                },
                "goods_ids": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Артикулы товаров (GoodsId), по которым есть претензии."
                },
                "warehouse_ids": {
                    "type": "array", "items": {"type": "string"},
                    "description": "ID складов."
                },
                "sorting_field": {
                    "type": "string",
                    "description": "Поле для сортировки (например, DateCreate, Summ)."
                },
                "sorting_param": {
                    "type": "integer",
                    "description": "0 — по возрастанию, 1 — по убыванию.",
                    "enum": [0, 1]
                },
                "limit": {"type": "integer", "default": 30},
                "page_number": {"type": "integer", "default": 1},
            },
            "required": [],
        },
    },
}


def params_to_api_claims(params: dict[str, Any]) -> dict[str, Any]:
    api_data: dict[str, Any] = {
        "OrderBuyGroupId": params.get("order_ids", []),
        "ClaimGroupId": [], # Not used in this method
        "CheckxSupplier": params.get("checkx_supplier", []),
        "WarehouseId": params.get("warehouse_ids", []),
        "GoodsId": params.get("goods_ids", []),
        "PageNumber": params.get("page_number", 1),
        "Limit": params.get("limit", 30),
        "Language": 1,
    }
    # Добавляем даты ТОЛЬКО если они реально заданы (иначе API падает)
    if params.get("date_begin"):
        api_data["DateBegin"] = params["date_begin"]
    if params.get("date_end"):
        api_data["DateEnd"] = params["date_end"]

    # Сортировка
    if params.get("sorting_field"):
        api_data["SortingField"] = params["sorting_field"]
    if params.get("sorting_param") is not None:
        api_data["SortingParam"] = params["sorting_param"]
        
    return api_data


# ==========================================================================
# 2. ДЕТАЛИЗАЦИЯ ПРЕТЕНЗИИ (DeliveryClaimDetailsGet)
# ==========================================================================
def parse_claim_details_response(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Парсит ответ DeliveryClaimDetailsGet."""
    table = raw.get("Table", [])
    if not isinstance(table, list):
        table = []

    items: list[dict[str, Any]] = []
    for row in table:
        if not isinstance(row, dict):
            continue

        items.append({
            "claim_id": row.get("ClaimGroupId"),
            "status": row.get("CheckxSupplier"),
            "info": row.get("Info2"),
            
            # Даты
            "date_provide": row.get("TimeUser7"),      # дата предоставления заказа
            "date_plan": row.get("Datex"),             # плановая дата поставки
            "date_fact": row.get("TimeUser3"),         # фактическая дата поставки
            "date_created": row.get("DateCreate"),     # дата создания претензии
            
            # Заказ и склад
            "order_id": row.get("OrderBuyGroupId"),
            "warehouse": row.get("WarehouseId"),
            
            # Товар
            "line_number": _to_int(row.get("LineNumber")),
            "goods_id": row.get("GoodsId"),
            "goods_name": row.get("GoodsName"),
            "is_promo": _to_int(row.get("IsPromo")),
            
            # Количества
            "qty_order": _to_float(row.get("QuantityOrder")),
            "qty_fact": _to_float(row.get("QuantityFact")),
            "qty_claim": _to_float(row.get("QuantityClaim")),
            
            # Цены и суммы
            "price": _to_float(row.get("Firstx")),
            "sum_claim": _to_float(row.get("SummClaim")), # сумма штрафа
            "shortage_percent": _to_float(row.get("ShortageCompensationPercent")), # % штрафа
            "compensation_sum": _to_float(row.get("CompensationSum")), # сумма недопоставленного
            "total_penalty": _to_float(row.get("Summ")), # общая сумма штрафа
            "currency": row.get("Currency", "BYN"),
            
            # Пагинация
            "amount": _to_int(row.get("Amount")),
        })

    logger.info(f"[tool_registry] parsed {len(items)} claim details")
    return items


SPEC_CLAIM_DETAILS = {
    "type": "function",
    "function": {
        "name": "get_claim_details",
        "description": (
            "Получить ПОДРОБНУЮ ИНФОРМАЦИЮ по конкретным претензиям (список товаров, "
            "суммы, компенсации, количество).\n\n"
            "✅ ИСПОЛЬЗУЙ КОГДА пользователь спрашивает:\n"
            "• «расскажи подробнее про претензию X»\n"
            "• «какие товары в претензии X»\n"
            "• «какая сумма компенсации по претензии X»\n"
            "• «покажи состав претензии X»\n\n"
            "⚠️ ВАЖНО: Параметр claim_ids ОБЯЗАТЕЛЕН. "
            "Если пользователь не назвал ID претензии — сначала вызови get_supplier_claims, "
            "чтобы найти нужный ID, а потом передай его сюда."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "claim_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "ID претензий (ClaimGroupId). Например: [122366]."
                },
                "sorting_field": {
                    "type": "string",
                    "description": "Поле для сортировки (например, LineNumber, Summ)."
                },
                "sorting_param": {
                    "type": "integer",
                    "description": "0 — по возрастанию, 1 — по убыванию.",
                    "enum": [0, 1]
                },
                "limit": {"type": "integer", "default": 30},
                "page_number": {"type": "integer", "default": 1},
            },
            "required": ["claim_ids"],
        },
    },
}


def params_to_api_claim_details(params: dict[str, Any]) -> dict[str, Any]:
    api_data = {
        "ClaimGroupId": params.get("claim_ids", []),
        "PageNumber": params.get("page_number", 1),
        "Limit": params.get("limit", 30),
        "Language": 1,
    }
    if params.get("sorting_field"):
        api_data["SortingField"] = params["sorting_field"]
    if params.get("sorting_param") is not None:
        api_data["SortingParam"] = params["sorting_param"]
    return api_data