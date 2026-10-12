"""
Реестр инструментов Tool-агента.

Каждый инструмент содержит:
- spec: JSON-схема для function calling (что видит LLM)
- api_method: имя метода SupplierPortal
- parser: функция маппинга ответа API → список карточек заказов

Добавить новый инструмент = добавить запись в TOOLS.
"""
from typing import Any, Callable

from loguru import logger

from app.tools.supplier_portal import supplier_portal


# ==========================================================================
# Маппинг статусов заказов
# ==========================================================================
STATUS_LABELS: dict[int, str] = {
    10: "Черновик",
    20: "На согласовании",
    30: "Согласован",
    40: "В сборке",
    50: "Подтверждён",
    51: "Подтверждён",
    60: "Отгружен",
    70: "В пути",
    80: "Доставлен",
    90: "Закрыт",
}


def normalize_status(code: int) -> int:
    """Статусы 51-89 мапим в 50 (по документации)."""
    if 51 <= code <= 89:
        return 50
    return code


def status_label(code: int) -> str:
    """Человекочитаемая метка статуса."""
    return STATUS_LABELS.get(normalize_status(code), f"Статус {code}")


# ==========================================================================
# Парсер ответа GetOrderBuyGroup
# ==========================================================================
def parse_orders_response(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Ответ API → список словарей для OrderCardData.
    Из ответа берём только записи с OrderBuyGroup (пропускаем NextOrderDate).
    """
    orders: list[dict[str, Any]] = []
    data = raw.get("Data", [])

    for item in data:
        if "OrderBuyGroup" not in item:
            continue
        o = item["OrderBuyGroup"]

        orders.append({
            "order_id": o.get("OrderBuyGroupId"),
            "contract_group_id": o.get("ContractGroupId"),
            "contractor": o.get("ContractorNameShort"),
            "contractor_id": o.get("ContractorId"),
            "manager": o.get("Manager1"),
            "phone": o.get("PhoneNumber"),
            "status": status_label(o.get("Checkx", 0)),
            "status_code": normalize_status(o.get("Checkx", 0)),
            "checkx": o.get("Checkx"),
            "checkx_code": o.get("CheckxFact"),
            "created_at": o.get("TTTime"),
            "datex": o.get("TTTime"),
            "next_order_date": o.get("NextOrderDate"),
            "shipment_date": o.get("Datex"),
            "sum": o.get("OrderSum"),
            "warehouse": o.get("WarehouseId"),
            "warehouse_name": o.get("WarehouseName"),
            "gln_warehouse": o.get("GLNWarehouse"),
            "cross_dock": o.get("CrossDock"),
            "brand": o.get("BrandName"),
            "address": o.get("ADDRESS5NAME"),
            "region": o.get("ADDRESS7NAME"),
            "district": o.get("ADDRESS6NAME"),
            "city": o.get("ADDRESS5NAME"),
            "revised": o.get("IsRevised"),
            "is_auto": o.get("IsAuto"),
        })

    logger.info(f"[tool_registry] parsed {len(orders)} orders")
    return orders


# ==========================================================================
# Реестр инструментов
# ==========================================================================
TOOLS: list[dict[str, Any]] = [
    {
        "spec": {
            "type": "function",
            "function": {
                "name": "get_supplier_orders",
                "description": (
                    "Получить список заказов поставщика. "
                    "Поддерживает фильтры: период даты поставки, период даты создания, "
                    "статус заказа, склад, бренд, область, район, город, "
                    "флаг обработки заказа поставщиком. "
                    "Если фильтры не заданы — вернёт все заказы за последние 3 месяца."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date_begin": {
                            "type": "string",
                            "description": "Дата поставки с (формат YYYY-MM-DD)",
                        },
                        "date_end": {
                            "type": "string",
                            "description": "Дата поставки по (YYYY-MM-DD)",
                        },
                        "tt_time_begin": {
                            "type": "string",
                            "description": "Дата создания заказа с (YYYY-MM-DD)",
                        },
                        "tt_time_end": {
                            "type": "string",
                            "description": "Дата создания заказа по (YYYY-MM-DD)",
                        },
                        "checkx": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": (
                                "Статусы заказов: "
                                "10-черновик, 50-подтверждён, 60-отгружен, 90-закрыт"
                            ),
                        },
                        "warehouse_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "ID складов",
                        },
                        "brand_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "ID брендов: 2-Хит!, 3-Грошык, 6-Евроопт",
                        },
                        "region_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "ID областей (Address7Id)",
                        },
                        "district_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "ID районов (Address6Id)",
                        },
                        "city_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "ID городов (Address5Id)",
                        },
                        "is_processed": {
                            "type": "integer",
                            "description": "0-не обработан, 1-обработан",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Максимум заказов на странице (по умолчанию 30)",
                            "default": 30,
                        },
                        "page_number": {
                            "type": "integer",
                            "description": "Номер страницы (по умолчанию 1)",
                            "default": 1,
                        },
                    },
                    "required": [],
                },
            },
        },
        "api_method": "SupplierPortal.GetOrderBuyGroup",
        "parser": parse_orders_response,
    },
]


# ==========================================================================
# Маппинг параметров LLM → API
# ==========================================================================
def params_to_api(params: dict[str, Any]) -> dict[str, Any]:
    """LLM даёт snake_case, API ждёт CamelCase. Мапим."""
    api_data: dict[str, Any] = {
        "OrderBuyGroupId": [],
        "GoodsId": [],
        "Checkx": params.get("checkx", []),
        "BrandId": params.get("brand_ids", []),
        "WarehouseId": params.get("warehouse_ids", []),
        "RegionId": params.get("region_ids", []),
        "DistrictId": params.get("district_ids", []),
        "CityId": params.get("city_ids", []),
        "PageNumber": params.get("page_number", 1),
        "Limit": params.get("limit", 30),
        "IsGrouping": 0,
        "GroupParamLevel": [],
        "Language": 1,
    }

    if params.get("date_begin"):
        api_data["DateBegin"] = params["date_begin"]
    if params.get("date_end"):
        api_data["DateEnd"] = params["date_end"]
    if params.get("tt_time_begin"):
        api_data["TTTimeBegin"] = params["tt_time_begin"]
    if params.get("tt_time_end"):
        api_data["TTTimeEnd"] = params["tt_time_end"]
    if params.get("is_processed") is not None:
        api_data["IsProcessed"] = params["is_processed"]

    return api_data


# ==========================================================================
# Хелпер: получить инструмент по имени
# ==========================================================================
def get_tool(name: str) -> dict[str, Any] | None:
    for t in TOOLS:
        if t["spec"]["function"]["name"] == name:
            return t
    return None


def get_all_specs() -> list[dict[str, Any]]:
    """Все spec-и для передачи в LLM."""
    return [t["spec"] for t in TOOLS]
