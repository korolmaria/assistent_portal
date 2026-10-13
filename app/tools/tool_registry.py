"""
Реестр инструментов Tool-агента.

Каждый инструмент содержит:
- spec: JSON-схема для function calling (что видит LLM)
- api_method: имя метода SupplierPortal
- parser: функция маппинга ответа API → список карточек заказов

Все справочники — из документации метода GetOrderBuyGroup.
Источник: Инструкция по методу GETORDERBUYGROUP.docx
"""
from typing import Any

from loguru import logger

from app.tools.supplier_portal import supplier_portal


# ==========================================================================
# Справочники (из документации GetOrderBuyGroup)
# ==========================================================================

# Бренды (BrandId)
# Документация: «В поле Бренд передаются: Хит - 2; Грошык - 3; Евроопт - 6»
BRAND_IDS: dict[str, int] = {
    "хит": 2, "хит!": 2, "hit": 2,
    "грошык": 3,
    "евроопт": 6,
}

# Статусы заказов (Checkx)
# Документация:
#   На сайте – 10
#   Автоподтвержден – 48
#   Подтвержден – 50
#   Выполнен – 90
ORDER_STATUS_IDS: dict[str, int] = {
    "на сайте": 10,
    "автоподтверждён": 48, "автоподтвержден": 48,
    "подтверждён": 50, "подтвержден": 50,
    "выполнен": 90,
}

# Человекочитаемые метки статусов (обратное отображение)
STATUS_LABELS: dict[int, str] = {
    10: "На сайте",
    48: "Автоподтверждён",
    50: "Подтверждён",
    90: "Выполнен",
}

# ==========================================================================
# Логика определения статуса заказа (перенесена с фронта)
# ==========================================================================
# Логика соответствует getOrderStatus из кабинета поставщика.
#
# Приоритет:
#   1. IsRevised = 2 (корректировка разрешена) + Checkx = 10 → "Корректировка разрешена"
#   2. IsRevised = 1 (запрошена корректировка) → "Запрошена корректировка"
#   3. IsRevised = 3 (отклонена) → "Корректировка отклонена"
#   4. Checkx = 10 (без корректировки) → "На сайте"
#   5. Checkx = 50 → "Подтверждён" / "Автоподтверждён" (по IsAuto)
#   6. Checkx = 90 → "Выполнен"
def compute_order_status(
    checkx: int | None,
    checkx_fact: int | None = None,
    is_revised: int | None = None,
    is_auto: int | None = None,
) -> tuple[str, str]:
    """
    Определяет финальный статус заказа.

    Возвращает:
        (status_text, status_type) — например, ("Подтверждён", "success")

    status_type используется для цвета на фронте:
        success_default, success, warning, error, default
    """
    status = str(checkx) if checkx is not None else None
    is_auto_str = str(is_auto) if is_auto is not None else None
    is_revised_str = str(is_revised) if is_revised is not None else None

    # 1. Корректировка разрешена (2) + заказ на сайте (10)
    if status == "10" and is_revised_str == "2":
        return "Корректировка разрешена", "success_default"

    # 2. Заказ на сайте, без активной корректировки
    if (not is_revised_str or is_revised_str == "99") and status == "10":
        return "На сайте", "success_default"

    # 3. Заказ подтверждён (50), без активной корректировки
    if (not is_revised_str or is_revised_str == "99") and status == "50":
        text = "Автоподтверждён" if is_auto_str in ("1",) else "Подтверждён"
        return text, "success"

    # 4. Запрошена корректировка (1)
    if status in ("50", "10") and is_revised_str == "1":
        return "Запрошена корректировка", "warning"

    # 5. Корректировка отклонена (3)
    if status in ("50", "10") and is_revised_str == "3":
        return "Корректировка отклонена", "error"

    # 6. Заказ выполнен (90)
    if status == "90":
        return "Выполнен", "default"

    # 7. Fallback
    fallback = STATUS_LABELS.get(checkx or 0, f"Статус {checkx}")
    return fallback, "warning"


# Старые метки — оставляем для обратной совместимости
def status_label(code: int) -> str:
    """Простая метка статуса (используется для fallback)."""
    return STATUS_LABELS.get(code, f"Статус {code}")

# Статус обработки (IsProcessed)
# Документация: «Не обработан - 0, Обработан - 1»
PROCESSED_IDS: dict[str, int] = {
    "не обработан": 0, "необработанный": 0, "новый": 0,
    "обработан": 1, "обработанный": 1,
}

# Области (RegionId = Address7Id)
# Документация: список из 12 значений (70-81)
REGION_IDS: dict[str, int] = {
    "брестская область": 70,
    "брест": 71,
    "витебская область": 72,
    "витебск": 73,
    "гомельская область": 74,
    "гомель": 75,
    "гродненская область": 76,
    "гродно": 77,
    "минск": 78,
    "минская область": 79,
    "могилевская область": 80,
    "могилев": 81, "могилёв": 81,
}


# ==========================================================================
# Хелперы для работы со справочниками
# ==========================================================================
def status_label(code: int) -> str:
    """Человекочитаемая метка статуса."""
    return STATUS_LABELS.get(code, f"Статус {code}")


def resolve_brand(name: str) -> int | None:
    """Название бренда → ID."""
    if not name:
        return None
    return BRAND_IDS.get(name.strip().lower())


def resolve_region(name: str) -> int | None:
    """Название области → ID."""
    if not name:
        return None
    return REGION_IDS.get(name.strip().lower())


def resolve_status(name: str) -> int | None:
    """Название статуса заказа → ID."""
    if not name:
        return None
    return ORDER_STATUS_IDS.get(name.strip().lower())


def resolve_processed(name: str) -> int | None:
    """Название статуса обработки → ID."""
    if not name:
        return None
    return PROCESSED_IDS.get(name.strip().lower())


# ==========================================================================
# Парсер ответа GetOrderBuyGroup
# ==========================================================================
def parse_orders_response(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Ответ API → список словарей для OrderCardData.

    Реальный формат ответа от SupplierPortal:
    {
      "Table": [
        {
          "OrderBuyGroupId": 98010205,
          "ContractorNameShort": "...",
          "Checkx": 50,
          ...
        }
      ],
      "Table2": []
    }

    Все поля заказа — на верхнем уровне в Table.
    """
    orders: list[dict[str, Any]] = []

    # Основные данные — в "Table"
    data = raw.get("Table", [])
    if not isinstance(data, list):
        data = []

    for o in data:
        # Если структура вложенная (OrderBuyGroup внутри) — разворачиваем
        if isinstance(o, dict) and "OrderBuyGroup" in o:
            o = o["OrderBuyGroup"]

        if not isinstance(o, dict):
            continue

        # Проверяем, что это заказ (есть OrderBuyGroupId)
        if "OrderBuyGroupId" not in o:
            continue

        status_text, status_type = compute_order_status(
            checkx=o.get("Checkx"),
            checkx_fact=o.get("CheckxFact"),
            is_revised=o.get("IsRevised"),
            is_auto=o.get("IsAuto"),
        )

        orders.append({
            "order_id": o.get("OrderBuyGroupId"),
            "contract_group_id": o.get("ContractGroupId"),
            "contractor": o.get("ContractorNameShort"),
            "contractor_id": o.get("ContractorId"),
            "manager": o.get("Manager1"),
            "phone": o.get("PhoneNumber"),
            "status":  status_text,
            "status_type": status_type,
            "status_code": o.get("Checkx"),
            "checkx": o.get("Checkx"),
            "checkx_code": o.get("CheckxFact"),
            "is_processed": o.get("IsProcessed"),
            "processed_label": {
                0: "Не обработан",
                1: "Обработан",
                2: "Обработан (недоступен)",
            }.get(o.get("IsProcessed"), "Не обработан"),   
            "created_at": o.get("TTTime"),
            "datex": o.get("Datex"),
            "next_order_date": o.get("NextOrderDate"),
            "shipment_date": o.get("Datex"),
            "sum": o.get("OrderSum"),
            "warehouse": o.get("WarehouseId"),
            "warehouse_name": o.get("WarehouseName"),
            "gln_warehouse": o.get("GLNWarehouse"),
            "cross_dock": o.get("CrossDock"),
            "brand": o.get("BrandName"),
            "address": (
                o.get("DepartmentAddress")
                or ", ".join(filter(None, [
                    o.get("ADDRESS7NAME"),
                    o.get("ADDRESS6NAME"),
                    o.get("ADDRESS5NAME"),
                ]))
            ),
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
                    "Получить список заказов поставщика из SupplierPortal. "
                    "Поддерживает фильтры: номер заказа, период поставки, "
                    "период создания, статус заказа, статус обработки, бренд, "
                    "склад, область, район, город. "
                    "Если фильтры не заданы — вернёт все заказы за последние 3-6 месяцев.\n\n"
                    "СПРАВОЧНИКИ ID (используй эти значения):\n"
                    "• Бренды (brand_ids): 2=Хит, 3=Грошык, 6=Евроопт\n"
                    "• Статусы заказа (checkx): 10=На сайте, 48=Автоподтверждён, "
                    "50=Подтверждён, 90=Выполнен\n"
                    "• Статус обработки (is_processed): 0=не обработан, 1=обработан\n"
                    "• Области (region_ids): 70=Брестская обл, 71=Брест, "
                    "72=Витебская обл, 73=Витебск, 74=Гомельская обл, 75=Гомель, "
                    "76=Гродненская обл, 77=Гродно, 78=Минск, 79=Минская обл, "
                    "80=Могилевская обл, 81=Могилев\n\n"
                    "ПРИМЕРЫ ЗАПРОСОВ:\n"
                    "• «покажи заказ 56» → order_ids: [56]\n"
                    "• «заказы 1 и 2» → order_ids: [1, 2]\n"
                    "• «подтверждённые заказы» → checkx: [50]\n"
                    "• «необработанные» → is_processed: 0\n"
                    "• «заказы по бренду Хит» → brand_ids: [2]\n"
                    "• «заказы по Минску» → region_ids: [78]\n"
                    "• «за сентябрь» → date_begin: 2026-09-01, date_end: 2026-09-30"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": (
                                "ID заказов (OrderBuyGroupId). "
                                "Используй, когда пользователь называет конкретные номера: "
                                "«заказ 56», «заказы 1 и 2»."
                            ),
                        },
                        "date_begin": {
                            "type": "string",
                            "description": (
                                "Дата поставки с (YYYY-MM-DD). "
                                "Например: '2026-09-01' для «за сентябрь»."
                            ),
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
                                "Статусы заказов (Checkx): "
                                "10=На сайте, 48=Автоподтверждён, "
                                "50=Подтверждён, 90=Выполнен. "
                                "Например, для «подтверждённые заказы» → [50]."
                            ),
                        },
                        "brand_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": (
                                "ID брендов: 2=Хит, 3=Грошык, 6=Евроопт. "
                                "Например, для «заказы по бренду Хит» → [2]."
                            ),
                        },
                        "warehouse_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "ID складов (WarehouseId)",
                        },
                        "region_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": (
                                "ID областей (RegionId=Address7Id): 70-81. "
                                "Например, для «заказы по Минску» → [78]."
                            ),
                        },
                        "district_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "ID районов (DistrictId=Address6Id)",
                        },
                        "city_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "ID городов (CityId=Address5Id)",
                        },
                        "is_processed": {
                            "type": "integer",
                            "description": (
                                "Статус обработки: 0=не обработан, 1=обработан. "
                                "Например, для «необработанные заказы» → 0."
                            ),
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
        "OrderBuyGroupId": params.get("order_ids", []),
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