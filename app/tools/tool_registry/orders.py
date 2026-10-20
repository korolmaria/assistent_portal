"""
Инструмент get_supplier_orders (SupplierPortal.GetOrderBuyGroup).
"""
from typing import Any

from loguru import logger

from app.tools.tool_registry.common import compute_order_status


def parse_orders_response(raw: dict[str, Any]) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    data = raw.get("Table", [])
    if not isinstance(data, list):
        data = []

    for o in data:
        if isinstance(o, dict) and "OrderBuyGroup" in o:
            o = o["OrderBuyGroup"]
        if not isinstance(o, dict):
            continue
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
            "status": status_text,
            "status_type": status_type,
            "status_code": o.get("Checkx"),
            "checkx": o.get("Checkx"),
            "checkx_code": o.get("CheckxFact"),
            "is_processed": o.get("IsProcessed"),
            "processed_label": {
                0: "Не обработан", 1: "Обработан",
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
                    o.get("ADDRESS7NAME"), o.get("ADDRESS6NAME"), o.get("ADDRESS5NAME"),
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


SPEC = {
    "type": "function",
    "function": {
        "name": "get_supplier_orders",
        "description": (
            "Получить СПИСОК ЗАКАЗОВ поставщика. "
            "Возвращает карточки заказов (номер, статус, сумма, склад, "
            "контрагент, адрес), БЕЗ списка товаров внутри.\n\n"
            "✅ ИСПОЛЬЗУЙ КОГДА пользователь спрашивает про ЗАКАЗЫ:\n"
            "• «покажи мои заказы» / «покажи заказы»\n"
            "• «заказы за сентябрь» / «заказы за последнюю неделю»\n"
            "• «подтверждённые заказы» / «необработанные заказы»\n"
            "• «заказы по бренду Хит» / «заказы по Минску»\n"
            "• «сколько заказов за сентябрь»\n"
            "• «покажи заказ 98010225»\n"
            "• «какой статус у заказа 98010225»\n"
            "• «какая сумма у заказа 98010225»\n"
            "• «покажи заказы на склад 700»\n"
            "• «заказы, созданные вчера» / «какие заказы я оформил за октябрь»\n\n"
            "❌ НЕ ИСПОЛЬЗУЙ ЕСЛИ в запросе есть слова «товары», "
            "«позиции», «артикул», «содержимое», «что заказано» — "
            "для этого есть get_order_items. "
            "И НЕ ИСПОЛЬЗУЙ для претензий/брака — для этого get_supplier_claims.\n\n"
            "СПРАВОЧНИКИ ID ДЛЯ ПАРАМЕТРОВ:\n"
            "• Бренды (brand_ids): 2=Хит, 3=Грошык, 6=Евроопт\n"
            "• Статусы (checkx): 10=На сайте, 48=Автоподтверждён, 50=Подтверждён, 90=Выполнен\n"
            "• Обработка (is_processed): 0=не обработан, 1=обработан. "
            "Ставь ТОЛЬКО если явно сказано «необработанные» или «обработанные». "
            "Слова «неподтвержденные» или «на сайте» — это СТАТУС (checkx), а НЕ обработка!\n"
            "• Области (region_ids): 70=Брестская обл, 71=Брест, 72=Витебская обл, "
            "73=Витебск, 74=Гомельская обл, 75=Гомель, 76=Гродненская обл, 77=Гродно, "
            "78=Минск, 79=Минская обл, 80=Могилевская обл, 81=Могилев\n"
            "• Склады (warehouse_ids): передавай ID склада числом, если пользователь "
            "явно указал склад (например, «на складе 700» → warehouse_ids: [700]).\n"
            "\n"
            "⚠️ РАЗНИЦА МЕЖДУ ДАТАМИ:\n"
            "1. date_begin / date_end — это ДАТА ПОСТАВКИ (когда товар должен приехать). "
            "Используй, если сказано «заказы на октябрь», «поставка на следующую неделю».\n"
            "2. tt_time_begin / tt_time_end — это ДАТА СОЗДАНИЯ ЗАКАЗА (когда его оформили на сайте). "
            "Используй, если сказано «какие заказы я создал», «оформленные за октябрь», «созданные вчера».\n"
            "Всегда ставь оба параметра (begin и end), если указан месяц.\n"
            "\n"
            "⚠️ ЕСЛИ В ЗАПРОСЕ ЕСТЬ НОМЕР ЗАКАЗА:\n"
            "Ставь ТОЛЬКО order_ids. НЕ ДОБАВЛЯЙ checkx, date_begin, brand_ids, "
            "даже если в тексте есть слова «подтверждён», «за октябрь» или «Хит»."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_ids": {
                    "type": "array", "items": {"type": "integer"},
                    "description": "ID заказов (OrderBuyGroupId)."
                },
                "date_begin": {"type": "string", "description": "Дата поставки с (YYYY-MM-DD)."},
                "date_end": {"type": "string", "description": "Дата поставки по (YYYY-MM-DD)."},
                "tt_time_begin": {"type": "string", "description": "Дата создания заказа с (YYYY-MM-DD)."},
                "tt_time_end": {"type": "string", "description": "Дата создания заказа по (YYYY-MM-DD)."},
                "checkx": {
                    "type": "array", "items": {"type": "integer"},
                    "description": "Статусы: 10, 48, 50, 90."
                },
                "brand_ids": {
                    "type": "array", "items": {"type": "integer"},
                    "description": "Бренды: 2=Хит, 3=Грошык, 6=Евроопт."
                },
                "is_processed": {"type": "integer", "description": "0 — не обработан, 1 — обработан."},
                "region_ids": {"type": "array", "items": {"type": "integer"}},
                "warehouse_ids": {
                    "type": "array", "items": {"type": "integer"},
                    "description": "ID складов (WarehouseId). Например, 700."
                },
                "limit": {"type": "integer", "default": 30},
                "page_number": {"type": "integer", "default": 1},
            },
            "required": [],
        },
    },
}


def params_to_api(params: dict[str, Any]) -> dict[str, Any]:
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