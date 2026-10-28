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
            "Получить СПИСОК ЗАКАЗОВ поставщика (номер, статус, сумма, склад, "
            "контрагент, адрес). БЕЗ списка товаров внутри — для товаров "
            "используй get_order_items. Для претензий — get_supplier_claims.\n\n"
            "Вызывай, когда спрашивают про заказы: «покажи заказы», "
            "«заказы за октябрь», «подтверждённые заказы», «заказы по бренду Хит», "
            "«покажи заказ 98010225», «заказы на склад 700».\n\n"
            "НЕ вызывай, если в запросе есть «товары», «позиции», «артикул», "
            "«содержимое», «что заказано» (это get_order_items), "
            "или «претензия», «брак», «возврат» (это get_supplier_claims)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_ids": {
                    "type": "array", "items": {"type": "integer"},
                    "description": (
                        "ID заказов (OrderBuyGroupId). "
                        "Ставь ТОЛЬКО когда пользователь явно назвал номер заказа "
                        "(«заказ 98010225», «покажи 101229916»). "
                        "Если order_ids заполнен — НЕ добавляй date_begin, date_end, "
                        "tt_time_begin, tt_time_end, checkx, brand_ids, is_processed, "
                        "region_ids, warehouse_ids: пользователь спрашивает про "
                        "конкретный заказ, а не про выборку."
                    )
                },
                "date_begin": {
                    "type": "string",
                    "description": (
                        "Дата ПОСТАВКИ с (YYYY-MM-DD). Ставь, когда речь про дату "
                        "привоза: «заказы на октябрь», «поставка на следующую неделю». "
                        "НЕ ставь для даты создания — для этого tt_time_begin. "
                        "Всегда вместе с date_end."
                    )
                },
                "date_end": {
                    "type": "string",
                    "description": "Дата ПОСТАВКИ по (YYYY-MM-DD). Всегда вместе с date_begin."
                },
                "tt_time_begin": {
                    "type": "string",
                    "description": (
                        "Дата СОЗДАНИЯ заказа с (YYYY-MM-DD). Ставь, когда речь про то, "
                        "когда заказ оформили: «созданные вчера», «я оформил за октябрь», "
                        "«какие заказы я создал». "
                        "НЕ ставь для даты поставки — для этого date_begin. "
                        "Всегда вместе с tt_time_end."
                    )
                },
                "tt_time_end": {
                    "type": "string",
                    "description": "Дата СОЗДАНИЯ заказа по (YYYY-MM-DD). Всегда вместе с tt_time_begin."
                },
                "checkx": {
                    "type": "array", "items": {"type": "integer"},
                    "description": (
                        "Статусы заказа: 10=На сайте, 48=Автоподтверждён, "
                        "50=Подтверждён, 90=Выполнен. "
                        "Ставь, только если пользователь назвал статус явно. "
                        "«Необработанные» — это НЕ статус, это is_processed."
                    )
                },
                "brand_ids": {
                    "type": "array", "items": {"type": "integer"},
                    "description": (
                        "Бренды: 2=Хит, 3=Грошык, 6=Евроопт. "
                        "Ставь, только если бренд назван явно."
                    )
                },
                "is_processed": {
                    "type": "integer",
                    "description": (
                        "0 — не обработан, 1 — обработан. "
                        "Ставь ТОЛЬКО при словах «необработанные» или «обработанные». "
                        "Слова «на сайте», «неподтверждённые» — это статус (checkx), "
                        "НЕ обработка."
                    )
                },
                "region_ids": {
                    "type": "array", "items": {"type": "integer"},
                    "description": (
                        "Области/города: 70=Брестская обл, 71=Брест, 72=Витебская обл, "
                        "73=Витебск, 74=Гомельская обл, 75=Гомель, 76=Гродненская обл, "
                        "77=Гродно, 78=Минск, 79=Минская обл, 80=Могилевская обл, "
                        "81=Могилев. Ставь, только если регион назван явно."
                    )
                },
                "warehouse_ids": {
                    "type": "array", "items": {"type": "integer"},
                    "description": (
                        "ID складов (WarehouseId). Ставь, только если пользователь "
                        "явно назвал склад числом: «на складе 700» → [700]."
                    )
                },
                "limit": {
                    "type": "integer", "default": 30,
                    "description": "Сколько записей вернуть (по умолчанию 30)."
                },
                "page_number": {
                    "type": "integer", "default": 1,
                    "description": "Номер страницы (по умолчанию 1)."
                },
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
