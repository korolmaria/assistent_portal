"""
Инструмент get_order_items (SupplierPortal.GetOrderBuy).
"""
from typing import Any

from loguru import logger

from app.tools.tool_registry.common import compute_order_status, _to_float, _to_int


def parse_order_items_response(raw: dict[str, Any]) -> dict[str, Any]:
    table = raw.get("Table", [])
    if not isinstance(table, list):
        table = []

    items: list[dict[str, Any]] = []
    amount = 0
    page_qty = 0

    for row in table:
        if not isinstance(row, dict):
            continue
        if "OrderBuyId" not in row:
            continue

        if not amount:
            amount = int(row.get("Amount") or 0)
            page_qty = int(row.get("PageQty") or 0)

        status_text, status_type = compute_order_status(
            checkx=row.get("Checkx"),
            checkx_fact=row.get("CheckxFact"),
            is_revised=row.get("IsRevised"),
            is_auto=row.get("IsAuto"),
        )

        items.append({
            "order_id": _to_int(row.get("OrderBuyGroupId")),
            "order_buy_id": _to_int(row.get("OrderBuyId")),
            "line_number": _to_int(row.get("LineNumber")),
            "row_number": _to_int(row.get("RowNumber")),
            "goods_id": _to_int(row.get("GoodsId")),
            "goods_name": row.get("GoodsName"),
            "barcode": row.get("Barcode1"),
            "brand": row.get("BrandName"),
            "warehouse_id": _to_int(row.get("WarehouseId")),
            "gln_warehouse": _to_int(row.get("GLNWarehouse")),
            "address": row.get("DepartmentAddress"),
            "status": status_text,
            "status_type": status_type,
            "checkx": _to_int(row.get("Checkx")),
            "checkx_fact": _to_int(row.get("CheckxFact")),
            "is_revised": _to_int(row.get("IsRevised")),
            "is_auto": _to_int(row.get("IsAuto")),
            "quantity_order": _to_float(row.get("QuantityOrder")),
            "quantity_supplier": _to_float(row.get("QuantitySupplier")),
            "quantity_add2": _to_float(row.get("QuantityAdd2")),
            "dif_quantity": _to_float(row.get("DifQuantity")),
            "limit_quantity": _to_float(row.get("LimitQuantity")),
            "quantity_in_pack": _to_float(row.get("QuantityInPack")),
            "quantity_multiple": _to_float(row.get("QuantityMultiple")),
            "delivery_quantity_pack_min": _to_float(row.get("DeliveryQuantityPackMin")),
            "quantity_order_promo": _to_float(row.get("QuantityOrderPromo")),
            "price": _to_float(row.get("Price")),
            "price2": _to_float(row.get("Price2")),
            "first_x": _to_float(row.get("Firstx")),
            "tax_id": _to_int(row.get("TaxId")),
            "order_sum": _to_float(row.get("OrderSum")),
            "sum2_add2": _to_float(row.get("Sum2Add2")),
            "is_promo": _to_int(row.get("IsPromo")),
            "plan_promo": row.get("PlanPromo"),
            "is_cross_dock": _to_int(row.get("IsCrossDock")),
            "is_processed": _to_int(row.get("IsProcessed")),
            "control_weight": _to_int(row.get("ControlWeight")),
            "info": row.get("Info"),
            "group_info_name": row.get("OrderBuyGroupInfoName"),
            "datex": row.get("Datex"),
            "shipment_date_last": row.get("ShipmentDateLast"),
        })

    logger.info(
        f"[tool_registry] parsed {len(items)} items, amount={amount}, page_qty={page_qty}"
    )
    return {"items": items, "amount": amount, "page_qty": page_qty}


SPEC = {
    "type": "function",
    "function": {
        "name": "get_order_items",
        "description": (
            "Получить СПИСОК ТОВАРОВ (позиций) ВНУТРИ заказов поставщика.\n\n"
            "✅ ИСПОЛЬЗУЙ КОГДА пользователь спрашивает про ТОВАРЫ в заказе:\n"
            "• «какие товары в заказе X»\n"
            "• «покажи содержимое заказа X»\n"
            "• «что заказано по заказу X»\n"
            "• «позиции заказа X»\n"
            "• «есть ли товар N в заказе X»\n"
            "• «какие товары заказали»\n\n"
            "❌ НЕ ИСПОЛЬЗУЙ:\n"
            "- Для списка самих заказов (без товаров) — для этого get_supplier_orders.\n"
            "- Для претензий/брака — для этого get_supplier_claims.\n\n"
            "ВАЖНО: order_ids ОБЯЗАТЕЛЕН. Если пользователь не назвал номер заказа, "
            "сначала вызови get_supplier_orders, чтобы найти нужный заказ.\n\n"
            "Можно запросить сразу НЕСКОЛЬКО заказов — передай их ID массивом в order_ids."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_ids": {
                    "type": "array", "items": {"type": "integer"},
                    "description": "ID заказов (OrderBuyGroupId). Например: [98010205]."
                },
                "goods_ids": {
                    "type": "array", "items": {"type": "integer"},
                    "description": "Артикулы товаров (GoodsId)."
                },
                "goods_name": {"type": "string", "description": "Краткое наименование товара."},
                "barcodes": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer", "default": 30},
                "page_number": {"type": "integer", "default": 1},
            },
            "required": ["order_ids"],
        },
    },
}


def params_to_api(params: dict[str, Any]) -> dict[str, Any]:
    api_data: dict[str, Any] = {
        "OrderBuyGroupId": params.get("order_ids", []),
        "GoodsId": params.get("goods_ids", []),
        "Barcode1": params.get("barcodes", []),
        "PageNumber": params.get("page_number", 1),
        "Limit": params.get("limit", 30),
        "Language": 1,
    }
    if params.get("goods_name"):
        api_data["GoodsName"] = params["goods_name"]
    if params.get("sorting_field"):
        api_data["SortingField"] = params["sorting_field"]
    if params.get("sorting_param") is not None:
        api_data["SortingParam"] = params["sorting_param"]
    return api_data