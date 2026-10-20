"""
Обёртки для методов заказов SupplierPortal.
"""
from typing import Any

from app.tools.supplier_portal.core import SupplierPortalClient


def get_order_buy_group(
    client: SupplierPortalClient,
    filters: dict[str, Any],
    jwt: str,
) -> Any:
    """Список заказов поставщика (статусы 10-90)."""
    return client.call(
        method="SupplierPortal.GetOrderBuyGroup",
        data=filters,
        jwt=jwt,
    )


def get_order_buy(
    client: SupplierPortalClient,
    filters: dict[str, Any],
    jwt: str,
) -> Any:
    """Содержимое заказов (позиции/строки)."""
    return client.call(
        method="SupplierPortal.GetOrderBuy",
        data=filters,
        jwt=jwt,
    )