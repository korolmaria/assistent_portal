"""
Обёртки для методов претензий SupplierPortal.
"""
from typing import Any

from app.tools.supplier_portal.core import SupplierPortalClient


async def get_delivery_claims(
    client: SupplierPortalClient,
    filters: dict[str, Any],
    jwt: str,
) -> Any:
    """Список претензий поставщика (DeliveryClaimGroupGet)."""
    return await client.call(
        method="SupplierPortal.DeliveryClaimGroupGet",
        data=filters,
        jwt=jwt,
    )