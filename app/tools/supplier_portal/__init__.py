"""
Фасад для SupplierPortal.

Экспортирует:
- supplier_portal: singleton клиента (для прямого вызова call)
- SupplierPortalError: исключение
- функции-обёртки для конкретных методов (если нужно)
"""
from app.tools.supplier_portal.core import SupplierPortalClient, SupplierPortalError
from app.tools.supplier_portal import orders, claims

# Singleton (используется в tool_agent через supplier_portal.call)
supplier_portal = SupplierPortalClient()

__all__ = [
    "supplier_portal",
    "SupplierPortalError",
    "orders",
    "claims",
]