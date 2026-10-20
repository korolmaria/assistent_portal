"""
Ядро клиента SupplierPortal — транспорт и проверка ошибок.
"""
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings


class SupplierPortalError(Exception):
    """Ошибка от SupplierPortal (вернулась в Table с Error/ErrorDescription)."""

    def __init__(self, error_id: int, description: str):
        self.error_id = error_id
        self.description = description
        super().__init__(f"SupplierPortal error {error_id}: {description}")


class SupplierPortalClient:
    """Универсальный клиент к AlwaysOn /Json (только транспорт)."""

    def __init__(self) -> None:
        self.base_url = settings.supplier_portal_url.rstrip("/")
        self.service_number = settings.supplier_portal_service_number
        self.timeout = settings.supplier_portal_timeout
        self.verify_ssl = settings.supplier_portal_verify_ssl

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def call(
        self,
        method: str,
        data: dict[str, Any],
        jwt: str,
    ) -> dict[str, Any]:
        """
        Универсальный вызов любого метода SupplierPortal.
        """
        payload = {
            "CRC": "",
            "Packet": {
                "JWT": jwt,
                "MethodName": method,
                "ServiceNumber": self.service_number,
                "Data": data,
            },
        }

        url = f"{self.base_url}/Json"
        logger.debug(f"[SupplierPortal] POST {method} → {url}")

        async with httpx.AsyncClient(
            verify=self.verify_ssl,
            timeout=self.timeout,
        ) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

        logger.info(f"[SupplierPortal] RAW ответ: {str(result)[:1000]}")

        # Проверяем ошибку внутри ответа
        if isinstance(result, dict) and "Table" in result:
            table = result["Table"]
            if table and isinstance(table, list) and "Error" in table[0]:
                err = table[0]
                raise SupplierPortalError(
                    error_id=int(err.get("Error", 0)),
                    description=err.get("ErrorDescription", "unknown"),
                )

        logger.debug(f"[SupplierPortal] {method} → OK")
        return result