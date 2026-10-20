"""
Фасад реестра инструментов.
"""
from typing import Any

from app.tools.tool_registry import orders, order_items, claims


TOOLS: list[dict[str, Any]] = [
    {
        "spec": orders.SPEC,
        "api_method": "SupplierPortal.GetOrderBuyGroup",
        "parser": orders.parse_orders_response,
        "mapper": orders.params_to_api,
    },
    {
        "spec": order_items.SPEC,
        "api_method": "SupplierPortal.GetOrderBuy",
        "parser": order_items.parse_order_items_response,
        "mapper": order_items.params_to_api,
    },
    {
        "spec": claims.SPEC_CLAIMS,
        "api_method": "SupplierPortal.DeliveryClaimGroupGet",
        "parser": claims.parse_claims_response,
        "mapper": claims.params_to_api_claims,
    },
    {
        "spec": claims.SPEC_CLAIM_DETAILS,
        "api_method": "SupplierPortal.DeliveryClaimDetailsGet",
        "parser": claims.parse_claim_details_response,
        "mapper": claims.params_to_api_claim_details,
    },
]


def get_tool(name: str) -> dict[str, Any] | None:
    for t in TOOLS:
        if t["spec"]["function"]["name"] == name:
            return t
    return None


def get_all_specs() -> list[dict[str, Any]]:
    return [t["spec"] for t in TOOLS]


def params_to_api(params: dict[str, Any], tool_name: str) -> dict[str, Any]:
    tool = get_tool(tool_name)
    if not tool:
        raise ValueError(f"Нет маппера для tool: {tool_name}")
    return tool["mapper"](params)