#!/usr/bin/env python3
"""
MCP-сервер для интеграции инструментов SupplierPortal с Hermes.
Использует существующие get_tool, params_to_api, supplier_portal из проекта.
Разрешены только методы на чтение.
"""
import asyncio
import json
import os
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.tools.tool_registry import get_tool, get_all_specs, params_to_api
from app.tools.supplier_portal import supplier_portal

import logging
logger = logging.getLogger("mcp")

# ⚠️ БЕЛЫЙ СПИСОК: только чтение
ALLOWED_TOOLS = [
    "get_orders",
    "get_order_items",
    "get_supplier_claims",
    "get_claim_details",
]

server = Server("supplier-portal")


@server.list_tools()
async def list_tools() -> list[Tool]:
    tools = []
    specs = {t["function"]["name"]: t["function"] for t in get_all_specs()}

    for name in ALLOWED_TOOLS:
        spec = specs.get(name)
        if not spec:
            logger.warning(f"[mcp] инструмент '{name}' не найден")
            continue

        tools.append(Tool(
            name=name,
            description=spec.get("description", ""),
            inputSchema=spec.get("parameters", {"type": "object", "properties": {}}),
        ))

    logger.info(f"[mcp] зарегистрировано {len(tools)} инструментов")
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name not in ALLOWED_TOOLS:
        return [TextContent(type="text", text=f"Инструмент '{name}' запрещён (только чтение).")]

    jwt = os.environ.get("SUPPLIER_JWT", "")
    if not jwt:
        return [TextContent(type="text", text="Ошибка: JWT не передан (SUPPLIER_JWT)")]

    tool = get_tool(name)
    if not tool:
        return [TextContent(type="text", text=f"Инструмент '{name}' не найден в реестре.")]

    try:
        api_data = params_to_api(arguments, name)
        logger.info(f"[mcp] вызов {tool['api_method']} с {api_data}")
        raw = await supplier_portal.call(method=tool["api_method"], data=api_data, jwt=jwt)
        parsed = tool["parser"](raw)
        return [TextContent(type="text", text=json.dumps(parsed, ensure_ascii=False, default=str))]
    except Exception as e:
        logger.exception(f"[mcp] ошибка вызова {name}")
        return [TextContent(type="text", text=f"Ошибка API: {e}")]


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
