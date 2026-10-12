"""Тест Supervisor — маршрутизация + SSE-события."""
import asyncio
from app.agents.supervisor import run_supervisor


async def main():
    queries = [
        # RAG-запросы
        ("auto", "Как подтвердить заказ в кабинете поставщика?"),
        ("auto", "Что такое Service Level?"),
        # Tool-запрос
        ("auto", "Покажи мои заказы за сентябрь"),
        # Явный выбор
        ("rag", "Требования к остаточному сроку годности"),
    ]

    for route, query in queries:
        print(f"\n{'=' * 70}")
        print(f"🔍 [{route}] {query}")
        print('=' * 70)

        async for event in run_supervisor(query, route=route, jwt="fake-jwt"):
            etype = event["type"]

            if etype == "route":
                print(f"  🎯 Route: {event['route']}")
                print(f"     Reason: {event['reason']}")

            elif etype == "step":
                print(f"  📍 {event['message']}")

            elif etype == "metadata":
                print(f"\n  📝 ОТВЕТ:")
                print(f"  {event.get('answer', '')[:400]}")
                print(f"\n  Источников: {len(event.get('sources', []))}")
                print(f"  Инструментов: {event.get('tools_used', [])}")
                if event.get("display"):
                    print(f"  Display: type={event['display'].get('type')}")

            elif etype == "error":
                print(f"  ❌ ОШИБКА: {event['error']}")


if __name__ == "__main__":
    asyncio.run(main())
