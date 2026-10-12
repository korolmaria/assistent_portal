"""Тест RAG + Synthesis."""
import asyncio
from app.agents.rag_agent import search
from app.agents.synthesis_agent import synthesize, extract_citations


async def main():
    queries = [
        "Какой допустимый остаток для товара со сроком годности 45 суток?",
        "Что будет с товаром с нарушением норм требований?",
    ]

    for query in queries:
        print(f"\n{'=' * 70}")
        print(f"🔍 {query}")
        print('=' * 70)

        # RAG
        chunks, _ = search(query, top_k=3)
        print(f"RAG: {len(chunks)} чанков\n")

        if not chunks:
            print("Ничего не найдено\n")
            continue

        # Synthesis
        answer, sources, tools = await synthesize(query, rag_chunks=chunks)

        print("ОТВЕТ:")
        print(answer)
        print()

        citations = extract_citations(answer)
        print(f"Цитат: {len(citations)}: {citations}")


if __name__ == "__main__":
    asyncio.run(main())
