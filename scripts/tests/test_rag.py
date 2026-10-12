"""Тест RAG-агента (поиск + реранкинг)."""
from app.agents.rag_agent import search, build_context


def main():
    queries = [
        "Какой допустимый остаток для товара со сроком годности 45 суток?",
        "Что будет с товаром с нарушением норм требований?",
        "Как подтвердить заказ в кабинете поставщика?",
    ]

    for query in queries:
        print(f"\n{'=' * 70}")
        print(f"🔍 {query}")
        print('=' * 70)

        chunks, sources = search(query, top_k=3)

        print(f"\nНайдено: {len(chunks)} чанков, {len(sources)} источников\n")

        for i, (ch, src) in enumerate(zip(chunks, sources), 1):
            print(f"[{i}] score={ch['score']:.4f}  rrf={ch['rrf_score']:.4f}")
            print(f"    doc: {src.doc_title}")
            print(f"    стр.: {src.page_label}  раздел: {src.section or '—'}")
            print(f"    {ch['text'][:200]}")
            print()

        print("--- CONTEXT для LLM ---")
        print(build_context(chunks)[:600])
        print()


if __name__ == "__main__":
    main()
