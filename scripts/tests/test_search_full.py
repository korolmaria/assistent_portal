"""Тест поиска по всей базе."""
from app.agents.rag_agent import search


def main():
    queries = [
        "Как подтвердить заказ в кабинете поставщика?",
        "Что делать с просроченными заказами?",
        "Какие отчёты доступны поставщику?",
        "Как работать с графиками поставок?",
        "Что такое Service Level?",
        "Как обрабатываются претензии?",
        "Требования к остаточному сроку годности",
    ]

    for query in queries:
        print(f"\n{'=' * 70}")
        print(f"🔍 {query}")
        print('=' * 70)

        chunks, sources = search(query, top_k=3)

        if not chunks:
            print("Ничего не найдено\n")
            continue

        for i, ch in enumerate(chunks, 1):
            payload = ch
            print(f"\n[{i}] score={ch['score']:.4f}")
            print(f"    doc: {ch.get('source_file', '')}")
            print(f"    стр.: {ch.get('page_label')}  раздел: {ch.get('section', '')[:50]}")
            print(f"    {ch['text'][:200]}")


if __name__ == "__main__":
    main()
