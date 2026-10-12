"""Тест гибридного поиска по Qdrant."""
from app.tools.qdrant_client import qdrant_db
from app.tools.triton_client import triton_client


def main():
    # Тестовые запросы
    queries = [
        "Какой допустимый остаток для товара со сроком годности 45 суток?",
        "Что будет с товаром с нарушением норм требований?",
        "Какой срок годности для товара до 15 суток?",
    ]

    for query in queries:
        print(f"\n{'=' * 70}")
        print(f"🔍 {query}")
        print('=' * 70)

        # Dense-эмбеддинг запроса
        dense = triton_client.embed([query])[0].tolist()

        # Гибридный поиск
        results = qdrant_db.hybrid_search(
            query=query,
            dense_vector=dense,
            top_k=3,
        )

        for i, r in enumerate(results, 1):
            payload = r["payload"]
            text = payload.get("text", "")[:200]
            print(f"\n[{i}] score={r['score']:.4f}")
            print(f"    section={payload.get('section', '')!r}")
            print(f"    type={payload.get('element_type')}")
            print(f"    {text}")


if __name__ == "__main__":
    main()
