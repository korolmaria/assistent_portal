"""Тест гибридного поиска + реранкинг."""
from app.tools.qdrant_client import qdrant_db
from app.tools.triton_client import triton_client


def main():
    queries = [
        "Какой допустимый остаток для товара со сроком годности 45 суток?",
        "Что будет с товаром с нарушением норм требований?",
        "Какой срок годности для товара до 15 суток?",
    ]

    for query in queries:
        print(f"\n{'=' * 70}")
        print(f"🔍 {query}")
        print('=' * 70)

        # 1. Dense-эмбеддинг
        dense = triton_client.embed([query])[0].tolist()

        # 2. Гибридный поиск (top-20)
        candidates = qdrant_db.hybrid_search(
            query=query,
            dense_vector=dense,
            top_k=20,
            prefetch_limit=40,
        )
        print(f"Кандидатов из Qdrant: {len(candidates)}")

        if not candidates:
            continue

        # 3. Реранкинг
        texts = [c["payload"].get("text", "") for c in candidates]
        rerank_scores = triton_client.rerank(query, texts)

        # 4. Сортируем по реранк-скору
        reranked = sorted(
            zip(candidates, rerank_scores),
            key=lambda x: -x[1],
        )

        print(f"\nТоп-3 после реранкинга:")
        for i, (c, r_score) in enumerate(reranked[:3], 1):
            payload = c["payload"]
            text = payload.get("text", "")[:200]
            print(f"\n[{i}] rerank={r_score:.4f}  (rrf={c['score']:.4f})")
            print(f"    section={payload.get('section', '')!r}")
            print(f"    {text}")


if __name__ == "__main__":
    main()
