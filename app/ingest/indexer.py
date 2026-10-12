"""
Чанки → эмбеддинги → Qdrant.

Batch-обработка: чанки отправляются в Triton по 32 штуки,
затем upsert в Qdrant по 100 точек.
"""
from typing import Any

from loguru import logger

from app.tools.qdrant_client import qdrant_db
from app.tools.triton_client import triton_client


EMBED_BATCH_SIZE = 32   # сколько текстов отправлять в Triton за раз
UPSERT_BATCH_SIZE = 100  # сколько точек писать в Qdrant за раз


def index_chunks(chunks: list[dict[str, Any]]) -> int:
    """
    Принимает чанки от chunker, считает эмбеддинги, кладёт в Qdrant.

    Args:
        chunks: список {"id": str, "text": str, "payload": dict}

    Returns:
        Количество загруженных точек.
    """
    if not chunks:
        logger.warning("[indexer] пустой список чанков")
        return 0

    logger.info(f"[indexer] индексирую {len(chunks)} чанков")

    # 1. Собираем все тексты
    texts = [ch["text"] for ch in chunks]

    # 2. Эмбеддинги батчами
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        logger.debug(f"[indexer] эмбеддинги {i + len(batch)}/{len(texts)}")
        emb = triton_client.embed(batch)  # (N, 1024)
        all_embeddings.append(emb)

    import numpy as np
    embeddings = np.vstack(all_embeddings) if all_embeddings else np.zeros((0, 1024))
    logger.info(f"[indexer] эмбеддинги: shape={embeddings.shape}")

    # 3. Формируем points для Qdrant
    points = []
    for ch, emb in zip(chunks, embeddings):
        points.append({
            "id": ch["id"],
            "dense": emb.tolist(),
            "text": ch["text"],     # для BM25 (sparse)
            "payload": {
                **ch["payload"],
                "text": ch["text"],   # дублируем текст в payload для возврата
            },
        })

    # 4. Upsert
    total = qdrant_db.upsert(points, batch_size=UPSERT_BATCH_SIZE)
    logger.info(f"[indexer] ✅ загружено {total} точек в Qdrant")
    return total


def delete_document(document_id: str) -> None:
    """Удаляет все чанки документа из Qdrant."""
    qdrant_db.delete_by_document(document_id)
    logger.info(f"[indexer] удалён document_id={document_id}")


def get_document_stats(document_id: str) -> dict[str, Any]:
    """Статистика по документу в Qdrant."""
    points = qdrant_db.scroll_by_document(document_id)
    return {
        "document_id": document_id,
        "chunks_count": len(points),
    }
