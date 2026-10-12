"""
RAG-агент: гибридный поиск + реранкинг + расширение соседями.

Возвращает:
- top-K чанков с метаданными (+ соседние чанки для сохранения контекста)
- список Source для фронта
- готовый контекст для LLM

Ключевая фича: expand_with_neighbors — подтягивает prev/next чанки,
чтобы технические описания (метод + URL + JSON), разрезанные между
чанками, попадали в контекст вместе.
"""
from typing import Any

from loguru import logger

from app.schemas import Source
from app.tools.qdrant_client import qdrant_db
from app.tools.triton_client import triton_client


CANDIDATES_TOP_K = 20
RERANK_TOP_K = 5
MIN_RERANK_SCORE = 0.01


def search(
    query: str,
    top_k: int = RERANK_TOP_K,
    min_score: float = MIN_RERANK_SCORE,
    filter_conditions: dict[str, Any] | None = None,
    expand_neighbors: bool = True,
) -> tuple[list[dict[str, Any]], list[Source]]:
    """
    RAG-поиск: dense + sparse + RRF + реранкинг + расширение соседями.

    Args:
        query: вопрос пользователя
        top_k: сколько чанков вернуть после реранкинга
        min_score: минимальный реранк-скор
        filter_conditions: фильтры Qdrant
        expand_neighbors: подтягивать ли соседние чанки (prev/next)

    Returns:
        (chunks, sources)
    """
    logger.info(f"[rag_agent] поиск: {query[:80]}")

    # 1. Dense-эмбеддинг запроса
    dense = triton_client.embed([query])[0].tolist()

    # 2. Гибридный поиск (dense + sparse + RRF)
    candidates = qdrant_db.hybrid_search(
        query=query,
        dense_vector=dense,
        top_k=CANDIDATES_TOP_K,
        prefetch_limit=CANDIDATES_TOP_K * 2,
        filter_conditions=filter_conditions,
    )
    logger.info(f"[rag_agent] кандидатов из Qdrant: {len(candidates)}")

    if not candidates:
        return [], []

    # 3. Реранкинг
    texts = [c["payload"].get("text", "") for c in candidates]
    rerank_scores = triton_client.rerank(query, texts)
    logger.info(
        f"[rag_agent] реранк: min={min(rerank_scores):.4f}, "
        f"max={max(rerank_scores):.4f}"
    )

    # 4. Сортировка по реранк-скору
    reranked = sorted(zip(candidates, rerank_scores), key=lambda x: -x[1])

    # 5. Фильтр + top_k + дедупликация sources по (doc_id, page)
    results: list[dict[str, Any]] = []
    sources: list[Source] = []
    seen_pages: set[tuple[str, int | None]] = set()

    for cand, r_score in reranked:
        if r_score < min_score:
            continue
        if len(results) >= top_k:
            break

        payload = cand["payload"]
        text = payload.get("text", "")
        doc_id = payload.get("document_id", "")
        page = payload.get("page")

        chunk = {
            "id": cand["id"],
            "text": text,
            "score": round(r_score, 4),
            "rrf_score": round(cand["score"], 4),
            "document_id": doc_id,
            "source_file": payload.get("source_file"),
            "page": page,
            "page_label": payload.get("page_label"),
            "section": payload.get("section"),
            "element_type": payload.get("element_type"),
            "chunk_index": payload.get("chunk_index"),
            "prev_chunk_id": payload.get("prev_chunk_id"),
            "next_chunk_id": payload.get("next_chunk_id"),
        }
        results.append(chunk)

        # В sources — только уникальные (doc_id, page)
        key = (doc_id, page)
        if key not in seen_pages:
            seen_pages.add(key)
            sources.append(Source(
                doc_id=doc_id,
                doc_title=payload.get("source_file", ""),
                page=page,
                page_label=payload.get("page_label"),
                section=payload.get("section", ""),
                content_type=payload.get("element_type", "text"),
                snippet=text[:200],
                grade_score=round(r_score, 4),
            ))

    logger.info(
        f"[rag_agent] финальных чанков: {len(results)}, "
        f"уникальных источников: {len(sources)}"
    )

    # 6. Расширение соседями (для сохранения контекста техописаний)
    if expand_neighbors:
        expanded = expand_with_neighbors(results, max_neighbors=1)
        logger.info(
            f"[rag_agent] после расширения: {len(expanded)} чанков"
        )
        return expanded, sources

    return results, sources


def expand_with_neighbors(
    chunks: list[dict[str, Any]],
    max_neighbors: int = 1,
) -> list[dict[str, Any]]:
    """
    Расширяет список чанков соседними (prev/next) из Qdrant.

    Это помогает сохранить технические детали, разрезанные между чанками:
    - Имя метода в одном чанке
    - URL в другом
    - JSON-параметры в третьем

    Дедупликация по (document_id, chunk_index).

    Args:
        chunks: исходные чанки от RAG
        max_neighbors: сколько соседей с каждой стороны подтягивать

    Returns:
        Расширенный список, отсортированный по (document_id, chunk_index).
    """
    if not chunks:
        return chunks

    # Дедупликация исходных по (document_id, chunk_index)
    seen_keys: set[tuple[str, int]] = set()
    unique_chunks: list[dict[str, Any]] = []
    for ch in chunks:
        key = (ch.get("document_id") or "", ch.get("chunk_index") or 0)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_chunks.append(ch)

    logger.info(
        f"[rag_agent] expand: исходных {len(chunks)} → "
        f"уникальных {len(unique_chunks)}"
    )

    extra: list[dict[str, Any]] = []

    for ch in unique_chunks:
        prev_id = ch.get("prev_chunk_id")
        next_id = ch.get("next_chunk_id")

        for neighbor_id in [prev_id, next_id]:
            if not neighbor_id:
                continue

            try:
                points = qdrant_db.client.retrieve(
                    collection_name=qdrant_db.collection,
                    ids=[neighbor_id],
                    with_payload=True,
                )
                if not points:
                    logger.debug(
                        f"[rag_agent] сосед {neighbor_id} не найден в Qdrant"
                    )
                    continue

                p = points[0]
                pl = p.payload or {}
                neighbor_key = (
                    pl.get("document_id") or "",
                    pl.get("chunk_index") or 0,
                )

                # Пропускаем, если такой (doc_id, chunk_index) уже есть
                if neighbor_key in seen_keys:
                    logger.debug(
                        f"[rag_agent] сосед {neighbor_key} уже есть"
                    )
                    continue
                seen_keys.add(neighbor_key)

                extra.append({
                    "id": p.id,
                    "text": pl.get("text", ""),
                    "score": 0.0,
                    "rrf_score": 0.0,
                    "document_id": pl.get("document_id"),
                    "source_file": pl.get("source_file"),
                    "page": pl.get("page"),
                    "page_label": pl.get("page_label"),
                    "section": pl.get("section"),
                    "element_type": pl.get("element_type"),
                    "chunk_index": pl.get("chunk_index"),
                    "is_neighbor": True,
                })

            except Exception as e:
                logger.warning(
                    f"[rag_agent] сосед {neighbor_id}: {e}"
                )

    all_chunks = unique_chunks + extra
    all_chunks.sort(key=lambda x: (
        x.get("document_id") or "",
        x.get("chunk_index") or 0,
    ))
    logger.info(
        f"[rag_agent] expand: итого {len(all_chunks)} чанков "
        f"(добавлено {len(extra)} соседей)"
    )
    return all_chunks


def build_context(chunks: list[dict[str, Any]], max_chars: int = 8000) -> str:
    """
    Собирает контекст для LLM из чанков.

    Формат:
        [1] (источник, стр. X, раздел)
        текст
        ---
        [2] ...
    """
    parts = []
    total = 0

    for i, ch in enumerate(chunks, 1):
        header = f"[{i}] {ch.get('source_file', '')}"
        if ch.get("page_label"):
            header += f", {ch['page_label']}"
        if ch.get("section"):
            header += f", раздел: {ch['section']}"

        block = f"{header}\n{ch['text']}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

    return "\n\n---\n\n".join(parts)
