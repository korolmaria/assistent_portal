"""
RAG-агент: гибридный поиск + реранкинг + расширение соседями.
"""
from typing import Any

from loguru import logger

from app.schemas import Source
from app.tools.qdrant_client import qdrant_db
from app.tools.triton_client import triton_client


CANDIDATES_TOP_K = 25     
RERANK_TOP_K = 8           
MIN_RERANK_SCORE = 0.01    # Вернули нормальный порог
RERANK_BATCH_SIZE = 32     # Лимит Triton


def _rerank_in_batches(query: str, texts: list[str]) -> list[float]:
    """Реранкинг с разбивкой на батчи (Triton не принимает > 32 текстов за раз)."""
    if not texts:
        return []
    scores: list[float] = []
    for i in range(0, len(texts), RERANK_BATCH_SIZE):
        batch = texts[i:i + RERANK_BATCH_SIZE]
        batch_scores = triton_client.rerank(query, batch)
        scores.extend(batch_scores)
        logger.debug(f"[rag_agent] реранкинг батча {i // RERANK_BATCH_SIZE + 1}: {len(batch)} текстов")
    return scores


def search(
    query: str,
    top_k: int = RERANK_TOP_K,
    min_score: float = MIN_RERANK_SCORE,
    filter_conditions: dict[str, Any] | None = None,
    expand_neighbors: bool = True,
) -> tuple[list[dict[str, Any]], list[Source]]:
    """
    RAG-поиск: dense + sparse + RRF + реранкинг + расширение соседями.
    """
    logger.info(f"[rag_agent] поиск: {query[:80]}")

    # 1. Dense-эмбеддинг запроса
    dense = triton_client.embed([query])[0].tolist()

    # 2. Гибридный поиск
    candidates = qdrant_db.hybrid_search(
        query=query,
        dense_vector=dense,
        top_k=CANDIDATES_TOP_K,
        prefetch_limit=CANDIDATES_TOP_K * 2,
        filter_conditions=filter_conditions,
    )
    logger.info(f"[rag_agent] кандидатов из Qdrant: {len(candidates)}")

    # === ФОЛБЭК ДЛЯ API-ЗАПРОСОВ ===
    query_lower = query.lower()
    is_api_query = any(kw in query_lower for kw in ["метод", "api", "запрос", "post", "get"])
    
    if is_api_query:
        logger.info("[rag_agent] API-запрос: применяю фолбэк-поиск по ключевым словам")
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchText
            
            scroll_res, _ = qdrant_db.client.scroll(
                collection_name=qdrant_db.collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="text", match=MatchText(text="MethodName")),
                    ]
                ),
                limit=10, 
                with_payload=True
            )
            
            for p in scroll_res:
                if not any(c["id"] == p.id for c in candidates):
                    candidates.append({
                        "id": p.id,
                        "payload": p.payload,
                        "score": 10.0,
                    })
                    logger.debug(f"[rag_agent] фолбэк: добавлен чанк {p.id} ({p.payload.get('source_file')})")
        except Exception as e:
            logger.warning(f"[rag_agent] фолбэк-поиск не удался: {e}")

    if not candidates:
        return [], []

    # 3. Реранкинг (с разбивкой на батчи)
    texts = [
        f"{c['payload'].get('section', '')}\n{c['payload'].get('text', '')}"
        for c in candidates
    ]
    logger.info(f"[rag_agent] отправляю на реранкинг {len(texts)} текстов (батчами по {RERANK_BATCH_SIZE})")
    rerank_scores = _rerank_in_batches(query, texts)
    logger.info(f"[rag_agent] реранкинг завершён для {len(rerank_scores)} текстов")
    
    # === ПРИНУДИТЕЛЬНЫЙ БУСТ ДЛЯ API-МЕТОДОВ ===
    boosted_count = 0
    for i, (cand, score) in enumerate(zip(candidates, rerank_scores)):
        text = cand["payload"].get("text", "")
        if "MethodName" in text or "POST /SUPPLIERPORTAL" in text or "POST SUPPLIERPORTAL" in text:
            if score < 0.9:
                rerank_scores[i] = 0.9
                boosted_count += 1
                logger.debug(f"[rag_agent] буст для API-чанка {cand['id']}: {score:.4f} -> 0.9")
    
    if boosted_count:
        logger.info(f"[rag_agent] применён буст для {boosted_count} API-чанков")
    
    # 4. Сортировка по реранк-скору
    reranked = sorted(zip(candidates, rerank_scores), key=lambda x: -x[1])

    # 5. Фильтр + top_k
    results: list[dict[str, Any]] = []
    sources: list[Source] = []
    seen_pages: set[tuple[str, int | None]] = set()
    force_expand_ids = set()

    for cand, r_score in reranked:
        if r_score < min_score:
            continue
        if len(results) >= top_k:
            break

        payload = cand["payload"]
        text = payload.get("text", "")
        doc_id = payload.get("document_id", "")
        page = payload.get("page")
        source_file = payload.get("source_file", "")

        chunk = {
            "id": cand["id"],
            "text": text,
            "score": round(r_score, 4),
            "rrf_score": round(cand["score"], 4),
            "document_id": doc_id,
            "source_file": source_file,
            "page": page,
            "page_label": payload.get("page_label"),
            "section": payload.get("section"),
            "element_type": payload.get("element_type"),
            "chunk_index": payload.get("chunk_index"),
            "prev_chunk_id": payload.get("prev_chunk_id"),
            "next_chunk_id": payload.get("next_chunk_id"),
        }
        results.append(chunk)

        if "MethodName" in text or "POST /SUPPLIERPORTAL" in text or "POST SUPPLIERPORTAL" in text:
            force_expand_ids.add(cand["id"])
            logger.debug(f"[rag_agent] найден API-метод, принудительное расширение для {cand['id']}")

        key = (doc_id, page)
        if key not in seen_pages:
            seen_pages.add(key)
            sources.append(Source(
                doc_id=doc_id,
                doc_title=source_file,
                page=page,
                page_label=payload.get("page_label"),
                section=payload.get("section", ""),
                content_type=payload.get("element_type", "text"),
                snippet=text[:200],
                grade_score=round(r_score, 4),
            ))

    # === ФИЛЬТРАЦИЯ ДЛЯ API-ЗАПРОСОВ ===
    # Если запрос про API, оставляем только технические чанки,
    # чтобы LLM не смотрела на "кнопки в интерфейсе" и не галлюцинировала.
    if is_api_query and results:
        api_results = [
            r for r in results 
            if "MethodName" in r["text"] 
            or "POST /SUPPLIERPORTAL" in r["text"]
            or "API_инструкция" in r["source_file"]
        ]
        if api_results:
            logger.info(f"[rag_agent] фильтр API: было {len(results)}, осталось {len(api_results)}")
            results = api_results
            # Обновляем sources
            sources = [s for s in sources if "API_инструкция" in s.doc_title or "api" in s.doc_title.lower()]

    logger.info(
        f"[rag_agent] финальных чанков: {len(results)}, "
        f"уникальных источников: {len(sources)}"
    )

    # 6. Расширение соседями
    if expand_neighbors or force_expand_ids:
        expanded = expand_with_neighbors(
            results, 
            max_neighbors=2, 
            force_expand_ids=force_expand_ids
        )
        logger.info(f"[rag_agent] после расширения: {len(expanded)} чанков")
        return expanded, sources

    return results, sources


def expand_with_neighbors(
    chunks: list[dict[str, Any]],
    max_neighbors: int = 1,
    force_expand_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Расширяет список чанков соседними.
    """
    if not chunks:
        return chunks

    force_expand_ids = force_expand_ids or set()

    seen_keys: set[tuple[str, int]] = set()
    unique_chunks: list[dict[str, Any]] = []
    for ch in chunks:
        key = (ch.get("document_id") or "", ch.get("chunk_index") or 0)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_chunks.append(ch)

    extra: list[dict[str, Any]] = []

    for ch in unique_chunks:
        doc_id = ch.get("document_id")
        chunk_idx = ch.get("chunk_index")
        
        is_forced = ch.get("id") in force_expand_ids
        neighbors_to_take = max_neighbors if is_forced else 1
        
        neighbor_ids = []
        if ch.get("prev_chunk_id"): neighbor_ids.append(ch["prev_chunk_id"])
        if ch.get("next_chunk_id"): neighbor_ids.append(ch["next_chunk_id"])

        for neighbor_id in neighbor_ids:
            try:
                points = qdrant_db.client.retrieve(
                    collection_name=qdrant_db.collection,
                    ids=[neighbor_id],
                    with_payload=True,
                )
                if points:
                    p = points[0]
                    pl = p.payload or {}
                    n_key = (pl.get("document_id") or "", pl.get("chunk_index") or 0)
                    if n_key not in seen_keys:
                        seen_keys.add(n_key)
                        extra.append({
                            "id": p.id, "text": pl.get("text", ""), "score": 0.0,
                            "document_id": pl.get("document_id"),
                            "source_file": pl.get("source_file"),
                            "page": pl.get("page"), "page_label": pl.get("page_label"),
                            "section": pl.get("section"), "element_type": pl.get("element_type"),
                            "chunk_index": pl.get("chunk_index"), "is_neighbor": True,
                        })
            except Exception as e:
                logger.warning(f"[rag_agent] ошибка получения соседа {neighbor_id}: {e}")

        if doc_id and chunk_idx is not None:
            for offset in range(1, neighbors_to_take + 1):
                for target_idx in [chunk_idx - offset, chunk_idx + offset]:
                    t_key = (doc_id, target_idx)
                    if t_key in seen_keys:
                        continue
                    
                    try:
                        from qdrant_client.models import Filter, FieldCondition, MatchValue
                        scroll_res, _ = qdrant_db.client.scroll(
                            collection_name=qdrant_db.collection,
                            scroll_filter=Filter(
                                must=[
                                    FieldCondition(key="document_id", match=MatchValue(value=doc_id)),
                                    FieldCondition(key="chunk_index", match=MatchValue(value=target_idx)),
                                ]
                            ),
                            limit=1,
                            with_payload=True
                        )
                        if scroll_res:
                            p = scroll_res[0]
                            pl = p.payload or {}
                            seen_keys.add(t_key)
                            extra.append({
                                "id": p.id, "text": pl.get("text", ""), "score": 0.0,
                                "document_id": pl.get("document_id"),
                                "source_file": pl.get("source_file"),
                                "page": pl.get("page"), "page_label": pl.get("page_label"),
                                "section": pl.get("section"), "element_type": pl.get("element_type"),
                                "chunk_index": pl.get("chunk_index"), "is_neighbor": True,
                            })
                    except Exception as e:
                        logger.debug(f"[rag_agent] не удалось найти соседа по индексу {t_key}: {e}")

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


def build_context(chunks: list[dict[str, Any]], max_chars: int = 16000) -> str:
    """
    Собирает контекст для LLM.
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