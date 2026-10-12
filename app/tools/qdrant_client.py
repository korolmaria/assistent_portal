"""Клиент к Qdrant: коллекция с dense + sparse, гибридный поиск через RRF."""
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient as _QdrantClient
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings


class QdrantDB:
    """Обёртка над Qdrant."""

    def __init__(self) -> None:
        self.client = _QdrantClient(url=settings.qdrant_url)
        self.collection = settings.qdrant_collection
        self.dim = settings.embedding_dim

    # ------------------------------------------------------------------
    # Создание коллекции
    # ------------------------------------------------------------------
    def create_collection(self, recreate: bool = False) -> None:
        """Создаёт коллекцию: dense_bge (1024, Cosine) + sparse (BM25)."""
        exists = self.client.collection_exists(self.collection)

        if exists and recreate:
            logger.warning(f"[qdrant] удаляю коллекцию '{self.collection}'")
            self.client.delete_collection(self.collection)
            exists = False

        if exists:
            logger.info(f"[qdrant] коллекция '{self.collection}' уже существует")
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense_bge": models.VectorParams(
                    size=self.dim,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )
        logger.info(f"[qdrant] коллекция '{self.collection}' создана")

    def create_payload_indexes(self) -> None:
        """Индексы для фильтрации по payload."""
        for field_name, field_type in [
            ("document_id", "keyword"),
            ("page", "integer"),
            ("element_type", "keyword"),
            ("source_file", "keyword"),
        ]:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=field_type,
                )
                logger.info(f"[qdrant] индекс '{field_name}' создан")
            except Exception as e:
                logger.debug(f"[qdrant] индекс '{field_name}': {e}")

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------
    def upsert(self, points: list[dict[str, Any]], batch_size: int = 100) -> int:
        """
        points: [{"id": str, "dense": list[float], "text": str, "payload": dict}]
        """
        total = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            qdrant_points = [
                models.PointStruct(
                    id=p["id"],
                    vector={
                        "dense_bge": p["dense"],
                        "sparse": models.Document(text=p["text"], model="Qdrant/bm25"),
                    },
                    payload=p["payload"],
                )
                for p in batch
            ]
            self.client.upsert(
                collection_name=self.collection,
                points=qdrant_points,
                wait=True,
            )
            total += len(batch)
            logger.debug(f"[qdrant] upsert {total}/{len(points)}")

        logger.info(f"[qdrant] upsert завершён: {total} точек")
        return total

    # ------------------------------------------------------------------
    # Гибридный поиск (dense + sparse + RRF)
    # ------------------------------------------------------------------
    def hybrid_search(
        self,
        query: str,
        dense_vector: list[float],
        top_k: int = 20,
        prefetch_limit: int = 40,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        qdrant_filter = None
        if filter_conditions:
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=k, match=models.MatchValue(value=v)
                    )
                    for k, v in filter_conditions.items()
                ]
            )

        try:
            results = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector,
                        using="dense_bge",
                        limit=prefetch_limit,
                        filter=qdrant_filter,
                    ),
                    models.Prefetch(
                        query=models.Document(text=query, model="Qdrant/bm25"),
                        using="sparse",
                        limit=prefetch_limit,
                        filter=qdrant_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
        except UnexpectedResponse as e:
            logger.error(f"[qdrant] hybrid_search error: {e}")
            return []

        output = []
        for point in results.points:
            output.append({
                "id": point.id,
                "score": point.score,
                "payload": point.payload or {},
            })

        logger.debug(f"[qdrant] hybrid_search: {len(output)} results")
        return output

    # ------------------------------------------------------------------
    # Служебные
    # ------------------------------------------------------------------
    def scroll_by_document(self, document_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        results, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
        )
        return [{"id": p.id, "payload": p.payload} for p in results]

    def delete_by_document(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
            wait=True,
        )
        logger.info(f"[qdrant] удалены точки document_id='{document_id}'")

    def info(self) -> dict[str, Any]:
        try:
            info = self.client.get_collection(self.collection)
            return {
                "exists": True,
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.warning(f"[qdrant] health check failed: {e}")
            return False


# Singleton
qdrant_db = QdrantDB()
