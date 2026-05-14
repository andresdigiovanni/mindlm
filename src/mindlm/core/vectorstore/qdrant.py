import contextlib
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from mindlm.core.config.models import VectorStoreConfig
from mindlm.core.models import Point, Result, SparseVector
from mindlm.core.vectorstore.base import VectorStore


class QdrantVectorStore(VectorStore):
    def __init__(self, config: VectorStoreConfig) -> None:
        self._client = QdrantClient(
            host=config.host,
            port=config.port,
            api_key=config.api_key,
            prefer_grpc=False,
        )
        self._collection = config.collection

    def upsert(self, points: list[Point]) -> None:
        structs = []
        for p in points:
            vector: dict[str, Any] = {"dense": p.vector}
            if p.sparse_vector is not None:
                vector["sparse"] = models.SparseVector(
                    indices=p.sparse_vector.indices,
                    values=p.sparse_vector.values,
                )
            structs.append(
                models.PointStruct(id=p.id, vector=vector, payload=p.payload)
            )
        self._client.upsert(collection_name=self._collection, points=structs)

    def search(
        self, query_vector: list[float], top_k: int, filters: dict | None
    ) -> list[Result]:
        hits = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            using="dense",
            limit=top_k,
            query_filter=self._build_filter(filters),
        )
        return [
            Result(id=str(h.id), score=h.score, payload=h.payload or {})
            for h in hits.points
        ]

    def search_hybrid(
        self,
        dense: list[float],
        sparse: SparseVector,
        top_k: int,
        filters: dict | None,
    ) -> list[Result]:
        hits = self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                models.Prefetch(query=dense, using="dense", limit=top_k),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse.indices, values=sparse.values
                    ),
                    using="sparse",
                    limit=top_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            query_filter=self._build_filter(filters),
        )
        return [
            Result(id=str(h.id), score=h.score, payload=h.payload or {})
            for h in hits.points
        ]

    def delete(self, ids: list[str]) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.PointIdsList(points=list(ids)),
        )

    def delete_by_filter(self, filters: dict) -> None:
        f = self._build_filter(filters)
        if f is not None:
            self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(filter=f),
            )

    def get_by_id(self, id: str) -> Point | None:
        results = self._client.retrieve(
            collection_name=self._collection,
            ids=[id],
            with_vectors=True,
            with_payload=True,
        )
        if not results:
            return None
        r = results[0]
        raw_vec = r.vector
        vector: list[float] = (
            raw_vec
            if isinstance(raw_vec, list)
            and raw_vec
            and not isinstance(raw_vec[0], list)
            else []
        )
        return Point(id=str(r.id), vector=vector, payload=r.payload or {})

    def scroll(
        self, filters: dict, limit: int, offset: str | None
    ) -> tuple[list[Point], str | None]:
        points, next_offset = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=self._build_filter(filters),
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        result_points = [
            Point(
                id=str(p.id),
                vector=p.vector
                if isinstance(p.vector, list)
                and p.vector
                and not isinstance(p.vector[0], list)
                else [],
                payload=p.payload or {},
            )
            for p in points
        ]
        return result_points, str(next_offset) if next_offset is not None else None

    def create_collection(self, name: str, dense_dim: int, sparse: bool) -> None:
        vectors_config: dict[str, VectorParams] = {
            "dense": VectorParams(size=dense_dim, distance=Distance.COSINE)
        }
        sparse_config: dict[str, SparseVectorParams] | None = None
        if sparse:
            sparse_config = {
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            }
        self._client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
        )

    def recreate_collection(self, name: str, dense_dim: int, sparse: bool) -> None:
        with contextlib.suppress(Exception):  # Collection may not exist yet
            self._client.delete_collection(name)  # Collection may not exist yet
        self.create_collection(name, dense_dim, sparse)

    def list_collections(self) -> list[str]:
        return [c.name for c in self._client.get_collections().collections]

    def _build_filter(self, filters: dict | None) -> models.Filter | None:
        if not filters:
            return None
        conditions: list[models.FieldCondition] = [
            models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in filters.items()
        ]
        must_conditions: list[
            models.FieldCondition
            | models.IsEmptyCondition
            | models.IsNullCondition
            | models.HasIdCondition
            | models.HasVectorCondition
            | models.NestedCondition
            | models.Filter
        ] = list(conditions)
        return models.Filter(must=must_conditions)
