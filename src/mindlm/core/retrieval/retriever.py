from typing import Any

from fastembed.sparse.bm25 import Bm25

from mindlm.core.config.models import RetrievalConfig
from mindlm.core.embeddings.base import EmbeddingProvider
from mindlm.core.models import Result, SparseVector
from mindlm.core.vectorstore.base import VectorStore


class Retriever:
    """Raw single-query retrieval: embed query → dense or hybrid vectorstore search."""

    def __init__(
        self,
        config: RetrievalConfig,
        vectorstore: VectorStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._config = config
        self._vectorstore = vectorstore
        self._embedding_provider = embedding_provider
        self._bm25: Bm25 | None = None

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        *,
        top_k: int | None = None,
    ) -> list[Result]:
        """Retrieve raw candidates from vectorstore for a single query.

        Args:
            query: Single query string
            filters: Optional vectorstore filters
            top_k: Override config.top_k for this call

        Returns:
            List of Result objects from vectorstore.

        Raises:
            ValueError: If strategy not in ('vector', 'hybrid')
        """
        effective_top_k = top_k if top_k is not None else self._config.top_k
        dense = self._embedding_provider.embed([query])[0]
        match self._config.strategy:
            case "vector":
                return self._vectorstore.search(dense, effective_top_k, filters)
            case "hybrid":
                sparse = self._compute_bm25(query)
                return self._vectorstore.search_hybrid(
                    dense, sparse, effective_top_k, filters
                )
            case _:  # pragma: no cover
                raise ValueError(
                    f"Unknown retrieval strategy: {self._config.strategy!r}"
                )

    def _compute_bm25(self, query: str) -> SparseVector:
        """Compute BM25 sparse vector for hybrid search (lazily initialized)."""
        if self._bm25 is None:
            self._bm25 = Bm25("Qdrant/bm25")
        result = next(iter(self._bm25.query_embed(query)))
        return SparseVector(
            indices=result.indices.tolist(), values=result.values.tolist()
        )
