from fastembed.sparse.bm25 import Bm25

from mindlm.core.config.models import RetrievalConfig
from mindlm.core.embeddings.base import EmbeddingProvider
from mindlm.core.models import Result, SparseVector
from mindlm.core.vectorstore.base import VectorStore


class Retriever:
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

    def retrieve(self, query: str, filters: dict | None = None) -> list[Result]:
        dense = self._embedding_provider.embed_one(query)
        match self._config.strategy:
            case "vector":
                return self._vectorstore.search(dense, self._config.top_k, filters)
            case "hybrid":
                sparse = self._compute_bm25(query)
                return self._vectorstore.search_hybrid(
                    dense, sparse, self._config.top_k, filters
                )
            case _:  # pragma: no cover
                raise ValueError(
                    f"Unknown retrieval strategy: {self._config.strategy!r}"
                )

    def _compute_bm25(self, query: str) -> SparseVector:
        if self._bm25 is None:
            self._bm25 = Bm25("Qdrant/bm25")
        result = next(iter(self._bm25.query_embed(query)))
        return SparseVector(
            indices=result.indices.tolist(), values=result.values.tolist()
        )
