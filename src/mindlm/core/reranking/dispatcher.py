from mindlm.core.config.models import RerankingConfig
from mindlm.core.embeddings.base import EmbeddingProvider
from mindlm.core.models import Result
from mindlm.core.reranking.base import BaseReranker
from mindlm.core.reranking.cross_encoder import CrossEncoderReranker
from mindlm.core.reranking.mmr import MMRReranker


class RerankerDispatcher:
    def __init__(
        self,
        config: RerankingConfig,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._config = config
        self._embedding_provider = embedding_provider
        self._reranker: BaseReranker | None = None
        if config.enabled:
            self._reranker = self._build()

    def _build(self) -> BaseReranker:
        match self._config.method:
            case "cross_encoder":
                return CrossEncoderReranker(self._config)
            case "mmr":
                if self._embedding_provider is None:
                    raise ValueError("EmbeddingProvider required for MMR reranking")
                return MMRReranker(self._config, self._embedding_provider)
            case _:
                raise ValueError(f"Unknown reranking method: {self._config.method}")

    def rerank(self, query: str, results: list[Result]) -> list[Result]:
        if not self._config.enabled or self._reranker is None:
            return results
        return self._reranker.rerank(query, results)
