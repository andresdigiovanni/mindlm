from dataclasses import replace

from sentence_transformers import CrossEncoder

from mindlm.core.config.models import RerankingConfig
from mindlm.core.models import Result
from mindlm.core.reranking.base import BaseReranker


class CrossEncoderReranker(BaseReranker):
    def __init__(self, config: RerankingConfig) -> None:
        self._model = CrossEncoder(
            config.model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, query: str, results: list[Result]) -> list[Result]:
        if not results:
            return results
        pairs = [(query, r.payload.get("content", "")) for r in results]
        scores: list[float] = self._model.predict(pairs).tolist()
        scored = sorted(
            zip(results, scores, strict=False), key=lambda x: x[1], reverse=True
        )
        return [replace(r, score=s) for r, s in scored]
