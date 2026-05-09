from mindlm.core.config.models import RerankingConfig
from mindlm.core.embeddings.base import EmbeddingProvider
from mindlm.core.models import Result
from mindlm.core.reranking.base import BaseReranker


def _cosine(a: list[float], b: list[float]) -> float:
    dot: float = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a: float = sum(x * x for x in a) ** 0.5
    norm_b: float = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class MMRReranker(BaseReranker):
    def __init__(
        self,
        _config: RerankingConfig,
        embedding_provider: EmbeddingProvider,
        lambda_mult: float = 0.5,
    ) -> None:
        self._provider = embedding_provider
        self._lambda = lambda_mult

    def rerank(self, query: str, results: list[Result]) -> list[Result]:
        if not results:
            return results
        contents = [r.payload.get("content", "") for r in results]
        query_vec = self._provider.embed_one(query)
        content_vecs = self._provider.embed(contents)

        query_sims = [_cosine(query_vec, v) for v in content_vecs]
        selected_indices: list[int] = []
        remaining = list(range(len(results)))

        while remaining:
            if not selected_indices:
                best = max(remaining, key=lambda i: query_sims[i])
            else:
                selected_vecs = [content_vecs[i] for i in selected_indices]

                def mmr_score(i: int, _sv: list[list[float]] = selected_vecs) -> float:
                    max_sim_selected = max(_cosine(content_vecs[i], sv) for sv in _sv)
                    return (
                        self._lambda * query_sims[i]
                        - (1 - self._lambda) * max_sim_selected
                    )

                best = max(remaining, key=mmr_score)
            selected_indices.append(best)
            remaining.remove(best)

        return [results[i] for i in selected_indices]
