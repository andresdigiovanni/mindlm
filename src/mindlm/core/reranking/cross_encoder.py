from dataclasses import replace

import torch
from langfuse.decorators import observe
from sentence_transformers import CrossEncoder

from mindlm.core.config.models import RerankingConfig
from mindlm.core.models import Result
from mindlm.core.reranking.base import BaseReranker


class CrossEncoderReranker(BaseReranker):
    def __init__(self, config: RerankingConfig) -> None:
        # activation_fn=Sigmoid() overrides the model's default activation and
        # ensures predict() always operates on raw logits → returns scores in (0, 1).
        # This avoids double-sigmoid on models (e.g. BAAI/bge-reranker-v2-m3)
        # that already declare Sigmoid as their default activation_fn.
        self._model = CrossEncoder(
            config.model or "BAAI/bge-reranker-v2-m3",
            activation_fn=torch.nn.Sigmoid(),
        )

    @observe(name="cross-encoder-rerank")
    def rerank(self, query: str, results: list[Result]) -> list[Result]:
        if not results:
            return results
        pairs = [
            (query, r.payload.get("matched_chunk") or r.payload.get("content", ""))
            for r in results
        ]
        scores: list[float] = self._model.predict(pairs).tolist()  # (0, 1) via Sigmoid
        scored = sorted(
            zip(results, scores, strict=False), key=lambda x: x[1], reverse=True
        )
        return [replace(r, score=s) for r, s in scored]
