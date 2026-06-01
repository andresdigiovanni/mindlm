from dataclasses import replace
from typing import Any

from langfuse.decorators import langfuse_context, observe

from mindlm.core.generation.base import LLMProvider
from mindlm.core.models import Result
from mindlm.core.query_processing.dispatcher import QueryProcessorDispatcher
from mindlm.core.retrieval.retriever import Retriever

_DEFAULT_RRF_K = 60


def _rrf_merge(
    ranked_lists: list[list[Result]], k: int = _DEFAULT_RRF_K
) -> list[Result]:
    """Reciprocal Rank Fusion of multiple ranked result lists.

    RRF(d) = sum(1 / (k + rank_i(d))) across all lists i.

    Args:
        ranked_lists: List of result lists, each sorted by relevance descending.
        k: RRF constant (default 60). Higher k reduces the impact of high ranks.

    Returns:
        Merged list sorted by accumulated RRF score descending.
        Payload is taken from the last list that contained the document.
    """
    scores: dict[str, float] = {}
    seen: dict[str, Result] = {}
    for ranked in ranked_lists:
        for rank_i, result in enumerate(ranked):
            scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (k + rank_i + 1)
            seen[result.id] = result
    merged = [
        replace(seen[doc_id], score=rrf_score) for doc_id, rrf_score in scores.items()
    ]
    merged.sort(key=lambda r: r.score, reverse=True)
    return merged


class FusionEngine:
    """Multi-query retrieval with Reciprocal Rank Fusion.

    Expands a query via QueryProcessorDispatcher (if provided), retrieves
    raw candidates for each query variant, and merges them using RRF.
    """

    def __init__(
        self,
        retriever: Retriever,
        query_processor: QueryProcessorDispatcher | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        if query_processor is not None and llm is None:
            raise ValueError("llm is required when query_processor is provided")
        self._retriever = retriever
        self._query_processor = query_processor
        self._llm = llm

    @observe(name="fusion-retrieve")
    def fuse(
        self,
        query: str,
        filters: dict[str, Any] | None,
        fused_top_k: int,
        *,
        per_query_top_k: int,
    ) -> list[Result]:
        """Expand query, retrieve per-query candidates, merge with RRF.

        Args:
            query: Original user query
            filters: Optional vectorstore filters
            fused_top_k: Number of results to keep after RRF merge.
            per_query_top_k: Number of candidates to fetch for each query variant.

        Returns:
            RRF-merged results sorted by score descending, truncated to top_k.
        """
        langfuse_context.update_current_observation(
            input=query,
            metadata={
                "fused_top_k": fused_top_k,
                "per_query_top_k": per_query_top_k,
            },
        )
        if self._query_processor is not None:
            if self._llm is None:  # pragma: no cover
                raise RuntimeError("llm must be set when query_processor is provided")
            queries: list[str] = self._query_processor.process(query, self._llm)
        else:
            queries = [query]

        ranked_lists = [
            self._retriever.retrieve(q, filters, top_k=per_query_top_k) for q in queries
        ]
        merged = _rrf_merge(ranked_lists)[:fused_top_k]
        langfuse_context.update_current_observation(
            metadata={
                "queries_count": len(queries),
                "merged_count": len(merged),
            },
        )
        return merged
