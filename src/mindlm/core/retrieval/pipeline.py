from typing import Any

from langfuse.decorators import langfuse_context, observe

from mindlm.core.config.models import RetrievalConfig
from mindlm.core.models import Result
from mindlm.core.retrieval.context_resolver import ContextResolver
from mindlm.core.retrieval.fusion import FusionEngine
from mindlm.core.retrieval.graph_augmenter import GraphAugmenter


class RetrievalPipeline:
    """Orchestrates the full retrieval pipeline.

    Pipeline: FusionEngine → ContextResolver → GraphAugmenter (optional)
    """

    def __init__(
        self,
        config: RetrievalConfig,
        fusion: FusionEngine,
        context_resolver: ContextResolver,
        graph_augmenter: GraphAugmenter | None = None,
    ) -> None:
        self._config = config
        self._fusion = fusion
        self._context_resolver = context_resolver
        self._graph_augmenter = graph_augmenter

    @observe(name="retrieve")
    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        *,
        top_k: int | None = None,
    ) -> list[Result]:
        """Execute the full retrieval pipeline.

        Args:
            query: User search query
            filters: Optional vectorstore filters
            top_k: Override config.top_k for this request

        Returns:
            Final list of results after all pipeline stages.
        """
        effective_top_k = top_k if top_k is not None else self._config.top_k
        langfuse_context.update_current_observation(
            input=query,
            metadata={
                "strategy": self._config.strategy,
                "top_k": effective_top_k,
            },
        )
        results = self._fusion.fuse(query, filters, effective_top_k)
        results = self._context_resolver.resolve(results)
        if self._graph_augmenter is not None:
            results = self._graph_augmenter.augment(results, effective_top_k)
        return results
