from collections.abc import Callable

from langfuse.decorators import langfuse_context, observe

from mindlm.core.config.models import QueryProcessingConfig
from mindlm.core.generation.base import LLMProvider
from mindlm.core.query_processing.base import BaseQueryProcessor
from mindlm.core.query_processing.decomposer import QueryDecomposer
from mindlm.core.query_processing.expander import QueryExpander
from mindlm.core.query_processing.hyde import HyDEProcessor
from mindlm.core.query_processing.multi_query import MultiQueryProcessor
from mindlm.core.query_processing.rewriter import QueryRewriter
from mindlm.core.query_processing.step_back import StepBackProcessor

_PROCESSOR_REGISTRY: list[
    tuple[str, Callable[[QueryProcessingConfig], BaseQueryProcessor]]
] = [
    ("rewriting", lambda _: QueryRewriter()),
    ("expansion", lambda _: QueryExpander()),
    ("hyde", lambda _: HyDEProcessor()),
    ("multi_query", lambda cfg: MultiQueryProcessor(cfg.multi_query)),
    ("decomposition", lambda cfg: QueryDecomposer(cfg.decomposition)),
    ("step_back", lambda _: StepBackProcessor()),
]


class QueryProcessorDispatcher:
    def __init__(self, config: QueryProcessingConfig) -> None:
        self._processors: list[BaseQueryProcessor] = [
            factory(config)
            for attr, factory in _PROCESSOR_REGISTRY
            if getattr(config, attr).enabled
        ]

    @observe(name="query-process")
    def process(self, query: str, llm: LLMProvider) -> list[str]:
        langfuse_context.update_current_observation(
            input=query,
            metadata={"processors": [type(p).__name__ for p in self._processors]},
        )
        seen: set[str] = {query}
        result: list[str] = [query]
        for processor in self._processors:
            for q in processor.process(query, llm):
                if q not in seen:
                    seen.add(q)
                    result.append(q)
        langfuse_context.update_current_observation(output=result)
        return result
