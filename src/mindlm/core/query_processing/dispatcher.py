from collections.abc import Callable

from langfuse.decorators import langfuse_context, observe

from mindlm.core.config.models import QueryProcessingConfig
from mindlm.core.generation.base import LLMProvider
from mindlm.core.query_processing.base import BaseQueryProcessor
from mindlm.core.query_processing.decomposer import QueryDecomposer
from mindlm.core.query_processing.expander import QueryExpander
from mindlm.core.query_processing.hyde import HyDEProcessor
from mindlm.core.query_processing.multi_query import MultiQueryProcessor
from mindlm.core.query_processing.planner import QueryPlanner
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
    def __init__(
        self,
        config: QueryProcessingConfig,
        planner: QueryPlanner | None = None,
    ) -> None:
        self._processors: dict[str, BaseQueryProcessor] = {
            attr: factory(config)
            for attr, factory in _PROCESSOR_REGISTRY
            if getattr(config, attr).enabled
        }
        self._planner = planner

    @observe(name="query-process")
    def process(self, query: str, llm: LLMProvider) -> list[str]:
        if self._planner is None:
            active = list(self._processors.values())
            active_names = list(self._processors.keys())
        else:
            plan = self._planner.plan(query, list(self._processors.keys()), llm)
            active = [
                self._processors[name]
                for name in plan.processors
                if name in self._processors
            ]
            active_names = [
                name for name in plan.processors if name in self._processors
            ]
        langfuse_context.update_current_observation(
            input=query,
            metadata={
                "configured_processors": list(self._processors.keys()),
                "active_processors": active_names,
            },
        )
        seen: set[str] = {query}
        result: list[str] = [query]
        for processor in active:
            for q in processor.process(query, llm):
                if q not in seen:
                    seen.add(q)
                    result.append(q)
        langfuse_context.update_current_observation(output=result)
        return result
