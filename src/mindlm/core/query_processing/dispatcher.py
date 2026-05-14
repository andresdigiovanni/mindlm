from __future__ import annotations

from typing import TYPE_CHECKING

from mindlm.core.query_processing.decomposer import QueryDecomposer
from mindlm.core.query_processing.expander import QueryExpander
from mindlm.core.query_processing.hyde import HyDEProcessor
from mindlm.core.query_processing.multi_query import MultiQueryProcessor
from mindlm.core.query_processing.rewriter import QueryRewriter
from mindlm.core.query_processing.step_back import StepBackProcessor

if TYPE_CHECKING:
    from mindlm.core.config.models import QueryProcessingConfig
    from mindlm.core.generation.base import LLMProvider
    from mindlm.core.query_processing.base import BaseQueryProcessor


class QueryProcessorDispatcher:
    def __init__(self, config: QueryProcessingConfig) -> None:
        self._processors: list[BaseQueryProcessor] = []
        if config.rewriting.enabled:
            self._processors.append(QueryRewriter())
        if config.expansion.enabled:
            self._processors.append(QueryExpander())
        if config.hyde.enabled:
            self._processors.append(HyDEProcessor())
        if config.multi_query.enabled:
            self._processors.append(MultiQueryProcessor(config.multi_query))
        if config.decomposition.enabled:
            self._processors.append(QueryDecomposer(config.decomposition))
        if config.step_back.enabled:
            self._processors.append(StepBackProcessor())

    def process(self, query: str, llm: LLMProvider) -> list[str]:
        seen: set[str] = {query}
        result: list[str] = [query]
        for processor in self._processors:
            for q in processor.process(query, llm):
                if q not in seen:
                    seen.add(q)
                    result.append(q)
        return result
