from mindlm.core.config.models import QueryDecompositionConfig
from mindlm.core.generation.base import LLMProvider
from mindlm.core.query_processing._parsing import parse_numbered_list
from mindlm.core.query_processing.base import BaseQueryProcessor

_PROMPT = (
    "Break the question below into at most {max_subqueries} simpler, independent "
    "sub-questions that together cover the full original question. "
    "If the question is already simple, return it as a single sub-question.\n\n"
    "Output ONLY the sub-questions, one per line, numbered:\n"
    "1. <sub-question>\n"
    "2. <sub-question>\n"
    "...\n\n"
    "Question: {query}"
)


class QueryDecomposer(BaseQueryProcessor):
    def __init__(self, config: QueryDecompositionConfig) -> None:
        self._max_subqueries = config.max_subqueries

    def process(self, query: str, llm: LLMProvider) -> list[str]:
        prompt = _PROMPT.format(max_subqueries=self._max_subqueries, query=query)
        response = llm.chat([{"role": "user", "content": prompt}]).strip()
        if not response:
            return [query]
        subqueries = parse_numbered_list(response)[: self._max_subqueries]
        return subqueries if subqueries else [query]
