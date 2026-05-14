from mindlm.core.generation.base import LLMProvider
from mindlm.core.query_processing.base import BaseQueryProcessor

_PROMPT = (
    "You are a search query optimizer. Rewrite the query below, incorporating relevant "
    "synonyms, related terms, and key concepts that improve document retrieval recall. "
    "Keep the rewritten query concise.\n\n"
    "Output ONLY the expanded query — no explanation, no prefix, no quotation marks.\n\n"
    "Query: {query}"
)


class QueryExpander(BaseQueryProcessor):
    def process(self, query: str, llm: LLMProvider) -> list[str]:
        response = llm.chat(
            [{"role": "user", "content": _PROMPT.format(query=query)}]
        ).strip()
        return [response] if response else [query]
