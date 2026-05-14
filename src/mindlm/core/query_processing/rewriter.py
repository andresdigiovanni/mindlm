from mindlm.core.generation.base import LLMProvider
from mindlm.core.query_processing.base import BaseQueryProcessor

_PROMPT = (
    "You are a search query optimizer. Reformulate the query below to be clearer "
    "and better suited for semantic document retrieval. Preserve the original intent exactly.\n\n"
    "Output ONLY the reformulated query — no explanation, no prefix, no quotation marks.\n\n"
    "Query: {query}"
)


class QueryRewriter(BaseQueryProcessor):
    def process(self, query: str, llm: LLMProvider) -> list[str]:
        response = llm.chat(
            [{"role": "user", "content": _PROMPT.format(query=query)}]
        ).strip()
        return [response] if response else [query]
