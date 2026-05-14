from mindlm.core.generation.base import LLMProvider
from mindlm.core.query_processing.base import BaseQueryProcessor

_PROMPT = (
    "Write a short factual passage (2-4 sentences) that directly answers the question below. "
    "This passage will be used for document retrieval only and will not be shown to users. "
    "Write as if from an authoritative source. Do not express uncertainty.\n\n"
    "Output ONLY the passage — no preamble, no prefix.\n\n"
    "Question: {query}"
)


class HyDEProcessor(BaseQueryProcessor):
    def process(self, query: str, llm: LLMProvider) -> list[str]:
        response = llm.chat(
            [{"role": "user", "content": _PROMPT.format(query=query)}]
        ).strip()
        return [response] if response else [query]
