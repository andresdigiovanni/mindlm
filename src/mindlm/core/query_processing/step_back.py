from mindlm.core.generation.base import LLMProvider
from mindlm.core.query_processing.base import BaseQueryProcessor

_PROMPT = (
    "Generate a more abstract, general version of the question below that captures "
    "the underlying concept or principle being asked about.\n\n"
    "Output ONLY the step-back question — no explanation, no prefix, no quotation marks.\n\n"
    "Original question: {query}"
)


class StepBackProcessor(BaseQueryProcessor):
    def process(self, query: str, llm: LLMProvider) -> list[str]:
        response = llm.chat(
            [{"role": "user", "content": _PROMPT.format(query=query)}]
        ).strip()
        return [response] if response else [query]
