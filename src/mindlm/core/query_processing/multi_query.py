from mindlm.core.config.models import MultiQueryConfig
from mindlm.core.generation.base import LLMProvider
from mindlm.core.query_processing._parsing import parse_numbered_list
from mindlm.core.query_processing.base import BaseQueryProcessor

_PROMPT = (
    "Generate {num_variants} alternative phrasings of the question below. "
    "Each phrasing must express the same information need using different vocabulary or structure.\n\n"
    "Output ONLY the variants, one per line, numbered:\n"
    "1. <variant>\n"
    "2. <variant>\n"
    "...\n\n"
    "Question: {query}"
)


class MultiQueryProcessor(BaseQueryProcessor):
    def __init__(self, config: MultiQueryConfig) -> None:
        self._num_variants = config.num_variants

    def process(self, query: str, llm: LLMProvider) -> list[str]:
        prompt = _PROMPT.format(num_variants=self._num_variants, query=query)
        response = llm.chat([{"role": "user", "content": prompt}]).strip()
        if not response:
            return [query]
        variants = parse_numbered_list(response)
        return variants if variants else [query]
