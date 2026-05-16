from langfuse.decorators import observe

from mindlm.core.config.models import ContextualRetrievalConfig
from mindlm.core.generation.base import LLMProvider


class Contextualizer:
    def __init__(self, config: ContextualRetrievalConfig, llm: LLMProvider) -> None:
        self._config = config
        self._llm = llm

    @observe(name="contextualize")
    def contextualize(self, document_text: str, chunk_text: str) -> str:
        prompt = self._config.prompt_template.format(
            document=document_text, chunk=chunk_text
        )
        context = self._llm.chat([{"role": "user", "content": prompt}]).strip()
        if not context:
            return chunk_text
        return f"{context} {chunk_text}"
