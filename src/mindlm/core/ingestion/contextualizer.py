from langfuse.decorators import observe

from mindlm.core.config.models import ContextualRetrievalConfig
from mindlm.core.generation.base import LLMProvider


class Contextualizer:
    def __init__(self, config: ContextualRetrievalConfig, llm: LLMProvider) -> None:
        self._config = config
        self._llm = llm

    @property
    def chunk_context_enabled(self) -> bool:
        return self._config.chunk_context_enabled

    @observe(name="contextualize")
    def contextualize(self, document_text: str, chunk_text: str) -> str:
        if not self._config.chunk_context_enabled:
            return ""
        # str.format required: prompt_template is user-configurable at runtime
        prompt = self._config.prompt_template.format(
            document=document_text, chunk=chunk_text
        )
        return self._llm.chat([{"role": "user", "content": prompt}]).strip()

    @observe(name="summarize")
    def summarize(self, document_text: str) -> str:
        if not self._config.document_summary_enabled:
            return ""
        # str.format required: document_summary_prompt_template is user-configurable
        prompt = self._config.document_summary_prompt_template.format(
            document=document_text
        )
        return self._llm.chat([{"role": "user", "content": prompt}]).strip()
