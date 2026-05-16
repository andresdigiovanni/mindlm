from dataclasses import replace

from langfuse.decorators import observe

from mindlm.core.generation.base import LLMProvider
from mindlm.core.models import Result
from mindlm.core.reranking.base import BaseReranker

_COMPRESS_PROMPT = (
    "Given the following query and document, extract only the parts of the document "
    "that are directly relevant to answering the query.\n"
    "Return the extracted text verbatim. If no part of the document is relevant, "
    "return an empty string. Do not add any explanation.\n\n"
    "Query: {query}\n\nDocument: {text}\n\nRelevant parts:"
)


class ContextualCompressor(BaseReranker):
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    @observe(name="contextual-compress")
    def rerank(self, query: str, results: list[Result]) -> list[Result]:
        if not results:
            return results
        compressed: list[Result] = []
        for r in results:
            content = r.payload.get("content", "")
            prompt = _COMPRESS_PROMPT.format(query=query, text=content)
            try:
                compressed_text = self._llm.chat(
                    [{"role": "user", "content": prompt}]
                ).strip()
            except (RuntimeError, OSError):
                compressed_text = content  # fallback: keep original
            if compressed_text:
                new_payload = {**r.payload, "content": compressed_text}
                compressed.append(replace(r, payload=new_payload))
        return compressed
