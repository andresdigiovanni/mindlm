from dataclasses import replace

from langfuse.decorators import observe

from mindlm.core.generation.base import LLMProvider
from mindlm.core.models import Result


class ContextualCompressor:
    """LLM-based content extraction: removes irrelevant parts from each result.

    Uses LLM to extract only the query-relevant portions of each result's content.
    Drops results where LLM returns empty string (all irrelevant).
    Falls back to original content if LLM fails.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    @observe(name="contextual-compress")
    def compress(self, query: str, results: list[Result]) -> list[Result]:
        """Compress each result's content via LLM.

        Args:
            query: User query for context
            results: Retrieved results

        Returns:
            Results with compressed content, filtered to remove empty results.
            Scores and all payload fields (except content) are preserved.
        """
        if not results:
            return results
        compressed: list[Result] = []
        for r in results:
            content = r.payload.get("content", "")
            prompt = (
                "Given the following query and document, extract only the parts of the "
                "document that are directly relevant to answering the query.\n"
                "Return the extracted text verbatim. If no part of the document is "
                "relevant, return an empty string. Do not add any explanation.\n\n"
                f"Query: {query}\n\nDocument: {content}\n\nRelevant parts:"
            )
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
