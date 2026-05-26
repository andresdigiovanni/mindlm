import re
from dataclasses import replace

from langfuse.decorators import observe

from mindlm.core.generation.base import LLMProvider
from mindlm.core.models import Result
from mindlm.core.reranking.base import BaseReranker

_RELEVANCE_PROMPT = (
    "Score the relevance of the following document to the query on a scale from 1 to 10.\n"
    "Respond with only a single integer between 1 and 10. No explanation.\n\n"
    "Query: {query}\n\nDocument: {document}\n\nScore:"
)


class LLMReranker(BaseReranker):
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    @observe(name="llm-rerank")
    def rerank(self, query: str, results: list[Result]) -> list[Result]:
        if not results:
            return results
        scored: list[tuple[Result, float]] = []
        for r in results:
            scored.append((r, self._score(query, r)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [replace(r, score=s) for r, s in scored]

    def _score(self, query: str, result: Result) -> float:
        content = result.payload.get("content", "")
        prompt = _RELEVANCE_PROMPT.format(query=query, document=content)
        try:
            response = self._llm.chat([{"role": "user", "content": prompt}])
            raw = self._parse_score(response)  # 1-10
            return (raw - 1.0) / 9.0  # normalize to [0, 1]
        except (RuntimeError, OSError, ValueError):
            return result.score

    @staticmethod
    def _parse_score(text: str) -> float:
        m = re.search(r"\b(10|[1-9])\b", text)
        if m:
            return float(m.group(1))
        raise ValueError(f"Cannot parse relevance score from: {text!r}")
