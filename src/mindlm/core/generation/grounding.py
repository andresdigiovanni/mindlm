import dataclasses
import json
import logging

from langfuse.decorators import observe

from mindlm.core.generation.base import LLMProvider
from mindlm.core.models import Result

_logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class GroundingResult:
    is_grounded: bool
    refined_query: str | None


def _format_context(results: list[Result]) -> str:
    blocks = []
    for r in results:
        parts = []
        if summary := r.payload.get("document_summary"):
            parts.append(f"[Document context: {summary}]")
        if ctx := r.payload.get("chunk_context"):
            parts.append(f"[Chunk context: {ctx}]")
        parts.append(r.payload.get("content", ""))
        blocks.append(f"[Source: {r.payload.get('source', '')}]\n" + "\n".join(parts))
    return "\n\n".join(blocks)


def _build_grounding_prompt(question: str, answer: str, context: str) -> str:
    return (
        "You are an evaluation assistant. Given a question, an answer, and the "
        "context used to generate the answer, determine whether the answer is "
        "fully grounded in the provided context.\n\n"
        "Return ONLY a JSON object with two fields:\n"
        '- "grounded": true if the answer is supported by the context, false otherwise\n'
        '- "refined_query": if not grounded, a refined search query that would '
        "retrieve better context; otherwise null\n\n"
        'Example (grounded): {"grounded": true, "refined_query": null}\n'
        'Example (not grounded): {"grounded": false, "refined_query": "specific refined search"}\n\n'
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer: {answer}"
    )


class GroundingChecker:
    """LLM-based grounding check: assesses whether an answer is supported by retrieved context.

    Falls back to is_grounded=True on any error (conservative — avoids infinite loops).
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    @observe(name="grounding-check")
    def check(
        self, question: str, answer: str, results: list[Result]
    ) -> GroundingResult:
        """Check whether answer is grounded in the provided results.

        Args:
            question: The original user question.
            answer: The LLM-generated answer to evaluate.
            results: The retrieved context chunks used to generate the answer.

        Returns:
            GroundingResult with is_grounded=True and refined_query=None as fallback
            on any error (conservative — avoids infinite retrieval loops).
        """
        _fallback = GroundingResult(is_grounded=True, refined_query=None)

        if not results:
            return _fallback

        context = _format_context(results)
        prompt = _build_grounding_prompt(question, answer, context)

        try:
            raw = self._llm.chat(
                [{"role": "user", "content": prompt}], json_mode=True
            ).strip()
        except (RuntimeError, OSError):
            _logger.warning("GroundingChecker: LLM call failed; assuming grounded")
            return _fallback

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            _logger.warning(
                "GroundingChecker: response is not valid JSON; assuming grounded. "
                "Response: %s",
                raw[:200],
            )
            return _fallback

        if not isinstance(parsed, dict):
            _logger.warning(
                "GroundingChecker: expected JSON object, got %s; assuming grounded",
                type(parsed).__name__,
            )
            return _fallback

        grounded = bool(parsed.get("grounded", True))
        raw_refined: object = parsed.get("refined_query")
        refined_query: str | None = None
        if raw_refined is not None:
            candidate = str(raw_refined).strip()
            refined_query = candidate if candidate else None

        return GroundingResult(is_grounded=grounded, refined_query=refined_query)
