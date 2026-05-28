from unittest.mock import MagicMock

import pytest

from mindlm.core.exceptions import LLMUnavailableError
from mindlm.core.generation.grounding import GroundingChecker, GroundingResult
from mindlm.core.models import Result


def _result(*, id_: str = "1", content: str = "some context") -> Result:
    return Result(id=id_, score=0.9, payload={"content": content})


class TestGroundingChecker:
    def test_should_return_grounded_when_llm_returns_grounded_true(self) -> None:
        mock_llm = MagicMock()
        mock_llm.chat.return_value = '{"grounded": true, "refined_query": null}'
        checker = GroundingChecker(llm=mock_llm)

        result = checker.check(
            question="What is RAG?",
            answer="RAG stands for Retrieval-Augmented Generation.",
            results=[_result()],
        )

        assert result == GroundingResult(is_grounded=True, refined_query=None)

    def test_should_return_not_grounded_with_refined_query_when_llm_returns_grounded_false(
        self,
    ) -> None:
        mock_llm = MagicMock()
        mock_llm.chat.return_value = (
            '{"grounded": false, "refined_query": "better query"}'
        )
        checker = GroundingChecker(llm=mock_llm)

        result = checker.check(
            question="What is RAG?",
            answer="Some answer.",
            results=[_result()],
        )

        assert result == GroundingResult(
            is_grounded=False, refined_query="better query"
        )

    @pytest.mark.parametrize(
        "exc",
        [
            LLMUnavailableError("down"),
            RuntimeError("timeout"),
            OSError("socket error"),
        ],
    )
    def test_should_fallback_when_llm_raises(self, exc: Exception) -> None:
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = exc
        checker = GroundingChecker(llm=mock_llm)

        result = checker.check(
            question="What is RAG?",
            answer="Some answer.",
            results=[_result()],
        )

        assert result == GroundingResult(is_grounded=True, refined_query=None)

    def test_should_fallback_when_llm_returns_invalid_json(self) -> None:
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "not json at all"
        checker = GroundingChecker(llm=mock_llm)

        result = checker.check(
            question="What is RAG?",
            answer="Some answer.",
            results=[_result()],
        )

        assert result == GroundingResult(is_grounded=True, refined_query=None)

    def test_should_fallback_when_llm_returns_json_array(self) -> None:
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "[]"
        checker = GroundingChecker(llm=mock_llm)

        result = checker.check(
            question="What is RAG?",
            answer="Some answer.",
            results=[_result()],
        )

        assert result == GroundingResult(is_grounded=True, refined_query=None)

    def test_should_return_grounded_when_results_empty(self) -> None:
        mock_llm = MagicMock()
        checker = GroundingChecker(llm=mock_llm)

        result = checker.check(
            question="What is RAG?",
            answer="Some answer.",
            results=[],
        )

        assert result == GroundingResult(is_grounded=True, refined_query=None)
        mock_llm.chat.assert_not_called()

    def test_should_coerce_whitespace_refined_query_to_none(self) -> None:
        mock_llm = MagicMock()
        mock_llm.chat.return_value = '{"grounded": false, "refined_query": "   "}'
        checker = GroundingChecker(llm=mock_llm)

        result = checker.check(
            question="What is RAG?",
            answer="Some answer.",
            results=[_result()],
        )

        assert result == GroundingResult(is_grounded=False, refined_query=None)

    def test_should_strip_refined_query_whitespace(self) -> None:
        mock_llm = MagicMock()
        mock_llm.chat.return_value = (
            '{"grounded": false, "refined_query": "  refined  "}'
        )
        checker = GroundingChecker(llm=mock_llm)

        result = checker.check(
            question="What is RAG?",
            answer="Some answer.",
            results=[_result()],
        )

        assert result == GroundingResult(is_grounded=False, refined_query="refined")
