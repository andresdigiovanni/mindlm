from unittest.mock import MagicMock

import pytest

from mindlm.core.models import Result
from mindlm.core.reranking.llm_reranker import LLMReranker


def _result(id: str, score: float, content: str = "test content") -> Result:
    return Result(id=id, score=score, payload={"content": content})


def _make_reranker(responses: list[str]) -> tuple[LLMReranker, MagicMock]:
    llm = MagicMock()
    llm.chat.side_effect = responses
    return LLMReranker(llm), llm


class TestLLMReranker:
    def test_empty_results_returns_empty(self) -> None:
        reranker, _ = _make_reranker([])

        output = reranker.rerank("query", [])

        assert output == []

    def test_results_sorted_by_llm_score(self) -> None:
        reranker, _ = _make_reranker(["3", "8", "6"])
        results = [_result("a", 0.9), _result("b", 0.8), _result("c", 0.7)]

        output = reranker.rerank("query", results)

        assert [r.id for r in output] == ["b", "c", "a"]

    def test_scores_replaced_with_llm_scores(self) -> None:
        reranker, _ = _make_reranker(["7"])
        results = [_result("x", 0.5)]

        output = reranker.rerank("query", results)

        assert output[0].score == 7.0

    def test_parse_failure_keeps_original_score(self) -> None:
        reranker, _ = _make_reranker(["not a number"])
        results = [_result("x", 0.42)]

        output = reranker.rerank("query", results)

        assert output[0].score == 0.42

    def test_llm_error_keeps_original_score(self) -> None:
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("LLM unavailable")
        reranker = LLMReranker(llm)
        results = [_result("x", 0.55)]

        output = reranker.rerank("query", results)

        assert output[0].score == 0.55

    def test_parse_score_plain_integer(self) -> None:
        assert LLMReranker._parse_score("7") == 7.0

    def test_parse_score_embedded_number(self) -> None:
        assert LLMReranker._parse_score("The relevance is 8 out of 10") == 8.0

    def test_parse_score_boundary_one(self) -> None:
        assert LLMReranker._parse_score("1") == 1.0

    def test_parse_score_boundary_ten(self) -> None:
        assert LLMReranker._parse_score("10") == 10.0

    def test_parse_score_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse"):
            LLMReranker._parse_score("0")

    def test_all_results_scored(self) -> None:
        reranker, llm = _make_reranker(["5", "6", "7"])
        results = [_result("a", 0.1), _result("b", 0.2), _result("c", 0.3)]

        reranker.rerank("query", results)

        assert llm.chat.call_count == 3


class TestParseScore:
    def test_parses_ten(self) -> None:
        assert LLMReranker._parse_score("Score: 10") == 10.0

    def test_invalid_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse"):
            LLMReranker._parse_score("no numbers at all")
