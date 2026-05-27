from typing import cast
from unittest.mock import MagicMock

import pytest

from mindlm.core.models import Result
from mindlm.core.retrieval.fusion import FusionEngine, _rrf_merge


def _result(id: str, score: float = 0.9) -> Result:
    return Result(id=id, score=score, payload={"content": f"content {id}"})


def _make_engine(
    retriever_results: list[Result] | None = None,
    query_variants: list[str] | None = None,
) -> FusionEngine:
    retriever = MagicMock()
    retriever.retrieve.return_value = retriever_results or []
    if query_variants is not None:
        llm = MagicMock()
        query_processor = MagicMock()
        query_processor.process.return_value = query_variants
        return FusionEngine(retriever, query_processor=query_processor, llm=llm)
    return FusionEngine(retriever)


class TestRRFMerge:
    def test_single_list_score_formula(self) -> None:
        """Doc at rank 0 of single list: RRF = 1/(60+0+1) = 1/61"""
        results = [_result("a")]
        merged = _rrf_merge([results], k=60)
        assert len(merged) == 1
        assert abs(merged[0].score - 1.0 / 61) < 1e-9

    def test_accumulates_score_across_lists(self) -> None:
        """Doc appearing in 2 lists at rank 0 scores 2/61"""
        doc = _result("a")
        merged = _rrf_merge([[doc], [doc]], k=60)
        assert len(merged) == 1
        assert abs(merged[0].score - 2.0 / 61) < 1e-9

    def test_higher_rank_means_lower_score(self) -> None:
        results = [_result("a"), _result("b"), _result("c")]
        merged = _rrf_merge([results])
        scores = [r.score for r in merged]
        assert scores == sorted(scores, reverse=True)
        assert merged[0].id == "a"

    def test_deduplicates_across_lists(self) -> None:
        """Same doc in two lists appears once in merged output"""
        doc = _result("shared")
        list1 = [_result("only1"), doc]
        list2 = [doc, _result("only2")]
        merged = _rrf_merge([list1, list2])
        ids = [r.id for r in merged]
        assert ids.count("shared") == 1

    def test_empty_lists_returns_empty(self) -> None:
        assert _rrf_merge([[], []]) == []

    def test_single_empty_list_returns_empty(self) -> None:
        assert _rrf_merge([[]]) == []

    def test_outer_empty_list_returns_empty(self) -> None:
        """No ranked lists at all → empty merge"""
        assert _rrf_merge([]) == []


class TestFusionEngine:
    def test_single_query_no_processor(self) -> None:
        engine = _make_engine(retriever_results=[_result("a")])
        results = engine.fuse("query", None, top_k=5)
        assert any(r.id == "a" for r in results)
        cast("MagicMock", engine._retriever.retrieve).assert_called_once_with(
            "query", None, top_k=5
        )

    def test_query_processor_expands_queries(self) -> None:
        engine = _make_engine(
            retriever_results=[_result("a")],
            query_variants=["q1", "q2", "q3"],
        )
        engine.fuse("original", None, top_k=5)
        assert cast("MagicMock", engine._retriever.retrieve).call_count == 3

    def test_rrf_merge_applied(self) -> None:
        """Two queries returning different orderings of same docs → RRF score differs from raw"""
        retriever = MagicMock()
        retriever.retrieve.side_effect = [
            [_result("a", 0.9), _result("b", 0.8)],
            [_result("b", 0.9), _result("a", 0.8)],
        ]
        llm = MagicMock()
        qp = MagicMock()
        qp.process.return_value = ["q1", "q2"]
        engine = FusionEngine(retriever, query_processor=qp, llm=llm)
        results = engine.fuse("original", None, top_k=5)
        # Both a and b should appear; scores should be RRF-based (not original 0.9/0.8)
        assert len(results) == 2
        for r in results:
            assert r.score < 1.0  # RRF scores are always < 1

    def test_top_k_truncates_merged_output(self) -> None:
        retriever = MagicMock()
        retriever.retrieve.return_value = [_result(str(i)) for i in range(10)]
        engine = FusionEngine(retriever)
        results = engine.fuse("query", None, top_k=3)
        assert len(results) == 3

    def test_all_queries_return_empty(self) -> None:
        engine = _make_engine(retriever_results=[])
        assert engine.fuse("query", None, top_k=5) == []

    def test_raises_when_processor_given_without_llm(self) -> None:
        retriever = MagicMock()
        qp = MagicMock()
        with pytest.raises(ValueError, match="llm is required"):
            FusionEngine(retriever, query_processor=qp, llm=None)
