from collections.abc import Mapping
from unittest.mock import MagicMock

import pytest

from mindlm.core.models import Point, Result
from mindlm.core.retrieval.graph_augmenter import GraphAugmenter


def _result(id: str, score: float = 0.8) -> Result:
    return Result(
        id=id, score=score, payload={"content": f"content {id}", "source": "doc.pdf"}
    )


def _point(id: str) -> Point:
    return Point(
        id=id,
        vector=[0.1, 0.2],
        payload={"content": f"content {id}", "source": "doc.pdf"},
    )


def _make_augmenter(
    related_ids: list[str], points: Mapping[str, Point | None]
) -> GraphAugmenter:
    graph_store = MagicMock()
    graph_store.get_related_chunk_ids.return_value = related_ids
    vectorstore = MagicMock()
    vectorstore.get_by_id.side_effect = lambda id_: points.get(id_)
    return GraphAugmenter(graph_store, vectorstore)


class TestGraphAugmenter:
    def test_adds_new_related_chunk(self) -> None:
        augmenter = _make_augmenter(["2"], {"2": _point("2")})
        results = [_result("1", score=0.8)]
        output = augmenter.augment(results, top_k=5)
        ids = [r.id for r in output]
        assert "2" in ids

    def test_skips_ids_already_in_results(self) -> None:
        augmenter = _make_augmenter(["1"], {"1": _point("1")})
        results = [_result("1", score=0.8)]
        output = augmenter.augment(results, top_k=5)
        assert len(output) == 1

    def test_expansion_score_is_half_of_min_score(self) -> None:
        augmenter = _make_augmenter(["2"], {"2": _point("2")})
        results = [_result("1", score=0.6)]
        output = augmenter.augment(results, top_k=5)
        new_result = next(r for r in output if r.id == "2")
        assert new_result.score == 0.3

    def test_respects_top_k_after_expansion(self) -> None:
        points = {str(i): _point(str(i)) for i in range(2, 10)}
        augmenter = _make_augmenter([str(i) for i in range(2, 10)], points)
        results = [_result("1", score=0.9)]
        output = augmenter.augment(results, top_k=3)
        assert len(output) == 3

    def test_missing_vectorstore_point_skipped(self) -> None:
        augmenter = _make_augmenter(["2"], {"2": None})
        results = [_result("1")]
        output = augmenter.augment(results, top_k=5)
        assert len(output) == 1
        assert output[0].id == "1"

    def test_empty_input_returns_empty(self) -> None:
        augmenter = _make_augmenter([], {})
        assert augmenter.augment([], top_k=5) == []

    def test_results_sorted_by_score_descending(self) -> None:
        augmenter = _make_augmenter(["2"], {"2": _point("2")})
        results = [_result("1", score=0.8)]
        output = augmenter.augment(results, top_k=5)
        scores = [r.score for r in output]
        assert scores == sorted(scores, reverse=True)

    def test_expansion_score_floor_when_all_scores_zero(self) -> None:
        """max(0.5*0, 1e-9) must floor to 1e-9, not zero"""
        augmenter = _make_augmenter(["2"], {"2": _point("2")})
        results = [_result("1", score=0.0)]
        output = augmenter.augment(results, top_k=5)
        new_result = next(r for r in output if r.id == "2")
        assert new_result.score == pytest.approx(1e-9)
