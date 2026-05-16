from unittest.mock import MagicMock, patch

import pytest

from mindlm.core.config.models import RerankingConfig
from mindlm.core.models import Result
from mindlm.core.reranking.dispatcher import RerankerDispatcher
from mindlm.core.reranking.mmr import MMRReranker


def _result(score: float) -> Result:
    return Result(id="1", score=score, payload={"content": "test"})


class TestRerankerDispatcher:
    def test_disabled_returns_unchanged(self) -> None:
        config = RerankingConfig(enabled=False)
        dispatcher = RerankerDispatcher(config)
        results = [_result(0.9), _result(0.5)]

        output = dispatcher.rerank("query", results)

        assert output is results

    def test_cross_encoder_sorts_by_score(self) -> None:
        config = RerankingConfig(
            enabled=True, method="cross_encoder", model="test-model"
        )
        with patch("mindlm.core.reranking.cross_encoder.CrossEncoder") as MockCE:
            import numpy as np

            mock_ce_instance = MagicMock()
            mock_ce_instance.predict.return_value = np.array([0.2, 0.9, 0.5])
            MockCE.return_value = mock_ce_instance

            dispatcher = RerankerDispatcher(config)
            results = [_result(0.1), _result(0.1), _result(0.1)]

            output = dispatcher.rerank("query", results)

            assert output[0].score == 0.9

    def test_llm_method_requires_llm_provider(self) -> None:
        config = RerankingConfig(enabled=True, method="llm")
        with pytest.raises(ValueError, match="LLMProvider required"):
            RerankerDispatcher(config)

    def test_compression_method_requires_llm_provider(self) -> None:
        config = RerankingConfig(enabled=True, method="compression")
        with pytest.raises(ValueError, match="LLMProvider required"):
            RerankerDispatcher(config)

    def test_llm_method_reranks_by_llm_score(self) -> None:
        llm = MagicMock()
        llm.chat.side_effect = ["3", "9", "5"]
        config = RerankingConfig(enabled=True, method="llm")
        dispatcher = RerankerDispatcher(config, llm=llm)
        results = [
            Result(id="a", score=0.9, payload={"content": "text a"}),
            Result(id="b", score=0.8, payload={"content": "text b"}),
            Result(id="c", score=0.7, payload={"content": "text c"}),
        ]

        output = dispatcher.rerank("query", results)

        assert output[0].id == "b"  # score 9 is highest

    def test_compression_method_drops_empty_results(self) -> None:
        llm = MagicMock()
        llm.chat.side_effect = ["relevant content", ""]
        config = RerankingConfig(enabled=True, method="compression")
        dispatcher = RerankerDispatcher(config, llm=llm)
        results = [
            Result(id="a", score=0.9, payload={"content": "text a"}),
            Result(id="b", score=0.8, payload={"content": "text b"}),
        ]

        output = dispatcher.rerank("query", results)

        assert len(output) == 1
        assert output[0].id == "a"


class TestMMRReranker:
    def test_mmr_reranking_selects_diverse_results(self) -> None:
        provider = MagicMock()
        provider.embed_one.return_value = [1.0, 0.0]
        provider.embed.return_value = [
            [1.0, 0.0],  # result 1: identical to query — high relevance
            [1.0, 0.0],  # result 2: same as result 1 — redundant
            [0.0, 1.0],  # result 3: different — novel
        ]
        config = RerankingConfig(enabled=True, method="mmr", model=None)
        reranker = MMRReranker(config, provider)
        results = [_result(0.9), _result(0.8), _result(0.7)]

        output = reranker.rerank("query", results)

        assert len(output) == 3

    def test_mmr_empty_results_returns_empty(self) -> None:
        provider = MagicMock()
        config = RerankingConfig(enabled=True, method="mmr")
        reranker = MMRReranker(config, provider)

        output = reranker.rerank("query", [])

        assert output == []

    def test_mmr_single_result_returns_it(self) -> None:
        provider = MagicMock()
        provider.embed_one.return_value = [1.0, 0.0]
        provider.embed.return_value = [[1.0, 0.0]]
        config = RerankingConfig(enabled=True, method="mmr", model=None)
        reranker = MMRReranker(config, provider)
        results = [_result(0.9)]

        output = reranker.rerank("query", results)

        assert len(output) == 1
        assert output[0] is results[0]
