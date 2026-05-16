from unittest.mock import MagicMock

from mindlm.core.models import Result
from mindlm.core.reranking.compressor import ContextualCompressor


def _result(id: str = "1", content: str = "original content") -> Result:
    return Result(id=id, score=0.9, payload={"content": content, "source": "doc.pdf"})


def _make_compressor(responses: list[str]) -> tuple[ContextualCompressor, MagicMock]:
    llm = MagicMock()
    llm.chat.side_effect = responses
    return ContextualCompressor(llm), llm


class TestContextualCompressor:
    def test_empty_results_returns_empty(self) -> None:
        compressor, _ = _make_compressor([])

        output = compressor.rerank("query", [])

        assert output == []

    def test_relevant_content_is_replaced(self) -> None:
        compressor, _ = _make_compressor(["compressed text"])
        results = [_result(content="original long content")]

        output = compressor.rerank("query", results)

        assert output[0].payload["content"] == "compressed text"

    def test_original_score_preserved(self) -> None:
        compressor, _ = _make_compressor(["some content"])
        results = [Result(id="1", score=0.77, payload={"content": "text"})]

        output = compressor.rerank("query", results)

        assert output[0].score == 0.77

    def test_empty_response_drops_result(self) -> None:
        compressor, _ = _make_compressor([""])
        results = [_result()]

        output = compressor.rerank("query", results)

        assert output == []

    def test_whitespace_only_response_drops_result(self) -> None:
        compressor, _ = _make_compressor(["   "])
        results = [_result()]

        output = compressor.rerank("query", results)

        assert output == []

    def test_partial_compression_keeps_non_empty(self) -> None:
        compressor, _ = _make_compressor(["relevant part", "", "also relevant"])
        results = [_result("a"), _result("b"), _result("c")]

        output = compressor.rerank("query", results)

        assert len(output) == 2
        assert output[0].id == "a"
        assert output[1].id == "c"

    def test_all_chunks_irrelevant_returns_empty(self) -> None:
        compressor, _ = _make_compressor(["", ""])
        results = [_result("a"), _result("b")]

        output = compressor.rerank("query", results)

        assert output == []

    def test_llm_called_once_per_result(self) -> None:
        compressor, llm = _make_compressor(["a", "b", "c"])
        results = [_result("a"), _result("b"), _result("c")]

        compressor.rerank("query", results)

        assert llm.chat.call_count == 3

    def test_non_content_payload_keys_preserved(self) -> None:
        compressor, _ = _make_compressor(["compressed"])
        results = [_result()]  # _result adds "source": "doc.pdf"

        output = compressor.rerank("query", results)

        assert output[0].payload["source"] == "doc.pdf"

    def test_llm_error_falls_back_to_original_content(self) -> None:
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("LLM down")
        compressor = ContextualCompressor(llm)
        results = [Result(id="1", score=0.9, payload={"content": "original text"})]

        output = compressor.rerank("query", results)

        assert len(output) == 1
        assert output[0].payload["content"] == "original text"
