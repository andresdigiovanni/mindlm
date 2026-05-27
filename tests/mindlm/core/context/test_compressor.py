from unittest.mock import MagicMock

from mindlm.core.context.compressor import ContextualCompressor
from mindlm.core.models import Result


def _result(id: str = "1", content: str = "original content") -> Result:
    return Result(id=id, score=0.9, payload={"content": content, "source": "doc.pdf"})


def _make_compressor(responses: list[str]) -> tuple[ContextualCompressor, MagicMock]:
    llm = MagicMock()
    llm.chat.side_effect = responses
    return ContextualCompressor(llm), llm


class TestContextualCompressor:
    def test_empty_results_returns_empty(self) -> None:
        compressor, _ = _make_compressor([])
        assert compressor.compress("query", []) == []

    def test_relevant_content_is_replaced(self) -> None:
        compressor, _ = _make_compressor(["compressed text"])
        results = [_result(content="original long content")]
        output = compressor.compress("query", results)
        assert output[0].payload["content"] == "compressed text"

    def test_original_score_preserved(self) -> None:
        compressor, _ = _make_compressor(["some content"])
        results = [Result(id="1", score=0.77, payload={"content": "text"})]
        output = compressor.compress("query", results)
        assert output[0].score == 0.77

    def test_empty_response_drops_result(self) -> None:
        compressor, _ = _make_compressor([""])
        assert compressor.compress("query", [_result()]) == []

    def test_whitespace_only_response_drops_result(self) -> None:
        compressor, _ = _make_compressor(["   "])
        assert compressor.compress("query", [_result()]) == []

    def test_partial_compression_keeps_non_empty(self) -> None:
        compressor, _ = _make_compressor(["relevant part", "", "also relevant"])
        results = [_result("a"), _result("b"), _result("c")]
        output = compressor.compress("query", results)
        assert len(output) == 2
        assert output[0].id == "a"
        assert output[1].id == "c"

    def test_all_chunks_irrelevant_returns_empty(self) -> None:
        compressor, _ = _make_compressor(["", ""])
        assert compressor.compress("query", [_result("a"), _result("b")]) == []

    def test_llm_called_once_per_result(self) -> None:
        compressor, llm = _make_compressor(["a", "b", "c"])
        compressor.compress("query", [_result("a"), _result("b"), _result("c")])
        assert llm.chat.call_count == 3

    def test_non_content_payload_keys_preserved(self) -> None:
        compressor, _ = _make_compressor(["compressed"])
        output = compressor.compress("query", [_result()])
        assert output[0].payload["source"] == "doc.pdf"

    def test_llm_runtime_error_falls_back_to_original_content(self) -> None:
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("LLM down")
        compressor = ContextualCompressor(llm)
        results = [Result(id="1", score=0.9, payload={"content": "original text"})]
        output = compressor.compress("query", results)
        assert len(output) == 1
        assert output[0].payload["content"] == "original text"

    def test_llm_os_error_falls_back_to_original_content(self) -> None:
        llm = MagicMock()
        llm.chat.side_effect = OSError("connection refused")
        compressor = ContextualCompressor(llm)
        results = [Result(id="1", score=0.9, payload={"content": "original text"})]
        output = compressor.compress("query", results)
        assert len(output) == 1
        assert output[0].payload["content"] == "original text"

    def test_brace_placeholders_in_content_are_safe(self) -> None:
        """Regression: f-string construction must not raise KeyError for {placeholder} in content."""
        compressor, _ = _make_compressor(["safe output"])
        results = [_result(content="text with {placeholder} and {0} and {key}")]
        output = compressor.compress("query about {topic}", results)
        assert output[0].payload["content"] == "safe output"
