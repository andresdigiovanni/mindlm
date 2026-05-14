from unittest.mock import MagicMock

from mindlm.core.query_processing.rewriter import QueryRewriter


class TestQueryRewriter:
    def test_calls_llm_with_query_in_prompt(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "reformulated query"
        rewriter = QueryRewriter()

        rewriter.process("my question", llm)

        call_args = llm.chat.call_args[0][0]
        assert "my question" in call_args[0]["content"]

    def test_returns_list_with_llm_response(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "reformulated query"
        rewriter = QueryRewriter()

        result = rewriter.process("my question", llm)

        assert result == ["reformulated query"]

    def test_strips_whitespace_from_response(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "  trimmed response  "
        rewriter = QueryRewriter()

        result = rewriter.process("query", llm)

        assert result == ["trimmed response"]

    def test_falls_back_to_original_when_llm_returns_empty(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = ""
        rewriter = QueryRewriter()

        result = rewriter.process("original", llm)

        assert result == ["original"]
