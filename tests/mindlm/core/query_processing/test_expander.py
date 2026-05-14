from unittest.mock import MagicMock

from mindlm.core.query_processing.expander import QueryExpander


class TestQueryExpander:
    def test_calls_llm_with_query_in_prompt(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "expanded query"
        expander = QueryExpander()

        expander.process("my question", llm)

        call_args = llm.chat.call_args[0][0]
        assert "my question" in call_args[0]["content"]

    def test_returns_list_with_llm_response(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "expanded query"
        expander = QueryExpander()

        result = expander.process("my question", llm)

        assert result == ["expanded query"]

    def test_strips_whitespace_from_response(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "  padded expansion  "
        expander = QueryExpander()

        result = expander.process("query", llm)

        assert result == ["padded expansion"]

    def test_falls_back_to_original_when_llm_returns_empty(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = ""
        expander = QueryExpander()

        result = expander.process("original", llm)

        assert result == ["original"]
