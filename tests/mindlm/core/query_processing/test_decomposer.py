from unittest.mock import MagicMock

from mindlm.core.config.models import QueryDecompositionConfig
from mindlm.core.query_processing.decomposer import QueryDecomposer


def _decomposer(max_subqueries: int = 4) -> QueryDecomposer:
    return QueryDecomposer(
        QueryDecompositionConfig(enabled=True, max_subqueries=max_subqueries)
    )


class TestQueryDecomposer:
    def test_calls_llm_with_query_in_prompt(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "1. sub one\n2. sub two"
        decomposer = _decomposer()

        decomposer.process("complex question", llm)

        call_args = llm.chat.call_args[0][0]
        assert "complex question" in call_args[0]["content"]

    def test_returns_list_with_llm_response(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "1. sub one\n2. sub two"
        decomposer = _decomposer()

        result = decomposer.process("question", llm)

        assert "sub one" in result
        assert "sub two" in result

    def test_strips_whitespace_and_prefix_from_response(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "  1.  padded sub  \n  2.  another  "
        decomposer = _decomposer()

        result = decomposer.process("q", llm)

        assert result == ["padded sub", "another"]

    def test_falls_back_to_original_when_llm_returns_empty(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = ""
        decomposer = _decomposer()

        result = decomposer.process("original", llm)

        assert result == ["original"]

    def test_parses_numbered_list(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "1. first sub\n2. second sub\n3. third sub"
        decomposer = _decomposer()

        result = decomposer.process("q", llm)

        assert result == ["first sub", "second sub", "third sub"]

    def test_truncates_to_max_subqueries(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "1. a\n2. b\n3. c\n4. d\n5. e"
        decomposer = _decomposer(max_subqueries=3)

        result = decomposer.process("q", llm)

        assert len(result) == 3

    def test_uses_max_subqueries_in_prompt(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "1. sub"
        decomposer = _decomposer(max_subqueries=6)

        decomposer.process("q", llm)

        call_args = llm.chat.call_args[0][0]
        assert "6" in call_args[0]["content"]
