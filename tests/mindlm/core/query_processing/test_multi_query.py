from unittest.mock import MagicMock

from mindlm.core.config.models import MultiQueryConfig
from mindlm.core.query_processing.multi_query import MultiQueryProcessor


def _processor(num_variants: int = 3) -> MultiQueryProcessor:
    return MultiQueryProcessor(
        MultiQueryConfig(enabled=True, num_variants=num_variants)
    )


class TestMultiQueryProcessor:
    def test_calls_llm_with_query_in_prompt(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "1. variant one\n2. variant two\n3. variant three"
        processor = _processor()

        processor.process("my question", llm)

        call_args = llm.chat.call_args[0][0]
        assert "my question" in call_args[0]["content"]

    def test_returns_list_with_llm_response(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "1. variant one\n2. variant two"
        processor = _processor()

        result = processor.process("my question", llm)

        assert "variant one" in result
        assert "variant two" in result

    def test_strips_whitespace_and_prefix_from_response(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "  1. trimmed variant  \n  2. another  "
        processor = _processor()

        result = processor.process("query", llm)

        assert result == ["trimmed variant", "another"]

    def test_falls_back_to_original_when_llm_returns_empty(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = ""
        processor = _processor()

        result = processor.process("original", llm)

        assert result == ["original"]

    def test_parses_numbered_list(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "1. first\n2. second\n3. third"
        processor = _processor()

        result = processor.process("q", llm)

        assert result == ["first", "second", "third"]

    def test_uses_num_variants_in_prompt(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "1. v1\n2. v2\n3. v3\n4. v4\n5. v5"
        processor = _processor(num_variants=5)

        processor.process("q", llm)

        call_args = llm.chat.call_args[0][0]
        assert "5" in call_args[0]["content"]

    def test_handles_fewer_lines_than_num_variants(self) -> None:
        llm = MagicMock()
        # LLM returns only 2 variants even though 3 were requested
        llm.chat.return_value = "1. only one\n2. only two"
        processor = _processor(num_variants=3)

        result = processor.process("q", llm)

        assert len(result) == 2
