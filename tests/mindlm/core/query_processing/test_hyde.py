from unittest.mock import MagicMock

from mindlm.core.query_processing.hyde import HyDEProcessor


class TestHyDEProcessor:
    def test_calls_llm_with_query_in_prompt(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "hypothetical passage"
        processor = HyDEProcessor()

        processor.process("what is RAG?", llm)

        call_args = llm.chat.call_args[0][0]
        assert "what is RAG?" in call_args[0]["content"]

    def test_returns_list_with_llm_response(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "hypothetical passage"
        processor = HyDEProcessor()

        result = processor.process("what is RAG?", llm)

        assert result == ["hypothetical passage"]

    def test_strips_whitespace_from_response(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "  passage with spaces  "
        processor = HyDEProcessor()

        result = processor.process("query", llm)

        assert result == ["passage with spaces"]

    def test_falls_back_to_original_when_llm_returns_empty(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = ""
        processor = HyDEProcessor()

        result = processor.process("original", llm)

        assert result == ["original"]
