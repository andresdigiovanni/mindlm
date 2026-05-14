from unittest.mock import MagicMock

from mindlm.core.query_processing.step_back import StepBackProcessor


class TestStepBackProcessor:
    def test_calls_llm_with_query_in_prompt(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "abstract version"
        processor = StepBackProcessor()

        processor.process("specific question", llm)

        call_args = llm.chat.call_args[0][0]
        assert "specific question" in call_args[0]["content"]

    def test_returns_list_with_llm_response(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "abstract version"
        processor = StepBackProcessor()

        result = processor.process("specific question", llm)

        assert result == ["abstract version"]

    def test_strips_whitespace_from_response(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "  abstract version  "
        processor = StepBackProcessor()

        result = processor.process("query", llm)

        assert result == ["abstract version"]

    def test_falls_back_to_original_when_llm_returns_empty(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = ""
        processor = StepBackProcessor()

        result = processor.process("original", llm)

        assert result == ["original"]
