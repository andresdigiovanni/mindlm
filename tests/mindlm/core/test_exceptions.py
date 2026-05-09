import pytest

from mindlm.core.exceptions import (
    CollectionNotFoundError,
    EmbeddingError,
    LLMUnavailableError,
    ParseError,
)


class TestExceptions:
    def test_llm_unavailable_error_message(self) -> None:
        exc = LLMUnavailableError("Ollama down")
        assert "Ollama down" in str(exc)

    def test_parse_error_message(self) -> None:
        exc = ParseError("/path/doc.png", "raw")
        assert "/path/doc.png" in str(exc)
        assert "raw" in str(exc)

    def test_collection_not_found_error(self) -> None:
        exc = CollectionNotFoundError("my_collection")
        assert "my_collection" in str(exc)

    def test_embedding_error(self) -> None:
        exc = EmbeddingError("model-name")
        assert "model-name" in str(exc)

    def test_exceptions_are_runtime_errors(self) -> None:
        with pytest.raises(RuntimeError):
            raise LLMUnavailableError("test")
