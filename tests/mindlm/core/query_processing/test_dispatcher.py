from unittest.mock import MagicMock

from mindlm.core.config.models import (
    HyDEConfig,
    MultiQueryConfig,
    QueryDecompositionConfig,
    QueryExpansionConfig,
    QueryProcessingConfig,
    QueryRewritingConfig,
    StepBackConfig,
)
from mindlm.core.query_processing.dispatcher import QueryProcessorDispatcher


def _make_dispatcher(**kwargs: bool) -> QueryProcessorDispatcher:
    config = QueryProcessingConfig(
        rewriting=QueryRewritingConfig(enabled=kwargs.get("rewriting", False)),
        expansion=QueryExpansionConfig(enabled=kwargs.get("expansion", False)),
        hyde=HyDEConfig(enabled=kwargs.get("hyde", False)),
        multi_query=MultiQueryConfig(enabled=kwargs.get("multi_query", False)),
        decomposition=QueryDecompositionConfig(
            enabled=kwargs.get("decomposition", False)
        ),
        step_back=StepBackConfig(enabled=kwargs.get("step_back", False)),
    )
    return QueryProcessorDispatcher(config)


class TestQueryProcessorDispatcher:
    def test_returns_original_query_when_no_processors_enabled(self) -> None:
        dispatcher = _make_dispatcher()
        llm = MagicMock()

        result = dispatcher.process("original", llm)

        assert result == ["original"]

    def test_always_includes_original_query(self) -> None:
        dispatcher = _make_dispatcher(rewriting=True)
        llm = MagicMock()
        llm.chat.return_value = "rewritten"

        result = dispatcher.process("original", llm)

        assert "original" in result

    def test_deduplicates_when_processor_returns_original(self) -> None:
        dispatcher = _make_dispatcher(rewriting=True)
        llm = MagicMock()
        # Processor returns the same string as original
        llm.chat.return_value = "original"

        result = dispatcher.process("original", llm)

        assert result.count("original") == 1

    def test_combines_multiple_processors(self) -> None:
        dispatcher = _make_dispatcher(rewriting=True, expansion=True)
        llm = MagicMock()
        llm.chat.side_effect = ["rewritten", "expanded"]

        result = dispatcher.process("original", llm)

        assert "rewritten" in result
        assert "expanded" in result

    def test_original_query_is_first(self) -> None:
        dispatcher = _make_dispatcher(rewriting=True)
        llm = MagicMock()
        llm.chat.return_value = "rewritten"

        result = dispatcher.process("original", llm)

        assert result[0] == "original"

    def test_deduplicates_across_processors(self) -> None:
        dispatcher = _make_dispatcher(rewriting=True, expansion=True)
        llm = MagicMock()
        # Both processors return the same result
        llm.chat.side_effect = ["same result", "same result"]

        result = dispatcher.process("original", llm)

        assert result.count("same result") == 1
