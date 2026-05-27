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
from mindlm.core.query_processing.planner import QueryPlan, QueryPlanner


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


def _make_dispatcher_with_planner(
    planner: QueryPlanner, **kwargs: bool
) -> QueryProcessorDispatcher:
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
    return QueryProcessorDispatcher(config, planner=planner)


def _make_planner(processors: list[str]) -> QueryPlanner:
    planner = MagicMock(spec=QueryPlanner)
    planner.plan.return_value = QueryPlan(processors=processors)
    return planner


class TestQueryProcessorDispatcherWithPlanner:
    def test_should_run_all_enabled_when_no_planner(self) -> None:
        dispatcher = _make_dispatcher(rewriting=True, expansion=True)
        llm = MagicMock()
        llm.chat.side_effect = ["rewritten", "expanded"]

        result = dispatcher.process("original", llm)

        assert "rewritten" in result
        assert "expanded" in result

    def test_should_run_only_planned_subset(self) -> None:
        planner = _make_planner(["rewriting"])
        dispatcher = _make_dispatcher_with_planner(
            planner, rewriting=True, expansion=True
        )
        llm = MagicMock()
        llm.chat.return_value = "rewritten"

        result = dispatcher.process("original", llm)

        assert "rewritten" in result
        # expansion should not have been called — only one chat call
        assert llm.chat.call_count == 1

    def test_should_return_only_original_when_planner_returns_empty_plan(
        self,
    ) -> None:
        planner = _make_planner([])
        dispatcher = _make_dispatcher_with_planner(
            planner, rewriting=True, expansion=True
        )
        llm = MagicMock()

        result = dispatcher.process("original", llm)

        assert result == ["original"]

    def test_should_ignore_unknown_names_in_plan(self) -> None:
        planner = _make_planner(["nonexistent"])
        dispatcher = _make_dispatcher_with_planner(planner, rewriting=True)
        llm = MagicMock()

        result = dispatcher.process("original", llm)

        assert result == ["original"]
        llm.chat.assert_not_called()

    def test_should_pass_enabled_processor_names_to_planner(self) -> None:
        planner = _make_planner(["rewriting"])
        dispatcher = _make_dispatcher_with_planner(
            planner, rewriting=True, expansion=True
        )
        llm = MagicMock()
        llm.chat.return_value = "rewritten"

        dispatcher.process("original", llm)

        planner.plan.assert_called_once()  # type: ignore[attr-defined]
        available = planner.plan.call_args.args[1]  # type: ignore[attr-defined]
        assert set(available) == {"rewriting", "expansion"}

    def test_should_not_raise_keyerror_when_plan_has_unknown_name(self) -> None:
        planner = _make_planner(["nonexistent", "rewriting"])
        dispatcher = _make_dispatcher_with_planner(planner, rewriting=True)
        llm = MagicMock()
        llm.chat.return_value = "rewritten"

        result = dispatcher.process("original", llm)

        assert "original" in result

    def test_should_record_configured_and_active_processors_in_langfuse(
        self,
    ) -> None:
        from unittest.mock import patch

        planner = _make_planner(["rewriting"])
        dispatcher = _make_dispatcher_with_planner(
            planner, rewriting=True, expansion=True
        )
        llm = MagicMock()
        llm.chat.return_value = "rewritten"

        with patch(
            "mindlm.core.query_processing.dispatcher.langfuse_context"
        ) as mock_langfuse:
            dispatcher.process("original", llm)

        first_call_metadata = mock_langfuse.update_current_observation.call_args_list[
            0
        ].kwargs["metadata"]
        assert set(first_call_metadata["configured_processors"]) == {
            "rewriting",
            "expansion",
        }
        assert first_call_metadata["active_processors"] == ["rewriting"]
