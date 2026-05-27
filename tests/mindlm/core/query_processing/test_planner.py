from unittest.mock import MagicMock

import pytest

from mindlm.core.exceptions import LLMUnavailableError
from mindlm.core.query_processing.planner import QueryPlan, QueryPlanner


def _planner() -> QueryPlanner:
    return QueryPlanner()


def _llm(return_value: str = "[]") -> MagicMock:
    llm = MagicMock()
    llm.chat.return_value = return_value
    return llm


class TestQueryPlannerPlan:
    def test_should_return_plan_instance(self) -> None:
        planner = _planner()
        result = planner.plan("query", ["rewriting"], _llm('["rewriting"]'))
        assert isinstance(result, QueryPlan)

    def test_should_return_all_available_when_llm_returns_full_list(self) -> None:
        planner = _planner()
        result = planner.plan(
            "query", ["rewriting", "hyde"], _llm('["rewriting", "hyde"]')
        )
        assert result.processors == ["rewriting", "hyde"]

    def test_should_return_subset_when_llm_returns_valid_subset(self) -> None:
        planner = _planner()
        result = planner.plan(
            "what is X", ["rewriting", "hyde", "expansion"], _llm('["rewriting"]')
        )
        assert result.processors == ["rewriting"]

    def test_should_filter_names_not_in_available(self) -> None:
        planner = _planner()
        result = planner.plan(
            "q", ["rewriting", "hyde"], _llm('["rewriting", "unknown_proc"]')
        )
        assert result.processors == ["rewriting"]

    @pytest.mark.parametrize(
        "exc",
        [
            LLMUnavailableError("service down"),
            LLMUnavailableError("timeout"),
        ],
    )
    def test_should_fallback_to_all_available_when_llm_raises(
        self, exc: LLMUnavailableError
    ) -> None:
        planner = _planner()
        llm = MagicMock()
        llm.chat.side_effect = exc
        result = planner.plan("q", ["rewriting", "hyde"], llm)
        assert set(result.processors) == {"rewriting", "hyde"}

    def test_should_fallback_when_llm_raises_generic_exception(self) -> None:
        planner = _planner()
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("boom")

        result = planner.plan("q", ["rewriting", "hyde"], llm)

        assert set(result.processors) == {"rewriting", "hyde"}

    def test_should_fallback_when_response_is_not_valid_json(self) -> None:
        planner = _planner()
        result = planner.plan("q", ["rewriting"], _llm("not json at all"))
        assert result.processors == ["rewriting"]

    def test_should_fallback_when_response_is_json_object_not_array(self) -> None:
        planner = _planner()
        result = planner.plan("q", ["rewriting"], _llm('{"processors": ["rewriting"]}'))
        assert result.processors == ["rewriting"]

    def test_should_return_empty_plan_without_llm_call_when_available_is_empty(
        self,
    ) -> None:
        planner = _planner()
        llm = MagicMock()
        result = planner.plan("q", [], llm)
        assert result.processors == []
        llm.chat.assert_not_called()

    def test_should_return_empty_list_when_llm_returns_empty_array(self) -> None:
        planner = _planner()
        result = planner.plan("q", ["rewriting"], _llm("[]"))
        assert result.processors == []

    def test_should_handle_unicode_query(self) -> None:
        planner = _planner()
        result = planner.plan(
            "¿Cómo funciona? 🔍", ["rewriting"], _llm('["rewriting"]')
        )
        assert result.processors == ["rewriting"]

    def test_should_deduplicate_when_llm_returns_duplicate_names(self) -> None:
        planner = _planner()
        result = planner.plan("q", ["rewriting"], _llm('["rewriting", "rewriting"]'))
        assert result.processors.count("rewriting") == 1

    def test_should_filter_non_string_items_in_json_array(self) -> None:
        planner = _planner()
        result = planner.plan("q", ["rewriting"], _llm('[1, "rewriting", null]'))
        assert result.processors == ["rewriting"]

    def test_should_filter_dict_items_in_json_array(self) -> None:
        planner = _planner()
        result = planner.plan(
            "q", ["rewriting"], _llm('[{"name": "rewriting"}, "rewriting"]')
        )
        assert result.processors == ["rewriting"]
