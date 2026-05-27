import dataclasses
import json
import logging

from mindlm.core.generation.base import LLMProvider

_logger = logging.getLogger(__name__)

_PROCESSOR_DESCRIPTIONS: dict[str, str] = {
    "rewriting": "Reformulates the query for clearer semantic retrieval without changing intent.",
    "expansion": "Adds related terms and synonyms to broaden the search.",
    "hyde": "Generates a hypothetical answer document to improve embedding-based retrieval.",
    "multi_query": "Produces multiple query variants from different angles to increase recall.",
    "decomposition": "Breaks a complex multi-part query into simpler focused sub-questions.",
    "step_back": "Abstracts the query to a higher-level principle for broader contextual retrieval.",
}


@dataclasses.dataclass
class QueryPlan:
    processors: list[str]


def _build_prompt(query: str, available: list[str]) -> str:
    descriptions = "\n".join(
        f"- {name}: {_PROCESSOR_DESCRIPTIONS.get(name, name)}" for name in available
    )
    return (
        "You are a search query optimizer. Given a user query, select which of the "
        "following query processors are useful to apply before retrieval.\n\n"
        "Available processors:\n"
        f"{descriptions}\n\n"
        "Return ONLY a JSON array of processor names to apply. "
        'Example: ["rewriting", "hyde"]\n'
        "If no processor is needed, return: []\n\n"
        f"Query: <query>{query}</query>"
    )


class QueryPlanner:
    def plan(self, query: str, available: list[str], llm: LLMProvider) -> QueryPlan:
        if not available:
            return QueryPlan(processors=[])
        prompt = _build_prompt(query, available)
        try:
            response = llm.chat([{"role": "user", "content": prompt}]).strip()
        except (
            Exception
        ):  # planner is best-effort; any failure must fall back gracefully
            _logger.warning(
                "QueryPlanner LLM call failed; falling back to all enabled processors"
            )
            return QueryPlan(processors=list(available))
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            _logger.warning(
                "QueryPlanner response is not valid JSON; falling back: %r", response
            )
            return QueryPlan(processors=list(available))
        if not isinstance(parsed, list):
            _logger.warning(
                "QueryPlanner response is not a JSON array; falling back: %r", parsed
            )
            return QueryPlan(processors=list(available))
        seen: set[str] = set()
        selected: list[str] = []
        for item in parsed:
            if isinstance(item, str) and item in available and item not in seen:
                seen.add(item)
                selected.append(item)
        return QueryPlan(processors=selected)
