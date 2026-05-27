from typing import Literal

from mindlm.core.config.models import ChunkingConfig
from mindlm.core.models import Result
from mindlm.core.retrieval.context_resolver import ContextResolver


def _result(id: str = "1", **payload_extra: object) -> Result:
    return Result(
        id=id,
        score=0.9,
        payload={"content": "original", "source": "doc.pdf", **payload_extra},
    )


def _resolver(
    *,
    parent_chunk_size: int | None = None,
    strategy: Literal[
        "fixed", "semantic", "sliding", "recursive", "sentence_window"
    ] = "fixed",
) -> ContextResolver:
    config = ChunkingConfig(strategy=strategy, parent_chunk_size=parent_chunk_size)
    return ContextResolver(config)


class TestContextResolverParent:
    def test_replaces_content_with_parent_content(self) -> None:
        resolver = _resolver(parent_chunk_size=1000)
        results = [_result(parent_content="parent text")]
        output = resolver.resolve(results)
        assert output[0].payload["content"] == "parent text"

    def test_leaves_content_unchanged_when_no_parent_content(self) -> None:
        resolver = _resolver(parent_chunk_size=1000)
        results = [_result()]
        output = resolver.resolve(results)
        assert output[0].payload["content"] == "original"

    def test_preserves_other_payload_fields(self) -> None:
        resolver = _resolver(parent_chunk_size=1000)
        results = [_result(parent_content="parent text")]
        output = resolver.resolve(results)
        assert output[0].payload["source"] == "doc.pdf"

    def test_empty_results_returns_empty(self) -> None:
        resolver = _resolver(parent_chunk_size=1000)
        assert resolver.resolve([]) == []


class TestContextResolverWindow:
    def test_replaces_content_with_window_context(self) -> None:
        resolver = _resolver(strategy="sentence_window")
        results = [_result(window_context="window text")]
        output = resolver.resolve(results)
        assert output[0].payload["content"] == "window text"

    def test_leaves_content_unchanged_when_no_window_context(self) -> None:
        resolver = _resolver(strategy="sentence_window")
        results = [_result()]
        output = resolver.resolve(results)
        assert output[0].payload["content"] == "original"

    def test_preserves_other_payload_fields(self) -> None:
        resolver = _resolver(strategy="sentence_window")
        results = [_result(window_context="window text")]
        output = resolver.resolve(results)
        assert output[0].payload["source"] == "doc.pdf"

    def test_empty_results_returns_empty(self) -> None:
        resolver = _resolver(strategy="sentence_window")
        assert resolver.resolve([]) == []


class TestContextResolverPassthrough:
    def test_no_resolution_returns_results_unchanged(self) -> None:
        resolver = _resolver()
        results = [_result("a"), _result("b")]
        assert resolver.resolve(results) == results
