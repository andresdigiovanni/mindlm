from unittest.mock import MagicMock

from mindlm.core.config.models import RetrievalConfig
from mindlm.core.models import Result
from mindlm.core.retrieval.pipeline import RetrievalPipeline


def _result(id: str, score: float = 0.9) -> Result:
    return Result(id=id, score=score, payload={"content": f"content {id}"})


def _make_pipeline(
    fusion_results: list[Result] | None = None,
    resolver_results: list[Result] | None = None,
    with_graph: bool = False,
) -> tuple[RetrievalPipeline, MagicMock, MagicMock, MagicMock | None]:
    config = RetrievalConfig(top_k=5)
    fusion = MagicMock()
    fusion.fuse.return_value = fusion_results or [_result("a")]
    resolver = MagicMock()
    resolver.resolve.return_value = resolver_results or [_result("a")]
    graph_augmenter: MagicMock | None = None
    if with_graph:
        graph_augmenter = MagicMock()
        graph_augmenter.augment.return_value = [_result("a")]
    pipeline = RetrievalPipeline(config, fusion, resolver, graph_augmenter)
    return pipeline, fusion, resolver, graph_augmenter


class TestRetrievalPipeline:
    def test_calls_fusion_with_query_filters_top_k(self) -> None:
        pipeline, fusion, _, _ = _make_pipeline()
        pipeline.retrieve("my query", {"field": "val"}, top_k=3)
        fusion.fuse.assert_called_once_with("my query", {"field": "val"}, 3)

    def test_passes_fusion_output_to_context_resolver(self) -> None:
        fusion_output = [_result("x")]
        pipeline, _, resolver, _ = _make_pipeline(fusion_results=fusion_output)
        pipeline.retrieve("q")
        resolver.resolve.assert_called_once_with(fusion_output)

    def test_calls_graph_augmenter_with_resolver_output(self) -> None:
        resolver_output = [_result("y")]
        pipeline, _, _, graph = _make_pipeline(
            resolver_results=resolver_output, with_graph=True
        )
        assert graph is not None
        pipeline.retrieve("q", top_k=7)
        graph.augment.assert_called_once_with(resolver_output, 7)

    def test_skips_graph_augmenter_when_none(self) -> None:
        pipeline, _, _, _ = _make_pipeline(with_graph=False)
        result = pipeline.retrieve("q")
        assert result is not None  # no error

    def test_config_top_k_used_when_not_overridden(self) -> None:
        pipeline, fusion, _, _ = _make_pipeline()
        pipeline.retrieve("q")
        _, _, called_top_k = fusion.fuse.call_args[0]
        assert called_top_k == 5  # from RetrievalConfig(top_k=5)

    def test_top_k_override_passed_to_fusion(self) -> None:
        pipeline, fusion, _, _ = _make_pipeline()
        pipeline.retrieve("q", top_k=10)
        _, _, called_top_k = fusion.fuse.call_args[0]
        assert called_top_k == 10
