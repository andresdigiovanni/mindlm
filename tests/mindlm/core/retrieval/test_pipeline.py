from unittest.mock import MagicMock

from mindlm.core.config.models import RetrievalConfig
from mindlm.core.models import Result
from mindlm.core.retrieval.pipeline import RetrievalPipeline


def _result(id: str, score: float = 0.9) -> Result:
    return Result(id=id, score=score, payload={"content": f"content {id}"})


def _make_pipeline(
    fusion_results: list[Result] | None = None,
    resolver_results: list[Result] | None = None,
) -> tuple[RetrievalPipeline, MagicMock, MagicMock]:
    config = RetrievalConfig(top_k=5)
    fusion = MagicMock()
    fusion.fuse.return_value = fusion_results or [_result("a")]
    resolver = MagicMock()
    resolver.resolve.return_value = resolver_results or [_result("a")]
    pipeline = RetrievalPipeline(config, fusion, resolver)
    return pipeline, fusion, resolver


class TestRetrievalPipeline:
    def test_calls_fusion_with_query_filters_top_k(self) -> None:
        pipeline, fusion, _ = _make_pipeline()
        pipeline.retrieve("my query", {"field": "val"}, top_k=3)
        fusion.fuse.assert_called_once_with(
            "my query",
            {"field": "val"},
            3,
            per_query_top_k=3,
        )

    def test_passes_fusion_output_to_context_resolver(self) -> None:
        fusion_output = [_result("x")]
        pipeline, _, resolver = _make_pipeline(fusion_results=fusion_output)
        pipeline.retrieve("q")
        resolver.resolve.assert_called_once_with(fusion_output)

    def test_config_top_k_used_when_not_overridden(self) -> None:
        pipeline, fusion, _ = _make_pipeline()
        pipeline.retrieve("q")
        _, _, called_fused_top_k = fusion.fuse.call_args[0]
        called_per_query_top_k = fusion.fuse.call_args.kwargs["per_query_top_k"]
        assert called_fused_top_k == 5  # from RetrievalConfig(top_k=5)
        assert called_per_query_top_k == 5

    def test_top_k_override_passed_to_fusion(self) -> None:
        pipeline, fusion, _ = _make_pipeline()
        pipeline.retrieve("q", top_k=10)
        _, _, called_fused_top_k = fusion.fuse.call_args[0]
        called_per_query_top_k = fusion.fuse.call_args.kwargs["per_query_top_k"]
        assert called_fused_top_k == 10
        assert called_per_query_top_k == 10

    def test_per_query_top_k_config_is_used_when_set(self) -> None:
        config = RetrievalConfig(top_k=5, per_query_top_k=12)
        fusion = MagicMock()
        fusion.fuse.return_value = [_result("a")]
        resolver = MagicMock()
        resolver.resolve.return_value = [_result("a")]
        pipeline = RetrievalPipeline(config, fusion, resolver)

        pipeline.retrieve("q")

        _, _, called_fused_top_k = fusion.fuse.call_args[0]
        called_per_query_top_k = fusion.fuse.call_args.kwargs["per_query_top_k"]
        assert called_fused_top_k == 5
        assert called_per_query_top_k == 12
