import pytest

from mindlm.core.config.models import (
    ChunkingConfig,
    MultiQueryConfig,
    QueryDecompositionConfig,
    QueryProcessingConfig,
)


class TestChunkingConfigValidator:
    def test_recursive_strategy_is_valid(self) -> None:
        config = ChunkingConfig(strategy="recursive", chunk_size=100, overlap=0)

        assert config.strategy == "recursive"

    def test_recursive_strategy_requires_nonempty_separators(self) -> None:
        with pytest.raises(ValueError, match="separators must not be empty"):
            ChunkingConfig(
                strategy="recursive", chunk_size=100, overlap=0, separators=[]
            )

    def test_parent_chunk_size_must_be_greater_than_chunk_size(self) -> None:
        with pytest.raises(
            ValueError, match="parent_chunk_size must be greater than chunk_size"
        ):
            ChunkingConfig(
                strategy="fixed", chunk_size=500, overlap=0, parent_chunk_size=100
            )

    def test_parent_chunk_size_equal_to_chunk_size_raises(self) -> None:
        with pytest.raises(
            ValueError, match="parent_chunk_size must be greater than chunk_size"
        ):
            ChunkingConfig(
                strategy="fixed", chunk_size=500, overlap=0, parent_chunk_size=500
            )

    def test_parent_chunk_size_none_is_valid(self) -> None:
        config = ChunkingConfig(
            strategy="fixed", chunk_size=500, overlap=0, parent_chunk_size=None
        )

        assert config.parent_chunk_size is None

    def test_parent_chunk_size_greater_than_chunk_size_is_valid(self) -> None:
        config = ChunkingConfig(
            strategy="fixed", chunk_size=500, overlap=0, parent_chunk_size=1000
        )

        assert config.parent_chunk_size == 1000


class TestQueryProcessingConfigDefaults:
    def test_query_processing_defaults_all_disabled(self) -> None:
        config = QueryProcessingConfig()

        assert config.rewriting.enabled is False
        assert config.expansion.enabled is False
        assert config.hyde.enabled is False
        assert config.multi_query.enabled is False
        assert config.decomposition.enabled is False
        assert config.step_back.enabled is False

    def test_multi_query_num_variants_minimum_2(self) -> None:
        with pytest.raises(ValueError, match="num_variants"):
            MultiQueryConfig(enabled=True, num_variants=1)

    def test_multi_query_num_variants_maximum_10(self) -> None:
        with pytest.raises(ValueError, match="num_variants"):
            MultiQueryConfig(enabled=True, num_variants=11)

    def test_multi_query_num_variants_boundary_values_are_valid(self) -> None:
        low = MultiQueryConfig(enabled=True, num_variants=2)
        high = MultiQueryConfig(enabled=True, num_variants=10)

        assert low.num_variants == 2
        assert high.num_variants == 10

    def test_decomposition_max_subqueries_minimum_2(self) -> None:
        with pytest.raises(ValueError, match="max_subqueries"):
            QueryDecompositionConfig(enabled=True, max_subqueries=1)

    def test_decomposition_max_subqueries_boundary_values_are_valid(self) -> None:
        config = QueryDecompositionConfig(enabled=True, max_subqueries=2)

        assert config.max_subqueries == 2
