import pytest
from pydantic import ValidationError

from mindlm.core.config.models import (
    ChunkingConfig,
    ContextualRetrievalConfig,
    MultiQueryConfig,
    ObservabilityConfig,
    QueryDecompositionConfig,
    QueryProcessingConfig,
    RAGConfig,
)


class TestContextualRetrievalConfig:
    def test_should_have_both_flags_disabled_by_default(self) -> None:
        config = ContextualRetrievalConfig()

        assert config.chunk_context_enabled is False
        assert config.document_summary_enabled is False

    def test_should_have_non_empty_prompt_templates_by_default(self) -> None:
        config = ContextualRetrievalConfig()

        assert config.prompt_template
        assert config.document_summary_prompt_template

    def test_should_allow_enabling_chunk_context_independently(self) -> None:
        config = ContextualRetrievalConfig(chunk_context_enabled=True)

        assert config.chunk_context_enabled is True
        assert config.document_summary_enabled is False

    def test_should_allow_enabling_document_summary_independently(self) -> None:
        config = ContextualRetrievalConfig(document_summary_enabled=True)

        assert config.chunk_context_enabled is False
        assert config.document_summary_enabled is True


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

    def test_sliding_strategy_raises_when_overlap_equals_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            ChunkingConfig(strategy="sliding", chunk_size=100, overlap=100)

    def test_sliding_strategy_raises_when_overlap_exceeds_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            ChunkingConfig(strategy="sliding", chunk_size=100, overlap=150)

    def test_sliding_strategy_valid_when_overlap_less_than_chunk_size(self) -> None:
        config = ChunkingConfig(strategy="sliding", chunk_size=100, overlap=50)

        assert config.overlap == 50

    def test_non_sliding_strategy_allows_overlap_equal_to_chunk_size(self) -> None:
        # For non-sliding strategies, overlap == chunk_size is allowed at config level
        config = ChunkingConfig(strategy="fixed", chunk_size=100, overlap=100)

        assert config.overlap == 100

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


class TestObservabilityConfig:
    def test_should_have_default_values_when_instantiated_without_arguments(
        self,
    ) -> None:
        cfg = ObservabilityConfig()
        assert cfg.public_key == "pk-lf-local-dev"
        assert cfg.secret_key == "sk-lf-local-dev"  # noqa: S105
        assert cfg.host == "http://langfuse:3000"
        assert cfg.flush_at == 15
        assert cfg.flush_interval == 0.5

    def test_should_raise_when_flush_at_is_zero(self) -> None:
        with pytest.raises(ValidationError):
            ObservabilityConfig(flush_at=0)

    def test_should_raise_when_flush_interval_is_zero(self) -> None:
        with pytest.raises(ValidationError):
            ObservabilityConfig(flush_interval=0.0)

    def test_should_raise_when_flush_at_is_negative(self) -> None:
        with pytest.raises(ValidationError):
            ObservabilityConfig(flush_at=-1)

    def test_should_include_observability_field_when_rag_config_is_instantiated(
        self,
    ) -> None:
        assert isinstance(RAGConfig().observability, ObservabilityConfig)


class TestRAGConfigSemanticModelValidator:
    def _make_semantic_rag_config(
        self, embeddings_model: str, semantic_model: str
    ) -> RAGConfig:
        from mindlm.core.config.models import (
            ChunkingConfig,
            EmbeddingsConfig,
        )

        return RAGConfig(
            embeddings=EmbeddingsConfig(
                provider="huggingface", model=embeddings_model, dimensions=384
            ),
            chunking=ChunkingConfig(
                strategy="semantic",
                chunk_size=500,
                overlap=50,
                semantic_model=semantic_model,
            ),
        )

    def test_should_raise_when_semantic_model_differs_from_embeddings_model(
        self,
    ) -> None:
        with pytest.raises(ValidationError, match="semantic_model"):
            self._make_semantic_rag_config(
                embeddings_model="model-a", semantic_model="model-b"
            )

    def test_should_be_valid_when_semantic_model_matches_embeddings_model(
        self,
    ) -> None:
        config = self._make_semantic_rag_config(
            embeddings_model="BAAI/bge-large-en-v1.5",
            semantic_model="BAAI/bge-large-en-v1.5",
        )

        assert config.chunking.semantic_model == config.embeddings.model

    def test_should_not_validate_when_strategy_is_not_semantic(self) -> None:
        from mindlm.core.config.models import ChunkingConfig, EmbeddingsConfig

        config = RAGConfig(
            embeddings=EmbeddingsConfig(
                provider="huggingface", model="model-a", dimensions=384
            ),
            chunking=ChunkingConfig(strategy="fixed", chunk_size=500, overlap=50),
        )

        assert config.chunking.strategy == "fixed"
