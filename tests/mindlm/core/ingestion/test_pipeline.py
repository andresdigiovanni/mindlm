from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mindlm.core.config.models import (
    ChunkingConfig,
    EmbeddingsConfig,
    IngestionConfig,
    LLMConfig,
    RAGConfig,
    RerankingConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from mindlm.core.ingestion.pipeline import IngestionPipeline
from mindlm.core.models import Chunk


def _make_rag_config(
    strategy: str = "vector", parent_chunk_size: int | None = None
) -> RAGConfig:
    return RAGConfig(
        llm=LLMConfig(provider="ollama", model="llama3", base_url="http://localhost"),
        embeddings=EmbeddingsConfig(provider="huggingface", model="test", dimensions=4),
        vector_store=VectorStoreConfig(
            provider="qdrant",
            mode="local",
            host="localhost",
            port=6333,
            collection="docs",
        ),
        ingestion=IngestionConfig(
            source_type=["markdown"], parsing_strategy="raw", deduplication=True
        ),
        chunking=ChunkingConfig(
            strategy="fixed",
            chunk_size=100,
            overlap=0,
            parent_chunk_size=parent_chunk_size,
        ),
        retrieval=RetrievalConfig(strategy=strategy, top_k=5),
        reranking=RerankingConfig(enabled=False),
    )


def _make_pipeline(
    config: RAGConfig | None = None,
) -> tuple[IngestionPipeline, MagicMock, MagicMock, MagicMock, MagicMock]:
    cfg = config or _make_rag_config()
    parser = MagicMock()
    chunker = MagicMock()
    embedding_provider = MagicMock()
    vectorstore = MagicMock()

    parser.parse.return_value = "some text content"
    chunker.chunk.return_value = [
        Chunk(text="chunk 1", index=0, metadata={}),
        Chunk(text="chunk 2", index=1, metadata={}),
    ]
    embedding_provider.embed.return_value = [[0.1] * 4, [0.2] * 4]

    pipeline = IngestionPipeline(cfg, parser, chunker, embedding_provider, vectorstore)
    return pipeline, parser, chunker, embedding_provider, vectorstore


class TestIngestionPipeline:
    def test_ingest_returns_chunk_count(self, tmp_path: Path) -> None:
        pipeline, _parser, _chunker, _ep, vs = _make_pipeline()
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        result = pipeline.ingest(doc)

        assert result == 2
        vs.upsert.assert_called_once()

    def test_ingest_empty_chunks_returns_zero(self, tmp_path: Path) -> None:
        pipeline, parser, chunker, _ep, vs = _make_pipeline()
        parser.parse.return_value = "text"
        chunker.chunk.return_value = []
        doc = tmp_path / "doc.md"
        doc.write_text("", encoding="utf-8")

        result = pipeline.ingest(doc)

        assert result == 0
        vs.upsert.assert_not_called()

    def test_payload_has_required_fields(self, tmp_path: Path) -> None:
        pipeline, _parser, _chunker, _ep, vs = _make_pipeline()
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        pipeline.ingest(doc)

        call_args = vs.upsert.call_args
        points = call_args[0][0]
        payload = points[0].payload
        assert "source" in payload
        assert "document_hash" in payload
        assert "chunk_index" in payload
        assert "total_chunks" in payload
        assert "ingested_at" in payload
        assert "modified_at" in payload
        assert "file_size" in payload

    def test_document_hash_is_sha256_prefixed(self, tmp_path: Path) -> None:
        pipeline, _parser, _chunker, _ep, vs = _make_pipeline()
        doc = tmp_path / "doc.md"
        doc.write_text("hello", encoding="utf-8")

        pipeline.ingest(doc)

        call_args = vs.upsert.call_args
        points = call_args[0][0]
        assert points[0].payload["document_hash"].startswith("sha256:")

    def test_ingest_hybrid_adds_sparse_vectors(self, tmp_path: Path) -> None:
        config = _make_rag_config(strategy="hybrid")
        pipeline, _parser, _chunker, _ep, vs = _make_pipeline(config)
        doc = tmp_path / "doc.md"
        doc.write_text("hello world", encoding="utf-8")

        mock_bm25 = MagicMock()
        mock_sparse_result = MagicMock()
        mock_sparse_result.indices.tolist.return_value = [0, 1]
        mock_sparse_result.values.tolist.return_value = [0.5, 0.3]
        mock_bm25.passage_embed.return_value = [mock_sparse_result]
        pipeline._bm25 = mock_bm25

        pipeline.ingest(doc)

        call_args = vs.upsert.call_args
        points = call_args[0][0]
        assert points[0].sparse_vector is not None

    def test_ingest_stores_correct_chunk_indices(self, tmp_path: Path) -> None:
        pipeline, _parser, _chunker, _ep, vs = _make_pipeline()
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        pipeline.ingest(doc)

        call_args = vs.upsert.call_args
        points = call_args[0][0]
        assert points[0].payload["chunk_index"] == 0
        assert points[1].payload["chunk_index"] == 1
        assert points[0].payload["total_chunks"] == 2

    @pytest.mark.parametrize("n_chunks", [1, 5, 10])
    def test_ingest_returns_exact_chunk_count(
        self, tmp_path: Path, n_chunks: int
    ) -> None:
        pipeline, _parser, chunker, ep, _vs = _make_pipeline()
        chunker.chunk.return_value = [
            Chunk(text=f"chunk {i}", index=i, metadata={}) for i in range(n_chunks)
        ]
        ep.embed.return_value = [[0.1] * 4] * n_chunks
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        result = pipeline.ingest(doc)

        assert result == n_chunks


class TestIngestionPipelineParentDoc:
    def _make_parent_pipeline(
        self,
        parent_chunk_size: int = 500,
        strategy: str = "vector",
    ) -> tuple[IngestionPipeline, MagicMock, MagicMock, MagicMock, MagicMock]:
        cfg = _make_rag_config(strategy=strategy, parent_chunk_size=parent_chunk_size)
        parser = MagicMock()
        chunker = MagicMock()
        embedding_provider = MagicMock()
        vectorstore = MagicMock()

        parser.parse.return_value = "parent text content here for testing"
        # child chunker returns 2 children per call
        chunker.chunk.return_value = [
            Chunk(text="child 1", index=0, metadata={}),
            Chunk(text="child 2", index=1, metadata={}),
        ]
        embedding_provider.embed.return_value = [[0.1] * 4, [0.2] * 4]

        pipeline = IngestionPipeline(
            cfg, parser, chunker, embedding_provider, vectorstore
        )
        return pipeline, parser, chunker, embedding_provider, vectorstore

    def test_normal_mode_when_parent_chunk_size_none(self, tmp_path: Path) -> None:
        # When parent_chunk_size is None, regular ingest path runs (chunker is called)
        cfg = _make_rag_config(strategy="vector", parent_chunk_size=None)
        parser = MagicMock()
        chunker = MagicMock()
        ep = MagicMock()
        vs = MagicMock()
        parser.parse.return_value = "text"
        chunker.chunk.return_value = [Chunk(text="chunk", index=0, metadata={})]
        ep.embed.return_value = [[0.1] * 4]
        pipeline = IngestionPipeline(cfg, parser, chunker, ep, vs)
        doc = tmp_path / "doc.md"
        doc.write_text("text", encoding="utf-8")

        pipeline.ingest(doc)

        chunker.chunk.assert_called_once_with("text")

    def test_includes_parent_content_in_payload(self, tmp_path: Path) -> None:
        pipeline, _parser, _chunker, _ep, vs = self._make_parent_pipeline(
            parent_chunk_size=500
        )
        doc = tmp_path / "doc.md"
        # Write enough content that a 500-char parent chunk contains it
        doc.write_text("x" * 30, encoding="utf-8")

        pipeline.ingest(doc)

        call_args = vs.upsert.call_args
        points = call_args[0][0]
        assert "parent_content" in points[0].payload

    def test_includes_parent_id_in_payload(self, tmp_path: Path) -> None:
        pipeline, _parser, _chunker, _ep, vs = self._make_parent_pipeline(
            parent_chunk_size=500
        )
        doc = tmp_path / "doc.md"
        doc.write_text("x" * 30, encoding="utf-8")

        pipeline.ingest(doc)

        call_args = vs.upsert.call_args
        points = call_args[0][0]
        assert "parent_id" in points[0].payload

    def test_children_of_same_parent_share_parent_id(self, tmp_path: Path) -> None:
        pipeline, _parser, _chunker, _ep, vs = self._make_parent_pipeline(
            parent_chunk_size=500
        )
        doc = tmp_path / "doc.md"
        # All text fits in a single parent chunk
        doc.write_text("x" * 30, encoding="utf-8")

        pipeline.ingest(doc)

        call_args = vs.upsert.call_args
        points = call_args[0][0]
        # Two children from one parent share the same parent_id
        assert points[0].payload["parent_id"] == points[1].payload["parent_id"]

    def test_children_of_different_parents_have_different_parent_ids(
        self, tmp_path: Path
    ) -> None:
        # Use parent_chunk_size=10 so two parents are created from 25-char text
        pipeline, parser, _chunker, _ep, vs = self._make_parent_pipeline(
            parent_chunk_size=200
        )
        # Make parser return text long enough for 2 parents of 200 chars each
        parser.parse.return_value = "a" * 400
        doc = tmp_path / "doc.md"
        doc.write_text("a" * 400, encoding="utf-8")

        pipeline.ingest(doc)

        call_args = vs.upsert.call_args
        points = call_args[0][0]
        # First 2 children are from parent 1, last 2 from parent 2
        assert points[0].payload["parent_id"] != points[2].payload["parent_id"]

    def test_only_child_points_upserted(self, tmp_path: Path) -> None:
        pipeline, _parser, _chunker, _ep, vs = self._make_parent_pipeline(
            parent_chunk_size=500
        )
        doc = tmp_path / "doc.md"
        doc.write_text("x" * 30, encoding="utf-8")

        pipeline.ingest(doc)

        vs.upsert.assert_called_once()
        call_args = vs.upsert.call_args
        points = call_args[0][0]
        # Only child points (2 children) are upserted, not parent points
        assert len(points) == 2

    def test_returns_child_count(self, tmp_path: Path) -> None:
        pipeline, _parser, _chunker, _ep, _vs = self._make_parent_pipeline(
            parent_chunk_size=500
        )
        doc = tmp_path / "doc.md"
        doc.write_text("x" * 30, encoding="utf-8")

        result = pipeline.ingest(doc)

        assert result == 2

    def test_returns_zero_for_empty_text(self, tmp_path: Path) -> None:
        pipeline, parser, _chunker, _ep, vs = self._make_parent_pipeline(
            parent_chunk_size=500
        )
        parser.parse.return_value = ""
        doc = tmp_path / "doc.md"
        doc.write_text("", encoding="utf-8")

        result = pipeline.ingest(doc)

        assert result == 0
        vs.upsert.assert_not_called()
