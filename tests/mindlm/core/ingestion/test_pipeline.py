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


def _make_rag_config(strategy: str = "vector") -> RAGConfig:
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
        chunking=ChunkingConfig(strategy="fixed", chunk_size=100, overlap=0),
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
