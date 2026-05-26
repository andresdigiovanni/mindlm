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
from mindlm.core.models import Chunk, ParsedDocument


def _make_rag_config(
    strategy: str = "vector",
    parent_chunk_size: int | None = None,
    allowed_base_dir: str = "/",
) -> RAGConfig:
    return RAGConfig(
        llm=LLMConfig(provider="ollama", model="gemma4", base_url="http://localhost"),
        embeddings=EmbeddingsConfig(provider="huggingface", model="test", dimensions=4),
        vector_store=VectorStoreConfig(
            provider="qdrant",
            mode="local",
            host="localhost",
            port=6333,
            collection="docs",
        ),
        ingestion=IngestionConfig(
            source_type=["markdown"],
            parsing_strategy="raw",
            deduplication=True,
            allowed_base_dir=allowed_base_dir,
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

    parser.parse.return_value = ParsedDocument(text="some text content", page_breaks=[])
    chunker.chunk.return_value = [
        Chunk(text="chunk 1", index=0, metadata={}, start_char=0, end_char=7),
        Chunk(text="chunk 2", index=1, metadata={}, start_char=7, end_char=14),
    ]
    embedding_provider.embed.return_value = [[0.1] * 4, [0.2] * 4]
    vectorstore.scroll.return_value = ([], None)  # no duplicate by default

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
        parser.parse.return_value = ParsedDocument(text="text", page_breaks=[])
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
            Chunk(
                text=f"chunk {i}",
                index=i,
                metadata={},
                start_char=i * 10,
                end_char=i * 10 + 7,
            )
            for i in range(n_chunks)
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

        parser.parse.return_value = ParsedDocument(
            text="parent text content here for testing", page_breaks=[]
        )
        # child chunker returns 2 children per call
        chunker.chunk.return_value = [
            Chunk(text="child 1", index=0, metadata={}, start_char=0, end_char=7),
            Chunk(text="child 2", index=1, metadata={}, start_char=7, end_char=14),
        ]
        embedding_provider.embed.return_value = [[0.1] * 4, [0.2] * 4]
        vectorstore.scroll.return_value = ([], None)  # no duplicate by default

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
        parser.parse.return_value = ParsedDocument(text="text", page_breaks=[])
        chunker.chunk.return_value = [
            Chunk(text="chunk", index=0, metadata={}, start_char=0, end_char=5)
        ]
        ep.embed.return_value = [[0.1] * 4]
        vs.scroll.return_value = ([], None)
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
        parser.parse.return_value = ParsedDocument(text="a" * 400, page_breaks=[])
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


class TestIngestionPipelineContextualizer:
    def _make_pipeline_with_contextualizer(
        self,
        contextualizer: MagicMock | None = None,
        chunks: list[Chunk] | None = None,
    ) -> tuple[IngestionPipeline, MagicMock, MagicMock, MagicMock, MagicMock]:
        cfg = _make_rag_config()
        parser = MagicMock()
        chunker = MagicMock()
        embedding_provider = MagicMock()
        vectorstore = MagicMock()

        parser.parse.return_value = ParsedDocument(
            text="some text content", page_breaks=[]
        )
        chunker.chunk.return_value = chunks or [
            Chunk(text="chunk 1", index=0, metadata={}, start_char=0, end_char=7),
            Chunk(text="chunk 2", index=1, metadata={}, start_char=7, end_char=14),
        ]
        embedding_provider.embed.return_value = [[0.1] * 4, [0.2] * 4]
        vectorstore.scroll.return_value = ([], None)

        pipeline = IngestionPipeline(
            cfg,
            parser,
            chunker,
            embedding_provider,
            vectorstore,
            contextualizer=contextualizer,
        )
        return pipeline, parser, chunker, embedding_provider, vectorstore

    def test_contextualizer_called_once_per_chunk(self, tmp_path: Path) -> None:
        ctx = MagicMock()
        ctx.contextualize.return_value = "Context sentence."
        ctx.summarize.return_value = "Doc summary."
        pipeline, _p, _c, _ep, _vs = self._make_pipeline_with_contextualizer(
            contextualizer=ctx
        )
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        pipeline.ingest(doc)

        assert ctx.contextualize.call_count == 2

    def test_raw_chunk_text_used_for_embedding(self, tmp_path: Path) -> None:
        ctx = MagicMock()
        ctx.contextualize.return_value = "Context sentence."
        ctx.summarize.return_value = "Doc summary."
        pipeline, _p, _c, ep, _vs = self._make_pipeline_with_contextualizer(
            contextualizer=ctx
        )
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        pipeline.ingest(doc)

        texts_embedded = ep.embed.call_args[0][0]
        assert texts_embedded == ["chunk 1", "chunk 2"]

    def test_contextualizer_none_does_not_change_chunks(self, tmp_path: Path) -> None:
        pipeline, _p, _c, ep, _vs = self._make_pipeline_with_contextualizer(
            contextualizer=None
        )
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        pipeline.ingest(doc)

        texts_embedded = ep.embed.call_args[0][0]
        assert texts_embedded == ["chunk 1", "chunk 2"]

    def test_window_context_written_to_payload(self, tmp_path: Path) -> None:
        window_chunks = [
            Chunk(
                text="sentence 1",
                index=0,
                metadata={"window_context": "sentence 1 sentence 2"},
                start_char=0,
                end_char=10,
            ),
            Chunk(
                text="sentence 2",
                index=1,
                metadata={"window_context": "sentence 1 sentence 2"},
                start_char=10,
                end_char=20,
            ),
        ]
        pipeline, _p, _c, _ep, vs = self._make_pipeline_with_contextualizer(
            chunks=window_chunks
        )
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        pipeline.ingest(doc)

        points = vs.upsert.call_args[0][0]
        assert "window_context" in points[0].payload
        assert points[0].payload["window_context"] == "sentence 1 sentence 2"

    def test_no_window_context_when_metadata_empty(self, tmp_path: Path) -> None:
        pipeline, _p, _c, _ep, vs = self._make_pipeline_with_contextualizer()
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        pipeline.ingest(doc)

        points = vs.upsert.call_args[0][0]
        assert "window_context" not in points[0].payload

    def test_returns_zero_for_empty_text(self, tmp_path: Path) -> None:
        pipeline, parser, chunker, _ep, vs = self._make_pipeline_with_contextualizer()
        parser.parse.return_value = ParsedDocument(text="", page_breaks=[])
        chunker.chunk.return_value = []
        doc = tmp_path / "doc.md"
        doc.write_text("", encoding="utf-8")

        result = pipeline.ingest(doc)

        assert result == 0
        vs.upsert.assert_not_called()

    def test_chunk_context_written_to_payload(self, tmp_path: Path) -> None:
        ctx = MagicMock()
        ctx.contextualize.return_value = "Context sentence."
        ctx.summarize.return_value = "Doc summary."
        pipeline, _p, _c, _ep, vs = self._make_pipeline_with_contextualizer(
            contextualizer=ctx
        )
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        pipeline.ingest(doc)

        points = vs.upsert.call_args[0][0]
        assert all(p.payload["chunk_context"] == "Context sentence." for p in points)

    def test_chunk_context_absent_when_contextualizer_none(
        self, tmp_path: Path
    ) -> None:
        pipeline, _p, _c, _ep, vs = self._make_pipeline_with_contextualizer(
            contextualizer=None
        )
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        pipeline.ingest(doc)

        points = vs.upsert.call_args[0][0]
        assert all("chunk_context" not in p.payload for p in points)

    def test_document_summary_written_to_all_chunk_payloads(
        self, tmp_path: Path
    ) -> None:
        ctx = MagicMock()
        ctx.contextualize.return_value = "Context sentence."
        ctx.summarize.return_value = "Doc summary."
        pipeline, _p, _c, _ep, vs = self._make_pipeline_with_contextualizer(
            contextualizer=ctx
        )
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        pipeline.ingest(doc)

        points = vs.upsert.call_args[0][0]
        assert all(p.payload["document_summary"] == "Doc summary." for p in points)

    def test_document_summary_absent_when_contextualizer_none(
        self, tmp_path: Path
    ) -> None:
        pipeline, _p, _c, _ep, vs = self._make_pipeline_with_contextualizer(
            contextualizer=None
        )
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        pipeline.ingest(doc)

        points = vs.upsert.call_args[0][0]
        assert all("document_summary" not in p.payload for p in points)

    def test_summarize_called_once_per_ingest(self, tmp_path: Path) -> None:
        ctx = MagicMock()
        ctx.contextualize.return_value = "Context sentence."
        ctx.summarize.return_value = "Doc summary."
        pipeline, _p, _c, _ep, _vs = self._make_pipeline_with_contextualizer(
            contextualizer=ctx
        )
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        pipeline.ingest(doc)

        assert ctx.summarize.call_count == 1

    def test_empty_context_not_written_to_payload(self, tmp_path: Path) -> None:
        ctx = MagicMock()
        ctx.contextualize.return_value = ""
        ctx.summarize.return_value = "Doc summary."
        pipeline, _p, _c, _ep, vs = self._make_pipeline_with_contextualizer(
            contextualizer=ctx
        )
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        pipeline.ingest(doc)

        points = vs.upsert.call_args[0][0]
        assert all("chunk_context" not in p.payload for p in points)

    def test_empty_document_summary_not_written_to_payload(
        self, tmp_path: Path
    ) -> None:
        ctx = MagicMock()
        ctx.contextualize.return_value = "Context sentence."
        ctx.summarize.return_value = ""
        pipeline, _p, _c, _ep, vs = self._make_pipeline_with_contextualizer(
            contextualizer=ctx
        )
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        pipeline.ingest(doc)

        points = vs.upsert.call_args[0][0]
        assert all("document_summary" not in p.payload for p in points)


class TestIngestionPipelineGraphExtraction:
    def _make_pipeline_with_graph(
        self,
        entity_extractor: MagicMock | None = None,
        graph_store: MagicMock | None = None,
    ) -> tuple[IngestionPipeline, MagicMock]:
        cfg = _make_rag_config()
        parser = MagicMock()
        chunker = MagicMock()
        embedding_provider = MagicMock()
        vectorstore = MagicMock()

        parser.parse.return_value = ParsedDocument(text="some text", page_breaks=[])
        chunker.chunk.return_value = [
            Chunk(text="chunk 1", index=0, metadata={}, start_char=0, end_char=7),
        ]
        embedding_provider.embed.return_value = [[0.1] * 4]
        vectorstore.scroll.return_value = ([], None)

        pipeline = IngestionPipeline(
            cfg,
            parser,
            chunker,
            embedding_provider,
            vectorstore,
            entity_extractor=entity_extractor,
            graph_store=graph_store,
        )
        return pipeline, vectorstore

    def test_ingest_without_extractor_does_not_call_graph_store(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        graph_store = MagicMock()
        pipeline, _vs = self._make_pipeline_with_graph(
            entity_extractor=None, graph_store=None
        )
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        # Act
        pipeline.ingest(doc)

        # Assert
        graph_store.upsert_entities.assert_not_called()

    def test_ingest_with_extractor_calls_upsert_entities(self, tmp_path: Path) -> None:
        # Arrange
        from mindlm.core.models import Entity

        entity_extractor = MagicMock()
        graph_store = MagicMock()
        mock_entity = Entity(
            id="e1", name="X", type="ORG", description="x", source_id="c1"
        )
        entity_extractor.extract.return_value = ([mock_entity], [])

        pipeline, _vs = self._make_pipeline_with_graph(
            entity_extractor=entity_extractor, graph_store=graph_store
        )
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        # Act
        pipeline.ingest(doc)

        # Assert
        graph_store.upsert_entities.assert_called_once()
        entities_arg = graph_store.upsert_entities.call_args[0][0]
        assert mock_entity in entities_arg

    def test_extractor_without_graph_store_raises(self) -> None:
        # Arrange
        cfg = _make_rag_config()
        entity_extractor = MagicMock()

        # Act / Assert
        with pytest.raises(ValueError, match="graph_store is required"):
            IngestionPipeline(
                cfg,
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                entity_extractor=entity_extractor,
                graph_store=None,
            )


class TestIngestionPipelineDeduplication:
    def _make_dedup_pipeline(
        self, deduplication: bool
    ) -> tuple[IngestionPipeline, MagicMock, MagicMock, MagicMock, MagicMock]:
        cfg = RAGConfig(
            llm=LLMConfig(
                provider="ollama", model="gemma4", base_url="http://localhost"
            ),
            embeddings=EmbeddingsConfig(
                provider="huggingface", model="test", dimensions=4
            ),
            vector_store=VectorStoreConfig(
                provider="qdrant",
                mode="local",
                host="localhost",
                port=6333,
                collection="docs",
            ),
            ingestion=IngestionConfig(
                source_type=["markdown"],
                parsing_strategy="raw",
                deduplication=deduplication,
                allowed_base_dir="/",
            ),
            chunking=ChunkingConfig(strategy="fixed", chunk_size=100, overlap=0),
            retrieval=RetrievalConfig(strategy="vector", top_k=5),
            reranking=RerankingConfig(enabled=False),
        )
        parser = MagicMock()
        chunker = MagicMock()
        ep = MagicMock()
        vs = MagicMock()
        parser.parse.return_value = ParsedDocument(text="text", page_breaks=[])
        chunker.chunk.return_value = [
            Chunk(text="chunk", index=0, metadata={}, start_char=0, end_char=5)
        ]
        ep.embed.return_value = [[0.1] * 4]
        pipeline = IngestionPipeline(cfg, parser, chunker, ep, vs)
        return pipeline, parser, chunker, ep, vs

    def test_should_skip_ingestion_when_duplicate_detected(
        self, tmp_path: Path
    ) -> None:
        pipeline, _parser, _chunker, _ep, vs = self._make_dedup_pipeline(
            deduplication=True
        )
        vs.scroll.return_value = ([MagicMock()], None)  # existing document found
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        result = pipeline.ingest(doc)

        assert result == 0
        vs.upsert.assert_not_called()

    def test_should_ingest_when_no_duplicate_exists(self, tmp_path: Path) -> None:
        pipeline, _parser, _chunker, _ep, vs = self._make_dedup_pipeline(
            deduplication=True
        )
        vs.scroll.return_value = ([], None)  # no existing document
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        result = pipeline.ingest(doc)

        assert result == 1
        vs.upsert.assert_called_once()

    def test_should_ingest_even_when_duplicate_exists_if_dedup_disabled(
        self, tmp_path: Path
    ) -> None:
        pipeline, _parser, _chunker, _ep, vs = self._make_dedup_pipeline(
            deduplication=False
        )
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        result = pipeline.ingest(doc)

        assert result == 1
        vs.scroll.assert_not_called()
        vs.upsert.assert_called_once()

    def test_should_scroll_with_document_hash_filter(self, tmp_path: Path) -> None:
        pipeline, _parser, _chunker, _ep, vs = self._make_dedup_pipeline(
            deduplication=True
        )
        vs.scroll.return_value = ([], None)
        doc = tmp_path / "doc.md"
        doc.write_text("hello", encoding="utf-8")

        pipeline.ingest(doc)

        call_args = vs.scroll.call_args
        filters = call_args.kwargs.get("filters") or call_args.args[0]
        assert "document_hash" in filters


class TestIngestionPipelineParentDocContextualizer:
    def _make_parent_pipeline_with_contextualizer(
        self,
        contextualizer: MagicMock | None = None,
        parent_chunk_size: int = 500,
    ) -> tuple[IngestionPipeline, MagicMock, MagicMock, MagicMock, MagicMock]:
        cfg = _make_rag_config(strategy="vector", parent_chunk_size=parent_chunk_size)
        parser = MagicMock()
        chunker = MagicMock()
        embedding_provider = MagicMock()
        vectorstore = MagicMock()

        parser.parse.return_value = ParsedDocument(
            text="parent text content here for testing", page_breaks=[]
        )
        chunker.chunk.return_value = [
            Chunk(text="child 1", index=0, metadata={}, start_char=0, end_char=7),
            Chunk(text="child 2", index=1, metadata={}, start_char=7, end_char=14),
        ]
        embedding_provider.embed.return_value = [[0.1] * 4, [0.2] * 4]
        vectorstore.scroll.return_value = ([], None)

        pipeline = IngestionPipeline(
            cfg,
            parser,
            chunker,
            embedding_provider,
            vectorstore,
            contextualizer=contextualizer,
        )
        return pipeline, parser, chunker, embedding_provider, vectorstore

    def test_parent_doc_chunk_context_written_to_payload(self, tmp_path: Path) -> None:
        ctx = MagicMock()
        ctx.contextualize.return_value = "Context sentence."
        ctx.summarize.return_value = "Doc summary."
        pipeline, _p, _c, _ep, vs = self._make_parent_pipeline_with_contextualizer(
            contextualizer=ctx
        )
        doc = tmp_path / "doc.md"
        doc.write_text("x" * 30, encoding="utf-8")

        pipeline.ingest(doc)

        points = vs.upsert.call_args[0][0]
        assert all("chunk_context" in p.payload for p in points)
        assert all(p.payload["chunk_context"] == "Context sentence." for p in points)

    def test_parent_doc_document_summary_written_to_all_children(
        self, tmp_path: Path
    ) -> None:
        ctx = MagicMock()
        ctx.contextualize.return_value = "Context sentence."
        ctx.summarize.return_value = "Doc summary."
        pipeline, _p, _c, _ep, vs = self._make_parent_pipeline_with_contextualizer(
            contextualizer=ctx
        )
        doc = tmp_path / "doc.md"
        doc.write_text("x" * 30, encoding="utf-8")

        pipeline.ingest(doc)

        points = vs.upsert.call_args[0][0]
        assert all(p.payload["document_summary"] == "Doc summary." for p in points)

    def test_parent_doc_summarize_called_once_not_per_parent_chunk(
        self, tmp_path: Path
    ) -> None:
        ctx = MagicMock()
        ctx.contextualize.return_value = "Context sentence."
        ctx.summarize.return_value = "Doc summary."
        pipeline, parser, _c, _ep, _vs = self._make_parent_pipeline_with_contextualizer(
            contextualizer=ctx, parent_chunk_size=200
        )
        # Two parent chunks worth of text
        parser.parse.return_value = ParsedDocument(text="a" * 400, page_breaks=[])
        doc = tmp_path / "doc.md"
        doc.write_text("a" * 400, encoding="utf-8")

        pipeline.ingest(doc)

        assert ctx.summarize.call_count == 1

    def test_parent_doc_contextualize_called_with_full_document_text(
        self, tmp_path: Path
    ) -> None:
        ctx = MagicMock()
        ctx.contextualize.return_value = "Context sentence."
        ctx.summarize.return_value = "Doc summary."
        full_text = "parent text content here for testing"
        pipeline, _p, _c, _ep, _vs = self._make_parent_pipeline_with_contextualizer(
            contextualizer=ctx
        )
        doc = tmp_path / "doc.md"
        doc.write_text("x" * 30, encoding="utf-8")

        pipeline.ingest(doc)

        for call in ctx.contextualize.call_args_list:
            assert call.args[0] == full_text

    def test_parent_doc_empty_document_summary_not_written_to_payload(
        self, tmp_path: Path
    ) -> None:
        ctx = MagicMock()
        ctx.contextualize.return_value = "Context sentence."
        ctx.summarize.return_value = ""
        pipeline, _p, _c, _ep, vs = self._make_parent_pipeline_with_contextualizer(
            contextualizer=ctx
        )
        doc = tmp_path / "doc.md"
        doc.write_text("x" * 30, encoding="utf-8")

        pipeline.ingest(doc)

        points = vs.upsert.call_args[0][0]
        assert all("document_summary" not in p.payload for p in points)


class TestIngestionPipelineCitations:
    def _make_pipeline_with_parsed_doc(
        self,
        parsed_doc: ParsedDocument,
        chunks: list[Chunk] | None = None,
    ) -> tuple[IngestionPipeline, MagicMock, MagicMock, MagicMock, MagicMock]:
        cfg = _make_rag_config()
        parser = MagicMock()
        chunker = MagicMock()
        embedding_provider = MagicMock()
        vectorstore = MagicMock()

        default_chunks = [
            Chunk(text="chunk 1", index=0, metadata={}, start_char=0, end_char=7),
        ]
        parser.parse.return_value = parsed_doc
        chunker.chunk.return_value = chunks if chunks is not None else default_chunks
        embedding_provider.embed.return_value = [[0.1] * 4] * len(
            chunker.chunk.return_value
        )
        vectorstore.scroll.return_value = ([], None)

        pipeline = IngestionPipeline(
            cfg, parser, chunker, embedding_provider, vectorstore
        )
        return pipeline, parser, chunker, embedding_provider, vectorstore

    def test_non_pdf_page_number_is_none(self, tmp_path: Path) -> None:
        # Arrange
        parsed_doc = ParsedDocument(text="some content", page_breaks=[])
        pipeline, _p, _c, _ep, vs = self._make_pipeline_with_parsed_doc(parsed_doc)
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        # Act
        pipeline.ingest(doc)

        # Assert
        payload = vs.upsert.call_args[0][0][0].payload
        assert payload["page_number"] is None

    def test_pdf_chunk_in_page_one(self, tmp_path: Path) -> None:
        # Arrange: page 1 ends at char 50, page 2 ends at char 100
        parsed_doc = ParsedDocument(text="a" * 100, page_breaks=[50, 100])
        chunk = Chunk(text="hello", index=0, metadata={}, start_char=10, end_char=15)
        pipeline, _p, _c, _ep, vs = self._make_pipeline_with_parsed_doc(
            parsed_doc, chunks=[chunk]
        )
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        # Act
        pipeline.ingest(doc)

        # Assert
        payload = vs.upsert.call_args[0][0][0].payload
        assert payload["page_number"] == 1

    def test_pdf_chunk_in_page_two(self, tmp_path: Path) -> None:
        # Arrange: page 1 ends at char 50, page 2 ends at char 100
        parsed_doc = ParsedDocument(text="a" * 100, page_breaks=[50, 100])
        chunk = Chunk(text="hello", index=0, metadata={}, start_char=60, end_char=65)
        pipeline, _p, _c, _ep, vs = self._make_pipeline_with_parsed_doc(
            parsed_doc, chunks=[chunk]
        )
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        # Act
        pipeline.ingest(doc)

        # Assert
        payload = vs.upsert.call_args[0][0][0].payload
        assert payload["page_number"] == 2

    def test_payload_char_start_equals_chunk_start_char(self, tmp_path: Path) -> None:
        # Arrange
        parsed_doc = ParsedDocument(text="hello world", page_breaks=[])
        chunk = Chunk(text="hello", index=0, metadata={}, start_char=0, end_char=5)
        pipeline, _p, _c, _ep, vs = self._make_pipeline_with_parsed_doc(
            parsed_doc, chunks=[chunk]
        )
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        # Act
        pipeline.ingest(doc)

        # Assert
        payload = vs.upsert.call_args[0][0][0].payload
        assert payload["char_start"] == chunk.start_char

    def test_payload_char_end_equals_chunk_end_char(self, tmp_path: Path) -> None:
        # Arrange
        parsed_doc = ParsedDocument(text="hello world", page_breaks=[])
        chunk = Chunk(text="hello", index=0, metadata={}, start_char=0, end_char=5)
        pipeline, _p, _c, _ep, vs = self._make_pipeline_with_parsed_doc(
            parsed_doc, chunks=[chunk]
        )
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")

        # Act
        pipeline.ingest(doc)

        # Assert
        payload = vs.upsert.call_args[0][0][0].payload
        assert payload["char_end"] == chunk.end_char

    def test_parent_doc_child_char_start_adjusted(self, tmp_path: Path) -> None:
        # Arrange: parent chunk starts at char 0, child starts at 0 within parent
        cfg = _make_rag_config(parent_chunk_size=200)
        parser = MagicMock()
        chunker = MagicMock()
        ep = MagicMock()
        vs = MagicMock()

        text = "a" * 200
        parsed_doc = ParsedDocument(text=text, page_breaks=[])
        parser.parse.return_value = parsed_doc
        # child starts at char 0 within the parent (parent itself starts at 0)
        child_chunk = Chunk(
            text="child", index=0, metadata={}, start_char=0, end_char=5
        )
        chunker.chunk.return_value = [child_chunk]
        ep.embed.return_value = [[0.1] * 4]
        vs.scroll.return_value = ([], None)

        pipeline = IngestionPipeline(cfg, parser, chunker, ep, vs)
        doc = tmp_path / "doc.md"
        doc.write_text(text, encoding="utf-8")

        # Act
        pipeline.ingest(doc)

        # Assert: parent_chunk.start_char = 0, child.start_char = 0 → payload = 0
        payload = vs.upsert.call_args[0][0][0].payload
        assert payload["char_start"] == 0
