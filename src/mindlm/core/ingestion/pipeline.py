import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastembed.sparse.bm25 import Bm25
from langfuse.decorators import langfuse_context, observe

from mindlm.core.chunking.dispatcher import ChunkerDispatcher
from mindlm.core.chunking.strategies.fixed import FixedChunker
from mindlm.core.config.models import ChunkingConfig, RAGConfig
from mindlm.core.embeddings.base import EmbeddingProvider
from mindlm.core.graph.base import GraphStore
from mindlm.core.graph.extractor import EntityExtractor
from mindlm.core.ingestion.contextualizer import Contextualizer
from mindlm.core.models import (
    Chunk,
    Entity,
    ParsedDocument,
    Point,
    Relationship,
    SparseVector,
)
from mindlm.core.parsing.dispatcher import ParserDispatcher
from mindlm.core.vectorstore.base import VectorStore


class IngestionPipeline:
    def __init__(
        self,
        config: RAGConfig,
        parser: ParserDispatcher,
        chunker: ChunkerDispatcher,
        embedding_provider: EmbeddingProvider,
        vectorstore: VectorStore,
        contextualizer: Contextualizer | None = None,
        entity_extractor: EntityExtractor | None = None,
        graph_store: GraphStore | None = None,
    ) -> None:
        if entity_extractor is not None and graph_store is None:
            raise ValueError(
                "graph_store is required when entity_extractor is provided"
            )
        self._config = config
        self._parser = parser
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._vectorstore = vectorstore
        self._contextualizer = contextualizer
        self._entity_extractor = entity_extractor
        self._graph_store = graph_store
        self._bm25: Bm25 | None = None

    @observe(name="ingest")
    def ingest(self, path: Path) -> int:
        self._check_allowed_path(path)
        langfuse_context.update_current_observation(
            input=str(path),
            metadata={"strategy": self._config.chunking.strategy},
        )
        parsed_doc = self._parser.parse(path)
        text = parsed_doc.text
        document_hash = self._calculate_hash(path)
        if self._config.ingestion.deduplication and self._is_duplicate(document_hash):
            return 0
        parent_size = self._config.chunking.parent_chunk_size
        if parent_size is not None:
            return self._ingest_parent_doc(
                path, text, parent_size, document_hash, parsed_doc
            )
        chunks = self._chunker.chunk(text)
        if not chunks:
            return 0
        document_summary = (
            self._contextualizer.summarize(text) if self._contextualizer else None
        )
        contexts = self._get_chunk_contexts(text, chunks)
        point_ids = [str(uuid4()) for _ in chunks]
        self._extract_and_store_entities(list(zip(chunks, point_ids, strict=True)))
        vectors = self._embedding_provider.embed([c.text for c in chunks])
        use_sparse = self._config.retrieval.strategy == "hybrid"
        points = [
            self._make_point(
                chunks[i],
                vectors[i],
                use_sparse,
                self._build_payload(
                    path,
                    chunks[i],
                    len(chunks),
                    document_hash,
                    parsed_doc,
                    chunk_context=contexts[i] if contexts else None,
                    document_summary=document_summary,
                ),
                point_ids[i],
            )
            for i in range(len(chunks))
        ]
        self._vectorstore.upsert(points)
        return len(points)

    def _is_duplicate(self, document_hash: str) -> bool:
        points, _ = self._vectorstore.scroll(
            filters={"document_hash": document_hash}, limit=1, offset=None
        )
        return len(points) > 0

    def _ingest_parent_doc(
        self,
        path: Path,
        text: str,
        parent_size: int,
        document_hash: str,
        parsed_doc: ParsedDocument,
    ) -> int:
        parent_config = ChunkingConfig(
            strategy="fixed", chunk_size=parent_size, overlap=0
        )
        parent_chunker = FixedChunker(parent_config)
        parent_chunks = parent_chunker.chunk(text)
        if not parent_chunks:
            return 0

        all_points: list[Point] = []
        use_sparse = self._config.retrieval.strategy == "hybrid"
        document_summary = (
            self._contextualizer.summarize(text) if self._contextualizer else None
        )

        for parent_chunk in parent_chunks:
            parent_id = str(uuid4())
            child_chunks = self._chunker.chunk(parent_chunk.text)
            if not child_chunks:
                continue
            adjusted_chunks = [
                Chunk(
                    text=c.text,
                    index=c.index,
                    metadata=c.metadata,
                    start_char=parent_chunk.start_char + c.start_char,
                    end_char=parent_chunk.start_char + c.end_char,
                )
                for c in child_chunks
            ]
            contexts = self._get_chunk_contexts(text, adjusted_chunks)
            child_point_ids = [str(uuid4()) for _ in adjusted_chunks]
            self._extract_and_store_entities(
                list(zip(adjusted_chunks, child_point_ids, strict=True))
            )
            child_texts = [c.text for c in adjusted_chunks]
            child_vectors = self._embedding_provider.embed(child_texts)
            for i, child in enumerate(adjusted_chunks):
                payload = self._build_payload(
                    path,
                    child,
                    len(adjusted_chunks),
                    document_hash,
                    parsed_doc,
                    chunk_context=contexts[i] if contexts else None,
                    document_summary=document_summary,
                )
                payload["parent_id"] = parent_id
                payload["parent_content"] = parent_chunk.text
                all_points.append(
                    self._make_point(
                        child, child_vectors[i], use_sparse, payload, child_point_ids[i]
                    )
                )

        if all_points:
            self._vectorstore.upsert(all_points)
        return len(all_points)

    def _extract_and_store_entities(
        self, chunk_point_pairs: list[tuple[Chunk, str]]
    ) -> None:
        if self._entity_extractor is None or self._graph_store is None:
            return
        all_entities: list[Entity] = []
        all_relationships: list[Relationship] = []
        for chunk, point_id in chunk_point_pairs:
            entities, relationships = self._entity_extractor.extract(
                chunk.text, point_id
            )
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        if all_entities:
            self._graph_store.upsert_entities(all_entities)
        if all_relationships:
            self._graph_store.upsert_relationships(all_relationships)

    def _get_chunk_contexts(self, document_text: str, chunks: list[Chunk]) -> list[str]:
        if (
            self._contextualizer is None
            or not self._contextualizer.chunk_context_enabled
        ):
            return []
        max_workers = min(len(chunks), self._config.contextual_retrieval.max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(
                executor.map(
                    lambda c: self._contextualizer.contextualize(document_text, c.text),  # type: ignore[union-attr]
                    chunks,
                )
            )

    def _check_allowed_path(self, path: Path) -> None:
        allowed = Path(self._config.ingestion.allowed_base_dir).resolve()
        resolved = path.resolve()
        if not resolved.is_relative_to(allowed):
            raise ValueError(
                f"Path {path!r} is outside the allowed base directory {allowed!r}"
            )

    def _calculate_hash(self, path: Path) -> str:
        return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"

    def _build_payload(
        self,
        path: Path,
        chunk: Chunk,
        total: int,
        document_hash: str,
        parsed_doc: ParsedDocument,
        chunk_context: str | None = None,
        document_summary: str | None = None,
    ) -> dict[str, Any]:
        stat = path.stat()
        payload = {
            "content": chunk.text,
            "source": str(path),
            "document_hash": document_hash,
            "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
            "file_size": stat.st_size,
            "chunk_index": chunk.index,
            "total_chunks": total,
            "ingested_at": datetime.now(tz=UTC).isoformat(),
            "char_start": chunk.start_char,
            "char_end": chunk.end_char,
            "page_number": parsed_doc.page_number_for(chunk.start_char),
        }
        if "window_context" in chunk.metadata:
            payload["window_context"] = chunk.metadata["window_context"]
        if chunk_context:
            payload["chunk_context"] = chunk_context
        if document_summary:
            payload["document_summary"] = document_summary
        return payload

    def _make_point(
        self,
        chunk: Chunk,
        vector: list[float],
        use_sparse: bool,
        payload: dict[str, Any],
        point_id: str | None = None,
    ) -> Point:
        return Point(
            id=point_id if point_id is not None else str(uuid4()),
            vector=vector,
            sparse_vector=self._compute_sparse(chunk.text) if use_sparse else None,
            payload=payload,
        )

    def _compute_sparse(self, text: str) -> SparseVector:
        if self._bm25 is None:
            self._bm25 = Bm25("Qdrant/bm25")
        result = next(iter(self._bm25.passage_embed(text)))
        return SparseVector(
            indices=result.indices.tolist(), values=result.values.tolist()
        )
