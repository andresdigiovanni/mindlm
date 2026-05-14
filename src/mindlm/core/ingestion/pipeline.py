import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastembed.sparse.bm25 import Bm25

from mindlm.core.chunking.dispatcher import ChunkerDispatcher
from mindlm.core.chunking.strategies.fixed import FixedChunker
from mindlm.core.config.models import ChunkingConfig, RAGConfig
from mindlm.core.embeddings.base import EmbeddingProvider
from mindlm.core.models import Chunk, Point, SparseVector
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
    ) -> None:
        self._config = config
        self._parser = parser
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._vectorstore = vectorstore
        self._bm25: Bm25 | None = None

    def ingest(self, path: Path) -> int:
        text = self._parser.parse(path)
        parent_size = self._config.chunking.parent_chunk_size
        if parent_size is not None:
            return self._ingest_parent_doc(path, text, parent_size)
        chunks = self._chunker.chunk(text)
        if not chunks:
            return 0
        document_hash = self._calculate_hash(path)
        vectors = self._embedding_provider.embed([c.text for c in chunks])
        use_sparse = self._config.retrieval.strategy == "hybrid"
        points = [
            self._make_point(
                chunks[i],
                vectors[i],
                use_sparse,
                self._build_payload(path, chunks[i], len(chunks), document_hash),
            )
            for i in range(len(chunks))
        ]
        self._vectorstore.upsert(points)
        return len(points)

    def _ingest_parent_doc(self, path: Path, text: str, parent_size: int) -> int:
        parent_config = ChunkingConfig(
            strategy="fixed", chunk_size=parent_size, overlap=0
        )
        parent_chunker = FixedChunker(parent_config)
        parent_chunks = parent_chunker.chunk(text)
        if not parent_chunks:
            return 0

        document_hash = self._calculate_hash(path)
        all_points: list[Point] = []
        use_sparse = self._config.retrieval.strategy == "hybrid"

        for parent_chunk in parent_chunks:
            parent_id = str(uuid4())
            child_chunks = self._chunker.chunk(parent_chunk.text)
            if not child_chunks:
                continue
            child_texts = [c.text for c in child_chunks]
            child_vectors = self._embedding_provider.embed(child_texts)
            for i, child in enumerate(child_chunks):
                payload = self._build_payload(
                    path, child, len(child_chunks), document_hash
                )
                payload["parent_id"] = parent_id
                payload["parent_content"] = parent_chunk.text
                all_points.append(
                    self._make_point(child, child_vectors[i], use_sparse, payload)
                )

        if all_points:
            self._vectorstore.upsert(all_points)
        return len(all_points)

    def _calculate_hash(self, path: Path) -> str:
        return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"

    def _build_payload(
        self, path: Path, chunk: Chunk, total: int, document_hash: str
    ) -> dict[str, Any]:
        stat = path.stat()
        return {
            "content": chunk.text,
            "source": str(path),
            "document_hash": document_hash,
            "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
            "file_size": stat.st_size,
            "chunk_index": chunk.index,
            "total_chunks": total,
            "ingested_at": datetime.now(tz=UTC).isoformat(),
        }

    def _make_point(
        self,
        chunk: Chunk,
        vector: list[float],
        use_sparse: bool,
        payload: dict[str, Any],
    ) -> Point:
        return Point(
            id=str(uuid4()),
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
