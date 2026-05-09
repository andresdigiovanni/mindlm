import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastembed.sparse.bm25 import Bm25

from mindlm.core.chunking.dispatcher import ChunkerDispatcher
from mindlm.core.config.models import RAGConfig
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
        chunks = self._chunker.chunk(text)
        if not chunks:
            return 0
        document_hash = self._calculate_hash(path)
        vectors = self._embedding_provider.embed([c.text for c in chunks])
        use_sparse = self._config.retrieval.strategy == "hybrid"
        points = [
            Point(
                id=str(uuid4()),
                vector=vectors[i],
                sparse_vector=self._compute_sparse(chunks[i].text)
                if use_sparse
                else None,
                payload=self._build_payload(
                    path, chunks[i], len(chunks), document_hash
                ),
            )
            for i in range(len(chunks))
        ]
        self._vectorstore.upsert(points)
        return len(points)

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

    def _compute_sparse(self, text: str) -> SparseVector:
        if self._bm25 is None:
            self._bm25 = Bm25("Qdrant/bm25")
        result = next(iter(self._bm25.passage_embed(text)))
        return SparseVector(
            indices=result.indices.tolist(), values=result.values.tolist()
        )
