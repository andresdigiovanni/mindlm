import hashlib
from pathlib import Path

from mindlm.core.ingestion.pipeline import IngestionPipeline
from mindlm.core.models import DocumentState, SyncResult
from mindlm.core.vectorstore.base import VectorStore


class Synchronizer:
    def __init__(self, vectorstore: VectorStore, pipeline: IngestionPipeline) -> None:
        self._vectorstore = vectorstore
        self._pipeline = pipeline

    def calculate_hash(self, path: Path) -> str:
        return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"

    def get_document_state(self, source: str) -> DocumentState | None:
        points, _ = self._vectorstore.scroll(
            filters={"source": source}, limit=1, offset=None
        )
        if not points:
            return None
        payload = points[0].payload
        return DocumentState(
            source=payload.get("source", source),
            document_hash=payload.get("document_hash", ""),
            modified_at=payload.get("modified_at", ""),
            file_size=int(payload.get("file_size", 0)),
            ingested_at=payload.get("ingested_at", ""),
        )

    def sync(self, paths: list[Path]) -> SyncResult:
        result = SyncResult()
        for path in paths:
            try:
                current_hash = self.calculate_hash(path)
                state = self.get_document_state(str(path))
                if state is None:
                    self._pipeline.ingest(path)
                    result.added += 1
                elif state.document_hash != current_hash:
                    self._vectorstore.delete_by_filter({"source": str(path)})
                    self._pipeline.ingest(path)
                    result.updated += 1
                else:
                    result.skipped += 1
            except Exception as exc:
                result.errors.append(f"{path}: {exc}")
        return result

    def full_reingest(
        self,
        paths: list[Path],
        collection: str,
        dense_dim: int,
        sparse: bool,
    ) -> SyncResult:
        self._vectorstore.recreate_collection(collection, dense_dim, sparse)
        result = SyncResult()
        for path in paths:
            try:
                self._pipeline.ingest(path)
                result.added += 1
            except Exception as exc:
                result.errors.append(f"{path}: {exc}")
        return result
