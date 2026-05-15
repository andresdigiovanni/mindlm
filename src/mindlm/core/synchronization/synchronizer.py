import hashlib
from collections.abc import Sequence
from pathlib import Path

from mindlm.core.ingestion.pipeline import IngestionPipeline
from mindlm.core.models import DocumentState, SyncResult
from mindlm.core.vectorstore.base import VectorStore

_SOURCE_TYPE_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "pdf": (".pdf",),
    "html": (".html", ".htm"),
    "markdown": (".md", ".markdown"),
    "png": (".png",),
    "jpeg": (".jpg", ".jpeg"),
    "pptx": (".pptx",),
    "docx": (".docx",),
}


class Synchronizer:
    def __init__(
        self,
        vectorstore: VectorStore,
        pipeline: IngestionPipeline,
        source_type: Sequence[str] | None = None,
    ) -> None:
        self._vectorstore = vectorstore
        self._pipeline = pipeline
        self._extensions: frozenset[str] = frozenset(
            ext
            for stype in (source_type or list(_SOURCE_TYPE_EXTENSIONS))
            for ext in _SOURCE_TYPE_EXTENSIONS.get(stype, ())
        )

    def _expand_paths(self, paths: list[Path]) -> list[Path]:
        """Expand directories recursively; pass through regular files as-is."""
        result: list[Path] = []
        for path in paths:
            if path.is_dir():
                result.extend(
                    p
                    for p in sorted(path.rglob("*"))
                    if p.is_file() and p.suffix.lower() in self._extensions
                )
            else:
                result.append(path)
        return result

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

    def sync(
        self,
        paths: list[Path],
        collection: str = "documents",
        dense_dim: int = 384,
        sparse: bool = False,
    ) -> SyncResult:
        self._vectorstore.ensure_collection(collection, dense_dim, sparse)
        result = SyncResult()
        for path in self._expand_paths(paths):
            try:
                current_hash = self.calculate_hash(path)
                state = self.get_document_state(str(path))
                if state is None:
                    result.chunks += self._pipeline.ingest(path)
                    result.added += 1
                elif state.document_hash != current_hash:
                    self._vectorstore.delete_by_filter({"source": str(path)})
                    result.chunks += self._pipeline.ingest(path)
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
        for path in self._expand_paths(paths):
            try:
                result.chunks += self._pipeline.ingest(path)
                result.added += 1
            except Exception as exc:
                result.errors.append(f"{path}: {exc}")
        return result
