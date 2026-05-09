import hashlib
from pathlib import Path
from unittest.mock import MagicMock

from mindlm.core.models import Point
from mindlm.core.synchronization.synchronizer import Synchronizer


def _make_synchronizer() -> tuple[Synchronizer, MagicMock, MagicMock]:
    vectorstore = MagicMock()
    pipeline = MagicMock()
    return Synchronizer(vectorstore, pipeline), vectorstore, pipeline


class TestSynchronizer:
    def test_sync_new_document(self, tmp_path: Path) -> None:
        sync, vs, pipeline = _make_synchronizer()
        doc = tmp_path / "doc.txt"
        doc.write_text("content", encoding="utf-8")
        vs.scroll.return_value = ([], None)

        result = sync.sync([doc])

        pipeline.ingest.assert_called_once_with(doc)
        assert result.added == 1
        assert result.skipped == 0

    def test_sync_unchanged_skips(self, tmp_path: Path) -> None:
        sync, vs, pipeline = _make_synchronizer()
        doc = tmp_path / "doc.txt"
        doc.write_text("content", encoding="utf-8")
        current_hash = f"sha256:{hashlib.sha256(b'content').hexdigest()}"
        mock_point = Point(
            id="1",
            vector=[0.1],
            payload={
                "source": str(doc),
                "document_hash": current_hash,
                "modified_at": "2025-01-01T00:00:00Z",
                "file_size": 7,
                "ingested_at": "2025-01-01T00:01:00Z",
            },
        )
        vs.scroll.return_value = ([mock_point], None)

        result = sync.sync([doc])

        pipeline.ingest.assert_not_called()
        assert result.skipped == 1

    def test_sync_modified_reingest(self, tmp_path: Path) -> None:
        sync, vs, pipeline = _make_synchronizer()
        doc = tmp_path / "doc.txt"
        doc.write_text("new content", encoding="utf-8")
        mock_point = Point(
            id="1",
            vector=[0.1],
            payload={
                "source": str(doc),
                "document_hash": "sha256:old_hash",
                "modified_at": "2025-01-01T00:00:00Z",
                "file_size": 5,
                "ingested_at": "2025-01-01T00:01:00Z",
            },
        )
        vs.scroll.return_value = ([mock_point], None)

        result = sync.sync([doc])

        vs.delete_by_filter.assert_called_once()
        pipeline.ingest.assert_called_once_with(doc)
        assert result.updated == 1

    def test_sync_error_continues(self, tmp_path: Path) -> None:
        sync, vs, pipeline = _make_synchronizer()
        doc1 = tmp_path / "doc1.txt"
        doc1.write_text("a", encoding="utf-8")
        doc2 = tmp_path / "doc2.txt"
        doc2.write_text("b", encoding="utf-8")
        vs.scroll.return_value = ([], None)
        pipeline.ingest.side_effect = [RuntimeError("fail"), None]

        result = sync.sync([doc1, doc2])

        assert len(result.errors) == 1
        assert result.added == 1

    def test_calculate_hash_deterministic(self, tmp_path: Path) -> None:
        sync, _, _ = _make_synchronizer()
        doc = tmp_path / "f.txt"
        doc.write_text("hello", encoding="utf-8")

        h1 = sync.calculate_hash(doc)
        h2 = sync.calculate_hash(doc)

        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_full_reingest_recreates_collection(self, tmp_path: Path) -> None:
        sync, vs, pipeline = _make_synchronizer()
        doc = tmp_path / "doc.txt"
        doc.write_text("text", encoding="utf-8")

        sync.full_reingest([doc], collection="docs", dense_dim=384, sparse=False)

        vs.recreate_collection.assert_called_once_with("docs", 384, False)
        pipeline.ingest.assert_called_once_with(doc)
