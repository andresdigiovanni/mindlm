from mindlm.core.models import DocumentState, SparseVector, SyncResult


class TestSyncResult:
    def test_sync_result_is_mutable(self) -> None:
        # Arrange
        r = SyncResult()

        # Act
        r.added += 1

        # Assert
        assert r.added == 1

    def test_sync_result_defaults(self) -> None:
        # Arrange / Act
        r = SyncResult()

        # Assert
        assert r.added == 0
        assert r.updated == 0
        assert r.skipped == 0
        assert r.chunks == 0
        assert r.errors == []


class TestDocumentState:
    def test_document_state_fields(self) -> None:
        # Arrange / Act
        state = DocumentState(
            source="/path/to/doc.pdf",
            document_hash="sha256:abc",
            modified_at="2025-01-01T00:00:00Z",
            file_size=1024,
            ingested_at="2025-01-01T00:01:00Z",
        )

        # Assert
        assert state.source == "/path/to/doc.pdf"
        assert state.document_hash == "sha256:abc"
        assert state.file_size == 1024


class TestSparseVector:
    def test_sparse_vector_fields(self) -> None:
        # Arrange / Act
        sv = SparseVector(indices=[0, 1], values=[0.5, 0.3])

        # Assert
        assert len(sv.indices) == 2
        assert len(sv.values) == 2
