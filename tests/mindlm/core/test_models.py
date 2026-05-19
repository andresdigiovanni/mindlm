from mindlm.core.models import DocumentState, ParsedDocument, SparseVector, SyncResult


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


class TestParsedDocument:
    def test_no_page_breaks_returns_none(self) -> None:
        # Arrange
        doc = ParsedDocument(text="hello", page_breaks=[])

        # Act
        result = doc.page_number_for(0)

        # Assert
        assert result is None

    def test_single_break_offset_before_returns_page_one(self) -> None:
        # Arrange
        doc = ParsedDocument(text="a" * 60, page_breaks=[50])

        # Act
        result = doc.page_number_for(0)

        # Assert
        assert result == 1

    def test_single_break_offset_just_before_break_returns_page_one(self) -> None:
        # Arrange
        doc = ParsedDocument(text="a" * 60, page_breaks=[50])

        # Act
        result = doc.page_number_for(49)

        # Assert
        assert result == 1

    def test_two_breaks_offset_on_first_page(self) -> None:
        # Arrange
        doc = ParsedDocument(text="a" * 30, page_breaks=[10, 20])

        # Act
        result = doc.page_number_for(9)

        # Assert
        assert result == 1

    def test_two_breaks_first_char_of_second_page(self) -> None:
        # Arrange
        doc = ParsedDocument(text="a" * 30, page_breaks=[10, 20])

        # Act
        result = doc.page_number_for(10)

        # Assert
        assert result == 2

    def test_offset_beyond_last_break_clamped_to_last_page(self) -> None:
        # Arrange
        doc = ParsedDocument(text="a" * 60, page_breaks=[50])

        # Act
        result = doc.page_number_for(100)

        # Assert
        assert result == 1
