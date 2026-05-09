from unittest.mock import MagicMock, patch

import pytest

from mindlm.core.config.models import VectorStoreConfig
from mindlm.core.models import Point, SparseVector
from mindlm.core.vectorstore.qdrant import QdrantVectorStore


def _config() -> VectorStoreConfig:
    return VectorStoreConfig(
        provider="qdrant",
        mode="local",
        host="localhost",
        port=6333,
        collection="test",
    )


@pytest.fixture
def store() -> QdrantVectorStore:
    with patch("mindlm.core.vectorstore.qdrant.QdrantClient"):
        return QdrantVectorStore(_config())


class TestQdrantVectorStore:
    def test_upsert_calls_client(self, store: QdrantVectorStore) -> None:
        point = Point(id="1", vector=[0.1, 0.2], payload={"content": "test"})

        store.upsert([point])

        store._client.upsert.assert_called_once()

    def test_upsert_with_sparse_vector(self, store: QdrantVectorStore) -> None:
        point = Point(
            id="1",
            vector=[0.1, 0.2],
            payload={},
            sparse_vector=SparseVector(indices=[0, 1], values=[0.5, 0.3]),
        )

        store.upsert([point])

        call_args = store._client.upsert.call_args
        point_struct = call_args[1]["points"][0]
        assert "sparse" in point_struct.vector

    def test_search_returns_result_list(self, store: QdrantVectorStore) -> None:
        mock_hit = MagicMock()
        mock_hit.id = "abc"
        mock_hit.score = 0.9
        mock_hit.payload = {"content": "x"}
        mock_qr = MagicMock()
        mock_qr.points = [mock_hit]
        store._client.query_points.return_value = mock_qr

        results = store.search([0.1, 0.2], top_k=5, filters=None)

        assert len(results) == 1
        assert results[0].score == 0.9
        assert results[0].id == "abc"

    def test_list_collections(self, store: QdrantVectorStore) -> None:
        col_a = MagicMock()
        col_a.name = "a"
        col_b = MagicMock()
        col_b.name = "b"
        store._client.get_collections.return_value = MagicMock(
            collections=[col_a, col_b]
        )

        result = store.list_collections()

        assert result == ["a", "b"]

    def test_get_by_id_returns_none_when_empty(self, store: QdrantVectorStore) -> None:
        store._client.retrieve.return_value = []

        result = store.get_by_id("nonexistent")

        assert result is None

    def test_create_collection_without_sparse(self, store: QdrantVectorStore) -> None:
        store.create_collection("col", dense_dim=384, sparse=False)

        call_kwargs = store._client.create_collection.call_args[1]
        assert call_kwargs.get("sparse_vectors_config") is None

    def test_create_collection_with_sparse(self, store: QdrantVectorStore) -> None:
        store.create_collection("col", dense_dim=384, sparse=True)

        call_kwargs = store._client.create_collection.call_args[1]
        assert call_kwargs.get("sparse_vectors_config") is not None

    def test_search_hybrid_calls_query_points(self, store: QdrantVectorStore) -> None:
        mock_response = MagicMock()
        mock_hit = MagicMock()
        mock_hit.id = "abc"
        mock_hit.score = 0.85
        mock_hit.payload = {"content": "hybrid result"}
        mock_response.points = [mock_hit]
        store._client.query_points.return_value = mock_response
        sparse = SparseVector(indices=[0, 1], values=[0.5, 0.3])

        results = store.search_hybrid([0.1, 0.2], sparse, top_k=5, filters=None)

        store._client.query_points.assert_called_once()
        assert len(results) == 1
        assert results[0].score == 0.85

    def test_delete_by_filter(self, store: QdrantVectorStore) -> None:
        store.delete_by_filter({"source": "/doc.pdf"})

        store._client.delete.assert_called_once()

    def test_delete_by_filter_empty_dict_is_noop(
        self, store: QdrantVectorStore
    ) -> None:
        store.delete_by_filter({})

        store._client.delete.assert_not_called()

    def test_get_by_id_returns_point(self, store: QdrantVectorStore) -> None:
        mock_point = MagicMock()
        mock_point.id = "abc"
        mock_point.vector = [0.1, 0.2]
        mock_point.payload = {"content": "test"}
        store._client.retrieve.return_value = [mock_point]

        result = store.get_by_id("abc")

        assert result is not None
        assert result.id == "abc"

    def test_scroll_returns_points_and_offset(self, store: QdrantVectorStore) -> None:
        mock_p = MagicMock()
        mock_p.id = "p1"
        mock_p.vector = [0.1]
        mock_p.payload = {"content": "x"}
        store._client.scroll.return_value = ([mock_p], "next_offset")

        points, offset = store.scroll({"source": "/doc"}, limit=10, offset=None)

        assert len(points) == 1
        assert offset == "next_offset"

    def test_scroll_none_offset_returns_none(self, store: QdrantVectorStore) -> None:
        store._client.scroll.return_value = ([], None)

        points, offset = store.scroll({}, limit=10, offset=None)

        assert points == []
        assert offset is None
