from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import mindlm.api.dependencies as deps
from mindlm.api.main import app
from mindlm.core.models import SyncResult


@pytest.fixture(autouse=True)
def clear_overrides() -> Generator[None, None, None]:
    yield
    app.dependency_overrides.clear()


class TestIngestEndpoint:
    def test_ingest_sync(self) -> None:
        mock_sync = MagicMock()
        mock_sync.sync.return_value = SyncResult(added=2, updated=0, skipped=1)
        mock_config = MagicMock()
        mock_config.ingestion.allowed_base_dir = "/data"
        app.dependency_overrides[deps.get_synchronizer] = lambda: mock_sync
        app.dependency_overrides[deps.get_config] = lambda: mock_config

        client = TestClient(app)
        response = client.post(
            "/ingest/sync", json={"paths": ["/data/doc1.pdf", "/data/doc2.pdf"]}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["added"] == 2
        assert data["skipped"] == 1

    def test_ingest_full(self) -> None:
        mock_sync = MagicMock()
        mock_sync.full_reingest.return_value = SyncResult(added=3, updated=0, skipped=0)
        mock_config = MagicMock()
        mock_config.retrieval.strategy = "vector"
        mock_config.vector_store.collection = "docs"
        mock_config.embeddings.dimensions = 384
        mock_config.ingestion.allowed_base_dir = "/data"
        app.dependency_overrides[deps.get_synchronizer] = lambda: mock_sync
        app.dependency_overrides[deps.get_config] = lambda: mock_config

        client = TestClient(app)
        response = client.post("/ingest/full", json={"paths": ["/data/doc1.pdf"]})

        assert response.status_code == 200
        data = response.json()
        assert data["added"] == 3
