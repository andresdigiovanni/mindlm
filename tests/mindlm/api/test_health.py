from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import mindlm.api.dependencies as deps
from mindlm.api.main import app


@pytest.fixture(autouse=True)
def clear_overrides() -> Generator[None, None, None]:
    yield
    app.dependency_overrides.clear()


class TestHealthEndpoint:
    def test_health_all_ok(self) -> None:
        mock_vs = MagicMock()
        mock_vs.list_collections.return_value = []
        mock_llm = MagicMock()
        mock_llm.healthcheck.return_value = True
        mock_emb = MagicMock()
        mock_emb.embed_one.return_value = [0.1]

        app.dependency_overrides[deps.get_vectorstore] = lambda: mock_vs
        app.dependency_overrides[deps.get_llm_provider] = lambda: mock_llm
        app.dependency_overrides[deps.get_embedding_provider] = lambda: mock_emb

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_degraded_ollama_down(self) -> None:
        mock_vs = MagicMock()
        mock_vs.list_collections.return_value = []
        mock_llm = MagicMock()
        mock_llm.healthcheck.return_value = False
        mock_emb = MagicMock()
        mock_emb.embed_one.return_value = [0.1]

        app.dependency_overrides[deps.get_vectorstore] = lambda: mock_vs
        app.dependency_overrides[deps.get_llm_provider] = lambda: mock_llm
        app.dependency_overrides[deps.get_embedding_provider] = lambda: mock_emb

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["services"]["ollama"] == "error"

    def test_health_error_qdrant_down(self) -> None:
        mock_vs = MagicMock()
        mock_vs.list_collections.side_effect = RuntimeError("connection refused")
        mock_llm = MagicMock()
        mock_llm.healthcheck.return_value = True
        mock_emb = MagicMock()
        mock_emb.embed_one.return_value = [0.1]

        app.dependency_overrides[deps.get_vectorstore] = lambda: mock_vs
        app.dependency_overrides[deps.get_llm_provider] = lambda: mock_llm
        app.dependency_overrides[deps.get_embedding_provider] = lambda: mock_emb

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
