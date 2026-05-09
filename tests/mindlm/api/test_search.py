from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import mindlm.api.dependencies as deps
from mindlm.api.main import app
from mindlm.core.exceptions import LLMUnavailableError
from mindlm.core.models import Result


@pytest.fixture(autouse=True)
def clear_overrides() -> Generator[None, None, None]:
    yield
    app.dependency_overrides.clear()


def _result() -> Result:
    return Result(
        id="1",
        score=0.9,
        payload={"content": "text", "source": "/doc.pdf", "chunk_index": 0},
    )


class TestSearchEndpoint:
    def test_search_returns_results(self) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [_result()]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [_result()]
        app.dependency_overrides[deps.get_retriever] = lambda: mock_retriever
        app.dependency_overrides[deps.get_reranker] = lambda: mock_reranker

        client = TestClient(app)
        response = client.post("/search", json={"query": "test"})

        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    def test_ask_success(self) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [_result()]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [_result()]
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "answer"
        app.dependency_overrides[deps.get_retriever] = lambda: mock_retriever
        app.dependency_overrides[deps.get_reranker] = lambda: mock_reranker
        app.dependency_overrides[deps.get_llm_provider] = lambda: mock_llm

        client = TestClient(app)
        response = client.post("/ask", json={"question": "What is RAG?"})

        assert response.status_code == 200
        assert response.json()["answer"] == "answer"

    def test_ask_503_llm_unavailable(self) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [_result()]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [_result()]
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = LLMUnavailableError("Ollama down")
        app.dependency_overrides[deps.get_retriever] = lambda: mock_retriever
        app.dependency_overrides[deps.get_reranker] = lambda: mock_reranker
        app.dependency_overrides[deps.get_llm_provider] = lambda: mock_llm

        client = TestClient(app)
        response = client.post("/ask", json={"question": "test?"})

        assert response.status_code == 503
        assert response.json()["error"] == "llm_unavailable"
