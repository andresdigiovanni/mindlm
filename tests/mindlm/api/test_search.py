from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import mindlm.api.dependencies as deps
from mindlm.api.main import app
from mindlm.api.routers.search import _format_context
from mindlm.core.exceptions import LLMUnavailableError
from mindlm.core.models import Result


@pytest.fixture(autouse=True)
def clear_overrides() -> Generator[None, None, None]:
    yield
    app.dependency_overrides.clear()


def _result(
    *,
    chunk_context: str | None = None,
    document_summary: str | None = None,
) -> Result:
    payload: dict[str, object] = {
        "content": "text",
        "source": "/doc.pdf",
        "chunk_index": 0,
    }
    if chunk_context is not None:
        payload["chunk_context"] = chunk_context
    if document_summary is not None:
        payload["document_summary"] = document_summary
    return Result(id="1", score=0.9, payload=payload)


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


class TestFormatContext:
    def test_plain_chunk_no_context_fields(self) -> None:
        result = _format_context([_result()])
        assert "[Source: /doc.pdf]" in result
        assert "text" in result
        assert "chunk_context" not in result
        assert "document_summary" not in result

    def test_chunk_context_included_when_present(self) -> None:
        result = _format_context([_result(chunk_context="This discusses cheese.")])
        assert "[Chunk context: This discusses cheese.]" in result
        assert "text" in result

    def test_document_summary_included_when_present(self) -> None:
        result = _format_context([_result(document_summary="Doc about dairy.")])
        assert "[Document context: Doc about dairy.]" in result
        assert "text" in result

    def test_document_summary_appears_before_chunk_context(self) -> None:
        result = _format_context(
            [_result(chunk_context="Chunk ctx.", document_summary="Doc summary.")]
        )
        assert result.index("[Document context:") < result.index("[Chunk context:")

    def test_chunk_context_appears_before_content(self) -> None:
        result = _format_context([_result(chunk_context="Chunk ctx.")])
        assert result.index("[Chunk context:") < result.index("text")

    def test_multiple_results_separated_by_blank_line(self) -> None:
        results = [_result(), _result()]
        formatted = _format_context(results)
        assert "\n\n" in formatted
