from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import mindlm.api.dependencies as deps
from mindlm.api.main import app
from mindlm.api.routers.search import _extract_sources, _format_context, _merge_results
from mindlm.core.exceptions import LLMUnavailableError
from mindlm.core.generation.grounding import GroundingChecker, GroundingResult
from mindlm.core.models import Result


@pytest.fixture(autouse=True)
def clear_overrides() -> Generator[None, None, None]:
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _result(
    *,
    id_: str = "1",
    chunk_context: str | None = None,
    document_summary: str | None = None,
    page_number: int | None = None,
    char_start: int | None = None,
    char_end: int | None = None,
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
    if page_number is not None:
        payload["page_number"] = page_number
    if char_start is not None:
        payload["char_start"] = char_start
    if char_end is not None:
        payload["char_end"] = char_end
    return Result(id=id_, score=0.9, payload=payload)


def _mock_config(score_threshold: float | None = None) -> MagicMock:
    cfg = MagicMock()
    cfg.reranking.score_threshold = score_threshold
    cfg.reranking.top_k = None
    cfg.reranking.enabled = True
    return cfg


def _mock_iterative_config(max_iterations: int = 3) -> MagicMock:
    cfg = MagicMock()
    cfg.reranking.score_threshold = None
    cfg.reranking.top_k = None
    cfg.reranking.enabled = True
    cfg.iterative_retrieval.max_iterations = max_iterations
    return cfg


class TestSearchEndpoint:
    def test_search_returns_results(self) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [_result()]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [_result()]
        app.dependency_overrides[deps.get_retriever] = lambda: mock_retriever
        app.dependency_overrides[deps.get_reranker] = lambda: mock_reranker
        app.dependency_overrides[deps.get_compressor] = lambda: None
        app.dependency_overrides[deps.get_config] = lambda: _mock_config()

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
        app.dependency_overrides[deps.get_compressor] = lambda: None
        app.dependency_overrides[deps.get_llm_provider] = lambda: mock_llm
        app.dependency_overrides[deps.get_config] = lambda: _mock_config()
        app.dependency_overrides[deps.get_grounding_checker] = lambda: None

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
        app.dependency_overrides[deps.get_compressor] = lambda: None
        app.dependency_overrides[deps.get_llm_provider] = lambda: mock_llm
        app.dependency_overrides[deps.get_config] = lambda: _mock_config()
        app.dependency_overrides[deps.get_grounding_checker] = lambda: None

        client = TestClient(app)
        response = client.post("/ask", json={"question": "test?"})

        assert response.status_code == 503
        assert response.json()["error"] == "llm_unavailable"

    def test_search_applies_compression_when_compressor_present(self) -> None:
        mock_compressor = MagicMock()
        mock_compressor.compress.return_value = [
            Result(
                id="1",
                score=0.9,
                payload={"content": "compressed", "source": "doc.pdf"},
            )
        ]
        app.dependency_overrides[deps.get_retriever] = lambda: MagicMock(
            retrieve=MagicMock(
                return_value=[
                    Result(
                        id="1",
                        score=0.9,
                        payload={"content": "original", "source": "doc.pdf"},
                    )
                ]
            )
        )
        app.dependency_overrides[deps.get_reranker] = lambda: MagicMock(
            rerank=MagicMock(
                return_value=[
                    Result(
                        id="1",
                        score=0.9,
                        payload={"content": "original", "source": "doc.pdf"},
                    )
                ]
            )
        )
        app.dependency_overrides[deps.get_compressor] = lambda: mock_compressor
        app.dependency_overrides[deps.get_config] = lambda: MagicMock(
            reranking=MagicMock(score_threshold=None, top_k=None, enabled=True),
        )
        client = TestClient(app)
        response = client.post("/search", json={"query": "test query"})
        assert response.status_code == 200
        mock_compressor.compress.assert_called_once()
        assert mock_compressor.compress.call_args[0][0] == "test query"

    def test_ask_applies_compression_when_compressor_present(self) -> None:
        mock_compressor = MagicMock()
        mock_compressor.compress.return_value = [
            Result(
                id="1",
                score=0.9,
                payload={"content": "compressed", "source": "doc.pdf"},
            )
        ]
        app.dependency_overrides[deps.get_retriever] = lambda: MagicMock(
            retrieve=MagicMock(
                return_value=[
                    Result(
                        id="1",
                        score=0.9,
                        payload={"content": "original", "source": "doc.pdf"},
                    )
                ]
            )
        )
        app.dependency_overrides[deps.get_reranker] = lambda: MagicMock(
            rerank=MagicMock(
                return_value=[
                    Result(
                        id="1",
                        score=0.9,
                        payload={"content": "original", "source": "doc.pdf"},
                    )
                ]
            )
        )
        app.dependency_overrides[deps.get_compressor] = lambda: mock_compressor
        app.dependency_overrides[deps.get_config] = lambda: MagicMock(
            reranking=MagicMock(score_threshold=None, top_k=None, enabled=True),
        )
        app.dependency_overrides[deps.get_llm_provider] = lambda: MagicMock(
            chat=MagicMock(return_value="answer")
        )
        app.dependency_overrides[deps.get_grounding_checker] = lambda: None
        client = TestClient(app)
        response = client.post("/ask", json={"question": "test question"})
        assert response.status_code == 200
        mock_compressor.compress.assert_called_once()
        assert mock_compressor.compress.call_args[0][0] == "test question"


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


class TestExtractSources:
    def test_citation_fields_populated_from_payload(self) -> None:
        # Arrange
        result = _result(page_number=2, char_start=100, char_end=200)

        # Act
        sources = _extract_sources([result])

        # Assert
        assert sources[0].page_number == 2
        assert sources[0].char_start == 100
        assert sources[0].char_end == 200

    def test_page_number_none_when_absent_from_payload(self) -> None:
        # Arrange
        result = _result()  # no page_number in payload

        # Act
        sources = _extract_sources([result])

        # Assert
        assert sources[0].page_number is None

    def test_char_start_and_end_default_to_none_when_absent(self) -> None:
        # Arrange
        result = _result()  # no char_start / char_end in payload

        # Act
        sources = _extract_sources([result])

        # Assert
        assert sources[0].char_start is None
        assert sources[0].char_end is None

    def test_char_start_cast_to_int_when_float_in_payload(self) -> None:
        # Arrange
        result = _result(char_start=100, char_end=200)
        # Simulate payload with float values
        result.payload["char_start"] = 100.0
        result.payload["char_end"] = 200.0

        # Act
        sources = _extract_sources([result])

        # Assert
        assert sources[0].char_start == 100
        assert isinstance(sources[0].char_start, int)


class TestSearchResultMatchedChunk:
    def _override(self, result: Result) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [result]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [result]
        app.dependency_overrides[deps.get_retriever] = lambda: mock_retriever
        app.dependency_overrides[deps.get_reranker] = lambda: mock_reranker
        app.dependency_overrides[deps.get_compressor] = lambda: None
        app.dependency_overrides[deps.get_config] = lambda: _mock_config()

    def test_matched_chunk_included_in_response_when_present(self) -> None:
        result = Result(
            id="1",
            score=0.9,
            payload={
                "content": "parent text",
                "matched_chunk": "original chunk",
                "source": "/doc.pdf",
                "chunk_index": 0,
            },
        )
        self._override(result)
        client = TestClient(app)

        response = client.post("/search", json={"query": "test"})

        assert response.status_code == 200
        item = response.json()["results"][0]
        assert item["matched_chunk"] == "original chunk"

    def test_matched_chunk_is_none_when_absent_from_payload(self) -> None:
        result = Result(
            id="1",
            score=0.9,
            payload={"content": "text", "source": "/doc.pdf", "chunk_index": 0},
        )
        self._override(result)
        client = TestClient(app)

        response = client.post("/search", json={"query": "test"})

        assert response.status_code == 200
        item = response.json()["results"][0]
        assert item["matched_chunk"] is None


class TestScoreThreshold:
    """Threshold is applied after reranking on the reranker's score scale."""

    def _make_result(self, id_: str, score: float) -> Result:
        return Result(id=id_, score=score, payload={"content": "t", "source": "s"})

    def _override(
        self, reranked: list[Result], config_threshold: float | None = None
    ) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = reranked
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = reranked
        app.dependency_overrides[deps.get_retriever] = lambda: mock_retriever
        app.dependency_overrides[deps.get_reranker] = lambda: mock_reranker
        app.dependency_overrides[deps.get_compressor] = lambda: None
        app.dependency_overrides[deps.get_config] = lambda: _mock_config(
            config_threshold
        )

    def test_no_threshold_returns_all_results(self) -> None:
        results = [self._make_result("1", 0.85), self._make_result("2", 0.2)]
        self._override(results)
        client = TestClient(app)
        response = client.post("/search", json={"query": "q"})
        assert response.status_code == 200
        assert len(response.json()["results"]) == 2

    def test_request_threshold_filters_low_scores(self) -> None:
        results = [self._make_result("1", 0.85), self._make_result("2", 0.2)]
        self._override(results)
        client = TestClient(app)
        response = client.post("/search", json={"query": "q", "score_threshold": 0.5})
        assert response.status_code == 200
        data = response.json()["results"]
        assert len(data) == 1
        assert data[0]["score"] == pytest.approx(0.85)

    def test_config_threshold_applied_when_no_request_threshold(self) -> None:
        results = [self._make_result("1", 0.85), self._make_result("2", 0.2)]
        self._override(results, config_threshold=0.5)
        client = TestClient(app)
        response = client.post("/search", json={"query": "q"})
        assert response.status_code == 200
        assert len(response.json()["results"]) == 1

    def test_request_threshold_overrides_config(self) -> None:
        results = [self._make_result("1", 0.9), self._make_result("2", 0.6)]
        self._override(results, config_threshold=0.5)  # config keeps both
        client = TestClient(app)
        response = client.post("/search", json={"query": "q", "score_threshold": 0.8})
        assert response.status_code == 200
        data = response.json()["results"]
        assert len(data) == 1
        assert data[0]["score"] == pytest.approx(0.9)

    def test_threshold_boundary_is_inclusive(self) -> None:
        results = [self._make_result("1", 0.7)]
        self._override(results, config_threshold=0.7)
        client = TestClient(app)
        response = client.post("/search", json={"query": "q"})
        assert response.status_code == 200
        assert len(response.json()["results"]) == 1

    def test_all_filtered_returns_empty_results(self) -> None:
        results = [self._make_result("1", 0.1), self._make_result("2", 0.05)]
        self._override(results, config_threshold=0.5)
        client = TestClient(app)
        response = client.post("/search", json={"query": "q"})
        assert response.status_code == 200
        assert response.json()["results"] == []


class TestMergeResults:
    def test_should_return_new_results_when_accumulated_is_empty(self) -> None:
        r1 = Result(id="1", score=0.9, payload={"content": "a"})
        r2 = Result(id="2", score=0.8, payload={"content": "b"})

        merged = _merge_results([], [r1, r2])

        assert merged == [r1, r2]

    def test_should_deduplicate_by_id(self) -> None:
        r1 = Result(id="1", score=0.9, payload={"content": "a"})
        r2 = Result(id="2", score=0.8, payload={"content": "b"})

        merged = _merge_results([r1], [r1, r2])

        assert merged == [r1, r2]

    def test_should_preserve_accumulated_order(self) -> None:
        r1 = Result(id="1", score=0.9, payload={"content": "a"})
        r2 = Result(id="2", score=0.8, payload={"content": "b"})
        r3 = Result(id="3", score=0.7, payload={"content": "c"})

        merged = _merge_results([r2, r1], [r3])

        assert merged == [r2, r1, r3]

    def test_should_return_empty_when_both_empty(self) -> None:
        merged = _merge_results([], [])

        assert merged == []


class TestAskIterativeRetrieval:
    def test_should_answer_in_one_pass_when_grounding_checker_is_none(
        self,
        client: TestClient,
    ) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [_result()]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [_result()]
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "answer"
        app.dependency_overrides[deps.get_retriever] = lambda: mock_retriever
        app.dependency_overrides[deps.get_reranker] = lambda: mock_reranker
        app.dependency_overrides[deps.get_llm_provider] = lambda: mock_llm
        app.dependency_overrides[deps.get_compressor] = lambda: None
        app.dependency_overrides[deps.get_config] = lambda: _mock_config()
        app.dependency_overrides[deps.get_grounding_checker] = lambda: None

        response = client.post("/ask", json={"question": "What is RAG?"})

        assert response.status_code == 200
        assert response.json()["answer"] == "answer"
        mock_retriever.retrieve.assert_called_once()

    def test_should_stop_on_first_iteration_when_answer_is_grounded(
        self,
        client: TestClient,
    ) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [_result()]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [_result()]
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "grounded answer"
        mock_checker = MagicMock(spec=GroundingChecker)
        mock_checker.check.return_value = GroundingResult(
            is_grounded=True, refined_query=None
        )
        app.dependency_overrides[deps.get_retriever] = lambda: mock_retriever
        app.dependency_overrides[deps.get_reranker] = lambda: mock_reranker
        app.dependency_overrides[deps.get_llm_provider] = lambda: mock_llm
        app.dependency_overrides[deps.get_compressor] = lambda: None
        app.dependency_overrides[deps.get_config] = lambda: _mock_iterative_config()
        app.dependency_overrides[deps.get_grounding_checker] = lambda: mock_checker

        response = client.post("/ask", json={"question": "What is RAG?"})

        assert response.status_code == 200
        mock_retriever.retrieve.assert_called_once()
        mock_checker.check.assert_called_once()

    def test_should_refine_query_and_retrieve_again_when_not_grounded(
        self,
        client: TestClient,
    ) -> None:
        r1 = _result(id_="1")
        r2 = _result(id_="2")
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = [[r1], [r2]]
        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = [[r1], [r2]]
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "answer"
        mock_checker = MagicMock(spec=GroundingChecker)
        mock_checker.check.side_effect = [
            GroundingResult(is_grounded=False, refined_query="refined query"),
            GroundingResult(is_grounded=True, refined_query=None),
        ]
        app.dependency_overrides[deps.get_retriever] = lambda: mock_retriever
        app.dependency_overrides[deps.get_reranker] = lambda: mock_reranker
        app.dependency_overrides[deps.get_llm_provider] = lambda: mock_llm
        app.dependency_overrides[deps.get_compressor] = lambda: None
        app.dependency_overrides[deps.get_config] = lambda: _mock_iterative_config()
        app.dependency_overrides[deps.get_grounding_checker] = lambda: mock_checker

        response = client.post("/ask", json={"question": "What is RAG?"})

        assert response.status_code == 200
        assert mock_retriever.retrieve.call_count == 2
        second_call_query = mock_retriever.retrieve.call_args_list[1][0][0]
        assert second_call_query == "refined query"

    def test_should_stop_at_max_iterations_when_never_grounded(
        self,
        client: TestClient,
    ) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [_result()]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [_result()]
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "answer"
        mock_checker = MagicMock(spec=GroundingChecker)
        mock_checker.check.return_value = GroundingResult(
            is_grounded=False, refined_query="refine again"
        )
        mock_config = MagicMock()
        mock_config.reranking.score_threshold = None
        mock_config.reranking.top_k = None
        mock_config.reranking.enabled = True
        mock_config.iterative_retrieval.max_iterations = 2
        app.dependency_overrides[deps.get_retriever] = lambda: mock_retriever
        app.dependency_overrides[deps.get_reranker] = lambda: mock_reranker
        app.dependency_overrides[deps.get_llm_provider] = lambda: mock_llm
        app.dependency_overrides[deps.get_compressor] = lambda: None
        app.dependency_overrides[deps.get_grounding_checker] = lambda: mock_checker
        app.dependency_overrides[deps.get_config] = lambda: mock_config

        response = client.post("/ask", json={"question": "What is RAG?"})

        assert response.status_code == 200
        assert mock_retriever.retrieve.call_count == 2

    def test_should_deduplicate_results_across_iterations(
        self,
        client: TestClient,
    ) -> None:
        r = _result(id_="dup")
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = [[r], [r]]
        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = [[r], [r]]
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "answer"
        mock_checker = MagicMock(spec=GroundingChecker)
        mock_checker.check.side_effect = [
            GroundingResult(is_grounded=False, refined_query="another query"),
            GroundingResult(is_grounded=True, refined_query=None),
        ]
        app.dependency_overrides[deps.get_retriever] = lambda: mock_retriever
        app.dependency_overrides[deps.get_reranker] = lambda: mock_reranker
        app.dependency_overrides[deps.get_llm_provider] = lambda: mock_llm
        app.dependency_overrides[deps.get_compressor] = lambda: None
        app.dependency_overrides[deps.get_config] = lambda: _mock_iterative_config()
        app.dependency_overrides[deps.get_grounding_checker] = lambda: mock_checker

        response = client.post("/ask", json={"question": "What is RAG?"})

        assert response.status_code == 200
        sources = response.json()["sources"]
        assert len(sources) == 1

    def test_should_return_503_when_llm_raises_on_first_pass(
        self,
        client: TestClient,
    ) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [_result()]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [_result()]
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = LLMUnavailableError("Ollama down")
        app.dependency_overrides[deps.get_retriever] = lambda: mock_retriever
        app.dependency_overrides[deps.get_reranker] = lambda: mock_reranker
        app.dependency_overrides[deps.get_llm_provider] = lambda: mock_llm
        app.dependency_overrides[deps.get_compressor] = lambda: None
        app.dependency_overrides[deps.get_config] = lambda: _mock_config()
        app.dependency_overrides[deps.get_grounding_checker] = lambda: None

        response = client.post("/ask", json={"question": "What is RAG?"})

        assert response.status_code == 503
        assert response.json()["error"] == "llm_unavailable"

    def test_should_stop_when_not_grounded_but_no_refined_query(
        self,
        client: TestClient,
    ) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [_result()]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [_result()]
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "answer"
        mock_checker = MagicMock(spec=GroundingChecker)
        mock_checker.check.return_value = GroundingResult(
            is_grounded=False, refined_query=None
        )
        app.dependency_overrides[deps.get_retriever] = lambda: mock_retriever
        app.dependency_overrides[deps.get_reranker] = lambda: mock_reranker
        app.dependency_overrides[deps.get_llm_provider] = lambda: mock_llm
        app.dependency_overrides[deps.get_compressor] = lambda: None
        app.dependency_overrides[deps.get_config] = lambda: _mock_iterative_config()
        app.dependency_overrides[deps.get_grounding_checker] = lambda: mock_checker

        response = client.post("/ask", json={"question": "What is RAG?"})

        assert response.status_code == 200
        mock_retriever.retrieve.assert_called_once()
