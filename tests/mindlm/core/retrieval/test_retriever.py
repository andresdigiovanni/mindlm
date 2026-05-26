from unittest.mock import MagicMock

import pytest

from mindlm.core.config.models import RetrievalConfig
from mindlm.core.models import Result
from mindlm.core.retrieval.retriever import Retriever


def _make_retriever(strategy: str) -> tuple[Retriever, MagicMock, MagicMock]:
    config = RetrievalConfig(strategy=strategy, top_k=5)
    vectorstore = MagicMock()
    vectorstore.search.return_value = [Result(id="1", score=0.9, payload={})]
    vectorstore.search_hybrid.return_value = [Result(id="1", score=0.9, payload={})]
    embedding_provider = MagicMock()
    embedding_provider.embed.return_value = [[0.1] * 10]
    retriever = Retriever(config, vectorstore, embedding_provider)
    return retriever, vectorstore, embedding_provider


class TestRetriever:
    def test_vector_strategy_calls_search(self) -> None:
        retriever, vs, _ep = _make_retriever("vector")

        retriever.retrieve("test query")

        vs.search.assert_called_once()
        vs.search_hybrid.assert_not_called()

    def test_hybrid_strategy_calls_search_hybrid(self) -> None:
        retriever, vs, _ep = _make_retriever("hybrid")

        # Mock BM25
        retriever._bm25 = MagicMock()
        mock_result = MagicMock()
        mock_result.indices.tolist.return_value = [0, 1]
        mock_result.values.tolist.return_value = [0.5, 0.3]
        retriever._bm25.query_embed.return_value = iter([mock_result])

        retriever.retrieve("test query")

        vs.search_hybrid.assert_called_once()
        vs.search.assert_not_called()

    def test_retrieve_embeds_query(self) -> None:
        retriever, _vs, ep = _make_retriever("vector")

        retriever.retrieve("my query")

        ep.embed.assert_called_once_with(["my query"])


class TestRetrieverParentDocResolution:
    def _make_retriever_with_results(
        self, results: list[Result], resolve_parents: bool = False
    ) -> Retriever:
        config = RetrievalConfig(strategy="vector", top_k=5)
        vectorstore = MagicMock()
        vectorstore.search.return_value = results
        ep = MagicMock()
        ep.embed.return_value = [[0.1] * 10]
        return Retriever(config, vectorstore, ep, resolve_parents=resolve_parents)

    def test_no_modification_when_resolve_parents_false(self) -> None:
        results = [Result(id="1", score=0.9, payload={"parent_content": "parent"})]
        retriever = self._make_retriever_with_results(results, resolve_parents=False)

        output = retriever.retrieve("q")

        assert output[0].payload.get("content") is None

    def test_replaces_content_with_parent_content(self) -> None:
        results = [
            Result(
                id="1",
                score=0.9,
                payload={"content": "child", "parent_content": "parent text"},
            )
        ]
        retriever = self._make_retriever_with_results(results, resolve_parents=True)

        output = retriever.retrieve("q")

        assert output[0].payload["content"] == "parent text"

    def test_leaves_content_unchanged_when_no_parent_content(self) -> None:
        results = [Result(id="1", score=0.9, payload={"content": "child text"})]
        retriever = self._make_retriever_with_results(results, resolve_parents=True)

        output = retriever.retrieve("q")

        assert output[0].payload["content"] == "child text"

    def test_preserves_other_payload_fields(self) -> None:
        results = [
            Result(
                id="1",
                score=0.9,
                payload={"source": "doc.md", "parent_content": "parent"},
            )
        ]
        retriever = self._make_retriever_with_results(results, resolve_parents=True)

        output = retriever.retrieve("q")

        assert output[0].payload["source"] == "doc.md"

    def test_handles_empty_results(self) -> None:
        retriever = self._make_retriever_with_results([], resolve_parents=True)

        output = retriever.retrieve("q")

        assert output == []


class TestRetrieverQueryProcessing:
    def _make_retriever_with_processor(
        self,
        queries_returned: list[str],
        search_results: list[Result] | None = None,
        top_k: int = 5,
    ) -> tuple[Retriever, MagicMock]:
        config = RetrievalConfig(strategy="vector", top_k=top_k)
        vectorstore = MagicMock()
        vectorstore.search.return_value = (
            search_results if search_results is not None else []
        )
        ep = MagicMock()
        ep.embed.return_value = [[0.1] * 10]
        llm = MagicMock()
        processor = MagicMock()
        processor.process.return_value = queries_returned
        retriever = Retriever(
            config, vectorstore, ep, llm=llm, query_processor=processor
        )
        return retriever, vectorstore

    def test_retrieves_once_with_original_when_no_processor(self) -> None:
        config = RetrievalConfig(strategy="vector", top_k=5)
        vs = MagicMock()
        vs.search.return_value = [Result(id="1", score=0.9, payload={})]
        ep = MagicMock()
        ep.embed.return_value = [[0.1] * 10]
        retriever = Retriever(config, vs, ep)

        retriever.retrieve("query")

        assert vs.search.call_count == 1

    def test_retrieves_for_each_query_from_processor(self) -> None:
        retriever, vs = self._make_retriever_with_processor(
            queries_returned=["q1", "q2", "q3"]
        )

        retriever.retrieve("original")

        assert vs.search.call_count == 3

    def test_deduplicates_by_id(self) -> None:
        duplicate = Result(id="dup", score=0.9, payload={})
        vs = MagicMock()
        vs.search.return_value = [duplicate]
        config = RetrievalConfig(strategy="vector", top_k=5)
        ep = MagicMock()
        ep.embed.return_value = [[0.1] * 10]
        llm = MagicMock()
        processor = MagicMock()
        processor.process.return_value = ["q1", "q2"]
        retriever = Retriever(config, vs, ep, llm=llm, query_processor=processor)

        results = retriever.retrieve("original")

        assert len(results) == 1
        assert results[0].id == "dup"

    def test_keeps_highest_score_on_id_collision(self) -> None:
        low = Result(id="x", score=0.5, payload={})
        high = Result(id="x", score=0.95, payload={})
        vs = MagicMock()
        vs.search.side_effect = [[low], [high]]
        config = RetrievalConfig(strategy="vector", top_k=5)
        ep = MagicMock()
        ep.embed.return_value = [[0.1] * 10]
        llm = MagicMock()
        processor = MagicMock()
        processor.process.return_value = ["q1", "q2"]
        retriever = Retriever(config, vs, ep, llm=llm, query_processor=processor)

        results = retriever.retrieve("original")

        assert results[0].score == 0.95

    def test_returns_at_most_top_k_results(self) -> None:
        many = [Result(id=str(i), score=float(i) / 10, payload={}) for i in range(10)]
        retriever, _vs = self._make_retriever_with_processor(
            queries_returned=["q"], search_results=many, top_k=3
        )

        results = retriever.retrieve("q")

        assert len(results) == 3

    def test_sorts_results_descending_by_score(self) -> None:
        results_data = [
            Result(id="a", score=0.3, payload={}),
            Result(id="b", score=0.9, payload={}),
            Result(id="c", score=0.6, payload={}),
        ]
        retriever, _vs = self._make_retriever_with_processor(
            queries_returned=["q"], search_results=results_data
        )

        results = retriever.retrieve("q")

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_raises_on_processor_without_llm(self) -> None:
        config = RetrievalConfig(strategy="vector", top_k=5)
        vs = MagicMock()
        ep = MagicMock()
        processor = MagicMock()

        with pytest.raises(ValueError, match="llm is required"):
            Retriever(config, vs, ep, query_processor=processor)

    def test_resolve_parents_and_windows_raises(self) -> None:
        config = RetrievalConfig(strategy="vector", top_k=5)
        vs = MagicMock()
        ep = MagicMock()
        with pytest.raises(ValueError, match="mutually exclusive"):
            Retriever(config, vs, ep, resolve_parents=True, resolve_windows=True)

    def test_handles_empty_results_from_all_queries(self) -> None:
        retriever, _vs = self._make_retriever_with_processor(
            queries_returned=["q1", "q2"], search_results=[]
        )

        results = retriever.retrieve("original")

        assert results == []


class TestRetrieverWindowResolution:
    def _make_retriever_with_results(
        self, results: list[Result], resolve_windows: bool = False
    ) -> Retriever:
        config = RetrievalConfig(strategy="vector", top_k=5)
        vectorstore = MagicMock()
        vectorstore.search.return_value = results
        ep = MagicMock()
        ep.embed.return_value = [[0.1] * 10]
        return Retriever(config, vectorstore, ep, resolve_windows=resolve_windows)

    def test_no_modification_when_resolve_windows_false(self) -> None:
        results = [Result(id="1", score=0.9, payload={"window_context": "window"})]
        retriever = self._make_retriever_with_results(results, resolve_windows=False)
        output = retriever.retrieve("q")
        assert output[0].payload.get("content") is None

    def test_replaces_content_with_window_context(self) -> None:
        results = [
            Result(
                id="1",
                score=0.9,
                payload={
                    "content": "single sentence",
                    "window_context": "A. single sentence. B.",
                },
            )
        ]
        retriever = self._make_retriever_with_results(results, resolve_windows=True)
        output = retriever.retrieve("q")
        assert output[0].payload["content"] == "A. single sentence. B."

    def test_leaves_content_unchanged_when_no_window_context(self) -> None:
        results = [Result(id="1", score=0.9, payload={"content": "text"})]
        retriever = self._make_retriever_with_results(results, resolve_windows=True)
        output = retriever.retrieve("q")
        assert output[0].payload["content"] == "text"

    def test_preserves_other_payload_fields(self) -> None:
        results = [
            Result(
                id="1", score=0.9, payload={"source": "doc.md", "window_context": "ctx"}
            )
        ]
        retriever = self._make_retriever_with_results(results, resolve_windows=True)
        output = retriever.retrieve("q")
        assert output[0].payload["source"] == "doc.md"

    def test_handles_empty_results(self) -> None:
        retriever = self._make_retriever_with_results([], resolve_windows=True)
        output = retriever.retrieve("q")
        assert output == []


class TestRetrieverGraphExpansion:
    def _make_retriever_with_graph(
        self,
        results: list[Result],
        graph_store: MagicMock | None = None,
        top_k: int = 5,
    ) -> tuple[Retriever, MagicMock]:
        config = RetrievalConfig(strategy="vector", top_k=top_k)
        vectorstore = MagicMock()
        vectorstore.search.return_value = results
        ep = MagicMock()
        ep.embed.return_value = [[0.1] * 10]
        retriever = Retriever(config, vectorstore, ep, graph_store=graph_store)
        return retriever, vectorstore

    def test_no_graph_store_passthrough(self) -> None:
        # Arrange
        results = [Result(id="1", score=0.9, payload={"content": "a"})]
        retriever, _vs = self._make_retriever_with_graph(results, graph_store=None)

        # Act
        output = retriever.retrieve("q")

        # Assert — results unchanged
        assert len(output) == 1
        assert output[0].id == "1"

    def test_expand_adds_related_chunk(self) -> None:
        # Arrange
        from mindlm.core.models import Point

        graph_store = MagicMock()
        graph_store.get_related_chunk_ids.return_value = ["1", "2"]  # "2" is new
        related_point = Point(id="2", vector=[], payload={"content": "related"})

        results = [Result(id="1", score=0.8, payload={})]
        retriever, vs = self._make_retriever_with_graph(
            results, graph_store=graph_store
        )
        vs.get_by_id.return_value = related_point

        # Act
        output = retriever.retrieve("q")

        # Assert — related chunk appended with halved score
        ids = {r.id for r in output}
        assert "1" in ids
        assert "2" in ids
        related = next(r for r in output if r.id == "2")
        assert related.score == pytest.approx(0.4)

    def test_expand_no_duplicates(self) -> None:
        # Arrange
        graph_store = MagicMock()
        # related IDs are all already in results
        graph_store.get_related_chunk_ids.return_value = ["1"]

        results = [Result(id="1", score=0.9, payload={})]
        retriever, _vs = self._make_retriever_with_graph(
            results, graph_store=graph_store
        )

        # Act
        output = retriever.retrieve("q")

        # Assert — no duplicates
        assert len(output) == 1

    def test_expand_get_by_id_returns_none_skipped(self) -> None:
        # Arrange
        graph_store = MagicMock()
        graph_store.get_related_chunk_ids.return_value = ["1", "missing"]

        results = [Result(id="1", score=0.8, payload={})]
        retriever, vs = self._make_retriever_with_graph(
            results, graph_store=graph_store
        )
        vs.get_by_id.return_value = None  # not found

        # Act
        output = retriever.retrieve("q")

        # Assert — "missing" silently skipped
        assert len(output) == 1
        assert output[0].id == "1"

    def test_expand_respects_top_k(self) -> None:
        # Arrange
        from mindlm.core.models import Point

        graph_store = MagicMock()
        # 3 existing + 5 related = 8 total, top_k=4
        existing = [
            Result(id=str(i), score=0.9 - i * 0.1, payload={}) for i in range(3)
        ]
        graph_store.get_related_chunk_ids.return_value = [str(i) for i in range(3, 8)]

        retriever, vs = self._make_retriever_with_graph(
            existing, graph_store=graph_store, top_k=4
        )
        vs.get_by_id.side_effect = lambda id_: Point(id=id_, vector=[], payload={})

        # Act
        output = retriever.retrieve("q")

        # Assert
        assert len(output) == 4


class TestRetrieverTopKOverride:
    def _make_retriever(self, results: list[Result], top_k: int = 10) -> Retriever:
        config = RetrievalConfig(strategy="vector", top_k=top_k)
        vs = MagicMock()
        vs.search.return_value = results
        ep = MagicMock()
        ep.embed.return_value = [[0.1] * 10]
        return Retriever(config, vs, ep)

    def test_retrieve_returns_all_within_top_k(self) -> None:
        results = [
            Result(id="1", score=0.9, payload={}),
            Result(id="2", score=0.3, payload={}),
        ]
        retriever = self._make_retriever(results)

        output = retriever.retrieve("q")

        assert len(output) == 2

    def test_request_top_k_overrides_config(self) -> None:
        results = [Result(id=str(i), score=float(i) / 10, payload={}) for i in range(8)]
        retriever = self._make_retriever(results, top_k=5)

        output = retriever.retrieve("q", top_k=3)

        assert len(output) == 3
