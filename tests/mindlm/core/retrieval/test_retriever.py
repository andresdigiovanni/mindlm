from unittest.mock import MagicMock

from mindlm.core.config.models import RetrievalConfig
from mindlm.core.models import Result
from mindlm.core.retrieval.retriever import Retriever


def _make_retriever(strategy: str) -> tuple[Retriever, MagicMock, MagicMock]:
    config = RetrievalConfig(strategy=strategy, top_k=5)
    vectorstore = MagicMock()
    vectorstore.search.return_value = [Result(id="1", score=0.9, payload={})]
    vectorstore.search_hybrid.return_value = [Result(id="1", score=0.9, payload={})]
    embedding_provider = MagicMock()
    embedding_provider.embed_one.return_value = [0.1] * 10
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

        ep.embed_one.assert_called_once_with("my query")
