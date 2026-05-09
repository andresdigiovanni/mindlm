import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import mindlm.mcp.server as server_module
from mindlm.core.exceptions import LLMUnavailableError
from mindlm.core.models import Result, SyncResult


@pytest.fixture(autouse=True)
def reset_mcp_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset MCP server global singletons before each test."""
    monkeypatch.setattr(server_module, "_retriever", None)
    monkeypatch.setattr(server_module, "_reranker", None)
    monkeypatch.setattr(server_module, "_llm", None)
    monkeypatch.setattr(server_module, "_synchronizer", None)
    monkeypatch.setattr(server_module, "_vectorstore", None)


@pytest.fixture
def mock_components(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Inject mock components into the MCP server."""
    mock_retriever = MagicMock()
    mock_reranker = MagicMock()
    mock_llm = MagicMock()
    mock_synchronizer = MagicMock()
    mock_vectorstore = MagicMock()

    mock_retriever.retrieve.return_value = [
        Result(id="1", score=0.9, payload={"content": "text", "source": "/doc.md"})
    ]
    mock_reranker.rerank.return_value = [
        Result(id="1", score=0.9, payload={"content": "text", "source": "/doc.md"})
    ]
    mock_llm.chat.return_value = "Generated answer"
    mock_synchronizer.sync.return_value = SyncResult(added=1)
    mock_synchronizer.full_reingest.return_value = SyncResult(added=2)
    mock_vectorstore.list_collections.return_value = ["col_a", "col_b"]

    monkeypatch.setattr(server_module, "_retriever", mock_retriever)
    monkeypatch.setattr(server_module, "_reranker", mock_reranker)
    monkeypatch.setattr(server_module, "_llm", mock_llm)
    monkeypatch.setattr(server_module, "_synchronizer", mock_synchronizer)
    monkeypatch.setattr(server_module, "_vectorstore", mock_vectorstore)

    return {
        "retriever": mock_retriever,
        "reranker": mock_reranker,
        "llm": mock_llm,
        "synchronizer": mock_synchronizer,
        "vectorstore": mock_vectorstore,
    }


class TestListTools:
    @pytest.mark.asyncio
    async def test_list_tools_returns_five(self) -> None:
        tools = await server_module.list_tools()

        assert len(tools) == 5
        names = {t.name for t in tools}
        assert names == {
            "search_documents",
            "ask_rag",
            "ingest_sync",
            "ingest_full",
            "list_collections",
        }


class TestCallToolSearchDocuments:
    @pytest.mark.asyncio
    async def test_search_documents(self, mock_components: dict) -> None:
        result = await server_module.call_tool("search_documents", {"query": "test"})

        assert len(result) == 1
        assert result[0].type == "text"
        data = json.loads(result[0].text)
        assert isinstance(data, list)
        mock_components["retriever"].retrieve.assert_called_once_with("test", None)

    @pytest.mark.asyncio
    async def test_search_documents_with_filters(self, mock_components: dict) -> None:
        filters = {"source": "/doc.md"}

        await server_module.call_tool(
            "search_documents", {"query": "test", "filters": filters}
        )

        mock_components["retriever"].retrieve.assert_called_once_with("test", filters)


@pytest.mark.usefixtures("mock_components")
class TestCallToolAskRag:
    @pytest.mark.asyncio
    async def test_ask_rag_returns_answer(self) -> None:
        result = await server_module.call_tool("ask_rag", {"question": "What is RAG?"})

        assert len(result) == 1
        assert "Generated answer" in result[0].text

    @pytest.mark.asyncio
    async def test_ask_rag_llm_unavailable(self, mock_components: dict) -> None:
        mock_components["llm"].chat.side_effect = LLMUnavailableError("Ollama down")

        result = await server_module.call_tool("ask_rag", {"question": "test?"})

        assert "Error:" in result[0].text


class TestCallToolIngestSync:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_components")
    async def test_ingest_sync(self, tmp_path: Path) -> None:
        with patch("mindlm.mcp.server.load_config") as mock_load_cfg:
            mock_cfg = MagicMock()
            mock_cfg.ingestion.allowed_base_dir = str(tmp_path)
            mock_cfg.vector_store.collection = "docs"
            mock_cfg.embeddings.dimensions = 384
            mock_cfg.retrieval.strategy = "vector"
            mock_load_cfg.return_value = mock_cfg

            doc = tmp_path / "doc.txt"
            doc.write_text("content", encoding="utf-8")

            result = await server_module.call_tool("ingest_sync", {"paths": [str(doc)]})

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "added" in data

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_components")
    async def test_ingest_sync_path_outside_allowed(self, tmp_path: Path) -> None:
        with patch("mindlm.mcp.server.load_config") as mock_load_cfg:
            mock_cfg = MagicMock()
            mock_cfg.ingestion.allowed_base_dir = str(tmp_path / "subdir")
            mock_load_cfg.return_value = mock_cfg

            result = await server_module.call_tool(
                "ingest_sync", {"paths": ["/etc/passwd"]}
            )

        assert any("Error:" in r.text or "outside" in r.text for r in result)


class TestCallToolIngestFull:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_components")
    async def test_ingest_full(self, tmp_path: Path) -> None:
        with patch("mindlm.mcp.server.load_config") as mock_load_cfg:
            mock_cfg = MagicMock()
            mock_cfg.ingestion.allowed_base_dir = str(tmp_path)
            mock_cfg.vector_store.collection = "docs"
            mock_cfg.embeddings.dimensions = 384
            mock_cfg.retrieval.strategy = "hybrid"
            mock_load_cfg.return_value = mock_cfg

            doc = tmp_path / "doc.txt"
            doc.write_text("content", encoding="utf-8")

            result = await server_module.call_tool("ingest_full", {"paths": [str(doc)]})

        data = json.loads(result[0].text)
        assert "added" in data

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_components")
    async def test_ingest_full_path_outside_allowed(self, tmp_path: Path) -> None:
        with patch("mindlm.mcp.server.load_config") as mock_load_cfg:
            mock_cfg = MagicMock()
            mock_cfg.ingestion.allowed_base_dir = str(tmp_path / "subdir")
            mock_load_cfg.return_value = mock_cfg

            result = await server_module.call_tool(
                "ingest_full", {"paths": ["/etc/passwd"]}
            )

        assert any("Error:" in r.text or "outside" in r.text for r in result)


@pytest.mark.usefixtures("mock_components")
class TestCallToolListCollections:
    @pytest.mark.asyncio
    async def test_list_collections(self) -> None:
        result = await server_module.call_tool("list_collections", {})

        data = json.loads(result[0].text)
        assert "col_a" in data
        assert "col_b" in data


@pytest.mark.usefixtures("mock_components")
class TestCallToolUnknown:
    @pytest.mark.asyncio
    async def test_unknown_tool(self) -> None:
        result = await server_module.call_tool("nonexistent_tool", {})

        assert "Unknown tool" in result[0].text
