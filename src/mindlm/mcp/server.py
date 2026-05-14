import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from mcp import types
from mcp.server import Server

from mindlm.core.chunking.dispatcher import ChunkerDispatcher
from mindlm.core.config.loader import load_config
from mindlm.core.embeddings.huggingface import HuggingFaceEmbeddingProvider
from mindlm.core.exceptions import LLMUnavailableError
from mindlm.core.generation.ollama import OllamaProvider
from mindlm.core.ingestion.pipeline import IngestionPipeline
from mindlm.core.parsing.dispatcher import ParserDispatcher
from mindlm.core.query_processing.dispatcher import QueryProcessorDispatcher
from mindlm.core.reranking.dispatcher import RerankerDispatcher
from mindlm.core.retrieval.retriever import Retriever
from mindlm.core.synchronization.synchronizer import Synchronizer
from mindlm.core.vectorstore.qdrant import QdrantVectorStore


def _build_components() -> tuple[
    Retriever, RerankerDispatcher, OllamaProvider, Synchronizer, QdrantVectorStore
]:
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    config = load_config(Path(config_path))
    embedding_provider = HuggingFaceEmbeddingProvider(config.embeddings)
    vectorstore = QdrantVectorStore(config.vector_store)
    llm = OllamaProvider(config.llm)
    query_processor = QueryProcessorDispatcher(config.query_processing)
    retriever = Retriever(
        config.retrieval,
        vectorstore,
        embedding_provider,
        llm=llm,
        query_processor=query_processor,
        resolve_parents=config.chunking.parent_chunk_size is not None,
    )
    reranker = RerankerDispatcher(config.reranking, embedding_provider)
    parser = ParserDispatcher(config.ingestion)
    chunker = ChunkerDispatcher(config.chunking, embedding_provider)
    pipeline = IngestionPipeline(
        config, parser, chunker, embedding_provider, vectorstore
    )
    synchronizer = Synchronizer(vectorstore, pipeline)
    return retriever, reranker, llm, synchronizer, vectorstore


server: Server = Server("mindlm-rag")
_retriever: Retriever | None = None
_reranker: RerankerDispatcher | None = None
_llm: OllamaProvider | None = None
_synchronizer: Synchronizer | None = None
_vectorstore: QdrantVectorStore | None = None


def _get_components() -> tuple[
    Retriever, RerankerDispatcher, OllamaProvider, Synchronizer, QdrantVectorStore
]:
    global _retriever, _reranker, _llm, _synchronizer, _vectorstore
    if _retriever is None:
        _retriever, _reranker, _llm, _synchronizer, _vectorstore = _build_components()
    assert _retriever is not None
    assert _reranker is not None
    assert _llm is not None
    assert _synchronizer is not None
    assert _vectorstore is not None
    return _retriever, _reranker, _llm, _synchronizer, _vectorstore


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_documents",
            description="Semantic search over indexed documents",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                    "filters": {"type": "object"},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="ask_rag",
            description="RAG question-answering with Ollama generation",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "filters": {"type": "object"},
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="ingest_sync",
            description="Incremental document synchronization",
            inputSchema={
                "type": "object",
                "properties": {"paths": {"type": "array", "items": {"type": "string"}}},
                "required": ["paths"],
            },
        ),
        types.Tool(
            name="ingest_full",
            description="Full document re-ingestion (drops and rebuilds index)",
            inputSchema={
                "type": "object",
                "properties": {"paths": {"type": "array", "items": {"type": "string"}}},
                "required": ["paths"],
            },
        ),
        types.Tool(
            name="list_collections",
            description="List all available Qdrant collections",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    retriever, reranker, llm, synchronizer, vectorstore = _get_components()

    match name:
        case "search_documents":
            results = retriever.retrieve(
                arguments["query"],
                arguments.get("filters"),
            )
            results = reranker.rerank(arguments["query"], results)
            text = json.dumps([r.payload for r in results], ensure_ascii=False)
            return [types.TextContent(type="text", text=text)]

        case "ask_rag":
            results = retriever.retrieve(
                arguments["question"],
                arguments.get("filters"),
            )
            results = reranker.rerank(arguments["question"], results)
            context = "\n\n".join(
                f"[Source: {r.payload.get('source', '')}]\n{r.payload.get('content', '')}"
                for r in results
            )
            system_msg = (
                "You are a helpful assistant. Answer using only the provided context."
            )
            user_msg = f"Context:\n{context}\n\nQuestion: {arguments['question']}"
            try:
                answer = llm.chat(
                    [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ]
                )
                return [types.TextContent(type="text", text=answer)]
            except LLMUnavailableError as exc:
                return [types.TextContent(type="text", text=f"Error: {exc}")]

        case "ingest_sync":
            config_path_env = os.environ.get("CONFIG_PATH", "config.yaml")
            cfg = load_config(Path(config_path_env))
            base = Path(cfg.ingestion.allowed_base_dir).resolve()
            safe_paths: list[Path] = []
            for p in arguments["paths"]:
                resolved = Path(p).resolve()
                try:
                    resolved.relative_to(base)
                    safe_paths.append(resolved)
                except ValueError:
                    return [
                        types.TextContent(
                            type="text",
                            text=f"Error: path outside allowed directory: {p}",
                        )
                    ]
            result = synchronizer.sync(safe_paths)
            return [
                types.TextContent(
                    type="text", text=json.dumps(dataclasses.asdict(result))
                )
            ]

        case "ingest_full":
            config_path_env = os.environ.get("CONFIG_PATH", "config.yaml")
            cfg = load_config(Path(config_path_env))
            base = Path(cfg.ingestion.allowed_base_dir).resolve()
            safe_paths_full: list[Path] = []
            for p in arguments["paths"]:
                resolved = Path(p).resolve()
                try:
                    resolved.relative_to(base)
                    safe_paths_full.append(resolved)
                except ValueError:
                    return [
                        types.TextContent(
                            type="text",
                            text=f"Error: path outside allowed directory: {p}",
                        )
                    ]
            result = synchronizer.full_reingest(
                safe_paths_full,
                collection=cfg.vector_store.collection,
                dense_dim=cfg.embeddings.dimensions,
                sparse=cfg.retrieval.strategy == "hybrid",
            )
            return [
                types.TextContent(
                    type="text", text=json.dumps(dataclasses.asdict(result))
                )
            ]

        case "list_collections":
            collections = vectorstore.list_collections()
            return [types.TextContent(type="text", text=json.dumps(collections))]

        case _:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
