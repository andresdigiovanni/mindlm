import functools
import os
from pathlib import Path

from mindlm.core.chunking.dispatcher import ChunkerDispatcher
from mindlm.core.config.loader import load_config
from mindlm.core.config.models import RAGConfig
from mindlm.core.embeddings.huggingface import HuggingFaceEmbeddingProvider
from mindlm.core.generation.ollama import OllamaProvider
from mindlm.core.ingestion.pipeline import IngestionPipeline
from mindlm.core.parsing.dispatcher import ParserDispatcher
from mindlm.core.query_processing.dispatcher import QueryProcessorDispatcher
from mindlm.core.reranking.dispatcher import RerankerDispatcher
from mindlm.core.retrieval.retriever import Retriever
from mindlm.core.synchronization.synchronizer import Synchronizer
from mindlm.core.vectorstore.qdrant import QdrantVectorStore


@functools.lru_cache(maxsize=1)
def get_config() -> RAGConfig:
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    return load_config(Path(config_path))


@functools.lru_cache(maxsize=1)
def get_embedding_provider() -> HuggingFaceEmbeddingProvider:
    return HuggingFaceEmbeddingProvider(get_config().embeddings)


def get_vectorstore() -> QdrantVectorStore:
    return QdrantVectorStore(get_config().vector_store)


def get_llm_provider() -> OllamaProvider:
    return OllamaProvider(get_config().llm)


def get_retriever() -> Retriever:
    config = get_config()
    return Retriever(
        config.retrieval,
        get_vectorstore(),
        get_embedding_provider(),
        llm=get_llm_provider(),
        query_processor=get_query_processor(),
        resolve_parents=config.chunking.parent_chunk_size is not None,
    )


@functools.lru_cache(maxsize=1)
def get_query_processor() -> QueryProcessorDispatcher:
    return QueryProcessorDispatcher(get_config().query_processing)


def get_reranker() -> RerankerDispatcher:
    config = get_config()
    return RerankerDispatcher(config.reranking, get_embedding_provider())


def get_pipeline() -> IngestionPipeline:
    config = get_config()
    parser = ParserDispatcher(config.ingestion)
    chunker = ChunkerDispatcher(config.chunking, get_embedding_provider())
    return IngestionPipeline(
        config, parser, chunker, get_embedding_provider(), get_vectorstore()
    )


def get_synchronizer() -> Synchronizer:
    return Synchronizer(get_vectorstore(), get_pipeline())
