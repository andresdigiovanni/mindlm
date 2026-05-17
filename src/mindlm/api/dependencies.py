import functools
import os
from pathlib import Path

from mindlm.core.chunking.dispatcher import ChunkerDispatcher
from mindlm.core.config.loader import load_config
from mindlm.core.config.models import RAGConfig
from mindlm.core.embeddings.huggingface import HuggingFaceEmbeddingProvider
from mindlm.core.generation.ollama import OllamaProvider
from mindlm.core.graph.base import GraphStore
from mindlm.core.graph.dispatcher import build_graph_store
from mindlm.core.graph.extractor import EntityExtractor
from mindlm.core.ingestion.contextualizer import Contextualizer
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
        # These two flags are mutually exclusive; ChunkingConfig._check_config
        # prevents parent_chunk_size + strategy='sentence_window' at config load time.
        resolve_parents=config.chunking.parent_chunk_size is not None,
        resolve_windows=config.chunking.strategy == "sentence_window",
        graph_store=get_graph_store(),
    )


@functools.lru_cache(maxsize=1)
def get_query_processor() -> QueryProcessorDispatcher:
    return QueryProcessorDispatcher(get_config().query_processing)


def get_reranker() -> RerankerDispatcher:
    config = get_config()
    return RerankerDispatcher(
        config.reranking, get_embedding_provider(), llm=get_llm_provider()
    )


def get_contextualizer() -> Contextualizer | None:
    config = get_config()
    if not config.contextual_retrieval.enabled:
        return None
    return Contextualizer(config.contextual_retrieval, get_llm_provider())


def get_graph_store() -> GraphStore | None:
    return build_graph_store(get_config().graph_rag)


def get_entity_extractor() -> EntityExtractor | None:
    config = get_config()
    if not config.graph_rag.enabled:
        return None
    return EntityExtractor(get_llm_provider())


def get_pipeline() -> IngestionPipeline:
    config = get_config()
    parser = ParserDispatcher(config.ingestion)
    chunker = ChunkerDispatcher(config.chunking, get_embedding_provider())
    return IngestionPipeline(
        config,
        parser,
        chunker,
        get_embedding_provider(),
        get_vectorstore(),
        contextualizer=get_contextualizer(),
        entity_extractor=get_entity_extractor(),
        graph_store=get_graph_store(),
    )


def get_synchronizer() -> Synchronizer:
    config = get_config()
    return Synchronizer(get_vectorstore(), get_pipeline(), config.ingestion.source_type)
