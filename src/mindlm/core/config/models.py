from typing import Literal

from pydantic import BaseModel, Field, model_validator


class LLMConfig(BaseModel):
    provider: Literal["ollama"] = "ollama"
    model: str = "gemma4"
    base_url: str = "http://ollama:11434"
    temperature: float = 0.7
    max_tokens: int = Field(default=1024, gt=0)


class EmbeddingsConfig(BaseModel):
    provider: Literal["huggingface"] = "huggingface"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: int = Field(default=384, gt=0)


class VectorStoreConfig(BaseModel):
    provider: Literal["qdrant"] = "qdrant"
    mode: Literal["local", "cloud"] = "local"
    host: str = "qdrant"
    port: int = 6333
    collection: str = "documents"
    api_key: str | None = None


class IngestionConfig(BaseModel):
    source_type: list[
        Literal["pdf", "html", "markdown", "png", "jpeg", "pptx", "docx"]
    ] = ["pdf", "html", "markdown", "png", "jpeg", "pptx", "docx"]
    parsing_strategy: Literal["raw", "structured", "ocr"] = "structured"
    deduplication: bool = True
    allowed_base_dir: str = "/data"


class ChunkingConfig(BaseModel):
    strategy: Literal[
        "fixed", "semantic", "sliding", "recursive", "sentence_window"
    ] = "fixed"
    chunk_size: int = Field(default=500, gt=0)
    overlap: int = Field(default=50, ge=0)
    window_size: int = Field(default=2, ge=1)
    semantic_model: str | None = None
    parent_chunk_size: int | None = Field(default=None, gt=0)
    separators: list[str] = Field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])

    @model_validator(mode="after")
    def _check_config(self) -> "ChunkingConfig":
        if self.strategy == "semantic" and self.semantic_model is None:
            raise ValueError("semantic_model required when strategy is semantic")
        if (
            self.parent_chunk_size is not None
            and self.parent_chunk_size <= self.chunk_size
        ):
            raise ValueError("parent_chunk_size must be greater than chunk_size")
        if self.strategy == "recursive" and len(self.separators) == 0:
            raise ValueError("separators must not be empty when strategy is recursive")
        if self.parent_chunk_size is not None and self.strategy == "sentence_window":
            raise ValueError(
                "parent_chunk_size cannot be used with strategy='sentence_window'"
            )
        if self.strategy == "sliding" and self.overlap >= self.chunk_size:
            raise ValueError(
                "overlap must be less than chunk_size when strategy is sliding"
            )
        return self


class RetrievalConfig(BaseModel):
    strategy: Literal["vector", "hybrid"] = "vector"
    top_k: int = Field(default=5, gt=0)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class RerankingConfig(BaseModel):
    enabled: bool = False
    method: Literal["cross_encoder", "mmr", "llm"] | None = None
    model: str | None = None

    @model_validator(mode="after")
    def _check_config(self) -> "RerankingConfig":
        if self.enabled and self.method is None:
            raise ValueError("method is required when reranking is enabled")
        return self


class ContextualRetrievalConfig(BaseModel):
    chunk_context_enabled: bool = False
    document_summary_enabled: bool = False
    prompt_template: str = (
        "Here is the full document:\n<document>\n{document}\n</document>\n\n"
        "Here is a chunk from the document:\n<chunk>\n{chunk}\n</chunk>\n\n"
        "Provide a brief, one-sentence context that situates this chunk within the "
        "overall document. Answer only with the context sentence, no additional text."
    )
    document_summary_prompt_template: str = (
        "Here is a document:\n<document>\n{document}\n</document>\n\n"
        "Provide a single-sentence summary of the document's main topic. "
        "Answer only with the summary sentence, no additional text."
    )


class QueryRewritingConfig(BaseModel):
    enabled: bool = False


class QueryExpansionConfig(BaseModel):
    enabled: bool = False


class HyDEConfig(BaseModel):
    enabled: bool = False


class MultiQueryConfig(BaseModel):
    enabled: bool = False
    num_variants: int = Field(default=3, ge=2, le=10)


class QueryDecompositionConfig(BaseModel):
    enabled: bool = False
    max_subqueries: int = Field(default=4, ge=2, le=10)


class StepBackConfig(BaseModel):
    enabled: bool = False


class QueryPlannerConfig(BaseModel):
    enabled: bool = False


class QueryProcessingConfig(BaseModel):
    rewriting: QueryRewritingConfig = QueryRewritingConfig()
    expansion: QueryExpansionConfig = QueryExpansionConfig()
    hyde: HyDEConfig = HyDEConfig()
    multi_query: MultiQueryConfig = MultiQueryConfig()
    decomposition: QueryDecompositionConfig = QueryDecompositionConfig()
    step_back: StepBackConfig = StepBackConfig()
    planner: QueryPlannerConfig = QueryPlannerConfig()


class ObservabilityConfig(BaseModel):
    enabled: bool = True
    public_key: str = "pk-lf-local-dev"
    secret_key: str = "sk-lf-local-dev"  # noqa: S105
    host: str = "http://langfuse:3000"
    flush_at: int = Field(default=15, gt=0)
    flush_interval: float = Field(default=0.5, gt=0)


class GraphStoreConfig(BaseModel):
    provider: Literal["neo4j"] = "neo4j"
    host: str = "neo4j"
    port: int = 7687
    username: str = "neo4j"
    password: str = "neo4j_password"  # noqa: S105


class GraphRAGConfig(BaseModel):
    enabled: bool = False
    store: GraphStoreConfig = GraphStoreConfig()


class CompressionConfig(BaseModel):
    enabled: bool = False


class RAGConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    ingestion: IngestionConfig = IngestionConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranking: RerankingConfig = RerankingConfig()
    contextual_retrieval: ContextualRetrievalConfig = Field(
        default_factory=ContextualRetrievalConfig
    )
    query_processing: QueryProcessingConfig = QueryProcessingConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    graph_rag: GraphRAGConfig = GraphRAGConfig()
    compression: CompressionConfig = CompressionConfig()

    @model_validator(mode="after")
    def _check_semantic_model_consistency(self) -> "RAGConfig":
        if (
            self.chunking.strategy == "semantic"
            and self.chunking.semantic_model != self.embeddings.model
        ):
            raise ValueError(
                f"chunking.semantic_model ({self.chunking.semantic_model!r}) "
                f"must match embeddings.model ({self.embeddings.model!r})"
            )
        return self
