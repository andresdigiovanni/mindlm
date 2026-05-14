from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AppConfig(BaseModel):
    name: str = "local-rag"


class LLMConfig(BaseModel):
    provider: Literal["ollama"] = "ollama"
    model: str = "llama3"
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
    strategy: Literal["fixed", "semantic", "sliding", "recursive"] = "fixed"
    chunk_size: int = Field(default=500, gt=0)
    overlap: int = Field(default=50, ge=0)
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
        return self


class RetrievalConfig(BaseModel):
    strategy: Literal["vector", "hybrid"] = "vector"
    top_k: int = Field(default=5, gt=0)


class RerankingConfig(BaseModel):
    enabled: bool = False
    method: Literal["cross_encoder", "mmr"] | None = None
    model: str | None = None


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


class QueryProcessingConfig(BaseModel):
    rewriting: QueryRewritingConfig = QueryRewritingConfig()
    expansion: QueryExpansionConfig = QueryExpansionConfig()
    hyde: HyDEConfig = HyDEConfig()
    multi_query: MultiQueryConfig = MultiQueryConfig()
    decomposition: QueryDecompositionConfig = QueryDecompositionConfig()
    step_back: StepBackConfig = StepBackConfig()


class RAGConfig(BaseModel):
    app: AppConfig = AppConfig()
    llm: LLMConfig = LLMConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    ingestion: IngestionConfig = IngestionConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranking: RerankingConfig = RerankingConfig()
    query_processing: QueryProcessingConfig = QueryProcessingConfig()
