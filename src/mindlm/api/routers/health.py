from typing import Literal

from fastapi import APIRouter, Depends

from mindlm.api.dependencies import (
    get_embedding_provider,
    get_llm_provider,
    get_vectorstore,
)
from mindlm.api.schemas import HealthResponse
from mindlm.core.embeddings.base import EmbeddingProvider
from mindlm.core.generation.base import LLMProvider
from mindlm.core.vectorstore.base import VectorStore

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def get_health(
    vectorstore: VectorStore = Depends(get_vectorstore),
    llm: LLMProvider = Depends(get_llm_provider),
    emb: EmbeddingProvider = Depends(get_embedding_provider),
) -> HealthResponse:
    services: dict[str, str] = {}

    try:
        vectorstore.list_collections()
        services["qdrant"] = "ok"
    except Exception:
        services["qdrant"] = "error"

    services["ollama"] = "ok" if llm.healthcheck() else "error"

    try:
        emb.embed_one("ping")
        services["embeddings"] = "ok"
    except Exception:
        services["embeddings"] = "error"

    status: Literal["ok", "degraded", "error"]
    if all(v == "ok" for v in services.values()):
        status = "ok"
    elif services.get("ollama") == "error" and services.get("qdrant") == "ok":
        status = "degraded"
    else:
        status = "error"

    return HealthResponse(status=status, services=services)
