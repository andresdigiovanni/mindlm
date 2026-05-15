import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mindlm.api.dependencies import get_embedding_provider, get_llm_provider
from mindlm.api.routers import collections, health, ingest, search
from mindlm.api.schemas import ErrorResponse
from mindlm.core.exceptions import (
    CollectionNotFoundError,
    LLMUnavailableError,
    ParseError,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("Pre-warming embedding model...")
    get_embedding_provider()
    logger.info("Embedding model ready.")
    logger.info("Ensuring Ollama model is available...")
    get_llm_provider().ensure_model()
    yield


app = FastAPI(title="MindLM RAG API", version="0.1.0", lifespan=lifespan)


@app.exception_handler(LLMUnavailableError)
async def llm_unavailable_handler(
    _request: Request, exc: LLMUnavailableError
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            error="llm_unavailable",
            message=str(exc),
            status_code=503,
        ).model_dump(),
    )


@app.exception_handler(ParseError)
async def parse_error_handler(_request: Request, exc: ParseError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="parse_error",
            message=str(exc),
            status_code=422,
        ).model_dump(),
    )


@app.exception_handler(CollectionNotFoundError)
async def not_found_handler(
    _request: Request, exc: CollectionNotFoundError
) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error="collection_not_found",
            message=str(exc),
            status_code=404,
        ).model_dump(),
    )


app.include_router(health.router)
app.include_router(collections.router)
app.include_router(ingest.router)
app.include_router(search.router)
