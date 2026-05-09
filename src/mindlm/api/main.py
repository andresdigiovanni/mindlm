from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mindlm.api.routers import collections, health, ingest, search
from mindlm.api.schemas import ErrorResponse
from mindlm.core.exceptions import (
    CollectionNotFoundError,
    LLMUnavailableError,
    ParseError,
)

app = FastAPI(title="MindLM RAG API", version="0.1.0")


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
