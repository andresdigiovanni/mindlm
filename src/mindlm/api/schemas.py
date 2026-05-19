from typing import Any, Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    paths: list[str]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    collection: str | None = None
    filters: dict[str, str] | None = None
    top_k: int = Field(default=5, gt=0)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    collection: str | None = None
    filters: dict[str, str] | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    services: dict[str, str]


class SearchResultItem(BaseModel):
    content: str
    score: float
    source: str
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]


class SourceRef(BaseModel):
    source: str
    score: float
    chunk_index: int
    page_number: int | None = None
    char_start: int | None = None
    char_end: int | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]


class SyncResponse(BaseModel):
    added: int
    updated: int
    skipped: int
    chunks: int
    errors: list[str]


class ErrorResponse(BaseModel):
    error: str
    message: str
    status_code: int
