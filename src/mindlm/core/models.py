import bisect
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class Point:
    id: str
    vector: list[float]
    payload: dict[str, Any]
    sparse_vector: SparseVector | None = None


@dataclass(frozen=True)
class Result:
    id: str
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int
    metadata: dict[str, Any]
    start_char: int
    end_char: int


@dataclass(frozen=True)
class ParsedDocument:
    text: str
    page_breaks: list[int]
    # page_breaks[i] = exclusive end char offset of page i.
    # For non-PDF formats: page_breaks = []

    def page_number_for(self, char_offset: int) -> int | None:
        if not self.page_breaks:
            return None
        idx = bisect.bisect_right(self.page_breaks, char_offset)
        return min(idx, len(self.page_breaks) - 1) + 1  # 1-indexed (page 1, 2, ...)


@dataclass(frozen=True)
class DocumentState:
    source: str
    document_hash: str
    modified_at: str
    file_size: int
    ingested_at: str


@dataclass
class SyncResult:
    added: int = 0
    updated: int = 0
    skipped: int = 0
    chunks: int = 0
    errors: list[str] = field(default_factory=list)
