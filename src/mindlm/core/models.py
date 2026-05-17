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


@dataclass(frozen=True)
class Entity:
    id: str
    name: str
    type: str
    description: str
    source_id: str  # chunk ID (Point.id) that produced this entity


@dataclass(frozen=True)
class Relationship:
    id: str
    source_entity_id: str
    target_entity_id: str
    type: str
    description: str
    weight: float  # 0.0-1.0
    source_id: str  # chunk ID that produced this relationship
