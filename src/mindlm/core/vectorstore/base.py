import contextlib
from abc import ABC, abstractmethod

from mindlm.core.models import Point, Result, SparseVector


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, points: list[Point]) -> None: ...

    @abstractmethod
    def search(
        self, query_vector: list[float], top_k: int, filters: dict | None
    ) -> list[Result]: ...

    @abstractmethod
    def search_hybrid(
        self,
        dense: list[float],
        sparse: SparseVector,
        top_k: int,
        filters: dict | None,
    ) -> list[Result]: ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None: ...

    @abstractmethod
    def delete_by_filter(self, filters: dict) -> None: ...

    @abstractmethod
    def get_by_id(self, id: str) -> Point | None: ...

    @abstractmethod
    def scroll(
        self, filters: dict, limit: int, offset: str | None
    ) -> tuple[list[Point], str | None]: ...

    @abstractmethod
    def create_collection(self, name: str, dense_dim: int, sparse: bool) -> None: ...

    @abstractmethod
    def recreate_collection(self, name: str, dense_dim: int, sparse: bool) -> None: ...

    def ensure_collection(self, name: str, dense_dim: int, sparse: bool) -> None:
        """Create the collection only if it does not already exist."""
        with contextlib.suppress(Exception):
            self.create_collection(name, dense_dim, sparse)

    @abstractmethod
    def list_collections(self) -> list[str]: ...
