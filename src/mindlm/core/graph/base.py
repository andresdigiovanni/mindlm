from abc import ABC, abstractmethod

from mindlm.core.models import Entity, Relationship


class GraphStore(ABC):
    @abstractmethod
    def upsert_entities(self, entities: list[Entity]) -> None: ...

    @abstractmethod
    def upsert_relationships(self, relationships: list[Relationship]) -> None: ...

    @abstractmethod
    def get_related_chunk_ids(self, chunk_ids: list[str], depth: int) -> list[str]: ...

    @abstractmethod
    def delete_by_source(self, source_id: str) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...
