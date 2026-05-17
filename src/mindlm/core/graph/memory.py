from collections import deque

from mindlm.core.graph.base import GraphStore
from mindlm.core.models import Entity, Relationship


class InMemoryGraphStore(GraphStore):
    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._relationships: list[Relationship] = []

    def upsert_entities(self, entities: list[Entity]) -> None:
        for e in entities:
            self._entities[e.id] = e

    def upsert_relationships(self, relationships: list[Relationship]) -> None:
        existing_ids = {r.id for r in self._relationships}
        for r in relationships:
            if r.id not in existing_ids:
                self._relationships.append(r)
                existing_ids.add(r.id)
            else:
                self._relationships = [
                    r if existing.id == r.id else existing
                    for existing in self._relationships
                ]

    def get_related_chunk_ids(self, chunk_ids: list[str], depth: int) -> list[str]:
        seed_entity_ids = {
            e.id for e in self._entities.values() if e.source_id in chunk_ids
        }
        if depth == 0 or not seed_entity_ids:
            return list(chunk_ids)

        adjacency: dict[str, set[str]] = {}
        for r in self._relationships:
            adjacency.setdefault(r.source_entity_id, set()).add(r.target_entity_id)
            adjacency.setdefault(r.target_entity_id, set()).add(r.source_entity_id)

        visited: set[str] = set(seed_entity_ids)
        frontier: deque[tuple[str, int]] = deque((eid, 0) for eid in seed_entity_ids)
        while frontier:
            eid, current_depth = frontier.popleft()
            if current_depth >= depth:
                continue
            for neighbor in adjacency.get(eid, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append((neighbor, current_depth + 1))

        result_chunk_ids = {
            entity.source_id
            for eid in visited
            if (entity := self._entities.get(eid)) is not None
        }
        return list(result_chunk_ids)

    def delete_by_source(self, source_id: str) -> None:
        deleted_entity_ids = {
            e.id for e in self._entities.values() if e.source_id == source_id
        }
        for eid in deleted_entity_ids:
            del self._entities[eid]
        self._relationships = [
            r
            for r in self._relationships
            if r.source_id != source_id
            and r.source_entity_id not in deleted_entity_ids
            and r.target_entity_id not in deleted_entity_ids
        ]

    def clear(self) -> None:
        self._entities.clear()
        self._relationships.clear()
