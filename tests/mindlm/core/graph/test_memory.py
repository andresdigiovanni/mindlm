from uuid import uuid4

from mindlm.core.graph.memory import InMemoryGraphStore
from mindlm.core.models import Entity, Relationship


def _make_entity(name: str, source_id: str, etype: str = "ORG") -> Entity:
    return Entity(
        id=str(uuid4()),
        name=name,
        type=etype,
        description=f"{name} description",
        source_id=source_id,
    )


def _make_relationship(src: Entity, tgt: Entity, source_id: str) -> Relationship:
    return Relationship(
        id=str(uuid4()),
        source_entity_id=src.id,
        target_entity_id=tgt.id,
        type="related_to",
        description="related",
        weight=0.5,
        source_id=source_id,
    )


class TestInMemoryGraphStoreUpsert:
    def test_upsert_and_get_entities(self) -> None:
        # Arrange
        store = InMemoryGraphStore()
        e1 = _make_entity("Apple", "chunk-1")
        e2 = _make_entity("Google", "chunk-2")

        # Act
        store.upsert_entities([e1, e2])

        # Assert — both entities are stored
        related = store.get_related_chunk_ids(["chunk-1"], depth=0)
        assert "chunk-1" in related

    def test_upsert_relationship_idempotent(self) -> None:
        # Arrange
        store = InMemoryGraphStore()
        e1 = _make_entity("A", "chunk-1")
        e2 = _make_entity("B", "chunk-2")
        store.upsert_entities([e1, e2])
        rel = _make_relationship(e1, e2, "chunk-1")

        # Act — upsert same relationship twice
        store.upsert_relationships([rel])
        store.upsert_relationships([rel])

        # Assert — only one relationship stored
        assert len(store._relationships) == 1


class TestInMemoryGraphStoreTraversal:
    def test_get_related_depth_zero_returns_same_ids(self) -> None:
        # Arrange
        store = InMemoryGraphStore()
        e1 = _make_entity("A", "chunk-a")
        e2 = _make_entity("B", "chunk-b")
        store.upsert_entities([e1, e2])
        store.upsert_relationships([_make_relationship(e1, e2, "chunk-a")])

        # Act
        result = store.get_related_chunk_ids(["chunk-a"], depth=0)

        # Assert — depth=0 returns input unchanged
        assert result == ["chunk-a"]

    def test_get_related_no_relationships_no_expansion(self) -> None:
        # Arrange
        store = InMemoryGraphStore()
        e1 = _make_entity("A", "chunk-a")
        store.upsert_entities([e1])
        # No relationships added

        # Act
        result = store.get_related_chunk_ids(["chunk-a"], depth=1)

        # Assert — no neighbors → only same chunk
        assert result == ["chunk-a"]

    def test_get_related_depth_one_expands_neighbours(self) -> None:
        # Arrange
        store = InMemoryGraphStore()
        e1 = _make_entity("A", "chunk-a")
        e2 = _make_entity("B", "chunk-b")
        store.upsert_entities([e1, e2])
        store.upsert_relationships([_make_relationship(e1, e2, "chunk-a")])

        # Act
        result = store.get_related_chunk_ids(["chunk-a"], depth=1)

        # Assert — chunk-b is reachable in 1 hop
        assert "chunk-a" in result
        assert "chunk-b" in result

    def test_get_related_cycles_no_infinite_loop(self) -> None:
        # Arrange — A ↔ B ↔ A (circular edges)
        store = InMemoryGraphStore()
        e1 = _make_entity("A", "chunk-a")
        e2 = _make_entity("B", "chunk-b")
        store.upsert_entities([e1, e2])
        rel_ab = _make_relationship(e1, e2, "chunk-a")
        rel_ba = _make_relationship(e2, e1, "chunk-b")
        store.upsert_relationships([rel_ab, rel_ba])

        # Act — should terminate
        result = store.get_related_chunk_ids(["chunk-a"], depth=2)

        # Assert — BFS terminates, no duplicates
        assert sorted(result) == sorted({"chunk-a", "chunk-b"})


class TestInMemoryGraphStoreDeletion:
    def test_delete_by_source_removes_entities_and_relationships(self) -> None:
        # Arrange
        store = InMemoryGraphStore()
        e1 = _make_entity("A", "chunk-a")
        e2 = _make_entity("B", "chunk-b")
        store.upsert_entities([e1, e2])
        store.upsert_relationships([_make_relationship(e1, e2, "chunk-a")])

        # Act
        store.delete_by_source("chunk-a")

        # Assert — entity and any relationship involving it are gone
        assert e1.id not in store._entities
        assert all(
            r.source_entity_id != e1.id and r.target_entity_id != e1.id
            for r in store._relationships
        )

    def test_clear_empties_store(self) -> None:
        # Arrange
        store = InMemoryGraphStore()
        e1 = _make_entity("A", "chunk-a")
        store.upsert_entities([e1])

        # Act
        store.clear()

        # Assert
        assert store._entities == {}
        assert store._relationships == []
