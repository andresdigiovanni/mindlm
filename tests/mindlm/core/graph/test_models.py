import pytest

from mindlm.core.models import Entity, Relationship


class TestEntity:
    def test_entity_is_frozen(self) -> None:
        # Arrange
        e = Entity(
            id="1", name="Apple", type="ORG", description="A company", source_id="c1"
        )

        # Act / Assert
        with pytest.raises(AttributeError):
            e.name = "Other"  # type: ignore[misc]

    def test_entity_equality_by_value(self) -> None:
        # Arrange
        e1 = Entity(
            id="1", name="Apple", type="ORG", description="A company", source_id="c1"
        )
        e2 = Entity(
            id="1", name="Apple", type="ORG", description="A company", source_id="c1"
        )

        # Assert
        assert e1 == e2

    def test_entity_is_hashable(self) -> None:
        # Arrange
        e = Entity(
            id="1", name="Apple", type="ORG", description="A company", source_id="c1"
        )

        # Act / Assert
        _ = {e}  # should not raise


class TestRelationship:
    def test_relationship_stores_weight_as_float(self) -> None:
        # Arrange / Act
        r = Relationship(
            id="r1",
            source_entity_id="e1",
            target_entity_id="e2",
            type="works_at",
            description="works at",
            weight=0.8,
            source_id="c1",
        )

        # Assert
        assert isinstance(r.weight, float)
        assert r.weight == 0.8

    def test_relationship_is_frozen(self) -> None:
        # Arrange
        r = Relationship(
            id="r1",
            source_entity_id="e1",
            target_entity_id="e2",
            type="works_at",
            description="works at",
            weight=0.8,
            source_id="c1",
        )

        # Act / Assert
        with pytest.raises(AttributeError):
            r.type = "other"  # type: ignore[misc]
