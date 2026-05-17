import json
from unittest.mock import MagicMock

from mindlm.core.graph.extractor import EntityExtractor


def _make_extractor() -> tuple[EntityExtractor, MagicMock]:
    llm = MagicMock()
    return EntityExtractor(llm), llm


class TestEntityExtractorValidResponse:
    def test_extract_valid_response_returns_entities_and_relationships(self) -> None:
        # Arrange
        extractor, llm = _make_extractor()
        payload = {
            "entities": [
                {"name": "Apple", "type": "ORG", "description": "A tech company"},
                {"name": "Tim Cook", "type": "PERSON", "description": "CEO"},
            ],
            "relationships": [
                {
                    "source": "Tim Cook",
                    "target": "Apple",
                    "type": "works_at",
                    "description": "Tim Cook is CEO",
                    "weight": 0.9,
                }
            ],
        }
        llm.chat.return_value = json.dumps(payload)

        # Act
        entities, relationships = extractor.extract("Some text", "chunk-1")

        # Assert
        assert len(entities) == 2
        assert len(relationships) == 1
        assert relationships[0].weight == 0.9

    def test_extract_sets_source_id_on_all_results(self) -> None:
        # Arrange
        extractor, llm = _make_extractor()
        payload = {
            "entities": [{"name": "X", "type": "ORG", "description": "x"}],
            "relationships": [],
        }
        llm.chat.return_value = json.dumps(payload)

        # Act
        entities, _relationships = extractor.extract("text", "my-source-id")

        # Assert
        assert all(e.source_id == "my-source-id" for e in entities)


class TestEntityExtractorEdgeCases:
    def test_extract_malformed_json_returns_empty(self) -> None:
        # Arrange
        extractor, llm = _make_extractor()
        llm.chat.return_value = "not valid json"

        # Act
        entities, relationships = extractor.extract("text", "c1")

        # Assert
        assert entities == []
        assert relationships == []

    def test_extract_empty_response_returns_empty(self) -> None:
        # Arrange
        extractor, llm = _make_extractor()
        llm.chat.return_value = ""

        # Act
        entities, relationships = extractor.extract("text", "c1")

        # Assert
        assert entities == []
        assert relationships == []

    def test_extract_missing_entities_key_returns_empty(self) -> None:
        # Arrange
        extractor, llm = _make_extractor()
        llm.chat.return_value = json.dumps({"relationships": []})

        # Act
        entities, relationships = extractor.extract("text", "c1")

        # Assert
        assert entities == []
        assert relationships == []

    def test_extract_relationship_with_unknown_entity_is_skipped(self) -> None:
        # Arrange
        extractor, llm = _make_extractor()
        payload = {
            "entities": [{"name": "Apple", "type": "ORG", "description": "company"}],
            "relationships": [
                {
                    "source": "Apple",
                    "target": "NonExistent",
                    "type": "related_to",
                    "description": "n/a",
                    "weight": 0.5,
                }
            ],
        }
        llm.chat.return_value = json.dumps(payload)

        # Act
        entities, relationships = extractor.extract("text", "c1")

        # Assert — relationship with unknown target is dropped
        assert len(entities) == 1
        assert relationships == []
