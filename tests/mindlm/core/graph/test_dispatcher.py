from unittest.mock import MagicMock, patch

from mindlm.core.config.models import GraphRAGConfig, GraphStoreConfig
from mindlm.core.graph.dispatcher import build_graph_store
from mindlm.core.graph.neo4j import Neo4jGraphStore


class TestGraphStoreDispatcher:
    def test_should_return_none_when_disabled(self) -> None:
        # Arrange
        config = GraphRAGConfig(enabled=False)

        # Act
        result = build_graph_store(config)

        # Assert
        assert result is None

    def test_should_return_neo4j_store_when_enabled(self) -> None:
        # Arrange
        config = GraphRAGConfig(enabled=True, store=GraphStoreConfig(provider="neo4j"))

        # Act
        with patch("mindlm.core.graph.neo4j.GraphDatabase") as mock_db:
            mock_db.driver.return_value = MagicMock()
            result = build_graph_store(config)

        # Assert
        assert isinstance(result, Neo4jGraphStore)
