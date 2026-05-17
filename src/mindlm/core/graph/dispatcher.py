from mindlm.core.config.models import GraphRAGConfig
from mindlm.core.graph.base import GraphStore
from mindlm.core.graph.neo4j import Neo4jGraphStore


def build_graph_store(config: GraphRAGConfig) -> GraphStore | None:
    if not config.enabled:
        return None
    return Neo4jGraphStore(config.store)
