from neo4j import GraphDatabase

from mindlm.core.config.models import GraphStoreConfig
from mindlm.core.graph.base import GraphStore
from mindlm.core.models import Entity, Relationship


class Neo4jGraphStore(GraphStore):
    def __init__(self, config: GraphStoreConfig) -> None:
        uri = f"bolt://{config.host}:{config.port}"
        self._driver = GraphDatabase.driver(
            uri, auth=(config.username, config.password)
        )

    def upsert_entities(self, entities: list[Entity]) -> None:
        with self._driver.session() as session:
            for e in entities:
                session.run(
                    "MERGE (n:Entity {id: $id}) "
                    "SET n.name=$name, n.type=$type, "
                    "n.description=$description, n.source_id=$source_id",
                    id=e.id,
                    name=e.name,
                    type=e.type,
                    description=e.description,
                    source_id=e.source_id,
                )

    def upsert_relationships(self, relationships: list[Relationship]) -> None:
        with self._driver.session() as session:
            for r in relationships:
                session.run(
                    "MATCH (s:Entity {id: $src}), (t:Entity {id: $tgt}) "
                    "MERGE (s)-[rel:RELATED {id: $id}]->(t) "
                    "SET rel.type=$type, rel.description=$description, "
                    "rel.weight=$weight, rel.source_id=$source_id",
                    src=r.source_entity_id,
                    tgt=r.target_entity_id,
                    id=r.id,
                    type=r.type,
                    description=r.description,
                    weight=r.weight,
                    source_id=r.source_id,
                )

    def get_related_chunk_ids(self, chunk_ids: list[str], depth: int) -> list[str]:
        with self._driver.session() as session:
            result = session.run(
                f"MATCH (start:Entity)-[*0..{depth}]-(related:Entity) "
                "WHERE start.source_id IN $chunk_ids "
                "RETURN DISTINCT related.source_id AS chunk_id",
                chunk_ids=chunk_ids,
            )
            return [record["chunk_id"] for record in result]

    def delete_by_source(self, source_id: str) -> None:
        with self._driver.session() as session:
            session.run(
                "MATCH (n:Entity {source_id: $source_id}) DETACH DELETE n",
                source_id=source_id,
            )

    def clear(self) -> None:
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
