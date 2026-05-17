import json
import logging
from uuid import uuid4

from langfuse.decorators import observe

from mindlm.core.generation.base import LLMProvider
from mindlm.core.models import Entity, Relationship

logger = logging.getLogger(__name__)

_PROMPT = (
    "Extract entities and relationships from the text chunk below.\n\n"
    "Text:\n{chunk_text}\n\n"
    "Respond with ONLY valid JSON matching this structure exactly:\n"
    '{{"entities": [{{"name": "...", "type": "...", "description": "..."}}], '
    '"relationships": [{{"source": "entity_name", "target": "entity_name", '
    '"type": "...", "description": "...", "weight": 0.8}}]}}\n'
    "Use entity names exactly as they appear in the text."
)


class EntityExtractor:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    @observe(name="graph-extract")
    def extract(
        self, chunk_text: str, source_id: str
    ) -> tuple[list[Entity], list[Relationship]]:
        prompt = _PROMPT.format(chunk_text=chunk_text)
        response = self._llm.chat([{"role": "user", "content": prompt}]).strip()
        if not response:
            return [], []
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("EntityExtractor: failed to parse LLM response as JSON")
            return [], []

        entity_map: dict[str, Entity] = {}
        for raw in data.get("entities", []):
            eid = str(uuid4())
            e = Entity(
                id=eid,
                name=raw.get("name", ""),
                type=raw.get("type", ""),
                description=raw.get("description", ""),
                source_id=source_id,
            )
            entity_map[e.name] = e

        relationships: list[Relationship] = []
        for raw in data.get("relationships", []):
            src = entity_map.get(raw.get("source", ""))
            tgt = entity_map.get(raw.get("target", ""))
            if src is None or tgt is None:
                continue
            relationships.append(
                Relationship(
                    id=str(uuid4()),
                    source_entity_id=src.id,
                    target_entity_id=tgt.id,
                    type=raw.get("type", ""),
                    description=raw.get("description", ""),
                    weight=float(raw.get("weight", 0.5)),
                    source_id=source_id,
                )
            )

        return list(entity_map.values()), relationships
