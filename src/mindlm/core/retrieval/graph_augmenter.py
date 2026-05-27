from mindlm.core.graph.base import GraphStore
from mindlm.core.models import Result
from mindlm.core.vectorstore.base import VectorStore


class GraphAugmenter:
    """Expands retrieval results with graph-related chunks (depth=1 neighbors)."""

    def __init__(self, graph_store: GraphStore, vectorstore: VectorStore) -> None:
        self._graph_store = graph_store
        self._vectorstore = vectorstore

    def augment(self, results: list[Result], top_k: int) -> list[Result]:
        """Augment results with graph neighbors, re-sorted and truncated to top_k.

        Args:
            results: Retrieved results to augment
            top_k: Maximum number of results to return after augmentation

        Returns:
            Results with graph neighbors appended, re-sorted by score and truncated.
        """
        if not results:
            return results
        chunk_ids = [r.id for r in results]
        related_ids = self._graph_store.get_related_chunk_ids(chunk_ids, depth=1)
        existing_ids: set[str] = set(chunk_ids)
        min_score = min(r.score for r in results)
        expansion_score = max(0.5 * min_score, 1e-9)
        expanded: list[Result] = list(results)
        for related_id in related_ids:
            if related_id in existing_ids:
                continue
            point = self._vectorstore.get_by_id(related_id)
            if point is None:
                continue
            expanded.append(
                Result(id=point.id, score=expansion_score, payload=point.payload)
            )
            existing_ids.add(related_id)
        expanded.sort(key=lambda r: r.score, reverse=True)
        return expanded[:top_k]
