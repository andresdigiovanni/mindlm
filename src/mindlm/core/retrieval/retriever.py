from dataclasses import replace
from typing import Any

from fastembed.sparse.bm25 import Bm25
from langfuse.decorators import langfuse_context, observe

from mindlm.core.config.models import RetrievalConfig
from mindlm.core.embeddings.base import EmbeddingProvider
from mindlm.core.generation.base import LLMProvider
from mindlm.core.graph.base import GraphStore
from mindlm.core.models import Result, SparseVector
from mindlm.core.query_processing.dispatcher import QueryProcessorDispatcher
from mindlm.core.vectorstore.base import VectorStore


class Retriever:
    def __init__(
        self,
        config: RetrievalConfig,
        vectorstore: VectorStore,
        embedding_provider: EmbeddingProvider,
        llm: LLMProvider | None = None,
        query_processor: QueryProcessorDispatcher | None = None,
        resolve_parents: bool = False,
        resolve_windows: bool = False,
        graph_store: GraphStore | None = None,
    ) -> None:
        if query_processor is not None and llm is None:
            raise ValueError("llm is required when query_processor is provided")
        if resolve_parents and resolve_windows:
            raise ValueError(
                "resolve_parents and resolve_windows are mutually exclusive: "
                "parent_chunk_size cannot be combined with strategy='sentence_window'"
            )
        self._config = config
        self._vectorstore = vectorstore
        self._embedding_provider = embedding_provider
        self._llm = llm
        self._query_processor = query_processor
        self._resolve_parents = resolve_parents
        self._resolve_windows = resolve_windows
        self._graph_store = graph_store
        self._bm25: Bm25 | None = None

    @observe(name="retrieve")
    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        *,
        top_k: int | None = None,
    ) -> list[Result]:
        effective_top_k = top_k if top_k is not None else self._config.top_k
        langfuse_context.update_current_observation(
            input=query,
            metadata={
                "strategy": self._config.strategy,
                "top_k": effective_top_k,
            },
        )
        if self._query_processor is not None:
            assert self._llm is not None
            queries: list[str] = self._query_processor.process(query, self._llm)
        else:
            queries = [query]
        all_results: dict[str, Result] = {}
        for q in queries:
            for result in self._retrieve_single(q, filters, effective_top_k):
                if (
                    result.id not in all_results
                    or result.score > all_results[result.id].score
                ):
                    all_results[result.id] = result
        merged = sorted(all_results.values(), key=lambda r: r.score, reverse=True)
        results = merged[:effective_top_k]
        if self._resolve_parents:
            results = self._apply_parent_resolution(results)
        if self._resolve_windows:
            results = self._apply_window_resolution(results)
        if self._graph_store is not None:
            results = self._expand_with_graph(results)
        return results

    def _retrieve_single(
        self, query: str, filters: dict[str, Any] | None, top_k: int
    ) -> list[Result]:
        dense = self._embedding_provider.embed([query])[0]
        match self._config.strategy:
            case "vector":
                return self._vectorstore.search(dense, top_k, filters)
            case "hybrid":
                sparse = self._compute_bm25(query)
                return self._vectorstore.search_hybrid(dense, sparse, top_k, filters)
            case _:  # pragma: no cover
                raise ValueError(
                    f"Unknown retrieval strategy: {self._config.strategy!r}"
                )

    def _expand_with_graph(self, results: list[Result]) -> list[Result]:
        chunk_ids = [r.id for r in results]
        related_ids = self._graph_store.get_related_chunk_ids(  # type: ignore[union-attr]
            chunk_ids, depth=1
        )
        existing_ids: set[str] = set(chunk_ids)
        min_score = min((r.score for r in results), default=0.0)
        expansion_score = 0.5 * min_score
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
        return expanded[: self._config.top_k]

    def _apply_parent_resolution(self, results: list[Result]) -> list[Result]:
        resolved = []
        for r in results:
            if "parent_content" in r.payload:
                new_payload = {**r.payload, "content": r.payload["parent_content"]}
                resolved.append(replace(r, payload=new_payload))
            else:
                resolved.append(r)
        return resolved

    def _apply_window_resolution(self, results: list[Result]) -> list[Result]:
        resolved: list[Result] = []
        for r in results:
            if "window_context" in r.payload:
                new_payload = {**r.payload, "content": r.payload["window_context"]}
                resolved.append(replace(r, payload=new_payload))
            else:
                resolved.append(r)
        return resolved

    def _compute_bm25(self, query: str) -> SparseVector:
        if self._bm25 is None:
            self._bm25 = Bm25("Qdrant/bm25")
        result = next(iter(self._bm25.query_embed(query)))
        return SparseVector(
            indices=result.indices.tolist(), values=result.values.tolist()
        )
