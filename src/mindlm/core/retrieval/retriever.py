from dataclasses import replace
from typing import Any

from fastembed.sparse.bm25 import Bm25
from langfuse.decorators import langfuse_context, observe

from mindlm.core.config.models import RetrievalConfig
from mindlm.core.embeddings.base import EmbeddingProvider
from mindlm.core.generation.base import LLMProvider
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
        self._bm25: Bm25 | None = None

    @observe(name="retrieve")
    def retrieve(
        self, query: str, filters: dict[str, Any] | None = None
    ) -> list[Result]:
        langfuse_context.update_current_observation(
            input=query,
            metadata={"strategy": self._config.strategy, "top_k": self._config.top_k},
        )
        if self._query_processor is not None:
            assert self._llm is not None
            queries: list[str] = self._query_processor.process(query, self._llm)
        else:
            queries = [query]
        all_results: dict[str, Result] = {}
        for q in queries:
            for result in self._retrieve_single(q, filters):
                if (
                    result.id not in all_results
                    or result.score > all_results[result.id].score
                ):
                    all_results[result.id] = result
        merged = sorted(all_results.values(), key=lambda r: r.score, reverse=True)
        results = merged[: self._config.top_k]
        if self._resolve_parents:
            results = self._apply_parent_resolution(results)
        if self._resolve_windows:
            results = self._apply_window_resolution(results)
        return results

    def _retrieve_single(
        self, query: str, filters: dict[str, Any] | None
    ) -> list[Result]:
        dense = self._embedding_provider.embed([query])[0]
        match self._config.strategy:
            case "vector":
                return self._vectorstore.search(dense, self._config.top_k, filters)
            case "hybrid":
                sparse = self._compute_bm25(query)
                return self._vectorstore.search_hybrid(
                    dense, sparse, self._config.top_k, filters
                )
            case _:  # pragma: no cover
                raise ValueError(
                    f"Unknown retrieval strategy: {self._config.strategy!r}"
                )

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
