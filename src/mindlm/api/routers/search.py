from fastapi import APIRouter, Depends
from langfuse.decorators import observe

from mindlm.api.dependencies import get_llm_provider, get_reranker, get_retriever
from mindlm.api.schemas import (
    AskRequest,
    AskResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SourceRef,
)
from mindlm.core.generation.base import LLMProvider
from mindlm.core.models import Result
from mindlm.core.reranking.base import BaseReranker
from mindlm.core.retrieval.retriever import Retriever


def _format_context(results: list[Result]) -> str:
    blocks = []
    for r in results:
        parts = []
        if summary := r.payload.get("document_summary"):
            parts.append(f"[Document context: {summary}]")
        if ctx := r.payload.get("chunk_context"):
            parts.append(f"[Chunk context: {ctx}]")
        parts.append(r.payload.get("content", ""))
        blocks.append(f"[Source: {r.payload.get('source', '')}]\n" + "\n".join(parts))
    return "\n\n".join(blocks)


def _extract_sources(results: list[Result]) -> list[SourceRef]:
    return [
        SourceRef(
            source=r.payload.get("source", ""),
            score=r.score,
            chunk_index=int(r.payload.get("chunk_index", 0)),
            page_number=r.payload.get("page_number"),
            char_start=int(v)
            if (v := r.payload.get("char_start")) is not None
            else None,
            char_end=int(v) if (v := r.payload.get("char_end")) is not None else None,
        )
        for r in results
    ]


router = APIRouter()


@router.post("/search", response_model=SearchResponse)
@observe(name="search")
async def search(
    request: SearchRequest,
    retriever: Retriever = Depends(get_retriever),
    reranker: BaseReranker = Depends(get_reranker),
) -> SearchResponse:
    results = retriever.retrieve(request.query, request.filters)
    results = reranker.rerank(request.query, results)
    items = [
        SearchResultItem(
            content=r.payload.get("content", ""),
            score=r.score,
            source=r.payload.get("source", ""),
            metadata={
                k: v for k, v in r.payload.items() if k not in ("content", "source")
            },
        )
        for r in results
    ]
    return SearchResponse(query=request.query, results=items)


@router.post("/ask", response_model=AskResponse)
@observe(name="ask")
async def ask(
    request: AskRequest,
    retriever: Retriever = Depends(get_retriever),
    reranker: BaseReranker = Depends(get_reranker),
    llm: LLMProvider = Depends(get_llm_provider),
) -> AskResponse:
    results = retriever.retrieve(request.question, request.filters)
    results = reranker.rerank(request.question, results)

    context_blocks = _format_context(results)
    system_msg = "You are a helpful assistant. Answer using only the provided context."
    user_msg = f"Context:\n{context_blocks}\n\nQuestion: {request.question}"

    answer = llm.chat(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
    )

    sources = _extract_sources(results)
    return AskResponse(answer=answer, sources=sources)
