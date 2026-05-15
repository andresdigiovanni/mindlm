from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from mindlm.api.dependencies import get_config, get_synchronizer
from mindlm.api.schemas import IngestRequest, SyncResponse
from mindlm.core.config.models import RAGConfig
from mindlm.core.synchronization.synchronizer import Synchronizer

router = APIRouter()


def _safe_paths(raw_paths: list[str], base_dir: str) -> list[Path]:
    base = Path(base_dir).resolve()
    result: list[Path] = []
    for raw in raw_paths:
        try:
            resolved = Path(raw).resolve()
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid path: {raw}") from exc
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Path outside allowed directory: {raw}",
            ) from exc
        result.append(resolved)
    return result


@router.post("/ingest/sync", response_model=SyncResponse)
async def ingest_sync(
    request: IngestRequest,
    sync: Synchronizer = Depends(get_synchronizer),
    config: RAGConfig = Depends(get_config),
) -> SyncResponse:
    paths = _safe_paths(request.paths, config.ingestion.allowed_base_dir)
    use_sparse = config.retrieval.strategy == "hybrid"
    result = sync.sync(
        paths,
        collection=config.vector_store.collection,
        dense_dim=config.embeddings.dimensions,
        sparse=use_sparse,
    )
    return SyncResponse(
        added=result.added,
        updated=result.updated,
        skipped=result.skipped,
        chunks=result.chunks,
        errors=result.errors,
    )


@router.post("/ingest/full", response_model=SyncResponse)
async def ingest_full(
    request: IngestRequest,
    sync: Synchronizer = Depends(get_synchronizer),
    config: RAGConfig = Depends(get_config),
) -> SyncResponse:
    paths = _safe_paths(request.paths, config.ingestion.allowed_base_dir)
    use_sparse = config.retrieval.strategy == "hybrid"
    result = sync.full_reingest(
        paths,
        collection=config.vector_store.collection,
        dense_dim=config.embeddings.dimensions,
        sparse=use_sparse,
    )
    return SyncResponse(
        added=result.added,
        updated=result.updated,
        skipped=result.skipped,
        chunks=result.chunks,
        errors=result.errors,
    )
