from fastapi import APIRouter, Depends

from mindlm.api.dependencies import get_vectorstore
from mindlm.core.vectorstore.base import VectorStore

router = APIRouter()


@router.get("/collections", response_model=list[str])
async def list_collections(vs: VectorStore = Depends(get_vectorstore)) -> list[str]:
    return vs.list_collections()
