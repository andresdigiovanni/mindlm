from mindlm.core.chunking.base import BaseChunker
from mindlm.core.config.models import ChunkingConfig
from mindlm.core.models import Chunk


class FixedChunker(BaseChunker):
    def __init__(self, config: ChunkingConfig) -> None:
        self._size = config.chunk_size

    def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []
        size = self._size
        segments = [text[i : i + size] for i in range(0, len(text), size)]
        return [Chunk(text=s, index=i, metadata={}) for i, s in enumerate(segments)]
