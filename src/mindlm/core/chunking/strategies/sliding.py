from mindlm.core.chunking.base import BaseChunker
from mindlm.core.config.models import ChunkingConfig
from mindlm.core.models import Chunk


class SlidingChunker(BaseChunker):
    def __init__(self, config: ChunkingConfig) -> None:
        self._size = config.chunk_size
        self._overlap = config.overlap
        step = config.chunk_size - config.overlap
        if step <= 0:
            raise ValueError("overlap must be less than chunk_size")
        self._step = step

    def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []
        chunks: list[Chunk] = []
        start = 0
        idx = 0
        while start < len(text):
            segment = text[start : start + self._size]
            chunks.append(Chunk(text=segment, index=idx, metadata={}))
            start += self._step
            idx += 1
        return chunks
