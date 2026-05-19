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
        chunks: list[Chunk] = []
        for idx, start in enumerate(range(0, len(text), size)):
            end = min(start + size, len(text))
            chunks.append(
                Chunk(
                    text=text[start:end],
                    index=idx,
                    metadata={},
                    start_char=start,
                    end_char=end,
                )
            )
        return chunks
