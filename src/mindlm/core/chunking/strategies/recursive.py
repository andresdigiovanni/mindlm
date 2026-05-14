from mindlm.core.chunking.base import BaseChunker
from mindlm.core.config.models import ChunkingConfig
from mindlm.core.models import Chunk


class RecursiveChunker(BaseChunker):
    def __init__(self, config: ChunkingConfig) -> None:
        self._chunk_size = config.chunk_size
        self._separators = config.separators

    def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []
        raw = self._split(text, sep_index=0)
        return [Chunk(text=t, index=i, metadata={}) for i, t in enumerate(raw)]

    def _split(self, text: str, sep_index: int) -> list[str]:
        if len(text) <= self._chunk_size:
            return [text]
        if sep_index >= len(self._separators):
            return [
                text[i : i + self._chunk_size]
                for i in range(0, len(text), self._chunk_size)
            ]
        sep = self._separators[sep_index]
        parts = text.split(sep) if sep else list(text)
        result: list[str] = []
        accumulator = ""
        for part in parts:
            candidate = (accumulator + sep + part) if accumulator else part
            if len(candidate) <= self._chunk_size:
                accumulator = candidate
            else:
                if accumulator:
                    result.append(accumulator)
                if len(part) > self._chunk_size:
                    result.extend(self._split(part, sep_index + 1))
                    accumulator = ""
                else:
                    accumulator = part
        if accumulator:
            result.append(accumulator)
        return [s for s in result if s]
