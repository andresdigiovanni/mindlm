import re

from mindlm.core.chunking.base import BaseChunker
from mindlm.core.config.models import ChunkingConfig
from mindlm.core.models import Chunk

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class SentenceWindowChunker(BaseChunker):
    def __init__(self, config: ChunkingConfig) -> None:
        self._window_size = config.window_size

    def chunk(self, text: str) -> list[Chunk]:
        if not text or not text.strip():
            return []
        raw = _SENTENCE_RE.split(text)
        sentences = [s.strip() for s in raw if s.strip()]
        if not sentences:
            return []
        w = self._window_size
        result: list[Chunk] = []
        for i, sentence in enumerate(sentences):
            start = max(0, i - w)
            end = min(len(sentences), i + w + 1)
            window_context = " ".join(sentences[start:end])
            result.append(
                Chunk(
                    text=sentence,
                    index=i,
                    metadata={"window_context": window_context},
                )
            )
        return result
