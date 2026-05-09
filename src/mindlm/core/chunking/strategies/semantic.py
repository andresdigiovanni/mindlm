import re

from mindlm.core.chunking.base import BaseChunker
from mindlm.core.chunking.strategies.fixed import FixedChunker
from mindlm.core.config.models import ChunkingConfig
from mindlm.core.embeddings.base import EmbeddingProvider
from mindlm.core.models import Chunk


def _cosine(a: list[float], b: list[float]) -> float:
    dot: float = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a: float = sum(x * x for x in a) ** 0.5
    norm_b: float = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticChunker(BaseChunker):
    def __init__(
        self, config: ChunkingConfig, embedding_provider: EmbeddingProvider
    ) -> None:
        self._config = config
        self._provider = embedding_provider

    def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s for s in sentences if s.strip()]
        if len(sentences) < 2:
            return [Chunk(text=text.strip(), index=0, metadata={})]

        vectors = self._provider.embed(sentences)
        similarities = [
            _cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)
        ]

        mean_sim = sum(similarities) / len(similarities)
        variance = sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)
        std_sim = variance**0.5
        threshold = mean_sim - 0.5 * std_sim

        groups: list[list[str]] = []
        current: list[str] = [sentences[0]]
        for i, sim in enumerate(similarities):
            if sim < threshold:
                groups.append(current)
                current = [sentences[i + 1]]
            else:
                current.append(sentences[i + 1])
        groups.append(current)

        fixed = FixedChunker(self._config)
        result: list[Chunk] = []
        idx = 0
        for group in groups:
            merged = " ".join(group)
            if len(merged) > self._config.chunk_size:
                sub = fixed.chunk(merged)
                for chunk in sub:
                    result.append(Chunk(text=chunk.text, index=idx, metadata={}))
                    idx += 1
            else:
                result.append(Chunk(text=merged, index=idx, metadata={}))
                idx += 1
        return result
