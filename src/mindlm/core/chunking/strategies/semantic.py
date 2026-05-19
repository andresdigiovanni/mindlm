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
            stripped = text.strip()
            start = text.find(stripped)
            return [
                Chunk(
                    text=stripped,
                    index=0,
                    metadata={},
                    start_char=start,
                    end_char=start + len(stripped),
                )
            ]

        sentence_offsets: list[int] = []
        search_from = 0
        for s in sentences:
            found = text.find(s, search_from)
            if found == -1:
                found = search_from
            sentence_offsets.append(found)
            search_from = found + len(s)

        vectors = self._provider.embed(sentences)
        similarities = [
            _cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)
        ]

        mean_sim = sum(similarities) / len(similarities)
        variance = sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)
        std_sim = variance**0.5
        threshold = mean_sim - 0.5 * std_sim

        groups: list[list[int]] = []
        current_indices: list[int] = [0]
        for i, sim in enumerate(similarities):
            if sim < threshold:
                groups.append(current_indices)
                current_indices = [i + 1]
            else:
                current_indices.append(i + 1)
        groups.append(current_indices)

        fixed = FixedChunker(self._config)
        result: list[Chunk] = []
        idx = 0
        for group_indices in groups:
            group_start = sentence_offsets[group_indices[0]]
            group_end = sentence_offsets[group_indices[-1]] + len(
                sentences[group_indices[-1]]
            )
            # Use the original text slice so start_char/end_char are exact.
            group_text = text[group_start:group_end]
            if len(group_text) > self._config.chunk_size:
                sub = fixed.chunk(group_text)
                for sub_chunk in sub:
                    result.append(
                        Chunk(
                            text=sub_chunk.text,
                            index=idx,
                            metadata={},
                            start_char=group_start + sub_chunk.start_char,
                            end_char=group_start + sub_chunk.end_char,
                        )
                    )
                    idx += 1
            else:
                result.append(
                    Chunk(
                        text=group_text,
                        index=idx,
                        metadata={},
                        start_char=group_start,
                        end_char=group_end,
                    )
                )
                idx += 1
        return result
