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
        raw = self._split(text, sep_index=0, offset=0)
        return [
            Chunk(text=t, index=i, metadata={}, start_char=s, end_char=e)
            for i, (t, s, e) in enumerate(raw)
        ]

    def _split(
        self, text: str, sep_index: int, offset: int = 0
    ) -> list[tuple[str, int, int]]:
        if not text:
            return []
        if len(text) <= self._chunk_size:
            return [(text, offset, offset + len(text))]
        if sep_index >= len(self._separators):
            result = []
            for i in range(0, len(text), self._chunk_size):
                s = text[i : i + self._chunk_size]
                result.append((s, offset + i, offset + i + len(s)))
            return result
        sep = self._separators[sep_index]
        if sep:
            parts_with_offsets: list[tuple[str, int]] = []
            pos = 0
            for part in text.split(sep):
                parts_with_offsets.append((part, pos))
                pos += len(part) + len(sep)
        else:
            parts_with_offsets = [(c, i) for i, c in enumerate(text)]

        result2: list[tuple[str, int, int]] = []
        accumulator = ""
        acc_start = 0
        for part, part_pos in parts_with_offsets:
            if not accumulator:
                candidate, cand_start = part, part_pos
            else:
                candidate = accumulator + sep + part
                cand_start = acc_start
            if len(candidate) <= self._chunk_size:
                accumulator = candidate
                acc_start = cand_start
            else:
                if accumulator:
                    result2.append(
                        (
                            accumulator,
                            offset + acc_start,
                            offset + acc_start + len(accumulator),
                        )
                    )
                if len(part) > self._chunk_size:
                    result2.extend(self._split(part, sep_index + 1, offset + part_pos))
                    accumulator = ""
                else:
                    accumulator = part
                    acc_start = part_pos
        if accumulator:
            result2.append(
                (accumulator, offset + acc_start, offset + acc_start + len(accumulator))
            )
        return [(t, s, e) for t, s, e in result2 if t]
