import pytest

from mindlm.core.chunking.strategies.sliding import SlidingChunker
from mindlm.core.config.models import ChunkingConfig


def _config(size: int, overlap: int) -> ChunkingConfig:
    return ChunkingConfig(strategy="sliding", chunk_size=size, overlap=overlap)


class TestSlidingChunker:
    def test_basic_sliding(self) -> None:
        chunker = SlidingChunker(_config(size=4, overlap=2))
        text = "abcdefgh"

        chunks = chunker.chunk(text)

        assert [c.text for c in chunks] == ["abcd", "cdef", "efgh", "gh"]

    def test_overlap_exceeds_size_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            SlidingChunker(_config(size=4, overlap=4))

    def test_empty_text(self) -> None:
        chunker = SlidingChunker(_config(size=10, overlap=2))

        chunks = chunker.chunk("")

        assert chunks == []

    def test_overlap_zero_same_as_fixed(self) -> None:
        chunker = SlidingChunker(_config(size=5, overlap=0))
        text = "abcdefghij"

        chunks = chunker.chunk(text)

        assert len(chunks) == 2
        assert chunks[0].text == "abcde"
        assert chunks[1].text == "fghij"
