from mindlm.core.chunking.strategies.fixed import FixedChunker
from mindlm.core.config.models import ChunkingConfig


def _config(size: int = 100, overlap: int = 0) -> ChunkingConfig:
    return ChunkingConfig(strategy="fixed", chunk_size=size, overlap=overlap)


class TestFixedChunker:
    def test_even_split(self) -> None:
        chunker = FixedChunker(_config(size=200))
        text = "a" * 1000

        chunks = chunker.chunk(text)

        assert len(chunks) == 5
        assert all(len(c.text) == 200 for c in chunks)

    def test_remainder(self) -> None:
        chunker = FixedChunker(_config(size=100))
        text = "a" * 250

        chunks = chunker.chunk(text)

        assert len(chunks) == 3
        assert len(chunks[2].text) == 50

    def test_empty_text(self) -> None:
        chunker = FixedChunker(_config())

        chunks = chunker.chunk("")

        assert chunks == []

    def test_chunk_indices_are_sequential(self) -> None:
        chunker = FixedChunker(_config(size=10))
        text = "a" * 30

        chunks = chunker.chunk(text)

        assert [c.index for c in chunks] == [0, 1, 2]

    def test_text_shorter_than_size(self) -> None:
        chunker = FixedChunker(_config(size=500))
        text = "short text"

        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].text == "short text"
