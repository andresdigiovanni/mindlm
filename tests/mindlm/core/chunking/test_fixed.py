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

    def test_first_chunk_start_end_char(self) -> None:
        chunker = FixedChunker(_config(size=2))
        text = "abcde"

        chunks = chunker.chunk(text)

        assert chunks[0].start_char == 0
        assert chunks[0].end_char == 2

    def test_last_chunk_end_char_equals_text_len(self) -> None:
        chunker = FixedChunker(_config(size=2))
        text = "abcde"

        chunks = chunker.chunk(text)

        assert chunks[-1].end_char == len(text)

    def test_slice_fidelity_all_chunks(self) -> None:
        chunker = FixedChunker(_config(size=3))
        text = "abcdefghij"

        chunks = chunker.chunk(text)

        assert all(text[c.start_char : c.end_char] == c.text for c in chunks)

    def test_start_char_sequential_for_divisible_text(self) -> None:
        chunker = FixedChunker(_config(size=5))
        text = "a" * 15

        chunks = chunker.chunk(text)

        assert [c.start_char for c in chunks] == [0, 5, 10]

    def test_span_length_matches_text_length(self) -> None:
        chunker = FixedChunker(_config(size=4))
        text = "abcdefghijkl"

        chunks = chunker.chunk(text)

        assert all(c.end_char - c.start_char == len(c.text) for c in chunks)
