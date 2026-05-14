from typing import Any

import pytest

from mindlm.core.chunking.strategies.recursive import RecursiveChunker
from mindlm.core.config.models import ChunkingConfig


def _config(
    size: int = 100,
    separators: list[str] | None = None,
) -> ChunkingConfig:
    kwargs: dict[str, Any] = {"strategy": "recursive", "chunk_size": size, "overlap": 0}
    if separators is not None:
        kwargs["separators"] = separators
    return ChunkingConfig(**kwargs)


class TestRecursiveChunker:
    def test_empty_text_returns_empty_list(self) -> None:
        chunker = RecursiveChunker(_config())

        chunks = chunker.chunk("")

        assert chunks == []

    def test_text_shorter_than_chunk_size_returns_single_chunk(self) -> None:
        chunker = RecursiveChunker(_config(size=200))
        text = "short text"

        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].text == "short text"

    def test_splits_at_double_newline_first(self) -> None:
        chunker = RecursiveChunker(_config(size=30))
        text = "paragraph one\n\nparagraph two\n\nparagraph three"

        chunks = chunker.chunk(text)

        assert all(len(c.text) <= 30 for c in chunks)
        assert any("paragraph one" in c.text for c in chunks)

    def test_recurses_to_single_newline(self) -> None:
        # No double newlines — must fall back to single newline
        chunker = RecursiveChunker(_config(size=20))
        text = "line one\nline two\nline three"

        chunks = chunker.chunk(text)

        assert all(len(c.text) <= 20 for c in chunks)
        assert len(chunks) > 1

    def test_falls_back_to_character_split(self) -> None:
        # Single long word with no separators matching — falls back to char split
        chunker = RecursiveChunker(_config(size=5, separators=["\n\n", "\n"]))
        text = "abcdefghij"

        chunks = chunker.chunk(text)

        assert all(len(c.text) <= 5 for c in chunks)
        assert "".join(c.text for c in chunks) == text

    def test_sequential_zero_based_indices(self) -> None:
        chunker = RecursiveChunker(_config(size=10))
        text = "a" * 40

        chunks = chunker.chunk(text)

        assert [c.index for c in chunks] == list(range(len(chunks)))

    @pytest.mark.parametrize("size", [10, 20, 50])
    def test_chunks_no_larger_than_chunk_size(self, size: int) -> None:
        chunker = RecursiveChunker(_config(size=size))
        text = "word " * 100

        chunks = chunker.chunk(text)

        assert all(len(c.text) <= size for c in chunks)

    def test_handles_unicode_text(self) -> None:
        chunker = RecursiveChunker(_config(size=20))
        text = "こんにちは\n\n世界\n\nPython 3"

        chunks = chunker.chunk(text)

        assert len(chunks) >= 1
        assert all(c.text for c in chunks)

    def test_custom_separators(self) -> None:
        chunker = RecursiveChunker(_config(size=30, separators=["|"]))
        text = "part one|part two|part three"

        chunks = chunker.chunk(text)

        assert len(chunks) >= 1
        assert all(len(c.text) <= 30 for c in chunks)

    def test_text_exactly_chunk_size_returns_single_chunk(self) -> None:
        chunker = RecursiveChunker(_config(size=10))
        text = "a" * 10

        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_empty_separator_triggers_character_split(self) -> None:
        chunker = RecursiveChunker(_config(size=3, separators=[""]))
        text = "abcdef"

        chunks = chunker.chunk(text)

        assert all(len(c.text) <= 3 for c in chunks)
        assert "".join(c.text for c in chunks) == text
