from unittest.mock import MagicMock

import pytest

from mindlm.core.chunking.strategies.semantic import SemanticChunker
from mindlm.core.config.models import ChunkingConfig
from mindlm.core.models import Chunk


def _config(size: int = 500) -> ChunkingConfig:
    return ChunkingConfig(
        strategy="semantic",
        chunk_size=size,
        overlap=50,
        semantic_model="test-model",
    )


class TestSemanticChunker:
    def test_single_sentence_returns_one_chunk(self) -> None:
        provider = MagicMock()
        chunker = SemanticChunker(_config(), provider)

        chunks = chunker.chunk("This is a single sentence.")

        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)

    def test_uses_embedding_provider(self) -> None:
        provider = MagicMock()
        provider.embed.return_value = [
            [0.1] * 10,
            [0.9] * 10,
            [0.2] * 10,
            [0.8] * 10,
            [0.3] * 10,
        ]
        chunker = SemanticChunker(_config(), provider)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."

        chunks = chunker.chunk(text)

        provider.embed.assert_called_once()
        assert len(chunks) > 0

    def test_empty_text_returns_empty(self) -> None:
        provider = MagicMock()
        chunker = SemanticChunker(_config(), provider)

        chunks = chunker.chunk("")

        assert chunks == []

    def test_single_sentence_start_char(self) -> None:
        provider = MagicMock()
        chunker = SemanticChunker(_config(), provider)
        text = "  Hello world."

        chunks = chunker.chunk(text)

        assert chunks[0].start_char == text.find(text.strip())

    def test_two_sentences_same_group_start_char_zero(self) -> None:
        # force same group with very similar vectors
        provider = MagicMock()
        provider.embed.return_value = [[0.9] * 10, [0.9] * 10]
        chunker = SemanticChunker(_config(size=500), provider)
        text = "First sentence. Second sentence."

        chunks = chunker.chunk(text)

        assert chunks[0].start_char == 0
        assert chunks[0].end_char >= len("First sentence. Second sentence.")

    def test_two_sentences_different_groups_second_start_char(self) -> None:
        provider = MagicMock()
        # Very different vectors → split into two groups
        provider.embed.return_value = [[1.0] + [0.0] * 9, [0.0] * 9 + [1.0]]
        chunker = SemanticChunker(_config(size=500), provider)
        text = "First sentence. Second sentence."

        chunks = chunker.chunk(text)

        if len(chunks) >= 2:
            assert chunks[1].start_char > 0
            assert chunks[1].start_char == text.find("Second sentence.")

    def test_chunk_text_exact_match_in_original_span(self) -> None:
        provider = MagicMock()
        provider.embed.return_value = [[0.5] * 10, [0.5] * 10, [0.5] * 10]
        chunker = SemanticChunker(_config(size=500), provider)
        text = "First sentence. Second sentence. Third sentence."

        chunks = chunker.chunk(text)

        for c in chunks:
            assert text[c.start_char : c.end_char] == c.text

    def test_repeated_sentence_second_has_higher_start_char(self) -> None:
        provider = MagicMock()
        # Force a split between the two identical sentences
        provider.embed.return_value = [[1.0] + [0.0] * 9, [0.0] * 9 + [1.0]]
        chunker = SemanticChunker(_config(size=500), provider)
        text = "Yes. Yes."

        chunks = chunker.chunk(text)

        if len(chunks) >= 2:
            assert chunks[1].start_char > chunks[0].start_char


class TestChunkerDispatcher:
    def test_requires_provider_for_semantic(self) -> None:
        from mindlm.core.chunking.dispatcher import ChunkerDispatcher

        config = _config()
        with pytest.raises(ValueError, match="EmbeddingProvider required"):
            ChunkerDispatcher(config, embedding_provider=None)

    def test_dispatcher_fixed_chunks(self) -> None:
        from mindlm.core.chunking.dispatcher import ChunkerDispatcher

        config = ChunkingConfig(strategy="fixed", chunk_size=5, overlap=0)
        dispatcher = ChunkerDispatcher(config)

        chunks = dispatcher.chunk("abcdefghij")

        assert len(chunks) == 2

    def test_dispatcher_sliding_chunks(self) -> None:
        from mindlm.core.chunking.dispatcher import ChunkerDispatcher

        config = ChunkingConfig(strategy="sliding", chunk_size=4, overlap=2)
        dispatcher = ChunkerDispatcher(config)

        chunks = dispatcher.chunk("abcdefgh")

        assert len(chunks) > 0
