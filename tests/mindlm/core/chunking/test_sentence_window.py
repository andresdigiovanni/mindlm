from mindlm.core.chunking.strategies.sentence_window import SentenceWindowChunker
from mindlm.core.config.models import ChunkingConfig


def _make_chunker(window_size: int = 2) -> SentenceWindowChunker:
    config = ChunkingConfig(
        strategy="sentence_window", chunk_size=500, overlap=0, window_size=window_size
    )
    return SentenceWindowChunker(config)


class TestSentenceWindowChunker:
    def test_empty_text_returns_empty(self) -> None:
        chunker = _make_chunker()
        assert chunker.chunk("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        chunker = _make_chunker()
        assert chunker.chunk("   ") == []

    def test_single_sentence_window_is_itself(self) -> None:
        chunker = _make_chunker(window_size=2)
        chunks = chunker.chunk("Hello world.")
        assert len(chunks) == 1
        assert chunks[0].metadata["window_context"] == "Hello world."

    def test_chunk_count_equals_sentence_count(self) -> None:
        chunker = _make_chunker()
        chunks = chunker.chunk("First. Second. Third. Fourth.")
        assert len(chunks) == 4

    def test_indices_are_sequential(self) -> None:
        chunker = _make_chunker()
        chunks = chunker.chunk("A. B. C.")
        assert [c.index for c in chunks] == [0, 1, 2]

    def test_chunk_text_is_single_sentence(self) -> None:
        chunker = _make_chunker()
        chunks = chunker.chunk("First sentence. Second sentence. Third sentence.")
        assert chunks[0].text == "First sentence."
        assert chunks[1].text == "Second sentence."
        assert chunks[2].text == "Third sentence."

    def test_first_sentence_window_has_no_left_neighbor(self) -> None:
        chunker = _make_chunker(window_size=1)
        chunks = chunker.chunk("A. B. C.")
        # index 0: window = [A, B] (no left neighbor)
        assert "A." in chunks[0].metadata["window_context"]
        assert "B." in chunks[0].metadata["window_context"]

    def test_last_sentence_window_has_no_right_neighbor(self) -> None:
        chunker = _make_chunker(window_size=1)
        chunks = chunker.chunk("A. B. C.")
        # index 2: window = [B, C] (no right neighbor)
        assert "B." in chunks[2].metadata["window_context"]
        assert "C." in chunks[2].metadata["window_context"]

    def test_middle_sentence_has_full_window(self) -> None:
        chunker = _make_chunker(window_size=2)
        chunks = chunker.chunk("A. B. C. D. E.")
        # index 2: window = [A, B, C, D, E]
        ctx = chunks[2].metadata["window_context"]
        for s in ["A.", "B.", "C.", "D.", "E."]:
            assert s in ctx

    def test_window_size_one(self) -> None:
        chunker = _make_chunker(window_size=1)
        chunks = chunker.chunk("A. B. C.")
        # middle sentence window = [A, B, C]
        assert "A." in chunks[1].metadata["window_context"]
        assert "B." in chunks[1].metadata["window_context"]
        assert "C." in chunks[1].metadata["window_context"]

    def test_window_larger_than_text(self) -> None:
        chunker = _make_chunker(window_size=10)
        chunks = chunker.chunk("A. B.")
        # all windows = full text
        assert (
            chunks[0].metadata["window_context"] == chunks[1].metadata["window_context"]
        )

    def test_text_without_sentence_boundaries(self) -> None:
        chunker = _make_chunker()
        chunks = chunker.chunk("no punctuation here")
        assert len(chunks) == 1
        assert chunks[0].text == "no punctuation here"

    def test_first_sentence_start_char_is_zero(self) -> None:
        chunker = _make_chunker()
        text = "First sentence. Second sentence."

        chunks = chunker.chunk(text)

        assert chunks[0].start_char == 0

    def test_second_sentence_start_char_matches_position(self) -> None:
        chunker = _make_chunker()
        text = "First sentence. Second sentence."

        chunks = chunker.chunk(text)

        assert chunks[1].start_char == text.find("Second sentence.")

    def test_slice_fidelity_all_chunks(self) -> None:
        chunker = _make_chunker()
        text = "First sentence. Second sentence. Third sentence."

        chunks = chunker.chunk(text)

        assert all(text[c.start_char : c.end_char] == c.text for c in chunks)

    def test_window_context_metadata_populated(self) -> None:
        chunker = _make_chunker(window_size=1)
        text = "A. B. C."

        chunks = chunker.chunk(text)

        assert all("window_context" in c.metadata for c in chunks)

    def test_single_sentence_start_char_zero_end_char(self) -> None:
        chunker = _make_chunker()
        text = "Hello world."

        chunks = chunker.chunk(text)

        assert chunks[0].start_char == 0
        assert chunks[0].end_char == len("Hello world.")
