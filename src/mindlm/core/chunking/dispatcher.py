from mindlm.core.chunking.base import BaseChunker
from mindlm.core.chunking.strategies.fixed import FixedChunker
from mindlm.core.chunking.strategies.recursive import RecursiveChunker
from mindlm.core.chunking.strategies.semantic import SemanticChunker
from mindlm.core.chunking.strategies.sentence_window import SentenceWindowChunker
from mindlm.core.chunking.strategies.sliding import SlidingChunker
from mindlm.core.config.models import ChunkingConfig
from mindlm.core.embeddings.base import EmbeddingProvider
from mindlm.core.models import Chunk


class ChunkerDispatcher:
    def __init__(
        self,
        config: ChunkingConfig,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        if config.strategy == "semantic" and embedding_provider is None:
            raise ValueError("EmbeddingProvider required for semantic chunking")
        self._config = config
        self._provider = embedding_provider
        self._chunker: BaseChunker = self._build()

    def _build(self) -> BaseChunker:
        match self._config.strategy:
            case "fixed":
                return FixedChunker(self._config)
            case "sliding":
                return SlidingChunker(self._config)
            case "semantic":
                assert self._provider is not None
                return SemanticChunker(self._config, self._provider)
            case "recursive":
                return RecursiveChunker(self._config)
            case "sentence_window":
                return SentenceWindowChunker(self._config)
            case _:  # pragma: no cover
                raise ValueError(
                    f"Unknown chunking strategy: {self._config.strategy!r}"
                )

    def chunk(self, text: str) -> list[Chunk]:
        return self._chunker.chunk(text)
