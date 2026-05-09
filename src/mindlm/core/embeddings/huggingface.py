import os

from sentence_transformers import SentenceTransformer

from mindlm.core.config.models import EmbeddingsConfig
from mindlm.core.embeddings.base import EmbeddingProvider
from mindlm.core.exceptions import EmbeddingError


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: EmbeddingsConfig) -> None:
        cache = os.environ.get("TRANSFORMERS_CACHE")
        self._model = SentenceTransformer(config.model, cache_folder=cache)
        self._config = config

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            result: list[list[float]] = self._model.encode(
                texts, convert_to_numpy=True
            ).tolist()
            return result
        except RuntimeError as exc:
            raise EmbeddingError(model=self._config.model) from exc

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimensions(self) -> int:
        return self._config.dimensions
