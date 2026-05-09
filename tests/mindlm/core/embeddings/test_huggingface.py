from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mindlm.core.config.models import EmbeddingsConfig
from mindlm.core.embeddings.huggingface import HuggingFaceEmbeddingProvider
from mindlm.core.exceptions import EmbeddingError


def _config() -> EmbeddingsConfig:
    return EmbeddingsConfig(provider="huggingface", model="test-model", dimensions=4)


class TestHuggingFaceEmbeddingProvider:
    def test_embed_one_shape(self) -> None:
        with patch("mindlm.core.embeddings.huggingface.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
            MockST.return_value = mock_model

            provider = HuggingFaceEmbeddingProvider(_config())
            result = provider.embed_one("test text")

        assert len(result) == 4
        assert isinstance(result[0], float)

    def test_embed_batch(self) -> None:
        with patch("mindlm.core.embeddings.huggingface.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[0.1] * 4, [0.2] * 4, [0.3] * 4])
            MockST.return_value = mock_model

            provider = HuggingFaceEmbeddingProvider(_config())
            result = provider.embed(["a", "b", "c"])

        assert len(result) == 3
        assert len(result[0]) == 4

    def test_embed_raises_embedding_error_on_runtime_error(self) -> None:
        with patch("mindlm.core.embeddings.huggingface.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.side_effect = RuntimeError("CUDA error")
            MockST.return_value = mock_model

            provider = HuggingFaceEmbeddingProvider(_config())

            with pytest.raises(EmbeddingError):
                provider.embed(["text"])

    def test_dimensions_property(self) -> None:
        with patch("mindlm.core.embeddings.huggingface.SentenceTransformer"):
            provider = HuggingFaceEmbeddingProvider(_config())

            assert provider.dimensions == 4

    def test_embed_one_delegates_to_embed(self) -> None:
        with patch("mindlm.core.embeddings.huggingface.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[0.5, 0.6, 0.7, 0.8]])
            MockST.return_value = mock_model

            provider = HuggingFaceEmbeddingProvider(_config())
            result = provider.embed_one("single text")

        assert len(result) == 4
        assert result[0] == pytest.approx(0.5)
