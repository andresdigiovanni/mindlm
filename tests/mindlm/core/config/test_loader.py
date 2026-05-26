from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mindlm.core.config.loader import load_config
from mindlm.core.config.models import RAGConfig

FIXTURES = Path(__file__).parent.parent.parent.parent / "fixtures"


class TestLoadConfig:
    def test_load_valid_config(self) -> None:
        # Arrange
        path = FIXTURES / "config_valid.yaml"

        # Act
        result = load_config(path)

        # Assert
        assert isinstance(result, RAGConfig)
        assert result.embeddings.dimensions == 384

    def test_load_missing_file(self) -> None:
        # Arrange
        path = Path("does_not_exist.yaml")

        # Act / Assert
        with pytest.raises(FileNotFoundError):
            load_config(path)

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        # Arrange
        bad = tmp_path / "bad.yaml"
        bad.write_text("key: [unclosed", encoding="utf-8")

        # Act / Assert
        with pytest.raises(yaml.YAMLError):
            load_config(bad)

    def test_semantic_requires_model(self) -> None:
        # Arrange
        path = FIXTURES / "config_invalid_semantic.yaml"

        # Act / Assert
        with pytest.raises(ValidationError, match="semantic_model"):
            load_config(path)

    def test_reranking_disabled_by_default(self, tmp_path: Path) -> None:
        # Arrange
        minimal = tmp_path / "minimal.yaml"
        minimal.write_text(
            "llm:\n  provider: ollama\n  model: gemma4\n  base_url: http://localhost\n"
            "embeddings:\n  provider: huggingface\n  model: test\n  dimensions: 1\n"
            "vector_store:\n  provider: qdrant\n  mode: local\n  host: localhost\n  port: 6333\n  collection: docs\n",
            encoding="utf-8",
        )

        # Act
        result = load_config(minimal)

        # Assert
        assert result.reranking.enabled is False

    def test_chunk_size_must_be_positive(self, tmp_path: Path) -> None:
        # Arrange
        bad = tmp_path / "bad_chunk.yaml"
        bad.write_text(
            "llm:\n  provider: ollama\n  model: gemma4\n  base_url: http://localhost\n"
            "embeddings:\n  provider: huggingface\n  model: test\n  dimensions: 1\n"
            "vector_store:\n  provider: qdrant\n  mode: local\n  host: localhost\n  port: 6333\n  collection: docs\n"
            "chunking:\n  strategy: fixed\n  chunk_size: 0\n  overlap: 0\n",
            encoding="utf-8",
        )

        # Act / Assert
        with pytest.raises(ValidationError):
            load_config(bad)

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        # Arrange
        empty = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")

        # Act / Assert
        with pytest.raises(ValueError, match="empty or not a valid YAML mapping"):
            load_config(empty)
