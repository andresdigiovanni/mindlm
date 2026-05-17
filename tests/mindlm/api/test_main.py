from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import mindlm.api.dependencies as deps
from mindlm.api.main import app
from mindlm.core.config.models import ObservabilityConfig, RAGConfig


@pytest.fixture(autouse=True)
def clear_state() -> Generator[None, None, None]:
    yield
    app.dependency_overrides.clear()
    deps.get_config.cache_clear()


def _make_config(**kwargs: object) -> RAGConfig:
    obs = ObservabilityConfig(**kwargs)
    return RAGConfig(observability=obs)


class TestLifespanLangfuseInit:
    def test_should_configure_langfuse_context_with_config_values(
        self,
    ) -> None:
        config = _make_config(
            public_key="pk-test",
            secret_key="sk-test",  # noqa: S106
            host="http://lf:3000",
        )

        with (
            patch("mindlm.api.main.get_config", return_value=config),
            patch("mindlm.api.main.get_embedding_provider", return_value=MagicMock()),
            patch("mindlm.api.main.get_llm_provider", return_value=MagicMock()),
            patch("mindlm.api.main.langfuse_context") as mock_ctx,
        ):
            with TestClient(app):
                pass
            mock_ctx.configure.assert_called_once()
            call_kwargs = mock_ctx.configure.call_args.kwargs
            assert call_kwargs["public_key"] == "pk-test"
            assert call_kwargs["secret_key"] == "sk-test"  # noqa: S105
            assert call_kwargs["host"] == "http://lf:3000"

    def test_should_pass_flush_params_from_config(self) -> None:
        config = _make_config(flush_at=5, flush_interval=1.0)

        with (
            patch("mindlm.api.main.get_config", return_value=config),
            patch("mindlm.api.main.get_embedding_provider", return_value=MagicMock()),
            patch("mindlm.api.main.get_llm_provider", return_value=MagicMock()),
            patch("mindlm.api.main.langfuse_context") as mock_ctx,
        ):
            with TestClient(app):
                pass
            call_kwargs = mock_ctx.configure.call_args.kwargs
            assert call_kwargs["flush_at"] == 5
            assert call_kwargs["flush_interval"] == 1.0

    def test_should_flush_context_on_shutdown(self) -> None:
        config = _make_config(
            public_key="pk-test",
            secret_key="sk-test",  # noqa: S106
        )

        with (
            patch("mindlm.api.main.get_config", return_value=config),
            patch("mindlm.api.main.get_embedding_provider", return_value=MagicMock()),
            patch("mindlm.api.main.get_llm_provider", return_value=MagicMock()),
            patch("mindlm.api.main.langfuse_context") as mock_ctx,
        ):
            with TestClient(app):
                pass
            mock_ctx.flush.assert_called_once()


class TestLifespanLangfuseDisabled:
    def test_langfuse_disabled_configure_not_called(self) -> None:
        # Arrange
        config = _make_config(enabled=False)

        with (
            patch("mindlm.api.main.get_config", return_value=config),
            patch("mindlm.api.main.get_embedding_provider", return_value=MagicMock()),
            patch("mindlm.api.main.get_llm_provider", return_value=MagicMock()),
            patch("mindlm.api.main.langfuse_context") as mock_ctx,
        ):
            # Act
            with TestClient(app):
                pass

            # Assert — configure never called
            mock_ctx.configure.assert_not_called()

    def test_langfuse_disabled_flush_not_called(self) -> None:
        # Arrange
        config = _make_config(enabled=False)

        with (
            patch("mindlm.api.main.get_config", return_value=config),
            patch("mindlm.api.main.get_embedding_provider", return_value=MagicMock()),
            patch("mindlm.api.main.get_llm_provider", return_value=MagicMock()),
            patch("mindlm.api.main.langfuse_context") as mock_ctx,
        ):
            # Act
            with TestClient(app):
                pass

            # Assert — flush never called when disabled
            mock_ctx.flush.assert_not_called()
