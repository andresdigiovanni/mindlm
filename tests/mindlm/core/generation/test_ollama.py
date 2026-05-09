import json as _json
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import httpx
import pytest

from mindlm.core.config.models import LLMConfig
from mindlm.core.exceptions import LLMUnavailableError
from mindlm.core.generation.ollama import OllamaProvider


def _config() -> LLMConfig:
    return LLMConfig(
        provider="ollama",
        model="llama3",
        base_url="http://localhost:11434",
        temperature=0.7,
        max_tokens=1024,
    )


class TestOllamaProvider:
    def test_healthcheck_ok(self) -> None:
        with patch("httpx.Client.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            provider = OllamaProvider(_config())

            result = provider.healthcheck()

            assert result is True

    def test_healthcheck_fail_connect_error(self) -> None:
        with patch("httpx.Client.get", side_effect=httpx.ConnectError("refused")):
            provider = OllamaProvider(_config())

            result = provider.healthcheck()

            assert result is False

    def test_healthcheck_fail_non_200(self) -> None:
        with patch("httpx.Client.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=503)
            provider = OllamaProvider(_config())

            result = provider.healthcheck()

            assert result is False

    def test_chat_returns_content(self) -> None:
        with (
            patch("httpx.Client.get") as mock_get,
            patch("httpx.Client.post") as mock_post,
        ):
            mock_get.return_value = MagicMock(status_code=200)
            mock_post.return_value = MagicMock(
                json=lambda: {"message": {"content": "answer"}}
            )
            provider = OllamaProvider(_config())

            result = provider.chat([{"role": "user", "content": "hi"}])

            assert result == "answer"

    def test_chat_raises_when_unhealthy(self) -> None:
        with patch("httpx.Client.get", side_effect=httpx.ConnectError("refused")):
            provider = OllamaProvider(_config())

            with pytest.raises(LLMUnavailableError):
                provider.chat([{"role": "user", "content": "hi"}])

    def test_stream_yields_chunks(self) -> None:
        class FakeStreamCtx:
            def __enter__(self) -> "FakeStreamCtx":
                return self

            def __exit__(self, *_a: object) -> None:
                pass

            def iter_lines(self) -> Iterator[str]:
                yield _json.dumps({"message": {"content": "chunk1"}, "done": False})
                yield _json.dumps({"message": {"content": "chunk2"}, "done": False})
                yield _json.dumps({"done": True})

        def fake_stream(*_args: object, **_kwargs: object) -> "FakeStreamCtx":
            return FakeStreamCtx()

        with (
            patch("httpx.Client.get") as mock_get,
            patch("httpx.Client.stream", fake_stream),
        ):
            mock_get.return_value = MagicMock(status_code=200)
            provider = OllamaProvider(_config())

            chunks = list(provider.stream([{"role": "user", "content": "hi"}]))

        assert chunks == ["chunk1", "chunk2"]

    def test_stream_raises_when_unhealthy(self) -> None:
        with patch("httpx.Client.get", side_effect=httpx.ConnectError("refused")):
            provider = OllamaProvider(_config())

            with pytest.raises(LLMUnavailableError):
                list(provider.stream([{"role": "user", "content": "hi"}]))
