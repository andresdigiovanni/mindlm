import contextlib
import json
from collections.abc import Iterator

import httpx

from mindlm.core.config.models import LLMConfig
from mindlm.core.exceptions import LLMUnavailableError
from mindlm.core.generation.base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=httpx.Timeout(120.0),
        )

    def healthcheck(self) -> bool:
        try:
            response = self._client.get("/api/tags")
            return bool(response.status_code == 200)
        except (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.ReadError,
            httpx.RemoteProtocolError,
        ):
            return False

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not self.healthcheck():
            raise LLMUnavailableError(
                f"Ollama not available at {self._config.base_url}. "
                "Verify that the service is running."
            )
        response = self._client.post(
            "/api/chat",
            json={
                "model": self._config.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self._config.temperature,
                    "num_predict": self._config.max_tokens,
                },
            },
        )
        return str(response.json()["message"]["content"])

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        if not self.healthcheck():
            raise LLMUnavailableError(
                f"Ollama not available at {self._config.base_url}. "
                "Verify that the service is running."
            )
        with self._client.stream(
            "POST",
            "/api/chat",
            json={
                "model": self._config.model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": self._config.temperature,
                    "num_predict": self._config.max_tokens,
                },
            },
        ) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("done"):
                    break
                yield str(data["message"]["content"])

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._client.close()
