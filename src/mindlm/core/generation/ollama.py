import contextlib
import json
import logging
from collections.abc import Iterator

import httpx
from langfuse.decorators import observe

from mindlm.core.config.models import LLMConfig
from mindlm.core.exceptions import LLMUnavailableError
from mindlm.core.generation.base import LLMProvider

logger = logging.getLogger(__name__)


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

    def ensure_model(self) -> None:
        """Pull the configured model if it is not already available locally."""
        if not self.healthcheck():
            raise LLMUnavailableError(
                f"Ollama not available at {self._config.base_url}. "
                "Verify that the service is running."
            )
        response = self._client.get("/api/tags")
        available = [m["model"] for m in response.json().get("models", [])]
        model = self._config.model
        if not any(name == model or name.startswith(model + ":") for name in available):
            logger.info("Pulling Ollama model '%s' (this may take a while)...", model)
            self._client.post(
                "/api/pull",
                json={"model": model, "stream": False},
                timeout=httpx.Timeout(3600.0),
            )
            logger.info("Ollama model '%s' ready.", model)
        else:
            logger.info("Ollama model '%s' already available.", model)

    @observe(as_type="generation", name="ollama-chat")
    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        if not self.healthcheck():
            raise LLMUnavailableError(
                f"Ollama not available at {self._config.base_url}. "
                "Verify that the service is running."
            )
        payload: dict[str, object] = {
            "model": self._config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self._config.temperature,
                "num_predict": self._config.max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"
        try:
            response = self._client.post("/api/chat", json=payload)
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMUnavailableError(
                f"Ollama request failed for model '{self._config.model}': {exc}"
            ) from exc
        if "error" in data:
            raise LLMUnavailableError(
                f"Ollama error for model '{self._config.model}': {data['error']}. "
                f"Pull the model first: ollama pull {self._config.model}"
            )
        return str(data["message"]["content"])

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
