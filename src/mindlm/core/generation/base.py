from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMProvider(ABC):
    @abstractmethod
    def chat(
        self, messages: list[dict[str, str]], *, json_mode: bool = False
    ) -> str: ...

    @abstractmethod
    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]: ...

    @abstractmethod
    def healthcheck(self) -> bool: ...
