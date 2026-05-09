from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_one(self, text: str) -> list[float]: ...

    @property
    @abstractmethod
    def dimensions(self) -> int: ...
