from abc import ABC, abstractmethod

from mindlm.core.models import Chunk


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]: ...
