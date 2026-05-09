from abc import ABC, abstractmethod

from mindlm.core.models import Result


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: list[Result]) -> list[Result]: ...
