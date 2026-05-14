from abc import ABC, abstractmethod

from mindlm.core.generation.base import LLMProvider


class BaseQueryProcessor(ABC):
    @abstractmethod
    def process(self, query: str, llm: LLMProvider) -> list[str]: ...
