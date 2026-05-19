from abc import ABC, abstractmethod
from pathlib import Path

from mindlm.core.models import ParsedDocument


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument: ...
