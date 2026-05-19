from pathlib import Path

from mindlm.core.config.models import IngestionConfig
from mindlm.core.models import ParsedDocument
from mindlm.core.parsing.base import DocumentParser
from mindlm.core.parsing.strategies.ocr import OcrParser
from mindlm.core.parsing.strategies.raw import RawParser
from mindlm.core.parsing.strategies.structured import StructuredParser


class ParserDispatcher:
    def __init__(self, config: IngestionConfig) -> None:
        self._config = config
        self._raw = RawParser()
        self._structured = StructuredParser()
        self._ocr = OcrParser()

    def _parser(self) -> DocumentParser:
        match self._config.parsing_strategy:
            case "raw":
                return self._raw
            case "structured":
                return self._structured
            case "ocr":
                return self._ocr
            case _:  # pragma: no cover
                raise ValueError(
                    f"Unknown parsing strategy: {self._config.parsing_strategy!r}"
                )

    def parse(self, path: Path) -> ParsedDocument:
        return self._parser().parse(path)
