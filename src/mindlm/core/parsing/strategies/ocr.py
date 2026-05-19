from pathlib import Path

import fitz
from PIL import Image
from surya.models import FoundationPredictor, RecognitionPredictor
from surya.settings import settings

from mindlm.core.exceptions import ParseError
from mindlm.core.models import ParsedDocument
from mindlm.core.parsing.base import DocumentParser

_IMAGE_EXTS = {".png", ".jpeg", ".jpg"}
_PDF_EXTS = {".pdf"}


def _load_predictor() -> RecognitionPredictor:
    return RecognitionPredictor(
        FoundationPredictor(checkpoint=settings.RECOGNITION_MODEL_CHECKPOINT)
    )


def _extract_text(results: list) -> list[str]:
    texts: list[str] = []
    for result in results:
        for line in result.text_lines:
            texts.append(line.text)
    return texts


class OcrParser(DocumentParser):
    def parse(self, path: Path) -> ParsedDocument:
        suffix = path.suffix.lower()
        if suffix in _IMAGE_EXTS:
            return self._parse_image(path)
        if suffix in _PDF_EXTS:
            return self._parse_pdf(path)
        raise ParseError(str(path), "ocr")

    def _parse_image(self, path: Path) -> ParsedDocument:
        predictor = _load_predictor()
        image = Image.open(path)
        results = predictor([image])
        return ParsedDocument(text="\n".join(_extract_text(results)), page_breaks=[])

    def _parse_pdf(self, path: Path) -> ParsedDocument:
        predictor = _load_predictor()
        doc = fitz.open(str(path))
        page_texts: list[str] = []
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            results = predictor([image])
            page_texts.append("\n".join(_extract_text(results)))
        pos = 0
        page_breaks: list[int] = []
        for i, pt in enumerate(page_texts):
            pos += len(pt)
            page_breaks.append(pos)
            if i < len(page_texts) - 1:
                pos += 1  # advance past the "\n" separator
        return ParsedDocument(text="\n".join(page_texts), page_breaks=page_breaks)
