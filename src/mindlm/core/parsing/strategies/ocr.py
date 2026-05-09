from pathlib import Path

from mindlm.core.exceptions import ParseError
from mindlm.core.parsing.base import DocumentParser

_IMAGE_EXTS = {".png", ".jpeg", ".jpg"}
_PDF_EXTS = {".pdf"}


class OcrParser(DocumentParser):
    def parse(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in _IMAGE_EXTS:
            return self._parse_image(path)
        if suffix in _PDF_EXTS:
            return self._parse_pdf(path)
        raise ParseError(str(path), "ocr")

    def _parse_image(self, path: Path) -> str:
        from PIL import Image
        from surya.model.recognition.model import load_model
        from surya.model.recognition.processor import load_processor
        from surya.recognition import run_recognition

        image = Image.open(path)
        model = load_model()
        processor = load_processor()
        results = run_recognition([image], [None], model, processor)
        texts: list[str] = []
        for result in results:
            for line in result.text_lines:
                texts.append(line.text)
        return "\n".join(texts)

    def _parse_pdf(self, path: Path) -> str:
        import fitz
        from PIL import Image
        from surya.model.recognition.model import load_model
        from surya.model.recognition.processor import load_processor
        from surya.recognition import run_recognition

        model = load_model()
        processor = load_processor()
        doc = fitz.open(str(path))
        all_texts: list[str] = []
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            results = run_recognition([image], [None], model, processor)
            for result in results:
                for line in result.text_lines:
                    all_texts.append(line.text)
        return "\n".join(all_texts)
