from pathlib import Path

import fitz
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pptx import Presentation

from mindlm.core.exceptions import ParseError
from mindlm.core.parsing.base import DocumentParser


class RawParser(DocumentParser):
    def parse(self, path: Path) -> str:
        suffix = path.suffix.lower()
        match suffix:
            case ".pdf":
                doc = fitz.open(str(path))
                return "\n".join(str(page.get_text()) for page in doc)
            case ".html" | ".htm":
                soup = BeautifulSoup(path.read_bytes(), "lxml")
                return str(soup.get_text(separator="\n", strip=True))
            case ".md" | ".markdown" | ".txt":
                return path.read_text(encoding="utf-8")
            case ".docx":
                doc = DocxDocument(str(path))
                return "\n".join(para.text for para in doc.paragraphs)
            case ".pptx":
                prs = Presentation(str(path))
                texts: list[str] = []
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if shape.has_text_frame:
                            texts.append(shape.text_frame.text)
                return "\n".join(texts)
            case _:
                raise ParseError(str(path), "raw")
