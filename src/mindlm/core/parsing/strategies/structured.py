from pathlib import Path

import fitz
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pptx import Presentation

from mindlm.core.exceptions import ParseError
from mindlm.core.parsing.base import DocumentParser


class StructuredParser(DocumentParser):
    def parse(self, path: Path) -> str:
        suffix = path.suffix.lower()
        match suffix:
            case ".pdf":
                doc = fitz.open(str(path))
                lines: list[str] = []
                for page in doc:
                    blocks = page.get_text("blocks")
                    sorted_blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
                    lines.extend(b[4].strip() for b in sorted_blocks if b[4].strip())
                return "\n".join(lines)
            case ".html" | ".htm":
                soup = BeautifulSoup(path.read_bytes(), "lxml")
                parts: list[str] = []
                for tag in soup.find_all(
                    ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th"]
                ):
                    name = tag.name
                    text = tag.get_text(strip=True)
                    if not text:
                        continue
                    match name:
                        case "h1":
                            parts.append(f"# {text}")
                        case "h2":
                            parts.append(f"## {text}")
                        case "h3" | "h4" | "h5" | "h6":
                            parts.append(f"### {text}")
                        case "li":
                            parts.append(f"- {text}")
                        case _:
                            parts.append(text)
                return "\n".join(parts)
            case ".md" | ".markdown":
                return path.read_text(encoding="utf-8")
            case ".docx":
                doc_x = DocxDocument(str(path))
                parts_d: list[str] = []
                for para in doc_x.paragraphs:
                    if not para.text.strip():
                        continue
                    style = para.style.name if para.style else ""
                    if "Heading 1" in style:
                        parts_d.append(f"# {para.text}")
                    elif "Heading 2" in style:
                        parts_d.append(f"## {para.text}")
                    else:
                        parts_d.append(para.text)
                return "\n".join(parts_d)
            case ".pptx":
                prs = Presentation(str(path))
                slide_texts: list[str] = []
                for i, slide in enumerate(prs.slides, 1):
                    title = ""
                    body_parts: list[str] = []
                    for shape in slide.shapes:
                        if shape.has_text_frame:
                            if shape.shape_type == 13:  # title placeholder
                                title = shape.text_frame.text
                            else:
                                body_parts.append(shape.text_frame.text)
                    header = f"## Slide {i}: {title}" if title else f"## Slide {i}"
                    slide_texts.append(header)
                    slide_texts.extend(body_parts)
                return "\n".join(slide_texts)
            case _:
                raise ParseError(str(path), "structured")
