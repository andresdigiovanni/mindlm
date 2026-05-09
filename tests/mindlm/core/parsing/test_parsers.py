from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mindlm.core.exceptions import ParseError
from mindlm.core.parsing.strategies.raw import RawParser
from mindlm.core.parsing.strategies.structured import StructuredParser

if TYPE_CHECKING:
    from pathlib import Path


class TestRawParser:
    def test_parse_markdown(self, tmp_path: Path) -> None:
        # Arrange
        doc = tmp_path / "doc.md"
        doc.write_text("# Title\nBody", encoding="utf-8")
        parser = RawParser()

        # Act
        result = parser.parse(doc)

        # Assert
        assert result == "# Title\nBody"

    def test_parse_png_raises(self, tmp_path: Path) -> None:
        # Arrange
        img = tmp_path / "img.png"
        img.write_bytes(b"fake")
        parser = RawParser()

        # Act / Assert
        with pytest.raises(ParseError):
            parser.parse(img)

    def test_parse_unknown_extension_raises(self, tmp_path: Path) -> None:
        # Arrange
        f = tmp_path / "file.xyz"
        f.write_text("data", encoding="utf-8")
        parser = RawParser()

        # Act / Assert
        with pytest.raises(ParseError):
            parser.parse(f)

    def test_parse_html(self, tmp_path: Path) -> None:
        # Arrange
        html = tmp_path / "page.html"
        html.write_bytes(b"<p>Hello World</p>")
        parser = RawParser()

        # Act
        result = parser.parse(html)

        # Assert
        assert "Hello World" in result

    def test_parse_txt(self, tmp_path: Path) -> None:
        # Arrange
        doc = tmp_path / "doc.txt"
        doc.write_text("plain text", encoding="utf-8")
        parser = RawParser()

        # Act
        result = parser.parse(doc)

        # Assert
        assert result == "plain text"


class TestStructuredParser:
    def test_parse_markdown_passthrough(self, tmp_path: Path) -> None:
        # Arrange
        doc = tmp_path / "doc.md"
        doc.write_text("# Heading\nContent", encoding="utf-8")
        parser = StructuredParser()

        # Act
        result = parser.parse(doc)

        # Assert
        assert result == "# Heading\nContent"

    def test_parse_html_extracts_headers(self, tmp_path: Path) -> None:
        # Arrange
        html = tmp_path / "page.html"
        html.write_bytes(b"<h1>Title</h1><p>Paragraph text</p>")
        parser = StructuredParser()

        # Act
        result = parser.parse(html)

        # Assert
        assert "# Title" in result
        assert "Paragraph text" in result

    def test_parse_unknown_extension_raises(self, tmp_path: Path) -> None:
        # Arrange
        f = tmp_path / "file.xyz"
        f.write_text("data", encoding="utf-8")
        parser = StructuredParser()

        # Act / Assert
        with pytest.raises(ParseError):
            parser.parse(f)

    def test_parse_markdown_file_extension(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.markdown"
        doc.write_text("# H1\n\nContent.", encoding="utf-8")
        parser = StructuredParser()

        result = parser.parse(doc)

        assert "# H1" in result

    def test_parse_html_h2_h3_li_elements(self, tmp_path: Path) -> None:
        html = tmp_path / "page.html"
        html.write_bytes(b"<h2>Sub</h2><h3>Deep</h3><li>Item</li>")
        parser = StructuredParser()

        result = parser.parse(html)

        assert "## Sub" in result
        assert "### Deep" in result
        assert "- Item" in result
