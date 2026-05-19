from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mindlm.core.exceptions import ParseError
from mindlm.core.parsing.strategies.raw import RawParser
from mindlm.core.parsing.strategies.structured import StructuredParser


class TestRawParser:
    def test_parse_markdown(self, tmp_path: Path) -> None:
        # Arrange
        doc = tmp_path / "doc.md"
        doc.write_text("# Title\nBody", encoding="utf-8")
        parser = RawParser()

        # Act
        result = parser.parse(doc)

        # Assert
        assert result.text == "# Title\nBody"

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
        assert "Hello World" in result.text

    def test_parse_txt(self, tmp_path: Path) -> None:
        # Arrange
        doc = tmp_path / "doc.txt"
        doc.write_text("plain text", encoding="utf-8")
        parser = RawParser()

        # Act
        result = parser.parse(doc)

        # Assert
        assert result.text == "plain text"

    def test_pdf_no_page_breaks_for_non_pdf(self, tmp_path: Path) -> None:
        # Arrange
        doc = tmp_path / "doc.md"
        doc.write_text("# Hello", encoding="utf-8")
        parser = RawParser()

        # Act
        result = parser.parse(doc)

        # Assert
        assert result.page_breaks == []

    def test_single_page_pdf_page_breaks(self, tmp_path: Path) -> None:
        # Arrange
        mock_page = MagicMock()
        mock_page.get_text.return_value = "hello"
        mock_doc = [mock_page]

        with patch(
            "mindlm.core.parsing.strategies.raw.fitz.open", return_value=mock_doc
        ):
            parser = RawParser()

            # Act
            result = parser.parse(tmp_path / "doc.pdf")

        # Assert
        assert result.page_breaks == [5]
        assert result.text == "hello"

    def test_two_page_pdf_page_breaks(self, tmp_path: Path) -> None:
        # Arrange
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "abc"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "def"
        mock_doc = [mock_page1, mock_page2]

        with patch(
            "mindlm.core.parsing.strategies.raw.fitz.open", return_value=mock_doc
        ):
            parser = RawParser()

            # Act
            result = parser.parse(tmp_path / "doc.pdf")

        # Assert
        assert result.page_breaks == [3, 7]
        assert result.text == "abc\ndef"

    def test_empty_pdf_returns_empty(self, tmp_path: Path) -> None:
        # Arrange
        mock_doc: list = []

        with patch(
            "mindlm.core.parsing.strategies.raw.fitz.open", return_value=mock_doc
        ):
            parser = RawParser()

            # Act
            result = parser.parse(tmp_path / "doc.pdf")

        # Assert
        assert result.page_breaks == []
        assert result.text == ""


class TestStructuredParser:
    def test_parse_markdown_passthrough(self, tmp_path: Path) -> None:
        # Arrange
        doc = tmp_path / "doc.md"
        doc.write_text("# Heading\nContent", encoding="utf-8")
        parser = StructuredParser()

        # Act
        result = parser.parse(doc)

        # Assert
        assert result.text == "# Heading\nContent"

    def test_parse_html_extracts_headers(self, tmp_path: Path) -> None:
        # Arrange
        html = tmp_path / "page.html"
        html.write_bytes(b"<h1>Title</h1><p>Paragraph text</p>")
        parser = StructuredParser()

        # Act
        result = parser.parse(html)

        # Assert
        assert "# Title" in result.text
        assert "Paragraph text" in result.text

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

        assert "# H1" in result.text

    def test_parse_html_h2_h3_li_elements(self, tmp_path: Path) -> None:
        html = tmp_path / "page.html"
        html.write_bytes(b"<h2>Sub</h2><h3>Deep</h3><li>Item</li>")
        parser = StructuredParser()

        result = parser.parse(html)

        assert "## Sub" in result.text
        assert "### Deep" in result.text
        assert "- Item" in result.text

    def test_markdown_has_no_page_breaks(self, tmp_path: Path) -> None:
        # Arrange
        doc = tmp_path / "doc.md"
        doc.write_text("# Hello", encoding="utf-8")
        parser = StructuredParser()

        # Act
        result = parser.parse(doc)

        # Assert
        assert result.page_breaks == []

    def test_two_page_pdf_has_two_page_breaks(self, tmp_path: Path) -> None:
        # Arrange
        block1 = (0, 0, 10, 10, "Block A", 0, 0)
        block2 = (0, 0, 10, 10, "Block B", 0, 0)
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = [block1]
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = [block2]
        mock_doc = [mock_page1, mock_page2]

        with patch(
            "mindlm.core.parsing.strategies.structured.fitz.open",
            return_value=mock_doc,
        ):
            parser = StructuredParser()

            # Act
            result = parser.parse(tmp_path / "doc.pdf")

        # Assert
        assert len(result.page_breaks) == 2
        assert result.page_breaks[-1] == len(result.text)

    def test_pdf_page_with_no_blocks(self, tmp_path: Path) -> None:
        # Arrange: one page returns empty blocks
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = []
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = []
        mock_doc = [mock_page1, mock_page2]

        with patch(
            "mindlm.core.parsing.strategies.structured.fitz.open",
            return_value=mock_doc,
        ):
            parser = StructuredParser()

            # Act
            result = parser.parse(tmp_path / "doc.pdf")

        # Assert: two pages → two entries in page_breaks
        assert len(result.page_breaks) == 2
