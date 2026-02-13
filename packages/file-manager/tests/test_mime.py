"""Tests for MIME type module."""

import pytest

from file_manager.mime import (
    get_content_type_header,
    get_extension_for_mime,
    get_file_category,
    get_mime_type,
    is_document,
    is_image,
    validate_extension,
)


class TestMimeType:
    """Tests for MIME type functions."""

    def test_get_mime_type_pdf(self):
        """Test MIME type for PDF files."""
        assert get_mime_type("document.pdf") == "application/pdf"

    def test_get_mime_type_docx(self):
        """Test MIME type for DOCX files."""
        mime = get_mime_type("document.docx")
        assert "wordprocessingml" in mime

    def test_get_mime_type_markdown(self):
        """Test MIME type for Markdown files."""
        assert get_mime_type("README.md") == "text/markdown"

    def test_get_mime_type_txt(self):
        """Test MIME type for text files."""
        assert get_mime_type("file.txt") == "text/plain"

    def test_get_mime_type_unknown(self):
        """Test MIME type for unknown extension."""
        assert get_mime_type("file.xyz123") == "application/octet-stream"

    def test_get_mime_type_epub(self):
        """Test MIME type for EPUB files."""
        assert get_mime_type("book.epub") == "application/epub+zip"


class TestExtensionForMime:
    """Tests for reverse MIME lookup."""

    def test_get_extension_for_pdf(self):
        """Test extension for PDF MIME type."""
        ext = get_extension_for_mime("application/pdf")
        assert ext == ".pdf"

    def test_get_extension_for_markdown(self):
        """Test extension for Markdown MIME type."""
        ext = get_extension_for_mime("text/markdown")
        assert ext == ".md"

    def test_get_extension_unknown(self):
        """Test extension for unknown MIME type."""
        ext = get_extension_for_mime("application/x-unknown-type")
        assert ext is None


class TestFileCategory:
    """Tests for file categorization."""

    def test_get_category_pdf(self):
        """Test PDF category."""
        assert get_file_category("document.pdf") == "pdf"

    def test_get_category_word(self):
        """Test Word document category."""
        assert get_file_category("document.docx") == "word"
        assert get_file_category("document.doc") == "word"

    def test_get_category_spreadsheet(self):
        """Test spreadsheet category."""
        assert get_file_category("data.xlsx") == "spreadsheet"
        assert get_file_category("data.csv") == "spreadsheet"

    def test_get_category_image(self):
        """Test image category."""
        assert get_file_category("photo.jpg") == "image"
        assert get_file_category("photo.png") == "image"

    def test_get_category_unknown(self):
        """Test unknown category."""
        assert get_file_category("file.xyz") is None


class TestIsDocument:
    """Tests for document type checking."""

    def test_is_document_pdf(self):
        """Test PDF is document."""
        assert is_document("document.pdf") is True

    def test_is_document_word(self):
        """Test Word file is document."""
        assert is_document("document.docx") is True

    def test_is_document_image(self):
        """Test image is not document."""
        assert is_document("photo.jpg") is False

    def test_is_document_unknown(self):
        """Test unknown is not document."""
        assert is_document("file.xyz") is False


class TestIsImage:
    """Tests for image type checking."""

    def test_is_image_jpg(self):
        """Test JPG is image."""
        assert is_image("photo.jpg") is True

    def test_is_image_png(self):
        """Test PNG is image."""
        assert is_image("photo.png") is True

    def test_is_image_pdf(self):
        """Test PDF is not image."""
        assert is_image("document.pdf") is False


class TestValidateExtension:
    """Tests for extension validation."""

    def test_valid_extension(self):
        """Test valid extension."""
        is_valid, error = validate_extension("document.pdf", [".pdf", ".docx"])
        assert is_valid is True
        assert error == ""

    def test_invalid_extension(self):
        """Test invalid extension."""
        is_valid, error = validate_extension("script.exe", [".pdf", ".docx"])
        assert is_valid is False
        assert ".exe" in error

    def test_no_extension(self):
        """Test file without extension."""
        is_valid, error = validate_extension("filename", [".pdf", ".docx"])
        assert is_valid is False
        assert "no extension" in error.lower()

    def test_case_insensitive(self):
        """Test case-insensitive validation."""
        is_valid, error = validate_extension("document.PDF", [".pdf"])
        assert is_valid is True


class TestContentTypeHeader:
    """Tests for Content-Type header generation."""

    def test_text_content_type(self):
        """Test text content type includes charset."""
        header = get_content_type_header("file.txt")
        assert "text/plain" in header
        assert "charset=utf-8" in header

    def test_binary_content_type(self):
        """Test binary content type without charset."""
        header = get_content_type_header("file.pdf")
        assert header == "application/pdf"
