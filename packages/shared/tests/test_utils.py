"""Tests for shared utilities."""

import pytest

from shared.utils import (
    clean_text,
    count_tokens_estimate,
    extract_title,
    normalize_whitespace,
    split_into_sentences,
    split_text,
    truncate_text,
)


class TestCleanText:
    """Tests for clean_text function."""

    def test_clean_empty_text(self):
        """Test cleaning empty string."""
        assert clean_text("") == ""

    def test_clean_whitespace(self):
        """Test normalizing whitespace."""
        assert clean_text("hello   world") == "hello world"

    def test_clean_excessive_newlines(self):
        """Test removing excessive newlines."""
        assert clean_text("hello\n\n\n\nworld") == "hello\n\nworld"

    def test_clean_control_characters(self):
        """Test removing control characters."""
        assert clean_text("hello\x00world") == "helloworld"

    def test_preserve_single_newlines(self):
        """Test preserving single newlines."""
        assert clean_text("hello\nworld") == "hello\nworld"


class TestSplitIntoSentences:
    """Tests for split_into_sentences function."""

    def test_split_empty_text(self):
        """Test splitting empty string."""
        assert split_into_sentences("") == []

    def test_split_single_sentence(self):
        """Test splitting single sentence."""
        result = split_into_sentences("Hello world.")
        assert len(result) == 1

    def test_split_multiple_sentences(self):
        """Test splitting multiple sentences."""
        result = split_into_sentences("Hello world. How are you? I am fine!")
        assert len(result) == 3


class TestSplitText:
    """Tests for split_text function."""

    def test_split_empty_text(self):
        """Test splitting empty string."""
        assert split_text("") == []

    def test_split_short_text(self):
        """Test text shorter than chunk size."""
        result = split_text("Hello world", chunk_size=100)
        assert len(result) == 1
        assert result[0] == "Hello world"

    def test_split_long_text(self):
        """Test splitting long text."""
        text = "A" * 2000
        result = split_text(text, chunk_size=500, chunk_overlap=100)
        assert len(result) > 1

    def test_split_respects_chunk_size(self):
        """Test chunks don't exceed chunk_size."""
        text = "word " * 500
        result = split_text(text, chunk_size=100, chunk_overlap=20)
        for chunk in result:
            assert len(chunk) <= 100 + 20  # Allow some flexibility for word boundaries

    def test_split_with_paragraphs(self):
        """Test splitting prefers paragraph boundaries."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = split_text(text, chunk_size=30, chunk_overlap=5)
        assert len(result) >= 2


class TestTruncateText:
    """Tests for truncate_text function."""

    def test_truncate_short_text(self):
        """Test text shorter than max_length."""
        assert truncate_text("Hello", max_length=10) == "Hello"

    def test_truncate_long_text(self):
        """Test truncating long text."""
        result = truncate_text("Hello world, how are you?", max_length=15)
        assert len(result) <= 15
        assert result.endswith("...")

    def test_truncate_empty_text(self):
        """Test truncating empty text."""
        assert truncate_text("", max_length=10) == ""

    def test_truncate_none_text(self):
        """Test truncating None."""
        assert truncate_text(None, max_length=10) == ""

    def test_truncate_custom_suffix(self):
        """Test custom suffix."""
        result = truncate_text("Hello world", max_length=8, suffix="…")
        assert result.endswith("…")


class TestNormalizeWhitespace:
    """Tests for normalize_whitespace function."""

    def test_normalize_empty(self):
        """Test normalizing empty string."""
        assert normalize_whitespace("") == ""

    def test_normalize_multiple_spaces(self):
        """Test collapsing multiple spaces."""
        assert normalize_whitespace("hello   world") == "hello world"

    def test_normalize_tabs_and_newlines(self):
        """Test converting tabs and newlines to spaces."""
        assert normalize_whitespace("hello\t\nworld") == "hello world"


class TestExtractTitle:
    """Tests for extract_title function."""

    def test_extract_from_empty(self):
        """Test extracting from empty string."""
        assert extract_title("") is None

    def test_extract_first_line(self):
        """Test extracting first line as title."""
        assert extract_title("My Title\n\nContent here") == "My Title"

    def test_extract_with_max_length(self):
        """Test title truncated to max_length."""
        title = extract_title("A" * 200, max_length=50)
        assert len(title) <= 50

    def test_skip_short_lines(self):
        """Test skipping very short lines."""
        assert extract_title("ab\nReal Title") == "Real Title"


class TestCountTokensEstimate:
    """Tests for count_tokens_estimate function."""

    def test_count_empty(self):
        """Test counting empty string."""
        assert count_tokens_estimate("") == 0

    def test_count_single_word(self):
        """Test counting single word."""
        result = count_tokens_estimate("hello")
        assert result >= 1

    def test_count_multiple_words(self):
        """Test counting multiple words."""
        result = count_tokens_estimate("hello world how are you")
        assert result >= 5  # At least word count
