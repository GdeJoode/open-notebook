"""
Text processing utilities.

Common text operations used across pipelines.
"""

import re
from typing import List, Optional


def clean_text(text: str) -> str:
    """
    Clean text by normalizing whitespace and removing control characters.

    Args:
        text: The input text to clean.

    Returns:
        Cleaned text with normalized whitespace.
    """
    if not text:
        return ""

    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # Normalize whitespace (collapse multiple spaces, preserve newlines)
    text = re.sub(r'[^\S\n]+', ' ', text)

    # Remove excessive newlines (more than 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Args:
        text: The input text to split.

    Returns:
        List of sentences.
    """
    if not text:
        return []

    # Simple sentence splitting on common terminators
    # Handles common abbreviations like Mr., Mrs., Dr., etc.
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text)

    return [s.strip() for s in sentences if s.strip()]


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separator: str = "\n\n",
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: The input text to split.
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of characters to overlap between chunks.
        separator: Preferred separator for splitting.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    current_pos = 0

    while current_pos < len(text):
        # Determine chunk end position
        end_pos = min(current_pos + chunk_size, len(text))

        # If not at the end, try to find a good break point
        if end_pos < len(text):
            # Look for separator within the last portion of the chunk
            search_start = max(current_pos, end_pos - chunk_overlap)
            search_region = text[search_start:end_pos]

            # Try to break at paragraph boundary
            last_sep = search_region.rfind(separator)
            if last_sep != -1:
                end_pos = search_start + last_sep + len(separator)
            else:
                # Fall back to sentence boundary
                last_period = search_region.rfind('. ')
                if last_period != -1:
                    end_pos = search_start + last_period + 2
                else:
                    # Fall back to word boundary
                    last_space = search_region.rfind(' ')
                    if last_space != -1:
                        end_pos = search_start + last_space + 1

        chunk = text[current_pos:end_pos].strip()
        if chunk:
            chunks.append(chunk)

        # Move position accounting for overlap
        if end_pos >= len(text):
            break

        current_pos = end_pos - chunk_overlap
        if current_pos >= len(text) - chunk_overlap:
            break

    return chunks


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length, adding suffix if truncated.

    Args:
        text: The input text to truncate.
        max_length: Maximum length including suffix.
        suffix: String to append when truncated.

    Returns:
        Truncated text with suffix if needed.
    """
    if not text or len(text) <= max_length:
        return text or ""

    truncate_at = max_length - len(suffix)
    if truncate_at <= 0:
        return suffix[:max_length]

    # Try to truncate at word boundary
    truncated = text[:truncate_at]
    last_space = truncated.rfind(' ')

    if last_space > truncate_at * 0.8:  # Only use space if reasonably close
        truncated = truncated[:last_space]

    return truncated.rstrip() + suffix


def normalize_whitespace(text: str) -> str:
    """
    Normalize all whitespace to single spaces.

    Args:
        text: The input text to normalize.

    Returns:
        Text with normalized whitespace.
    """
    if not text:
        return ""
    return ' '.join(text.split())


def extract_title(text: str, max_length: int = 100) -> Optional[str]:
    """
    Extract a title from the first meaningful line of text.

    Args:
        text: The input text.
        max_length: Maximum title length.

    Returns:
        Extracted title or None.
    """
    if not text:
        return None

    # Get first non-empty line
    for line in text.split('\n'):
        line = line.strip()
        if line and len(line) > 3:  # Skip very short lines
            return truncate_text(line, max_length)

    return None


def token_count(text: str) -> int:
    """
    Count tokens using tiktoken's ``o200k_base`` encoding.

    Falls back to :func:`count_tokens_estimate` when tiktoken is not installed.

    Args:
        text: The input text.

    Returns:
        Token count.
    """
    if not text:
        return 0
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text))
    except ImportError:
        return count_tokens_estimate(text)


def count_tokens_estimate(text: str) -> int:
    """
    Estimate token count for text (rough approximation).

    Uses simple word count * 1.3 factor as a rough estimate.
    For accurate counts, use a proper tokenizer.

    Args:
        text: The input text.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0

    # Rough estimate: words * 1.3
    word_count = len(text.split())
    return int(word_count * 1.3)


# ---------------------------------------------------------------------------
# Thinking-content utilities (migrated from open_notebook.utils.text_utils)
# ---------------------------------------------------------------------------

# Pattern for matching thinking content in AI responses
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def parse_thinking_content(content: str) -> tuple[str, str]:
    """
    Parse message content to extract thinking content from <think> tags.

    Args:
        content: The original message content.

    Returns:
        Tuple of (thinking_content, cleaned_content).
    """
    if not isinstance(content, str):
        return "", str(content) if content is not None else ""

    # Limit processing for very large content (100KB limit)
    if len(content) > 100000:
        return "", content

    thinking_matches = THINK_PATTERN.findall(content)

    if not thinking_matches:
        return "", content

    thinking_content = "\n\n".join(match.strip() for match in thinking_matches)
    cleaned_content = THINK_PATTERN.sub("", content)
    cleaned_content = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned_content).strip()

    return thinking_content, cleaned_content


def clean_thinking_content(content: str) -> str:
    """
    Remove thinking content from AI responses, returning only the cleaned content.

    Args:
        content: The original message content with potential <think> tags.

    Returns:
        Content with <think> blocks removed and whitespace cleaned.
    """
    _, cleaned_content = parse_thinking_content(content)
    return cleaned_content
