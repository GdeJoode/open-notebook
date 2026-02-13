"""Tests for ChunkRegrouper."""

import pytest

from ingestion.chunking.extractor import Chunk
from ingestion.chunking.regrouper import ChunkRegrouper
from ingestion.config import RegroupingConfig, RegroupingStrategy


def _make_chunk(
    text: str,
    order: int = 0,
    element_type: str = "paragraph",
    physical_page: int = 1,
    section_path: list[str] | None = None,
    section_level: int = 1,
    chunk_type: str = "original",
) -> Chunk:
    """Helper to create test chunks."""
    return Chunk(
        text=text,
        order=order,
        element_type=element_type,
        physical_page=physical_page,
        section_path=section_path or [],
        section_level=section_level,
        chunk_type=chunk_type,
    )


class TestStructuralMerge:
    """Tests for structural merge strategy."""

    def test_basic_merge_same_section(self):
        """5 short paragraphs under same heading -> 1 merged chunk."""
        chunks = [
            _make_chunk(
                text=f"Paragraph {i} with some content.",
                order=i,
                section_path=["Chapter 1"],
            )
            for i in range(5)
        ]
        config = RegroupingConfig(
            strategy=RegroupingStrategy.STRUCTURAL,
            target_tokens=800,  # Well above total text
            prepend_section_context=False,
        )
        regrouper = ChunkRegrouper(config)
        merged = regrouper.regroup(chunks)

        assert len(merged) == 1
        assert merged[0].chunk_type == "merged"
        assert merged[0].source_element_count == 5
        assert merged[0].section_path == ["Chapter 1"]
        # All paragraph texts should be present
        for i in range(5):
            assert f"Paragraph {i}" in merged[0].text

    def test_section_boundary_splits(self):
        """Paragraphs under different headings -> separate chunks."""
        chunks = [
            _make_chunk(text="Para A1", order=0, section_path=["Heading A"]),
            _make_chunk(text="Para A2", order=1, section_path=["Heading A"]),
            _make_chunk(text="Para B1", order=2, section_path=["Heading B"]),
            _make_chunk(text="Para B2", order=3, section_path=["Heading B"]),
        ]
        config = RegroupingConfig(
            strategy=RegroupingStrategy.STRUCTURAL,
            target_tokens=800,
            prepend_section_context=False,
        )
        regrouper = ChunkRegrouper(config)
        merged = regrouper.regroup(chunks)

        assert len(merged) == 2
        assert "Para A1" in merged[0].text
        assert "Para A2" in merged[0].text
        assert "Para B1" in merged[1].text
        assert "Para B2" in merged[1].text

    def test_token_limit_splits(self):
        """Many paragraphs exceeding target_tokens -> multiple chunks."""
        # Each paragraph ~100 chars = ~25 tokens at 4 chars/token
        # Target 50 tokens = 200 chars -> should split every ~2 paragraphs
        chunks = [
            _make_chunk(
                text="X" * 100,
                order=i,
                section_path=["Chapter 1"],
            )
            for i in range(6)
        ]
        config = RegroupingConfig(
            strategy=RegroupingStrategy.STRUCTURAL,
            target_tokens=50,  # 200 chars
            max_tokens=75,  # 300 chars
            prepend_section_context=False,
        )
        regrouper = ChunkRegrouper(config)
        merged = regrouper.regroup(chunks)

        # Should produce multiple chunks since each paragraph is 100 chars
        # and target is 200 chars
        assert len(merged) > 1
        assert all(c.section_path == ["Chapter 1"] for c in merged)

    def test_table_standalone(self):
        """Table gets own chunk with section prefix."""
        chunks = [
            _make_chunk(text="Intro paragraph", order=0, section_path=["Methods"]),
            _make_chunk(
                text="| Col A | Col B |\n|-------|-------|\n| 1 | 2 |",
                order=1,
                element_type="table",
                section_path=["Methods"],
                chunk_type="table",
            ),
            _make_chunk(text="Follow-up paragraph", order=2, section_path=["Methods"]),
        ]
        config = RegroupingConfig(
            strategy=RegroupingStrategy.STRUCTURAL,
            target_tokens=800,
            tables_as_standalone=True,
            prepend_section_context=True,
        )
        regrouper = ChunkRegrouper(config)
        merged = regrouper.regroup(chunks)

        # Should be 3 chunks: intro, table, follow-up
        # (intro and follow-up are in separate groups because table flushes buffer)
        assert any(c.element_type == "table" for c in merged)
        table_chunk = next(c for c in merged if c.element_type == "table")
        assert table_chunk.chunk_type == "table"
        assert "[Section: Methods]" in table_chunk.text

    def test_heading_folds_into_content(self):
        """Heading text included in next merged chunk."""
        chunks = [
            _make_chunk(
                text="Introduction",
                order=0,
                element_type="heading",
                section_path=["Introduction"],
            ),
            _make_chunk(
                text="This is the body text of the introduction.",
                order=1,
                section_path=["Introduction"],
            ),
        ]
        config = RegroupingConfig(
            strategy=RegroupingStrategy.STRUCTURAL,
            target_tokens=800,
            prepend_section_context=False,
        )
        regrouper = ChunkRegrouper(config)
        merged = regrouper.regroup(chunks)

        assert len(merged) == 1
        assert "Introduction" in merged[0].text
        assert "body text" in merged[0].text
        assert merged[0].source_element_count == 2

    def test_section_prefix(self):
        """Verify [Section: X > Y] prefix on merged chunks."""
        chunks = [
            _make_chunk(
                text="Some content",
                order=0,
                section_path=["Chapter 1", "Section 1.2"],
            ),
        ]
        config = RegroupingConfig(
            strategy=RegroupingStrategy.STRUCTURAL,
            target_tokens=800,
            prepend_section_context=True,
        )
        regrouper = ChunkRegrouper(config)
        merged = regrouper.regroup(chunks)

        assert len(merged) == 1
        assert merged[0].text.startswith("[Section: Chapter 1 > Section 1.2]")

    def test_no_section_prefix_when_disabled(self):
        """No prefix when prepend_section_context is False."""
        chunks = [
            _make_chunk(
                text="Some content",
                order=0,
                section_path=["Chapter 1"],
            ),
        ]
        config = RegroupingConfig(
            strategy=RegroupingStrategy.STRUCTURAL,
            target_tokens=800,
            prepend_section_context=False,
        )
        regrouper = ChunkRegrouper(config)
        merged = regrouper.regroup(chunks)

        assert not merged[0].text.startswith("[Section:")


class TestSlidingWindow:
    """Tests for sliding window strategy."""

    def test_basic_window(self):
        """Long text -> overlapping windows."""
        # Create chunks totaling ~1000 chars
        chunks = [
            _make_chunk(text="A" * 200, order=i)
            for i in range(5)
        ]
        config = RegroupingConfig(
            strategy=RegroupingStrategy.SLIDING_WINDOW,
            target_tokens=100,  # 400 chars
            overlap_tokens=25,  # 100 chars overlap
        )
        regrouper = ChunkRegrouper(config)
        merged = regrouper.regroup(chunks)

        # Total text ~1000 chars (200*5) + separators
        # Window 400 chars, step 300 chars -> should get multiple chunks
        assert len(merged) > 1

    def test_overlap_check(self):
        """Adjacent chunks share approximately overlap_tokens text."""
        text_per_chunk = "word " * 100  # 500 chars per chunk
        chunks = [
            _make_chunk(text=text_per_chunk, order=i)
            for i in range(4)
        ]
        config = RegroupingConfig(
            strategy=RegroupingStrategy.SLIDING_WINDOW,
            target_tokens=200,  # 800 chars
            overlap_tokens=50,  # 200 chars overlap
        )
        regrouper = ChunkRegrouper(config)
        merged = regrouper.regroup(chunks)

        if len(merged) >= 2:
            # The end of chunk[0] should overlap with start of chunk[1]
            end_of_first = merged[0].text[-200:]
            start_of_second = merged[1].text[:200]
            # They should share some common text
            assert end_of_first == start_of_second


class TestNoneStrategy:
    """Tests for NONE strategy (passthrough)."""

    def test_passthrough(self):
        """Input == output when strategy is NONE."""
        chunks = [
            _make_chunk(text=f"Para {i}", order=i)
            for i in range(3)
        ]
        config = RegroupingConfig(strategy=RegroupingStrategy.NONE)
        regrouper = ChunkRegrouper(config)
        result = regrouper.regroup(chunks)

        assert len(result) == 3
        for i, c in enumerate(result):
            assert c.text == f"Para {i}"


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_input(self):
        """Empty list returns empty list."""
        config = RegroupingConfig(strategy=RegroupingStrategy.STRUCTURAL)
        regrouper = ChunkRegrouper(config)
        assert regrouper.regroup([]) == []

    def test_single_chunk(self):
        """Single chunk passes through."""
        chunks = [_make_chunk(text="Only chunk", order=0, section_path=["Intro"])]
        config = RegroupingConfig(
            strategy=RegroupingStrategy.STRUCTURAL,
            prepend_section_context=False,
        )
        regrouper = ChunkRegrouper(config)
        merged = regrouper.regroup(chunks)

        assert len(merged) == 1
        assert merged[0].text == "Only chunk"
        assert merged[0].chunk_type == "original"
        assert merged[0].source_element_count == 1

    def test_no_section_path(self):
        """Chunks without section_path group together."""
        chunks = [
            _make_chunk(text="Para 1", order=0, section_path=[]),
            _make_chunk(text="Para 2", order=1, section_path=[]),
        ]
        config = RegroupingConfig(
            strategy=RegroupingStrategy.STRUCTURAL,
            target_tokens=800,
            prepend_section_context=False,
        )
        regrouper = ChunkRegrouper(config)
        merged = regrouper.regroup(chunks)

        assert len(merged) == 1
        assert "Para 1" in merged[0].text
        assert "Para 2" in merged[0].text
