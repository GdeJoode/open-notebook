"""
Chunk regrouper for merging 1:1 element chunks into semantically meaningful units.

Provides strategies:
- Structural merge: Group by section headings, merge until target token count
- Sliding window: Fixed-size with overlap for headingless documents
- None: Keep 1:1 mapping (legacy)
"""

from __future__ import annotations

from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    pass

from ingestion.chunking.extractor import Chunk, ChunkPosition
from ingestion.config import RegroupingConfig, RegroupingStrategy


class ChunkRegrouper:
    """Regroups 1:1 element chunks into semantically meaningful units."""

    def __init__(self, config: RegroupingConfig):
        self.config = config

    def regroup(self, chunks: list[Chunk]) -> list[Chunk]:
        """Route to configured strategy.

        Filters out noise chunks (is_content=False) before regrouping.
        The is_content flag is set by ChunkExtractor at extraction time.
        """
        if not chunks:
            return []

        # Only regroup content chunks; noise was flagged by the extractor
        content_chunks = [c for c in chunks if c.is_content]
        if not content_chunks:
            return []

        noise_count = len(chunks) - len(content_chunks)
        if noise_count:
            logger.info(
                f"Regrouper skipped {noise_count} noise chunks "
                f"({len(content_chunks)} content chunks)"
            )

        if self.config.strategy == RegroupingStrategy.STRUCTURAL:
            return self._structural_merge(content_chunks)
        elif self.config.strategy == RegroupingStrategy.SLIDING_WINDOW:
            return self._sliding_window(content_chunks)
        return content_chunks  # NONE

    def _structural_merge(self, chunks: list[Chunk]) -> list[Chunk]:
        """Group by section_path, merge until target_tokens reached.

        Rules:
        1. Never merge across different section_path values
        2. Tables always standalone (with section context prefix)
        3. Headings fold into the next paragraph group
        4. When accumulated text > target_tokens, flush as merged chunk
        5. Leftover text at section boundary -> flush regardless of size
        """
        merged: list[Chunk] = []
        buffer: list[Chunk] = []
        buffer_text = ""
        current_section: list[str] | None = None
        order = 0

        target_chars = int(self.config.target_tokens * self.config.chars_per_token)
        max_chars = int(self.config.max_tokens * self.config.chars_per_token)

        for chunk in chunks:
            # Tables always standalone
            if self.config.tables_as_standalone and chunk.element_type == "table":
                # Flush any buffered text first
                if buffer:
                    merged_chunk = self._flush_buffer(
                        buffer, buffer_text, order, current_section
                    )
                    merged.append(merged_chunk)
                    order += 1
                    buffer = []
                    buffer_text = ""

                # Emit table as standalone chunk
                table_text = chunk.text
                if self.config.prepend_section_context and chunk.section_path:
                    table_text = self._build_section_prefix(chunk.section_path) + table_text

                table_chunk = Chunk(
                    text=table_text,
                    order=order,
                    element_type=chunk.element_type,
                    physical_page=chunk.physical_page,
                    printed_page=chunk.printed_page,
                    positions=list(chunk.positions),
                    chapter=chunk.chapter,
                    section=chunk.section,
                    metadata=dict(chunk.metadata),
                    section_path=list(chunk.section_path),
                    section_level=chunk.section_level,
                    chunk_type="table",
                    source_element_count=1,
                )
                merged.append(table_chunk)
                order += 1
                continue

            chunk_section = tuple(chunk.section_path)

            # Section boundary: flush buffer if section changed
            if current_section is not None and chunk_section != current_section:
                if buffer:
                    merged_chunk = self._flush_buffer(
                        buffer, buffer_text, order, list(current_section)
                    )
                    merged.append(merged_chunk)
                    order += 1
                    buffer = []
                    buffer_text = ""

            current_section = chunk_section

            # Check if adding this chunk would exceed max
            new_text = f"{buffer_text}\n\n{chunk.text}" if buffer_text else chunk.text
            if self._estimate_chars(new_text) > max_chars and buffer:
                # Flush current buffer first
                merged_chunk = self._flush_buffer(
                    buffer, buffer_text, order, list(current_section)
                )
                merged.append(merged_chunk)
                order += 1
                buffer = [chunk]
                buffer_text = chunk.text
            else:
                buffer.append(chunk)
                buffer_text = new_text

            # Check if we've reached target
            if self._estimate_chars(buffer_text) >= target_chars:
                merged_chunk = self._flush_buffer(
                    buffer, buffer_text, order, list(current_section)
                )
                merged.append(merged_chunk)
                order += 1
                buffer = []
                buffer_text = ""

        # Flush remaining buffer
        if buffer:
            merged_chunk = self._flush_buffer(
                buffer, buffer_text, order,
                list(current_section) if current_section is not None else []
            )
            merged.append(merged_chunk)

        return merged

    def _flush_buffer(
        self,
        buffer: list[Chunk],
        text: str,
        order: int,
        section_path: list[str] | None,
    ) -> Chunk:
        """Create a merged chunk from the buffer."""
        section_path = section_path or []
        first = buffer[0]

        # Prepend section context
        final_text = text
        if self.config.prepend_section_context and section_path:
            final_text = self._build_section_prefix(section_path) + text

        # Collect all positions
        all_positions: list[ChunkPosition] = []
        for c in buffer:
            all_positions.extend(c.positions)

        # Determine element type: if all same, keep it; otherwise "mixed"
        element_types = {c.element_type for c in buffer}
        # Filter out heading types since they fold into content
        content_types = element_types - {"heading", "title"}
        if len(content_types) == 1:
            element_type = content_types.pop()
        elif not content_types:
            element_type = first.element_type
        else:
            element_type = "mixed"

        is_merged = len(buffer) > 1
        return Chunk(
            text=final_text,
            order=order,
            element_type=element_type,
            physical_page=first.physical_page,
            printed_page=first.printed_page,
            positions=all_positions,
            chapter=first.chapter,
            section=first.section,
            metadata=first.metadata,
            section_path=list(section_path),
            section_level=first.section_level,
            chunk_type="merged" if is_merged else "original",
            source_element_count=len(buffer),
        )

    def _sliding_window(self, chunks: list[Chunk]) -> list[Chunk]:
        """Fixed-size window with overlap for headingless documents.

        Concatenate all text, slide window of target_tokens with
        overlap_tokens overlap. Preserve first chunk's metadata.
        """
        if not chunks:
            return []

        # Concatenate all text
        full_text = "\n\n".join(c.text for c in chunks)
        target_chars = int(self.config.target_tokens * self.config.chars_per_token)
        overlap_chars = int(self.config.overlap_tokens * self.config.chars_per_token)
        step = max(target_chars - overlap_chars, 1)

        result: list[Chunk] = []
        order = 0
        pos = 0

        while pos < len(full_text):
            window_text = full_text[pos:pos + target_chars]
            if not window_text.strip():
                break

            # Find the source chunk for metadata
            source_chunk = chunks[0]
            char_count = 0
            for c in chunks:
                if char_count + len(c.text) + 2 >= pos:  # +2 for \n\n
                    source_chunk = c
                    break
                char_count += len(c.text) + 2

            chunk_type = "overlap" if pos > 0 and overlap_chars > 0 else "original"

            result.append(Chunk(
                text=window_text.strip(),
                order=order,
                element_type=source_chunk.element_type,
                physical_page=source_chunk.physical_page,
                printed_page=source_chunk.printed_page,
                positions=list(source_chunk.positions),
                chapter=source_chunk.chapter,
                section=source_chunk.section,
                metadata=dict(source_chunk.metadata),
                section_path=list(source_chunk.section_path),
                section_level=source_chunk.section_level,
                chunk_type=chunk_type,
                source_element_count=1,
            ))
            order += 1
            pos += step

        return result

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length."""
        return int(len(text) / self.config.chars_per_token)

    def _estimate_chars(self, text: str) -> int:
        """Return character length (for comparison with char-based limits)."""
        return len(text)

    def _build_section_prefix(self, section_path: list[str]) -> str:
        """Build a section context prefix string."""
        if not section_path or not self.config.prepend_section_context:
            return ""
        return f"[Section: {' > '.join(section_path)}]\n\n"
