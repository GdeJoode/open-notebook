"""Chunk merge/split mutations for the inspect workspace (Track I.D-3).

This service implements two structural edits an operator can make to a
source's chunk sequence from the inspect workspace:

* **merge** — fold a chunk into its immediate successor (by ``order``),
  concatenating text + positions and compacting the now-vacant ``order``.
* **split** — break a chunk at a character offset into two chunks,
  apportioning the bounding boxes proportionally to the text fraction.

Atomicity (AC3 / Q-I-D3-1)
--------------------------
Each mutation runs as a single SurrealQL ``BEGIN ... COMMIT`` transaction
sent through the canonical ``execute_query`` seam. SurrealDB applies the
block atomically server-side: if any statement inside the block raises,
the whole transaction is rolled back and ``execute_query`` re-raises, so
the chunk table is never left half-mutated.

Two things to know about the seam (both verified against surrealdb
1.0.6's ``AsyncSurreal.query``): a multi-statement query returns only the
*first* statement's result, and any statement error surfaces as a raised
exception. Because of the former we do not read mutated rows back from the
transaction's return value — callers re-fetch via the chunk repository
after the commit succeeds. All *validation* (ownership, adjacency, offset
bounds) happens in Python before the transaction is built, so the common
failure modes never reach the database at all.

Audit trail (AC4): each op logs a before/after summary via loguru. The
durable ``chunk_edit`` table is deferred to phase I.H2 and is intentionally
not built here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger
from shared.models import Chunk
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query

# Text separator inserted between two merged chunks. Two newlines mirror a
# paragraph break, which is how Docling/MinerU chunks read when adjacent.
MERGE_SEPARATOR = "\n\n"


class ChunkMutationError(Exception):
    """Raised when a merge/split cannot be performed as requested.

    The router maps this to an HTTP 4xx; it signals an operator/input
    problem (e.g. no adjacent chunk, offset out of range), not a server
    fault.
    """


def _split_positions(
    positions: List[List[float]], fraction: float
) -> tuple[List[List[float]], List[List[float]]]:
    """Apportion bounding boxes between the two halves of a split.

    Positions are ``[page, x1, x2, y1, y2]`` with coordinates normalized
    to 0–1 (post I.C). We split *vertically* by the text fraction: the
    first half keeps the top ``fraction`` of each box's height, the second
    half keeps the remainder. This is an approximation — we don't know the
    true line geometry — but it is monotonic, conserves the union of the
    original boxes, and keeps each half's overlay anchored to the correct
    region of the page.

    Single-box chunks (the common case) are cut into a top and bottom
    band. Empty position lists yield two empty lists.
    """
    if not positions:
        return [], []

    fraction = max(0.0, min(1.0, fraction))
    first: List[List[float]] = []
    second: List[List[float]] = []
    for pos in positions:
        if len(pos) < 5:
            # Malformed box — assign whole to both is wrong; keep it on the
            # first half so it isn't silently dropped.
            first.append(list(pos))
            continue
        page, x1, x2, y1, y2 = pos[0], pos[1], pos[2], pos[3], pos[4]
        cut = y1 + (y2 - y1) * fraction
        first.append([page, x1, x2, y1, cut])
        second.append([page, x1, x2, cut, y2])
    return first, second


class ChunkMutator:
    """Service performing atomic merge/split edits on a source's chunks.

    The service is stateless apart from an optional DB config; it reads
    the current chunk rows, validates the request, then writes the result
    inside one SurrealQL transaction.
    """

    def __init__(self, config: Optional[SurrealDBConfig] = None) -> None:
        self.config = config

    async def _get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        rows = await execute_query(
            "SELECT * FROM $id",
            {"id": ensure_record_id(chunk_id)},
            self.config,
        )
        return Chunk(**rows[0]) if rows else None

    async def _get_ordered_chunks(self, source_id: str) -> List[Chunk]:
        rows = await execute_query(
            "SELECT * FROM chunk WHERE source = $source ORDER BY order ASC",
            {"source": ensure_record_id(source_id)},
            self.config,
        )
        return [Chunk(**row) for row in rows]

    @staticmethod
    def _belongs_to(chunk: Chunk, source_id: str) -> bool:
        return str(chunk.source) == str(source_id)

    async def merge(
        self,
        source_id: str,
        chunk_id: str,
        target_chunk_id: Optional[str] = None,
    ) -> Chunk:
        """Merge ``chunk_id`` with an adjacent chunk into a single chunk.

        Direction (documented): by default the chunk is merged with its
        immediate successor by ``order`` (the next chunk). A caller may
        instead pass ``target_chunk_id`` to merge with a specific chunk;
        the two must be adjacent in ``order``. The lower-``order`` chunk is
        kept and receives the concatenated text/positions; the higher one
        is deleted and the trailing ``order`` values are compacted.

        Returns the surviving (merged) chunk.

        Raises:
            ChunkMutationError: if the chunk has no adjacent neighbour, the
                supplied target isn't adjacent, or either chunk doesn't
                belong to ``source_id``.
        """
        chunks = await self._get_ordered_chunks(source_id)
        by_id = {str(c.id): c for c in chunks}

        base = by_id.get(str(chunk_id))
        if base is None:
            raise ChunkMutationError("Chunk not found for this source")

        index = next(
            i for i, c in enumerate(chunks) if str(c.id) == str(chunk_id)
        )

        if target_chunk_id is not None:
            target = by_id.get(str(target_chunk_id))
            if target is None:
                raise ChunkMutationError(
                    "Target chunk not found for this source"
                )
            target_index = next(
                i
                for i, c in enumerate(chunks)
                if str(c.id) == str(target_chunk_id)
            )
            if abs(target_index - index) != 1:
                raise ChunkMutationError(
                    "Chunks must be adjacent to merge"
                )
        else:
            if index + 1 >= len(chunks):
                raise ChunkMutationError(
                    "No following chunk to merge with"
                )
            target = chunks[index + 1]

        # Keep the lower-order chunk; remove the higher-order one.
        keep, drop = (
            (base, target) if base.order <= target.order else (target, base)
        )

        merged_text = f"{keep.text}{MERGE_SEPARATOR}{drop.text}"
        merged_positions = list(keep.positions) + list(drop.positions)

        keep_id = ensure_record_id(str(keep.id))
        drop_id = ensure_record_id(str(drop.id))

        # Atomic block: update the survivor, delete the absorbed chunk, then
        # compact the order of everything that followed the deleted chunk.
        transaction = (
            "BEGIN TRANSACTION;\n"
            "UPDATE $keep_id SET text = $text, positions = $positions, "
            "source_element_count = source_element_count + "
            "$drop_elem, chunk_type = 'merged', updated = time::now();\n"
            "DELETE $drop_id;\n"
            "UPDATE chunk SET order = order - 1 "
            "WHERE source = $source AND order > $drop_order;\n"
            "COMMIT TRANSACTION;"
        )
        await execute_query(
            transaction,
            {
                "keep_id": keep_id,
                "drop_id": drop_id,
                "text": merged_text,
                "positions": merged_positions,
                "drop_elem": drop.source_element_count,
                "source": ensure_record_id(source_id),
                "drop_order": drop.order,
            },
            self.config,
        )

        logger.info(
            "chunk merge: source={} kept={} dropped={} "
            "len {}+{}->{} positions {}+{}->{}",
            source_id,
            keep.id,
            drop.id,
            len(keep.text),
            len(drop.text),
            len(merged_text),
            len(keep.positions),
            len(drop.positions),
            len(merged_positions),
        )

        result = await self._get_chunk(str(keep.id))
        if result is None:  # pragma: no cover - survivor must exist post-commit
            raise ChunkMutationError("Merged chunk vanished after commit")
        return result

    async def split(
        self,
        source_id: str,
        chunk_id: str,
        cursor_offset: int,
    ) -> List[Chunk]:
        """Split ``chunk_id`` at ``cursor_offset`` into two chunks.

        The original chunk keeps ``text[:cursor_offset]``; a new chunk
        receives ``text[cursor_offset:]`` and is inserted immediately after
        (``order + 1``), with all later chunks shifted up by one. Bounding
        boxes are apportioned proportionally to the text fraction (see
        ``_split_positions``).

        Returns ``[original, new]`` after the commit.

        Raises:
            ChunkMutationError: if the chunk doesn't belong to the source or
                the offset is not strictly inside the text (0 or end would
                produce an empty chunk).
        """
        chunk = await self._get_chunk(chunk_id)
        if chunk is None or not self._belongs_to(chunk, source_id):
            raise ChunkMutationError("Chunk not found for this source")

        text_len = len(chunk.text)
        if cursor_offset <= 0 or cursor_offset >= text_len:
            raise ChunkMutationError(
                "cursorOffset must be strictly inside the chunk text "
                f"(1..{text_len - 1})"
            )

        first_text = chunk.text[:cursor_offset]
        second_text = chunk.text[cursor_offset:]
        fraction = cursor_offset / text_len
        first_positions, second_positions = _split_positions(
            chunk.positions, fraction
        )

        chunk_id_rec = ensure_record_id(str(chunk.id))
        new_chunk_content: Dict[str, Any] = {
            "source": ensure_record_id(source_id),
            "text": second_text,
            "order": chunk.order + 1,
            "physical_page": chunk.physical_page,
            "printed_page": chunk.printed_page,
            "chapter": chunk.chapter,
            "paragraph_number": chunk.paragraph_number,
            "element_type": chunk.element_type,
            "positions": second_positions,
            "metadata": {**chunk.metadata, "split_from": str(chunk.id)},
            "section_path": list(chunk.section_path),
            "section_level": chunk.section_level,
            "chunk_type": "split",
            "source_element_count": 1,
            "is_content": chunk.is_content,
        }

        # Atomic block: open a slot at order+1 (shift later chunks up),
        # shrink the original to its first half, create the second half.
        transaction = (
            "BEGIN TRANSACTION;\n"
            "UPDATE chunk SET order = order + 1 "
            "WHERE source = $source AND order > $order;\n"
            "UPDATE $chunk_id SET text = $first_text, "
            "positions = $first_positions, chunk_type = 'split', "
            "updated = time::now();\n"
            "CREATE chunk CONTENT $new_chunk;\n"
            "COMMIT TRANSACTION;"
        )
        await execute_query(
            transaction,
            {
                "source": ensure_record_id(source_id),
                "order": chunk.order,
                "chunk_id": chunk_id_rec,
                "first_text": first_text,
                "first_positions": first_positions,
                "new_chunk": new_chunk_content,
            },
            self.config,
        )

        logger.info(
            "chunk split: source={} chunk={} offset={} "
            "len {}->{}+{} positions {}->{}+{}",
            source_id,
            chunk.id,
            cursor_offset,
            text_len,
            len(first_text),
            len(second_text),
            len(chunk.positions),
            len(first_positions),
            len(second_positions),
        )

        original = await self._get_chunk(str(chunk.id))
        siblings = await self._get_ordered_chunks(source_id)
        new_chunk = next(
            (
                c
                for c in siblings
                if c.order == chunk.order + 1 and str(c.id) != str(chunk.id)
            ),
            None,
        )
        if original is None or new_chunk is None:  # pragma: no cover
            raise ChunkMutationError("Split result incomplete after commit")
        return [original, new_chunk]
