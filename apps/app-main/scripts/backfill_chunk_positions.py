"""Backfill ``chunk.positions`` to the canonical 0–1 convention (Track I.C).

Before I.C, ``BoundingBox.from_docling`` wrote raw PDF points into
``chunk.positions`` (``[page, x1, x2, y1, y2]``) while MinerU wrote 0–1
normalized coords into the same field. This script rewrites the legacy
raw-point rows in place so every row is 0–1 (TOPLEFT origin), matching the
fixed extraction path and the frontend (which no longer sniffs the format).

How normalization is sourced
----------------------------
The plan referenced ``source.page_dimensions``; that field does not exist
in this codebase. Page dimensions are not persisted on the source — they
are read live from the source's PDF via ``pypdfium2`` (exactly what the
``/page-count`` endpoint does). This script does the same: for each chunk
it resolves ``chunk.source.asset.file_path`` and reads the page size from
the PDF, caching per file so each PDF is opened once.

Idempotency
-----------
A position whose x/y are already within [0, 1] is left untouched, so a row
is only rewritten once and re-running the script is a no-op on already-
normalized data. "Needs normalization" is detected the same way the old
frontend heuristic did: any of x1/x2/y1/y2 > 1.0 (the page index pos[0] is
never inspected).

Fallback
--------
Rows whose PDF is missing/unreadable, or whose page is out of range, are
logged and skipped — the operator re-ingests those sources (the only way
to recover dimensions that no longer exist on disk).

Run via::

    uv run --project apps/app-main \
        python apps/app-main/scripts/backfill_chunk_positions.py [--dry-run] [--batch-size N]

Performance: batched at 1000 rows/query; per-PDF page sizes are cached, so
the dominant cost is a single UPDATE per legacy row. On a dev machine 10K
chunks complete well under 60s (estimated/by-design — not measured against
a 10K live corpus in this sandbox).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# Allow ``python apps/app-main/scripts/backfill_chunk_positions.py`` to import
# the workspace package without installing a script entry point.
_APP_MAIN_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_APP_MAIN_SRC) not in sys.path:
    sys.path.insert(0, str(_APP_MAIN_SRC))

from surrealdb_service.connection import execute_query  # noqa: E402


DEFAULT_BATCH_SIZE = 1000


@dataclass
class BackfillStats:
    """Tally of what the backfill did."""

    processed: int = 0          # rows examined
    normalized: int = 0         # rows rewritten to 0–1
    already_normalized: int = 0  # rows already in 0–1 (left unchanged)
    skipped_no_positions: int = 0
    skipped_missing_pdf: int = 0
    skipped_bad_page: int = 0
    skipped_files: set[str] = field(default_factory=set)

    def summary(self) -> str:
        return (
            "Backfill complete:\n"
            f"  processed:           {self.processed}\n"
            f"  normalized:          {self.normalized}\n"
            f"  already normalized:  {self.already_normalized}\n"
            f"  skipped (no pos):    {self.skipped_no_positions}\n"
            f"  skipped (no PDF):    {self.skipped_missing_pdf}\n"
            f"  skipped (bad page):  {self.skipped_bad_page}\n"
            f"  distinct PDFs unresolved: {len(self.skipped_files)}"
        )


def _needs_normalization(position: list[Any]) -> bool:
    """A position is legacy (raw points) if any coordinate exceeds 1.0.

    ``position`` is ``[page, x1, x2, y1, y2]``; only the four coords are
    inspected — never the page index.
    """
    if not isinstance(position, (list, tuple)) or len(position) < 5:
        return False
    return any(float(c) > 1.0 for c in position[1:5])


def _normalize_position(
    position: list[Any], page_width: float, page_height: float
) -> list[float]:
    """Divide x by width, y by height, and clamp to [0, 1].

    Coordinates are assumed to already be in TOPLEFT image space (raw
    points). Docling's y-flip happens at extraction; legacy raw rows were
    written post-flip, so we only rescale here.
    """
    page = position[0]
    x1, x2, y1, y2 = (float(c) for c in position[1:5])

    def clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    return [
        page,
        clamp(x1 / page_width),
        clamp(x2 / page_width),
        clamp(y1 / page_height),
        clamp(y2 / page_height),
    ]


class _PageDimensionCache:
    """Reads + caches page dimensions per PDF file via pypdfium2."""

    def __init__(self) -> None:
        self._cache: dict[str, Optional[dict[int, tuple[float, float]]]] = {}

    def get(self, pdf_path: Optional[str]) -> Optional[dict[int, tuple[float, float]]]:
        """Return {page_number(1-indexed): (width, height)} or None if unreadable."""
        if not pdf_path:
            return None
        if pdf_path in self._cache:
            return self._cache[pdf_path]

        dims = self._read(pdf_path)
        self._cache[pdf_path] = dims
        return dims

    @staticmethod
    def _read(pdf_path: str) -> Optional[dict[int, tuple[float, float]]]:
        try:
            import pypdfium2 as pdfium
        except ImportError:
            logger.error("pypdfium2 not installed; cannot read page dimensions")
            return None
        if not Path(pdf_path).exists():
            return None
        try:
            pdf = pdfium.PdfDocument(pdf_path)
            dims: dict[int, tuple[float, float]] = {}
            for i in range(len(pdf)):
                page = pdf[i]
                dims[i + 1] = (page.get_width(), page.get_height())
            return dims
        except Exception as exc:  # pragma: no cover - depends on file contents
            logger.warning(f"Failed to read page dimensions from {pdf_path}: {exc}")
            return None


def _page_dims_for(
    dims: dict[int, tuple[float, float]], position: list[Any]
) -> Optional[tuple[float, float]]:
    """Resolve (width, height) for a position's page, tolerating 0/1 indexing."""
    page = int(position[0])
    if page in dims:
        return dims[page]
    # Legacy rows may carry a 0-indexed page; try page+1.
    if (page + 1) in dims:
        return dims[page + 1]
    return None


async def _fetch_batch(offset: int, batch_size: int) -> list[dict[str, Any]]:
    query = (
        "SELECT id, positions, source.asset.file_path AS pdf_path "
        "FROM chunk ORDER BY id LIMIT $limit START $start"
    )
    return await execute_query(query, {"limit": batch_size, "start": offset})


async def _write_positions(chunk_id: Any, positions: list[list[float]]) -> None:
    await execute_query(
        "UPDATE $id SET positions = $positions",
        {"id": chunk_id, "positions": positions},
    )


async def backfill(
    *, batch_size: int = DEFAULT_BATCH_SIZE, dry_run: bool = False
) -> BackfillStats:
    """Paginate the chunk table and normalize legacy raw-point positions."""
    stats = BackfillStats()
    dim_cache = _PageDimensionCache()
    offset = 0

    while True:
        rows = await _fetch_batch(offset, batch_size)
        if not rows:
            break

        for row in rows:
            stats.processed += 1
            positions = row.get("positions") or []
            if not positions:
                stats.skipped_no_positions += 1
                continue

            legacy = [p for p in positions if _needs_normalization(p)]
            if not legacy:
                stats.already_normalized += 1
                continue

            pdf_path = row.get("pdf_path")
            dims = dim_cache.get(pdf_path)
            if dims is None:
                stats.skipped_missing_pdf += 1
                if pdf_path:
                    stats.skipped_files.add(pdf_path)
                logger.warning(
                    f"Skipping chunk {row.get('id')}: page dimensions "
                    f"unavailable (pdf_path={pdf_path!r}); re-ingest required"
                )
                continue

            new_positions: list[list[float]] = []
            bad_page = False
            for pos in positions:
                if not _needs_normalization(pos):
                    new_positions.append(pos)
                    continue
                page_dims = _page_dims_for(dims, pos)
                if page_dims is None:
                    bad_page = True
                    break
                w, h = page_dims
                if w <= 0 or h <= 0:
                    bad_page = True
                    break
                new_positions.append(_normalize_position(pos, w, h))

            if bad_page:
                stats.skipped_bad_page += 1
                logger.warning(
                    f"Skipping chunk {row.get('id')}: page out of range for "
                    f"its PDF; re-ingest required"
                )
                continue

            stats.normalized += 1
            if not dry_run:
                await _write_positions(row["id"], new_positions)

        offset += batch_size

    return stats


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Rows per query batch (default {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing.",
    )
    return parser.parse_args(argv)


async def _main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if args.dry_run:
        logger.info("Running in DRY-RUN mode — no writes will be performed.")
    stats = await backfill(batch_size=args.batch_size, dry_run=args.dry_run)
    logger.info("\n" + stats.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
