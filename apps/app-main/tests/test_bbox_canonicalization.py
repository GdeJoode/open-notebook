"""Coordinate canonicalization tests for ``BoundingBox`` (Track I.C).

Both PDF parsers feed ``chunk.positions`` from ``BoundingBox``:

- Docling reports raw PDF points with a BOTTOMLEFT origin
  (``BoundingBox.from_docling`` normalizes + flips them).
- MinerU reports 0–1000 (pipeline back-end) or 0–1 (vlm) coords with a
  TOPLEFT origin (``_bbox_from_mineru`` divides by the back-end factor).

The bug fixed in I.C: ``from_docling`` used to emit raw points (x/width
never divided by page width; y-flip relying on a usually-missing
``prov.page_height``). These tests pin the 0–1 TOPLEFT convention and the
cross-parser parity that protects Track A (MinerU).

No live Docling run is needed — Docling provenance is a tiny stub mirroring
the attributes ``from_docling`` actually reads (``bbox.l/b/r/t`` and
optional ``bbox.coord_origin``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ingestion.models import BoundingBox
from app_main.services.parsing.mineru_layout_parser import _bbox_from_mineru


# ---------------------------------------------------------------------------
# Docling provenance stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubBBox:
    """Mimics docling_core's BoundingBox: l/b/r/t + optional coord_origin."""

    l: float  # noqa: E741 - matches Docling's attribute name
    b: float
    r: float
    t: float
    coord_origin: Optional[str] = None


@dataclass
class _StubProv:
    """Mimics a Docling provenance entry: only ``bbox`` is read here."""

    bbox: _StubBBox


# A4 in points — the page size from doc.pages[n].size that callers thread in.
PAGE_W = 595.0
PAGE_H = 842.0


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# Scenario 1 — Docling 0–1 round-trip
# ---------------------------------------------------------------------------


def test_docling_emits_normalized_0_1_coords():
    """from_docling divides by real page dims and flips y → all coords in 0–1."""
    # BOTTOMLEFT raw points: left=119, right=297.5 (→ 0.2..0.5 of width),
    # top=673.6, bottom=505.2 (measured from the bottom).
    bbox = _StubBBox(l=119.0, b=505.2, r=297.5, t=673.6)
    out = BoundingBox.from_docling(_StubProv(bbox), page=1,
                                   page_width=PAGE_W, page_height=PAGE_H)

    for v in (out.x, out.y, out.x2, out.y2):
        assert 0.0 <= v <= 1.0

    assert _approx(out.x, 119.0 / PAGE_W)
    assert _approx(out.x2, 297.5 / PAGE_W)
    # y-flip: top edge (image space) = page_height - t
    assert _approx(out.y, (PAGE_H - 673.6) / PAGE_H)
    assert _approx(out.y2, (PAGE_H - 505.2) / PAGE_H)
    assert out.width > 0 and out.height > 0


def test_docling_topleft_origin_not_flipped():
    """A TOPLEFT coord_origin is divided but not y-flipped."""
    bbox = _StubBBox(l=59.5, b=168.4, r=297.5, t=84.2, coord_origin="TOPLEFT")
    out = BoundingBox.from_docling(_StubProv(bbox), page=2,
                                   page_width=PAGE_W, page_height=PAGE_H)
    assert _approx(out.y, 84.2 / PAGE_H)
    assert _approx(out.y2, 168.4 / PAGE_H)
    assert out.page == 2


# ---------------------------------------------------------------------------
# Scenario 2 — MinerU 0–1 round-trip
# ---------------------------------------------------------------------------


def test_mineru_pipeline_factor_round_trips_to_0_1():
    """MinerU pipeline bbox (0–1000) → 0–1 TOPLEFT box."""
    out = _bbox_from_mineru([200.0, 187.6, 500.0, 387.6], page=1, factor=1000.0)
    assert out is not None
    assert _approx(out.x, 0.2)
    assert _approx(out.y, 0.1876)
    assert _approx(out.x2, 0.5)
    assert _approx(out.y2, 0.3876)
    for v in (out.x, out.y, out.x2, out.y2):
        assert 0.0 <= v <= 1.0


def test_mineru_vlm_factor_round_trips_to_0_1():
    """MinerU vlm bbox (already 0–1) passes through factor=1.0 unchanged."""
    out = _bbox_from_mineru([0.2, 0.1876, 0.5, 0.3876], page=1, factor=1.0)
    assert out is not None
    assert _approx(out.x, 0.2)
    assert _approx(out.y2, 0.3876)


# ---------------------------------------------------------------------------
# Scenario 3 — cross-parser parity (the Track A guard)
# ---------------------------------------------------------------------------


def test_docling_and_mineru_emit_identical_box_for_same_page_region():
    """Same physical region on the same page → identical 0–1 box from both.

    Target box (TOPLEFT, 0–1): x=0.2, x2=0.5, y=0.1876, y2=0.3876.

    MinerU (pipeline, factor 1000) expresses that directly as
    [200, 187.6, 500, 387.6].

    Docling expresses it as raw BOTTOMLEFT points on the A4 page:
      left  = 0.2  * 595   = 119.0
      right = 0.5  * 595   = 297.5
      t     = 842 - 0.1876*842 = 684.0392   (top edge, from bottom)
      b     = 842 - 0.3876*842 = 515.6008   (bottom edge, from bottom)
    """
    mineru = _bbox_from_mineru([200.0, 187.6, 500.0, 387.6], page=3, factor=1000.0)

    t = PAGE_H - 0.1876 * PAGE_H
    b = PAGE_H - 0.3876 * PAGE_H
    docling = BoundingBox.from_docling(
        _StubProv(_StubBBox(l=119.0, b=b, r=297.5, t=t)),
        page=3, page_width=PAGE_W, page_height=PAGE_H,
    )

    assert mineru is not None
    tol = 1e-6
    assert _approx(docling.x, mineru.x, tol)
    assert _approx(docling.y, mineru.y, tol)
    assert _approx(docling.x2, mineru.x2, tol)
    assert _approx(docling.y2, mineru.y2, tol)
    assert _approx(docling.width, mineru.width, tol)
    assert _approx(docling.height, mineru.height, tol)
    assert docling.page == mineru.page


# ---------------------------------------------------------------------------
# Scenario 4 — missing page dimensions
# ---------------------------------------------------------------------------


def test_missing_page_dimensions_yields_zero_box_not_raw_points():
    """Without page dims we cannot normalize → zero box (never raw points)."""
    raw = _StubBBox(l=119.0, b=505.2, r=297.5, t=673.6)

    none_dims = BoundingBox.from_docling(_StubProv(raw), page=1,
                                         page_width=None, page_height=None)
    assert (none_dims.x, none_dims.y, none_dims.width, none_dims.height) == (0, 0, 0, 0)

    zero_dims = BoundingBox.from_docling(_StubProv(raw), page=1,
                                         page_width=0, page_height=0)
    assert (zero_dims.x, zero_dims.width) == (0, 0)
    assert zero_dims.page == 1


def test_missing_bbox_yields_zero_box():
    class _NoBBox:
        pass

    out = BoundingBox.from_docling(_NoBBox(), page=4, page_width=PAGE_W,
                                   page_height=PAGE_H)
    assert (out.x, out.y, out.width, out.height) == (0, 0, 0, 0)
    assert out.page == 4


# ---------------------------------------------------------------------------
# Scenario 5 — y-flip correctness
# ---------------------------------------------------------------------------


def test_y_flip_top_of_page_maps_to_small_y():
    """An element near the physical top of the page (large Docling t/b, since
    Docling measures from the bottom) must map to a *small* normalized y.
    """
    # Near the top: t≈830, b≈800 on an 842pt page.
    top_box = BoundingBox.from_docling(
        _StubProv(_StubBBox(l=50.0, b=800.0, r=545.0, t=830.0)),
        page=1, page_width=PAGE_W, page_height=PAGE_H,
    )
    # Near the bottom: t≈42, b≈12.
    bottom_box = BoundingBox.from_docling(
        _StubProv(_StubBBox(l=50.0, b=12.0, r=545.0, t=42.0)),
        page=1, page_width=PAGE_W, page_height=PAGE_H,
    )

    assert top_box.y < 0.05          # top of page → small y
    assert bottom_box.y > 0.9        # bottom of page → large y
    assert top_box.y < bottom_box.y  # ordering preserved
    # Each box keeps a positive height (top edge above bottom edge).
    assert top_box.height > 0 and bottom_box.height > 0
