"""Regenerate the three synthetic PDFs used by Track A integration tests.

The fixtures are intentionally tiny (<100 KB each) and exist so:

1. ``parser-engine-integration.spec.ts`` has deterministic file paths to
   upload during the Playwright integration spec.
2. ``threshold-tuning.md`` includes a reproducible baseline row that
   maintainers can re-run on any machine.

They are NOT meant to give realistic confidence scores — a four-line
synthetic PDF will of course score very low on heading_rate and
text_density. The "real" tuning runs against developer-local PDFs in
``docling_input/`` (gitignored).

Generator strategy
------------------

We deliberately avoid adding ``reportlab`` as a workspace dependency
(see ``docs/tracks/A-mineru/plan-A.3.md`` §13 — risk + mitigation).
Two of the three PDFs are emitted directly using the stdlib (a
hand-rolled PDF writer is < 40 LOC for the "clean text" and
"table-heavy" shapes we need). The third (``synthetic_image_only.pdf``)
uses ``Pillow.Image.save(..., format="PDF")``, which is already a
transitive dep of docling and friends.

The PDFs are committed alongside this script so reviewers can grep for
the literal bytes; running the script just refreshes them.
"""

from __future__ import annotations

import io
import zlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FIXTURES_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Minimal PDF writer (text-only)
# ---------------------------------------------------------------------------


def _write_pdf(path: Path, page_streams: list[str]) -> None:
    """Write a one-page-per-stream PDF using only stdlib bytes.

    Each ``page_streams`` entry is the body of a ``content stream`` —
    PDF operators like ``BT /F1 12 Tf 72 720 Td (text) Tj ET``.
    Pages share a single Helvetica Type1 font (no embedding).
    """
    # Object 1: catalog. Object 2: pages dict. Object 3: font.
    # Pages start at object 4, contents at object 4 + len(pages).
    n_pages = len(page_streams)
    page_obj_ids = [4 + i for i in range(n_pages)]
    contents_obj_ids = [4 + n_pages + i for i in range(n_pages)]

    objects: list[bytes] = []

    # 1: catalog
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")

    # 2: pages
    kids = " ".join(f"{i} 0 R" for i in page_obj_ids)
    objects.append(
        f"<< /Type /Pages /Count {n_pages} /Kids [ {kids} ] >>".encode()
    )

    # 3: font
    objects.append(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    )

    # 4..: pages
    for page_id, contents_id in zip(page_obj_ids, contents_obj_ids):
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {contents_id} 0 R >>"
            ).encode()
        )

    # contents streams (uncompressed for grep-ability — these are tiny)
    for stream in page_streams:
        body = stream.encode()
        objects.append(
            (
                f"<< /Length {len(body)} >>\nstream\n".encode()
                + body
                + b"\nendstream"
            )
        )

    # Assemble
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")  # binary marker
    offsets = [0]  # xref index 0 is the "free" head
    for idx, obj in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{idx} 0 obj\n".encode())
        out.write(obj)
        out.write(b"\nendobj\n")
    xref_offset = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    )
    out.write(f"startxref\n{xref_offset}\n%%EOF\n".encode())

    path.write_bytes(out.getvalue())


def _escape_pdf_string(text: str) -> str:
    """PDF text literal — escape `(`, `)`, `\\`."""
    return text.replace("\\", "\\\\").replace("(", r"\(").replace(")", r"\)")


def _text_lines_stream(lines: list[tuple[float, float, int, str]]) -> str:
    """Build a content stream from (x, y, font_size, text) tuples."""
    parts = ["BT"]
    for x, y, size, text in lines:
        parts.append(f"/F1 {size} Tf")
        parts.append(f"1 0 0 1 {x:.1f} {y:.1f} Tm")
        parts.append(f"({_escape_pdf_string(text)}) Tj")
    parts.append("ET")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def build_clean_text_pdf() -> None:
    """Single-page lipsum-style PDF — clean, dense prose."""
    paragraph = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do "
        "eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut "
        "enim ad minim veniam, quis nostrud exercitation ullamco laboris "
        "nisi ut aliquip ex ea commodo consequat."
    )
    lines: list[tuple[float, float, int, str]] = [
        (72.0, 740.0, 18, "Synthetic Clean Text Fixture"),  # heading
        (72.0, 710.0, 12, paragraph[:80]),
        (72.0, 692.0, 12, paragraph[80:160]),
        (72.0, 674.0, 12, paragraph[160:240]),
        (72.0, 656.0, 12, paragraph[240:]),
        (72.0, 620.0, 14, "Section 1: Background"),  # second heading
        (72.0, 595.0, 12, "This document anchors the Track A integration"),
        (72.0, 577.0, 12, "tests with a deterministic small-PDF fixture."),
    ]
    _write_pdf(
        FIXTURES_DIR / "synthetic_clean_text.pdf",
        [_text_lines_stream(lines)],
    )


def build_table_heavy_pdf() -> None:
    """Single-page PDF with a primitive 4x8 grid rendered as text rows."""
    lines: list[tuple[float, float, int, str]] = [
        (72.0, 740.0, 16, "Synthetic Table-Heavy Fixture"),
        (72.0, 715.0, 11, "Col A    | Col B    | Col C    | Col D"),
    ]
    # 8 data rows, evenly spaced
    y = 698.0
    for i in range(8):
        lines.append(
            (
                72.0,
                y,
                11,
                f"row {i+1:>2} | val {i*2:>3} | val {i*3:>3} | val {i*5:>3}",
            )
        )
        y -= 18.0
    _write_pdf(
        FIXTURES_DIR / "synthetic_table_heavy.pdf",
        [_text_lines_stream(lines)],
    )


def build_image_only_pdf() -> None:
    """Single-page PDF where the only content is a rendered PNG.

    Stands in for the OCR-heavy / scanned-doc category — docling will
    see one image element and almost no text layer.
    """
    img = Image.new("RGB", (612, 792), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    draw.rectangle([(60, 60), (552, 732)], outline="black", width=2)
    draw.text(
        (80, 80),
        "Synthetic image-only fixture\n(rendered as raster, no text layer)",
        fill="black",
        font=font,
    )
    # Pillow's PDF writer embeds the image directly — exactly the
    # "scanned doc" shape we want to simulate.
    img.save(
        FIXTURES_DIR / "synthetic_image_only.pdf",
        format="PDF",
        resolution=72.0,
    )


def main() -> None:
    build_clean_text_pdf()
    build_table_heavy_pdf()
    build_image_only_pdf()
    print(f"Regenerated 3 fixtures into {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
