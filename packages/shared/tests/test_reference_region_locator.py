"""Unit tests for V.1 reference-region location (pure, offline).

Exercise the structure-first and full_text-fallback location paths plus the
no-reference case, against committed synthetic fixtures (no DB, no PDF, no
network). Chunk structures are literal :class:`ReferenceChunk` lists that mirror
the persisted-chunk shapes measured on staging (generic ``heading`` /
``section_header`` / ``list_item`` / ``text`` element types; a ``References``
heading with entries beneath it).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from shared.references.region_locator import (
    LocatedRegion,
    ReferenceChunk,
    locate_reference_region,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "references"


def _fixture(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


# -- structure-first: section_path membership -------------------------------


def _structured_chunks(heading: str = "References") -> List[ReferenceChunk]:
    """A doc whose reference entries carry ``section_path=(heading,)``."""
    return [
        ReferenceChunk(
            text="Introduction", section_path=("Introduction",),
            element_type="heading", order=0,
        ),
        ReferenceChunk(
            text="Body of the introduction.", section_path=("Introduction",),
            element_type="text", order=1,
        ),
        ReferenceChunk(
            text=heading, section_path=(heading,),
            element_type="section_header", order=2,
        ),
        ReferenceChunk(
            text="Acemoglu, D., & Robinson, J. A. (2012). Why Nations Fail. Crown.",
            section_path=(heading,), element_type="list_item", order=3,
        ),
        ReferenceChunk(
            text="Rodríguez-Pose, A. (2018). The revenge of the places. CJRES.",
            section_path=(heading,), element_type="list_item", order=4,
        ),
    ]


def test_locates_region_via_section_path_structure():
    region = locate_reference_region(_structured_chunks())
    assert region.located_via == "structure"
    assert region.found
    # The heading line itself is excluded — only the entries.
    assert "Acemoglu" in region.text
    assert "Rodríguez-Pose" in region.text
    assert region.text.strip().startswith("Acemoglu")
    # Body content ("Introduction") must NOT leak into the region.
    assert "Introduction" not in region.text


def test_dutch_heading_literatuur_is_recognized():
    region = locate_reference_region(_structured_chunks(heading="Literatuur"))
    assert region.located_via == "structure"
    assert "Acemoglu" in region.text


def test_structure_ignores_order_field_scrambling():
    """Region assembly uses ``order``, not list position."""
    chunks = list(reversed(_structured_chunks()))
    region = locate_reference_region(chunks)
    assert region.located_via == "structure"
    # Entries come out in document order despite the scrambled input list.
    assert region.text.index("Acemoglu") < region.text.index("Rodríguez-Pose")


# -- structure-first: standalone heading + following body -------------------


def test_locates_region_via_standalone_heading_chunk():
    """Heading chunk with entries that carry NO matching section_path."""
    chunks = [
        ReferenceChunk(text="Conclusion", element_type="heading", order=0),
        ReferenceChunk(text="Final remarks.", element_type="text", order=1),
        ReferenceChunk(text="References", element_type="heading", order=2),
        ReferenceChunk(
            text="Acemoglu, D. (2012). Why Nations Fail. Crown.",
            element_type="text", order=3,
        ),
        ReferenceChunk(
            text="Barca, F. (2012). The case for regional development. JRS.",
            element_type="text", order=4,
        ),
        ReferenceChunk(text="Appendix", element_type="heading", order=5),
        ReferenceChunk(text="Extra table.", element_type="text", order=6),
    ]
    region = locate_reference_region(chunks)
    assert region.located_via == "structure"
    assert "Acemoglu" in region.text and "Barca" in region.text
    # The next section (Appendix) terminates the region.
    assert "Extra table" not in region.text
    assert "Final remarks" not in region.text


# -- full_text fallback -----------------------------------------------------


def test_locates_region_via_full_text_fallback():
    full_text = _fixture("full_text_with_bib.txt")
    # No structural chunks at all → must use the full_text path.
    region = locate_reference_region([], full_text)
    assert region.located_via == "full_text"
    assert region.found
    assert "Acemoglu" in region.text
    # The Appendix AFTER the references must not be swallowed.
    assert "Data sources" not in region.text
    assert "Table A1" not in region.text
    # The Introduction/Conclusion BEFORE the references must not be included.
    assert "This paper examines" not in region.text


def test_full_text_span_round_trips():
    full_text = _fixture("full_text_with_bib.txt")
    region = locate_reference_region([], full_text)
    assert region.span is not None
    start, end = region.span
    assert full_text[start:end] == region.text


def test_structure_wins_over_full_text_when_both_present():
    """A structural heading is preferred even when full_text also has one."""
    chunks = _structured_chunks()
    region = locate_reference_region(chunks, _fixture("full_text_with_bib.txt"))
    assert region.located_via == "structure"


# -- no-reference case ------------------------------------------------------


def test_no_reference_region_returns_none_located_via():
    full_text = _fixture("full_text_no_refs.txt")
    chunks = [
        ReferenceChunk(text="Aanleiding", section_path=("Aanleiding",),
                       element_type="heading", order=0),
        ReferenceChunk(text="Partijen sluiten deze Regio Deal.",
                       section_path=("Aanleiding",), element_type="text", order=1),
    ]
    region = locate_reference_region(chunks, full_text)
    assert region.located_via == "none"
    assert region.text == ""
    assert region.span is None
    assert not region.found


def test_empty_inputs_never_crash():
    region = locate_reference_region([], "")
    assert region == LocatedRegion(text="", located_via="none", span=None)


def test_body_heading_named_sources_is_not_a_false_match():
    """A body heading like 'Data sources' must not match the 'bronnen'/'sources'
    vocabulary (only whole/prefix reference labels count)."""
    chunks = [
        ReferenceChunk(text="Data sources", section_path=("Data sources",),
                       element_type="heading", order=0),
        ReferenceChunk(text="Eurostat regional accounts.",
                       section_path=("Data sources",), element_type="text", order=1),
    ]
    region = locate_reference_region(chunks)
    assert region.located_via == "none"
