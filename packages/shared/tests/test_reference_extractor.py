"""Integration tests for the V.1→V.3 chained extractor (pure, offline).

Drive the full producer pipeline (locate → segment → parse) end-to-end on both
input shapes — a STRUCTURED chunk list and a FULL_TEXT-only document — against the
committed synthetic bibliography fixtures, and assert it yields well-formed
``ParsedReference`` objects. No DB, no network, no LLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from shared.references.reference_extractor import extract_references
from shared.references.region_locator import ReferenceChunk
from shared.retrieval.cites_matching import ParsedReference

_FIXTURES = Path(__file__).parent / "fixtures" / "references"


def _fixture(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


def _apa_structured_chunks() -> List[ReferenceChunk]:
    """A structured doc: a References heading + one list_item chunk per entry.

    Built from the APA fixture so the structure and full_text variants exercise
    the same underlying bibliography.
    """
    entries = [
        blk.replace("\n", " ").strip()
        for blk in _fixture("references_apa.txt").split("\n\n")
        if blk.strip()
    ]
    chunks: List[ReferenceChunk] = [
        ReferenceChunk(text="Conclusion", section_path=("Conclusion",),
                       element_type="heading", order=0),
        ReferenceChunk(text="Final remarks.", section_path=("Conclusion",),
                       element_type="text", order=1),
        ReferenceChunk(text="References", section_path=("References",),
                       element_type="section_header", order=2),
    ]
    for i, entry in enumerate(entries):
        chunks.append(
            ReferenceChunk(text=entry, section_path=("References",),
                           element_type="list_item", order=3 + i)
        )
    return chunks


# -- structured input -------------------------------------------------------


def test_extract_from_structured_chunks_full_bibliography():
    refs = extract_references(_apa_structured_chunks())
    assert len(refs) == 4
    assert all(isinstance(r, ParsedReference) for r in refs)
    # First entry: authors + year parsed.
    assert refs[0].authors[0].startswith("Acemoglu")
    assert refs[0].year == 2012
    # A DOI-bearing entry resolved its normalized DOI.
    dois = [r.doi for r in refs if r.doi]
    assert "10.1257/aer.91.5.1369" in dois
    # raw_text is always present.
    assert all(r.raw_text for r in refs)


# -- full_text-only input ---------------------------------------------------


def test_extract_from_full_text_only_document():
    refs = extract_references([], _fixture("full_text_with_bib.txt"))
    assert len(refs) == 3
    assert all(r.raw_text for r in refs)
    years = {r.year for r in refs}
    assert {2012, 2001, 2018} <= years
    # The DOI on the middle entry is normalized.
    assert any(r.doi == "10.1257/aer.91.5.1369" for r in refs)
    # No Appendix/body text leaked in as a spurious "reference".
    assert all("Data sources" not in r.raw_text for r in refs)


def test_structure_and_full_text_variants_agree_on_core_fields():
    """The same bibliography via structure vs full_text yields the same works."""
    struct = extract_references(_apa_structured_chunks())
    # full_text variant covers only the 3 entries present in that fixture.
    full = extract_references([], _fixture("full_text_with_bib.txt"))
    struct_years = {r.year for r in struct}
    full_years = {r.year for r in full}
    assert full_years <= struct_years


# -- no-reference / empty ---------------------------------------------------


def test_extract_returns_empty_when_no_reference_region():
    refs = extract_references([], _fixture("full_text_no_refs.txt"))
    assert refs == []


def test_extract_on_empty_inputs_returns_empty():
    assert extract_references([], "") == []


# -- ambiguity-resolver seam threads through --------------------------------


def test_extractor_threads_ambiguity_resolver_seam():
    def resolver(region_text, heuristic):
        return ["Solo, S. (2020). One entry. Publisher."]

    refs = extract_references(
        [], _fixture("full_text_with_bib.txt"), ambiguity_resolver=resolver
    )
    assert len(refs) == 1
    assert refs[0].year == 2020
    assert refs[0].authors == ("Solo, S.",)
