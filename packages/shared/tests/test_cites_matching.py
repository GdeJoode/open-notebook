"""Unit tests for the pure citation matcher (Track U.3).

These exercise the PRECISION logic in isolation — the part that, if wrong,
fabricates a citation. The acceptance criteria they map to:

* AC1 — a confident reference (DOI exact / strong title+author) → a match.
* AC2/AC3 — external and near-miss references → NO match (precision guard).
* AC2 — no self-citation (a reference to its own source is suppressed).
* AC4 — deterministic / stable output (idempotent materialization upstream).
* AC7 — empty input → empty result (the clean 0-edge no-op on the live corpus).
"""

from __future__ import annotations

from shared.retrieval.cites_matching import (
    MIN_TITLE_AUTHOR_CONFIDENCE,
    ParsedReference,
    SourceRecord,
    match_corpus_references,
    match_reference,
    normalize_doi,
)

# --------------------------------------------------------------------------
# Fixtures: a tiny in-corpus source set
# --------------------------------------------------------------------------

ALI = SourceRecord(
    source_id="source:ali",
    title="Cohesion Policy and Institutional Quality",
    authors=("Ali",),
    doi="10.1111/jcms.13501",
    year=2025,
)
BRAKMAN = SourceRecord(
    source_id="source:brakman",
    title="Economics without equilibrium",
    authors=("Brakman", "Garretsen"),
    doi="10.1093/cjres/rsab012",
    year=2021,
)
CORPUS = [ALI, BRAKMAN]


# --------------------------------------------------------------------------
# DOI exact match (the gold path) — AC1
# --------------------------------------------------------------------------


def test_doi_exact_match_resolves_with_certainty():
    ref = ParsedReference(
        raw_text="Ali (2025). doi:10.1111/jcms.13501",
        title="",  # title absent — DOI alone must resolve it
        doi="10.1111/jcms.13501",
    )
    match = match_reference(ref, CORPUS)
    assert match is not None
    assert match.source_id == "source:ali"
    assert match.method == "doi"
    assert match.confidence == 1.0
    assert match.reference.raw_text == ref.raw_text


def test_doi_match_ignores_prefix_and_case():
    # The reference carries a URL-form, upper-cased DOI; the source a bare one.
    ref = ParsedReference(
        raw_text="https://doi.org/10.1093/CJRES/RSAB012",
        doi="https://doi.org/10.1093/CJRES/RSAB012",
    )
    match = match_reference(ref, CORPUS)
    assert match is not None and match.source_id == "source:brakman"


def test_normalize_doi_strips_forms():
    assert normalize_doi("doi:10.1/X") == "10.1/x"
    assert normalize_doi("https://doi.org/10.1/X") == "10.1/x"
    assert normalize_doi("http://dx.doi.org/10.1/X") == "10.1/x"
    assert normalize_doi(None) == ""
    assert normalize_doi("  ") == ""


def test_two_empty_dois_never_match():
    # A reference with no DOI must not DOI-match a source with no DOI.
    src = SourceRecord(source_id="source:x", title="Some other title")
    ref = ParsedReference(raw_text="No DOI here", title="Totally unrelated work")
    assert match_reference(ref, [src]) is None


# --------------------------------------------------------------------------
# Title + author fuzzy path (no DOI) — AC1
# --------------------------------------------------------------------------


def test_fuzzy_title_author_match_above_threshold():
    # Slightly noisy title + correct author, no DOI → confident title_author hit.
    ref = ParsedReference(
        raw_text="Ali, D. (2025) Cohesion policy & institutional quality.",
        title="Cohesion Policy & Institutional Quality",
        authors=("Ali, D.",),
    )
    match = match_reference(ref, CORPUS)
    assert match is not None
    assert match.source_id == "source:ali"
    assert match.method == "title_author"
    assert match.confidence >= MIN_TITLE_AUTHOR_CONFIDENCE


def test_title_match_but_wrong_author_is_rejected():
    # Title matches Ali but the author is someone else → author can't confirm.
    ref = ParsedReference(
        raw_text="Smith, J. Cohesion Policy and Institutional Quality.",
        title="Cohesion Policy and Institutional Quality",
        authors=("Smith, J.",),
    )
    assert match_reference(ref, CORPUS) is None


def test_author_match_but_weak_title_is_rejected():
    # Right author (Ali) but a totally different title → title gate fails.
    ref = ParsedReference(
        raw_text="Ali (2019) A different paper on trade flows.",
        title="A different paper on trade flows",
        authors=("Ali",),
    )
    assert match_reference(ref, CORPUS) is None


# --------------------------------------------------------------------------
# Precision guard: near-miss rejection — AC3
# --------------------------------------------------------------------------


def test_near_miss_title_below_threshold_no_edge():
    """A title that is SIMILAR but not confident → NO edge (precision > recall).

    "Cohesion Policy and Institutional Capacity" shares most words with the Ali
    title but differs in the load-bearing final word (Quality vs Capacity). With
    the right author it is the classic dangerous near-miss; the guard must reject
    it rather than fabricate a citation.
    """
    ref = ParsedReference(
        raw_text="Ali (2025) Cohesion Policy and Regional Development Outcomes.",
        title="Cohesion Policy and Regional Development Outcomes",
        authors=("Ali",),
    )
    match = match_reference(ref, CORPUS)
    assert match is None, (
        "near-miss title must NOT match — a fabricated citation is worse than "
        "a missing one"
    )


def test_external_reference_no_match():
    # A genuine external academic work cited by a paper — not in the corpus.
    ref = ParsedReference(
        raw_text="Acemoglu, D. & Robinson, J. (2012) Why Nations Fail.",
        title="Why Nations Fail",
        authors=("Acemoglu", "Robinson"),
        doi="10.5555/whynationsfail",
    )
    assert match_reference(ref, CORPUS) is None


# --------------------------------------------------------------------------
# No self-citation — AC2
# --------------------------------------------------------------------------


def test_no_self_citation():
    """A reference that would resolve to its own source produces NO edge."""
    ref = ParsedReference(
        raw_text="Ali (2025). doi:10.1111/jcms.13501",
        doi="10.1111/jcms.13501",
    )
    # Without the origin exclusion it WOULD match Ali...
    assert match_reference(ref, CORPUS) is not None
    # ...but excluding the origin (the reference came from Ali) suppresses it.
    match = match_reference(ref, CORPUS, from_source_id="source:ali")
    assert match is None


# --------------------------------------------------------------------------
# Ambiguity guard — AC3
# --------------------------------------------------------------------------


def test_ambiguous_two_viable_targets_no_edge():
    """Two near-identical sources both clearing the bar → NO confident target."""
    twin_a = SourceRecord(
        source_id="source:twin_a",
        title="Regional Economics Handbook",
        authors=("Jones",),
    )
    twin_b = SourceRecord(
        source_id="source:twin_b",
        title="Regional Economics Handbook",
        authors=("Jones",),
    )
    ref = ParsedReference(
        raw_text="Jones. Regional Economics Handbook.",
        title="Regional Economics Handbook",
        authors=("Jones",),
    )
    # Both twins match equally well → ambiguous → no edge.
    assert match_reference(ref, [twin_a, twin_b]) is None


def test_clear_winner_over_runner_up_still_matches():
    """A confident DOI winner is not blocked by a weak runner-up."""
    other = SourceRecord(
        source_id="source:other",
        title="Cohesion Policy and Institutional Quality",  # same title, no DOI
        authors=("Ali",),
    )
    ref = ParsedReference(
        raw_text="Ali (2025) doi:10.1111/jcms.13501",
        title="Cohesion Policy and Institutional Quality",
        authors=("Ali",),
        doi="10.1111/jcms.13501",
    )
    # DOI (1.0) on Ali beats the title-only 'other' by more than the margin.
    match = match_reference(ref, [ALI, other])
    assert match is not None and match.source_id == "source:ali"
    assert match.method == "doi"


# --------------------------------------------------------------------------
# Corpus-level aggregation + telemetry — AC4, AC7
# --------------------------------------------------------------------------


def test_match_corpus_classifies_outcomes():
    refs_by_source = {
        # Brakman cites Ali (intra-corpus, DOI) → 1 edge.
        "source:brakman": [
            ParsedReference(
                raw_text="Ali (2025) doi:10.1111/jcms.13501",
                doi="10.1111/jcms.13501",
            ),
            # ...and an external work → no edge.
            ParsedReference(
                raw_text="Kaldor (1970) The case for regional policies.",
                title="The case for regional policies",
                authors=("Kaldor",),
            ),
        ],
        # Ali 'cites' itself (should be suppressed) ...
        "source:ali": [
            ParsedReference(
                raw_text="Ali (2025) doi:10.1111/jcms.13501",
                doi="10.1111/jcms.13501",
            ),
        ],
    }
    result = match_corpus_references(refs_by_source, CORPUS)
    assert len(result.matches) == 1
    edge = result.matches[0]
    assert edge.source_id == "source:ali"
    assert edge.method == "doi"
    assert result.external == 1
    assert result.self_citations_skipped == 1
    assert result.total_references == 3


def test_empty_input_is_clean_no_op():
    """AC7: no reference input → 0 matches, no crash (the live-corpus reality)."""
    result = match_corpus_references({}, CORPUS)
    assert result.matches == []
    assert result.total_references == 0
    assert result.external == 0
    assert result.self_citations_skipped == 0


def test_no_sources_means_every_reference_is_external():
    refs = {
        "source:x": [
            ParsedReference(raw_text="Some ref", title="T", authors=("A",))
        ]
    }
    result = match_corpus_references(refs, [])
    assert result.matches == []
    assert result.external == 1


def test_matching_is_deterministic():
    """AC4 upstream: identical inputs yield an identical match (stable winner)."""
    ref = ParsedReference(
        raw_text="Ali (2025) doi:10.1111/jcms.13501",
        doi="10.1111/jcms.13501",
    )
    m1 = match_reference(ref, CORPUS)
    m2 = match_reference(ref, list(reversed(CORPUS)))
    assert m1 is not None and m2 is not None
    assert m1.source_id == m2.source_id == "source:ali"
    assert m1.confidence == m2.confidence
