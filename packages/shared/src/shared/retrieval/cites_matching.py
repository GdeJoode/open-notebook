"""Pure reference→source citation matching (Track U.3).

This module is the **deciding half** of the ``cites`` (source→source) mechanism:
given a parsed reference (the Track V boundary, :class:`ParsedReference`) and the
in-corpus sources, it returns the single best in-corpus match with a confidence,
or no match. It is deliberately PURE (no I/O) so the precision logic — the part
that, if wrong, *fabricates a citation* — is unit-testable in isolation and the
materialization service just persists what this returns.

The Track V boundary
====================
Track V (reference / footnote extraction, not built yet) turns each document's
bibliography / footnotes into a ``List[ParsedReference]``. Track U.3 consumes that
list and matches each entry against the corpus. :class:`ParsedReference` is the
single, documented contract between the two tracks — V produces it, U.3 consumes
it. Until V exists there is no live reference input, so on the current corpus this
mechanism runs and produces **0 edges** (U.1 measured 0 intra-corpus citations).

Precision over recall (why this is conservative)
================================================
A WRONG match invents a citation that does not exist — strictly worse than a
missing one, because it pollutes the graph with a false fact. So matching is
PRECISION-FIRST, mirroring the Track-K resolution discipline (a confident match
or none — never an ambiguous guess):

* **DOI exact match** is the gold signal. A normalized DOI equality is treated as
  a certain match (:data:`DOI_CONFIDENCE`): DOIs are globally unique identifiers,
  so an exact match cannot be a coincidence.
* **Title + author similarity** is the fallback when there is no DOI. BOTH the
  title similarity AND an author-surname agreement must clear their thresholds —
  a title match alone (two papers can share a generic title fragment) or an
  author match alone (many works share an author) is NOT enough. The combined
  confidence must exceed :data:`MIN_TITLE_AUTHOR_CONFIDENCE`.
* **Ambiguity guard** — if two different in-corpus sources both clear the bar,
  the result is NO match (an ambiguous reference must not be force-resolved onto
  one of several plausible targets). A clear winner must out-score the runner-up
  by :data:`AMBIGUITY_MARGIN`.
* **No self-citation** — a reference is never matched to the source it was
  extracted from (``from_source_id``), so a document can never cite itself.

Everything here is plain string math (normalized Levenshtein, no external deps)
so the precision rules are transparent and the thresholds are auditable.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Precision thresholds (the auditable knobs — see module docstring)
# --------------------------------------------------------------------------

#: Confidence assigned to a DOI-exact match. DOIs are unique identifiers, so an
#: exact (normalized) match is treated as certain.
DOI_CONFIDENCE = 1.0

#: Minimum normalized title similarity (0–1) for the title+author path. Set high
#: because a single generic shared phrase must not pass on title alone.
MIN_TITLE_SIMILARITY = 0.85

#: Minimum combined title+author confidence to accept a no-DOI match.
MIN_TITLE_AUTHOR_CONFIDENCE = 0.80

#: How much the best candidate must out-score the runner-up to be unambiguous.
#: If two distinct sources both clear the bar within this margin → NO match.
AMBIGUITY_MARGIN = 0.05

#: Blend weight of the title vs author component in the combined confidence.
#: Title carries most of the signal; author agreement is a required confirmer.
_TITLE_WEIGHT = 0.7
_AUTHOR_WEIGHT = 0.3


@dataclass(frozen=True)
class ParsedReference:
    """A single structured reference — the **Track V → U.3 boundary**.

    Track V (reference/footnote extraction) produces a ``List[ParsedReference]``
    per source from the document's bibliography/footnotes; Track U.3 consumes it
    to materialize ``cites`` edges. This is the contract between the two tracks;
    keep it stable. All fields except ``raw_text`` are best-effort — a reference
    may carry only a DOI, or only a title+authors; the matcher uses whatever is
    present.

    Attributes:
        authors: Author names as parsed (any order/format). The matcher compares
            on normalized SURNAMES, so "Acemoglu, D." and "Daron Acemoglu" both
            contribute the surname "acemoglu".
        title: The work's title (may be empty if only a DOI was parsed).
        year: Publication year, when parseable.
        doi: The DOI, when present (the gold matching signal). Compared after
            normalization (lowercased, ``doi:``/URL prefix stripped).
        venue: Journal / conference / publisher (carried for provenance; not yet
            used in matching — a future precision confirmer).
        raw_text: The verbatim reference string as it appeared in the document.
            ALWAYS populated; it is what a created ``cites`` edge records as the
            human-readable citation, so a reviewer can see exactly what matched.
    """

    raw_text: str
    title: str = ""
    authors: Tuple[str, ...] = field(default_factory=tuple)
    year: Optional[int] = None
    doi: Optional[str] = None
    venue: Optional[str] = None

    def __post_init__(self) -> None:
        # Normalize ``authors`` to a tuple so the dataclass stays hashable/frozen
        # even when a caller passes a list (the natural shape from Track V).
        if not isinstance(self.authors, tuple):
            object.__setattr__(self, "authors", tuple(self.authors or ()))


@dataclass(frozen=True)
class SourceRecord:
    """An in-corpus source the matcher can resolve a reference TO.

    The minimal projection of a ``source`` row needed for matching — id plus the
    bibliographic fields. The materialization service builds these from the DB;
    the matcher only sees this pure shape.

    Attributes:
        source_id: The ``source:...`` record id (the ``cites`` edge ``out``).
        title: The source's title.
        authors: The source's authors (surnames compared, as for the reference).
        doi: The source's DOI when known (enables the gold DOI-exact path).
        year: The source's publication year when known.
    """

    source_id: str
    title: str = ""
    authors: Tuple[str, ...] = field(default_factory=tuple)
    doi: Optional[str] = None
    year: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.authors, tuple):
            object.__setattr__(self, "authors", tuple(self.authors or ()))


@dataclass(frozen=True)
class CitationMatch:
    """The result of matching ONE reference to the corpus.

    Attributes:
        source_id: The matched in-corpus ``source:...`` id (the ``cites`` ``out``).
        confidence: Match confidence in [0, 1] — ``DOI_CONFIDENCE`` for a DOI hit,
            else the blended title+author score (≥ ``MIN_TITLE_AUTHOR_CONFIDENCE``).
        method: ``"doi"`` or ``"title_author"`` — which precision path resolved it
            (recorded on the edge so a reviewer knows WHY it matched).
        reference: The :class:`ParsedReference` that produced the match (its
            ``raw_text`` is carried onto the edge as the human-readable citation).
    """

    source_id: str
    confidence: float
    method: str
    reference: ParsedReference


# --------------------------------------------------------------------------
# Normalization helpers (pure)
# --------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """Lowercase, NFKC-fold, strip punctuation, collapse whitespace.

    The shared normalization for title comparison: removes diacritics-insensitive
    casing and punctuation so "Cohesion Policy & Institutional Quality." and
    "cohesion policy and institutional quality" compare as near-equal. ``&`` is
    expanded to "and" first so an ampersand title and its spelled-out form align.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().replace("&", " and ")
    # Drop anything that is not a word char or whitespace.
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_doi(doi: Optional[str]) -> str:
    """Canonicalize a DOI for exact comparison.

    Strips a leading ``doi:`` / ``https://doi.org/`` / ``http://dx.doi.org/``
    prefix and lowercases (DOIs are case-insensitive). Returns ``""`` for a
    missing/blank DOI so two empty DOIs never compare equal (an empty DOI must
    NOT be a match signal).
    """
    if not doi:
        return ""
    d = doi.strip().lower()
    d = re.sub(r"^\s*doi:\s*", "", d)
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return d.strip()


def _surnames(authors: Sequence[str]) -> set:
    """Extract a set of normalized author SURNAMES from author strings.

    Handles both "Surname, Initials" and "Given Surname" forms: for a comma form
    the surname is the part before the comma; otherwise the last whitespace token.
    Comparing on surnames makes "Acemoglu, D.", "D. Acemoglu" and "Daron
    Acemoglu" all yield "acemoglu", so an author confirmer is robust to citation
    formatting without over-matching on initials.
    """
    out: set = set()
    for raw in authors or ():
        name = _normalize_text(raw)
        if not name:
            continue
        if "," in raw:
            surname = _normalize_text(raw.split(",", 1)[0])
        else:
            surname = name.split()[-1] if name.split() else ""
        if surname:
            out.add(surname)
    return out


def _levenshtein_similarity(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in [0, 1] (1 == identical).

    Single-row DP (O(min(m, n)) space). Mirrors the entity-filtering fuzzy
    resolver's metric so title matching uses the SAME string-distance discipline
    Track K resolution relies on, without importing a private method.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) > len(b):
        a, b = b, a
    len_a, len_b = len(a), len(b)
    current = list(range(len_a + 1))
    for i in range(1, len_b + 1):
        previous, current = current, [i] + [0] * len_a
        for j in range(1, len_a + 1):
            cost = 0 if a[j - 1] == b[i - 1] else 1
            current[j] = min(
                previous[j] + 1,      # delete
                current[j - 1] + 1,   # insert
                previous[j - 1] + cost,  # substitute
            )
    max_len = max(len_a, len_b)
    return 1.0 - current[len_a] / max_len if max_len else 1.0


def _author_agreement(ref_surnames: set, src_surnames: set) -> float:
    """Fraction of the reference's surnames found among the source's surnames.

    Uses the REFERENCE as the denominator (Jaccard-of-reference): a reference
    listing one author that the source shares scores 1.0 even if the source lists
    more co-authors. Returns 0.0 when either side has no parseable surname (no
    author signal → the title+author path cannot confirm → handled by the caller
    as "no author confirmation").
    """
    if not ref_surnames or not src_surnames:
        return 0.0
    shared = ref_surnames & src_surnames
    return len(shared) / len(ref_surnames)


def _title_author_confidence(
    ref: ParsedReference, src: SourceRecord
) -> Tuple[float, float, float]:
    """Compute (combined, title_sim, author_agreement) for the no-DOI path.

    Returns the blended confidence plus its two components so the caller can both
    threshold the combined score AND require the title component to clear
    ``MIN_TITLE_SIMILARITY`` (a high author agreement must not carry a poor
    title over the line — author is a CONFIRMER, not a substitute for the title).
    """
    title_sim = _levenshtein_similarity(
        _normalize_text(ref.title), _normalize_text(src.title)
    )
    author_agree = _author_agreement(_surnames(ref.authors), _surnames(src.authors))
    combined = _TITLE_WEIGHT * title_sim + _AUTHOR_WEIGHT * author_agree
    return combined, title_sim, author_agree


def _candidate_confidence(
    ref: ParsedReference, src: SourceRecord
) -> Tuple[float, str]:
    """Score one (reference, source) pair → (confidence, method).

    Confidence 0.0 means "not a viable match" (below precision floors). A viable
    match returns its confidence and the path that produced it. The DOI path
    short-circuits — a DOI hit is certain and never overridden by a weaker
    title+author score.
    """
    ref_doi = normalize_doi(ref.doi)
    src_doi = normalize_doi(src.doi)
    if ref_doi and src_doi and ref_doi == src_doi:
        return DOI_CONFIDENCE, "doi"

    combined, title_sim, author_agree = _title_author_confidence(ref, src)
    # Both gates must hold: the title must be a strong match AND the combined
    # title+author confidence must clear the bar (author confirms the title).
    if (
        title_sim >= MIN_TITLE_SIMILARITY
        and author_agree > 0.0
        and combined >= MIN_TITLE_AUTHOR_CONFIDENCE
    ):
        return combined, "title_author"
    return 0.0, ""


def match_reference(
    reference: ParsedReference,
    sources: Sequence[SourceRecord],
    *,
    from_source_id: Optional[str] = None,
) -> Optional[CitationMatch]:
    """Resolve ONE reference to its best in-corpus source, or ``None``.

    Pure and precision-first: scores the reference against every in-corpus source
    (excluding ``from_source_id`` so a document never cites itself), keeps only
    candidates that clear the precision floors, and returns the single confident
    winner. Returns ``None`` — meaning "no edge" — when:

    * no source clears the precision floor (an external reference), OR
    * two distinct sources clear it within ``AMBIGUITY_MARGIN`` (ambiguous — do
      not force-resolve), OR
    * the only candidate is the source the reference came from (self-citation).

    Args:
        reference: The parsed reference to resolve (the Track V output unit).
        sources: The in-corpus sources to match against.
        from_source_id: The source this reference was extracted from; it is
            excluded as a candidate so no self-citation can be produced.

    Returns:
        A :class:`CitationMatch` on a confident, unambiguous, non-self match;
        ``None`` otherwise (the precision guard / external-reference case).
    """
    scored: List[Tuple[float, str, SourceRecord]] = []
    for src in sources:
        # No self-citation: a reference never resolves to its own document.
        if from_source_id is not None and src.source_id == from_source_id:
            continue
        confidence, method = _candidate_confidence(reference, src)
        if confidence > 0.0:
            scored.append((confidence, method, src))

    if not scored:
        return None

    # Order: DOI (certain) before title_author, then by confidence, then a
    # deterministic source_id tie-break so identical inputs always yield the same
    # winner (idempotent materialization). DOI sorts first because a DOI-exact
    # hit is a unique-identifier certainty and must outrank any fuzzy candidate.
    def _rank(t: Tuple[float, str, SourceRecord]) -> Tuple[int, float, str]:
        conf, method, src = t
        doi_first = 0 if method == "doi" else 1
        return (doi_first, -conf, src.source_id)

    scored.sort(key=_rank)
    best_conf, best_method, best_src = scored[0]

    # Ambiguity guard: a clear winner must out-score any OTHER candidate.
    #   - A DOI-exact winner is a unique-identifier certainty: it is ambiguous
    #     ONLY against another DOI-exact hit (two sources can't truthfully share a
    #     DOI, but if the data did, force-resolving would be wrong). A fuzzy
    #     runner-up never blocks a DOI winner.
    #   - A fuzzy (title_author) winner must beat the next fuzzy candidate by the
    #     margin; two near-identical fuzzy targets → no confident target.
    if len(scored) > 1:
        runner_conf, runner_method, runner_src = scored[1]
        if runner_src.source_id != best_src.source_id:
            if best_method == "doi":
                if runner_method == "doi":
                    return None  # two distinct DOI-exact hits — refuse to guess
            elif best_conf - runner_conf < AMBIGUITY_MARGIN:
                return None

    return CitationMatch(
        source_id=best_src.source_id,
        confidence=round(best_conf, 4),
        method=best_method,
        reference=reference,
    )


@dataclass(frozen=True)
class CorpusMatchResult:
    """The outcome of matching a whole corpus' references (telemetry).

    Attributes:
        matches: The confident intra-corpus matches → the ``cites`` edges to
            create (one per resolved reference).
        external: Count of references that resolved to NO in-corpus source — the
            external works (recorded as a count; Track V may later stub them).
        ambiguous: Count rejected by the ambiguity guard (cleared the floor but
            had no clear winner — surfaced separately from plain external).
        self_citations_skipped: Count where the only candidate was the origin
            source (a self-citation that was correctly suppressed).
        total_references: Total references considered across all sources.
    """

    matches: List[CitationMatch]
    external: int
    ambiguous: int
    self_citations_skipped: int
    total_references: int


def match_corpus_references(
    references_by_source: "dict[str, Sequence[ParsedReference]]",
    sources: Sequence[SourceRecord],
) -> CorpusMatchResult:
    """Match every source's references against the corpus (pure aggregate).

    Drives :func:`match_reference` for each ``(origin_source, reference)`` pair,
    passing the origin id as ``from_source_id`` so self-citations are suppressed,
    and tallies the precision-guard outcomes (external / ambiguous / self) for
    telemetry. No I/O — the materialization service feeds it the loaded sources
    and the V-produced references, then persists ``result.matches``.

    Args:
        references_by_source: ``{origin_source_id: [ParsedReference, ...]}`` — the
            per-source reference lists Track V produces.
        sources: The in-corpus sources to resolve references against.

    Returns:
        A :class:`CorpusMatchResult` carrying the confident matches plus the
        per-outcome counts. On the current corpus (no V input) the input is
        empty → ``matches == []`` (the clean 0-edge no-op).
    """
    matches: List[CitationMatch] = []
    external = 0
    ambiguous = 0
    self_skipped = 0
    total = 0

    # Pre-compute, per origin, whether a same-origin candidate even exists so we
    # can attribute a None to "self-citation suppressed" vs "external". This is
    # telemetry only; it never changes whether an edge is created.
    source_ids = {s.source_id for s in sources}

    for origin_id, refs in references_by_source.items():
        for ref in refs or ():
            total += 1
            match = match_reference(ref, sources, from_source_id=origin_id)
            if match is not None:
                matches.append(match)
                continue
            # No edge — classify why for telemetry. Re-run WITHOUT the self
            # exclusion: if a confident match appears that points back at the
            # origin, the None was a suppressed self-citation; otherwise it was
            # external (or ambiguity-rejected).
            unguarded = match_reference(ref, sources, from_source_id=None)
            if (
                unguarded is not None
                and unguarded.source_id == origin_id
                and origin_id in source_ids
            ):
                self_skipped += 1
            elif unguarded is None and _has_viable_candidate(
                ref, sources, origin_id
            ):
                # A viable candidate existed but the ambiguity guard rejected it.
                ambiguous += 1
            else:
                external += 1

    return CorpusMatchResult(
        matches=matches,
        external=external,
        ambiguous=ambiguous,
        self_citations_skipped=self_skipped,
        total_references=total,
    )


def _has_viable_candidate(
    reference: ParsedReference,
    sources: Sequence[SourceRecord],
    origin_id: Optional[str],
) -> bool:
    """True if ≥2 distinct non-origin sources clear the precision floor.

    Telemetry helper: distinguishes an ambiguity-guard rejection (multiple viable
    targets) from a plain external reference (zero viable targets), without
    affecting any edge decision.
    """
    viable = {
        src.source_id
        for src in sources
        if src.source_id != origin_id
        and _candidate_confidence(reference, src)[0] > 0.0
    }
    return len(viable) >= 2


__all__ = [
    "ParsedReference",
    "SourceRecord",
    "CitationMatch",
    "CorpusMatchResult",
    "match_reference",
    "match_corpus_references",
    "normalize_doi",
    "DOI_CONFIDENCE",
    "MIN_TITLE_SIMILARITY",
    "MIN_TITLE_AUTHOR_CONFIDENCE",
    "AMBIGUITY_MARGIN",
]
