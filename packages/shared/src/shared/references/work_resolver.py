"""External work-resolution cascade (Track V.4).

Track V turns a document's bibliography/footnotes into ``ParsedReference`` units
(the Track V → U.3 boundary). V.4 is the *outward-looking* half: given ONE
``ParsedReference``, resolve it to a single canonical :class:`ResolvedWork` in an
external authority (OpenAlex / Crossref / DataCite / arXiv / overheid.nl /
RePEc). U.3 matches references against the IN-corpus sources; V.4 resolves the
EXTERNAL ones. This module is pure resolution — no DB, no ``cites`` wiring; a
later phase feeds a :class:`ResolvedWork` into materialization.

Precision over recall (the K.4 discipline, mirrored)
====================================================
A WRONG resolution attaches a false identity to a citation — strictly worse than
leaving it unresolved. So every leg is guarded exactly as the K.4 Crossref
provider guards its auto-link:

* A **DOI-exact** hit (the query DOI equals the returned, normalized DOI) is the
  gold signal — a unique identifier cannot match by coincidence → confidence 1.0.
* A **title/author** hit must clear a real lexical title-overlap floor
  (:data:`MIN_TITLE_OVERLAP`, token-Jaccard on normalized titles, same metric as
  ``crossref_provider._title_overlap``) AND, when the reference carries authors,
  an author-surname confirmer. A high API relevance score alone is NOT trusted
  (it is query-length dependent, not a probability). The blended confidence must
  clear :data:`MIN_MATCH_CONFIDENCE`.
* Below-threshold ⇒ the leg returns ``None`` and the cascade falls through. If no
  leg clears its guard the reference stays **unresolved** (``None``).

Shape-first routing
===================
The cascade routes by the *shape* of the reference (a DOI, an arXiv id, a Dutch
Kamerstuk, an econ working paper, or a bare title) so each authority only sees
the references it can actually answer — see :class:`ResolverCascade`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from loguru import logger

from shared.retrieval.cites_matching import ParsedReference, normalize_doi
from shared.utils.name_normalizer import normalize_entity_name

# --------------------------------------------------------------------------
# Precision thresholds (the auditable knobs — mirror crossref_provider)
# --------------------------------------------------------------------------

#: Confidence of a DOI-exact resolution. A DOI is a globally unique identifier,
#: so an exact (normalized) match is treated as certain.
DOI_MATCH_CONFIDENCE = 1.0

#: Minimum token-Jaccard overlap between the reference title and a candidate's
#: title for the title/author path. Mirrors ``crossref_provider._MIN_TITLE_OVERLAP``
#: — enough shared lexical content to reject a wholly-unrelated title.
MIN_TITLE_OVERLAP = 0.5

#: Minimum blended title+author confidence to accept a no-DOI resolution.
MIN_MATCH_CONFIDENCE = 0.5

#: Blend weight of the title vs the author confirmer in the combined confidence.
#: Title carries the signal; author agreement confirms it (as in cites_matching).
_TITLE_WEIGHT = 0.7
_AUTHOR_WEIGHT = 0.3


# --------------------------------------------------------------------------
# Result type — a DB-free, testable canonical work (mirrors VocabMatch)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedWork:
    """A reference resolved to a canonical external work.

    Plain, DB-free dataclass (like ``vocabulary.VocabMatch``) so resolution logic
    is unit-testable without a database. A later phase turns a ``ResolvedWork``
    into whatever the ``cites`` materialization needs; this type carries no DB id.

    Attributes:
        title: The canonical title from the resolving authority.
        source: The resolver ``name`` that produced this (``"openalex"`` …) — the
            provenance of the resolution.
        method: Which precision path resolved it — ``"doi"`` (gold, exact),
            ``"arxiv_id"``, ``"sru"`` (overheid id/title), or ``"title_author"``.
        confidence: Resolution confidence in [0, 1]; 1.0 for a DOI/id-exact hit,
            else the blended title+author score (≥ ``MIN_MATCH_CONFIDENCE``).
        doi: The canonical DOI (normalized), when the authority has one.
        authors: Canonical author display names.
        year: Publication year, when known.
        venue: Journal / publisher / series, when known.
        external_id: The authority-native id (OpenAlex ``W…`` URL, arXiv id, DOI,
            overheid identifier) — the stable handle for the resolved work.
        external_uri: A canonical, dereferenceable URL for the work.
        institutions: Affiliation display names (populated by OpenAlex; the input
            ROR enrichment resolves).
        author_ids: ORCID iDs for the authors (from the authority, or added by the
            opt-in enrichment step). Never required.
        institution_ids: ROR ids for the institutions (as ``author_ids``).
    """

    title: str
    source: str
    method: str
    confidence: float
    doi: Optional[str] = None
    authors: Tuple[str, ...] = field(default_factory=tuple)
    year: Optional[int] = None
    venue: Optional[str] = None
    external_id: str = ""
    external_uri: str = ""
    institutions: Tuple[str, ...] = field(default_factory=tuple)
    author_ids: Tuple[str, ...] = field(default_factory=tuple)
    institution_ids: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Coerce sequence fields to tuples so the frozen dataclass stays stable
        # even when a provider builds them from lists.
        for attr in ("authors", "institutions", "author_ids", "institution_ids"):
            value = getattr(self, attr)
            if not isinstance(value, tuple):
                object.__setattr__(self, attr, tuple(value or ()))


# --------------------------------------------------------------------------
# Resolver protocol — one adapter per external authority
# --------------------------------------------------------------------------


@runtime_checkable
class WorkResolver(Protocol):
    """Adapter for one external work authority.

    Every resolver:

    * ``name`` — a stable short id (also ``ResolvedWork.source``).
    * ``resolve(ref)`` — resolve ONE reference to a confident ``ResolvedWork`` or
      ``None``. MUST NOT raise on a network failure (fail-soft to ``None``) and
      MUST NOT return a below-threshold guess (precision guard).
    """

    @property
    def name(self) -> str:
        """Stable short identifier (also ``ResolvedWork.source``)."""
        ...

    async def resolve(self, ref: ParsedReference) -> Optional[ResolvedWork]:
        """Resolve ``ref`` to a confident work, or ``None`` (unresolved)."""
        ...


class WorkEnricher(Protocol):
    """Opt-in post-match ID enricher (ORCID / ROR). Never required."""

    async def enrich(self, work: ResolvedWork) -> ResolvedWork:
        """Return ``work`` augmented with author/institution ids (best-effort)."""
        ...


# --------------------------------------------------------------------------
# Pure matching helpers (shared by the title/author providers)
# --------------------------------------------------------------------------


def title_overlap(a: str, b: str) -> float:
    """Token-Jaccard overlap of two titles after normalization.

    Both sides run through :func:`normalize_entity_name` before tokenizing on
    whitespace; returns intersection-over-union of the token sets (0.0 when either
    side has no tokens). Identical to the K.4 Crossref title gate so V.4 uses the
    SAME lexical-overlap discipline the entity resolver trusts.
    """
    a_tokens = set(normalize_entity_name(a).split())
    b_tokens = set(normalize_entity_name(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    union = a_tokens | b_tokens
    return len(a_tokens & b_tokens) / len(union)


def _surnames(authors: Sequence[str]) -> set:
    """Normalized author SURNAMES from author strings (see cites_matching).

    A comma form ("Acemoglu, D.") takes the part before the comma; otherwise the
    last whitespace token. Comparing surnames makes citation-format variants align
    without over-matching on initials.
    """
    out: set = set()
    for raw in authors or ():
        if not raw:
            continue
        if "," in raw:
            surname = normalize_entity_name(raw.split(",", 1)[0])
        else:
            tokens = normalize_entity_name(raw).split()
            surname = tokens[-1] if tokens else ""
        if surname:
            out.add(surname)
    return out


def _author_agreement(ref_authors: Sequence[str], cand_authors: Sequence[str]) -> float:
    """Fraction of the reference's surnames present among the candidate's.

    Reference is the denominator: a reference listing one author the candidate
    shares scores 1.0 even if the candidate lists more co-authors. Returns 0.0
    when either side has no parseable surname (no author confirmation available).
    """
    ref = _surnames(ref_authors)
    cand = _surnames(cand_authors)
    if not ref or not cand:
        return 0.0
    return len(ref & cand) / len(ref)


def title_author_confidence(
    ref: ParsedReference,
    cand_title: str,
    cand_authors: Sequence[str],
) -> Tuple[float, float, float]:
    """Return ``(combined, overlap, author_agreement)`` for a title/author hit.

    The caller both thresholds the combined score AND requires the title-overlap
    component to clear :data:`MIN_TITLE_OVERLAP` (an author agreement must not
    carry a poor title over the line — author confirms, it does not substitute).
    """
    overlap = title_overlap(ref.title, cand_title)
    author_agree = _author_agreement(ref.authors, cand_authors)
    combined = _TITLE_WEIGHT * overlap + _AUTHOR_WEIGHT * author_agree
    return combined, overlap, author_agree


def passes_title_author_guard(
    ref: ParsedReference,
    cand_title: str,
    cand_authors: Sequence[str],
) -> Tuple[bool, float]:
    """Precision gate for a title/author candidate → ``(accepted, confidence)``.

    Accepts only when the title overlap clears :data:`MIN_TITLE_OVERLAP`, an
    author confirmer agrees (when the reference lists authors), and the blended
    confidence clears :data:`MIN_MATCH_CONFIDENCE`. Otherwise ``(False, 0.0)`` —
    the leg falls through and the reference stays unresolved.
    """
    if not ref.title:
        return False, 0.0
    combined, overlap, author_agree = title_author_confidence(
        ref, cand_title, cand_authors
    )
    if overlap < MIN_TITLE_OVERLAP:
        return False, 0.0
    if ref.authors and author_agree <= 0.0:
        return False, 0.0
    if combined < MIN_MATCH_CONFIDENCE:
        return False, 0.0
    return True, round(combined, 4)


# --------------------------------------------------------------------------
# Shape detection (pure, testable) — drives cascade routing
# --------------------------------------------------------------------------

_ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(\S+)", re.IGNORECASE)
_ARXIV_NEW = re.compile(r"arxiv[:\s]\s*(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
_ARXIV_OLD = re.compile(
    r"arxiv[:\s]\s*([a-z][a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", re.IGNORECASE
)
_ARXIV_BARE_NEW = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")

_NL_POLICY_MARKERS = (
    "kamerstuk",
    "staatsblad",
    "staatscourant",
    "tweede kamer",
    "eerste kamer",
    "handelingen der",
    "aanhangsel",
    "koninklijk besluit",
)
# A Dutch parliamentary vergaderjaar, e.g. "2019/20", "2019-2020".
_VERGADERJAAR = re.compile(r"\b(?:19|20)\d{2}[\-/](?:\d{2}|\d{4})\b")

_ECON_WP_MARKERS = (
    "working paper",
    "discussion paper",
    "nber",
    "iza",
    "cepr",
    "cesifo",
    "repec",
    "mimeo",
)


def extract_arxiv_id(ref: ParsedReference) -> Optional[str]:
    """Return the arXiv id carried by ``ref`` (DOI or text), else ``None``.

    Recognizes the arXiv DOI form (``10.48550/arXiv.<id>``) and an explicit
    ``arXiv:<id>`` mention (new ``YYMM.NNNNN`` or legacy ``archive/YYMMNNN``). A
    bare ``YYMM.NNNNN`` is only accepted when the reference actually mentions
    "arxiv" (a bare number is too ambiguous to route on alone).

    # TODO(V.4-live): confirm the id-form coverage against real bibliographies;
    # a live smoke should sample how references write arXiv ids in practice.
    """
    if ref.doi:
        m = _ARXIV_DOI.search(ref.doi)
        if m:
            return m.group(1).lower().rstrip(".")
    hay = " ".join(filter(None, [ref.raw_text, ref.venue or ""]))
    if "arxiv" in hay.lower():
        for pattern in (_ARXIV_NEW, _ARXIV_OLD, _ARXIV_BARE_NEW):
            m = pattern.search(hay)
            if m:
                return m.group(1)
    return None


def looks_like_nl_policy(ref: ParsedReference) -> bool:
    """Heuristic: does ``ref`` look like a Dutch Kamerstuk / policy document?

    True on an explicit marker ("Kamerstuk", "Staatsblad", "Tweede Kamer", …) or a
    vergaderjaar pattern co-occurring with a chamber mention. Marker-primary so a
    bare year-range (which also appears in page ranges) never routes on its own.
    """
    t = (ref.raw_text or "").lower()
    if any(marker in t for marker in _NL_POLICY_MARKERS):
        return True
    if _VERGADERJAAR.search(t) and ("kamer" in t or "handelingen" in t):
        return True
    return False


def looks_like_econ_wp(ref: ParsedReference) -> bool:
    """Heuristic: does ``ref`` look like an economics working/discussion paper?"""
    t = (ref.raw_text or "").lower()
    return any(marker in t for marker in _ECON_WP_MARKERS)


# --------------------------------------------------------------------------
# The cascade — shape-route, stop at first confident hit
# --------------------------------------------------------------------------


class ResolverCascade:
    """Route a reference across providers, stopping at the first confident hit.

    Routing is shape-first so each authority only sees references it can answer:

    1. **DOI present** → Crossref, then DataCite.
    2. **arXiv id present** → arXiv.
    3. **NL Kamerstuk / policy shape** → overheid.nl.
    4. **econ working-paper shape AND RePEc configured** → RePEc.
    5. **Fallback (title/author)** → OpenAlex (broadest recall).

    Each leg runs in order; on a network failure or a below-guard confidence the
    leg yields ``None`` and the cascade falls through to the next. The first
    ``ResolvedWork`` a leg returns wins (optionally ID-enriched); if no leg clears
    its precision guard the result is ``None`` (unresolved — preferred over a
    wrong match).

    An **unavailable** resolver (a resolver exposing ``available == False``, e.g.
    an unconfigured RePEc) is skipped silently: it never raises, warns, or blocks.
    """

    def __init__(
        self,
        *,
        openalex: Optional[WorkResolver] = None,
        crossref: Optional[WorkResolver] = None,
        datacite: Optional[WorkResolver] = None,
        arxiv: Optional[WorkResolver] = None,
        overheid: Optional[WorkResolver] = None,
        repec: Optional[WorkResolver] = None,
        enricher: Optional[WorkEnricher] = None,
    ) -> None:
        self._openalex = openalex
        self._crossref = crossref
        self._datacite = datacite
        self._arxiv = arxiv
        self._overheid = overheid
        self._repec = repec
        self._enricher = enricher

    def _route(self, ref: ParsedReference) -> List[WorkResolver]:
        """Ordered, de-duplicated resolver legs for ``ref`` (shape-first)."""
        legs: List[Optional[WorkResolver]] = []

        if normalize_doi(ref.doi):
            legs.append(self._crossref)
            legs.append(self._datacite)

        if extract_arxiv_id(ref):
            legs.append(self._arxiv)

        if looks_like_nl_policy(ref):
            legs.append(self._overheid)

        if looks_like_econ_wp(ref) and self._is_available(self._repec):
            legs.append(self._repec)

        # OpenAlex is the always-on fallback leg (broadest recall).
        legs.append(self._openalex)

        ordered: List[WorkResolver] = []
        seen: set = set()
        for resolver in legs:
            if resolver is None or not self._is_available(resolver):
                continue
            if id(resolver) in seen:
                continue
            seen.add(id(resolver))
            ordered.append(resolver)
        return ordered

    @staticmethod
    def _is_available(resolver: Optional[WorkResolver]) -> bool:
        """A resolver is available unless it explicitly reports ``available False``.

        The config-gate seam: an unconfigured RePEc sets ``available = False`` and
        is skipped silently. Any resolver without the attribute is available.
        """
        if resolver is None:
            return False
        return bool(getattr(resolver, "available", True))

    async def resolve(self, ref: ParsedReference) -> Optional[ResolvedWork]:
        """Resolve ``ref`` to a canonical work, or ``None`` (unresolved)."""
        for resolver in self._route(ref):
            try:
                result = await resolver.resolve(ref)
            except Exception as exc:  # a resolver should fail soft; guard anyway
                logger.warning(
                    "work resolver {name} raised (treated as no-match): {exc}",
                    name=getattr(resolver, "name", "?"),
                    exc=exc,
                )
                result = None
            if result is None:
                continue
            if self._enricher is not None:
                try:
                    result = await self._enricher.enrich(result)
                except Exception as exc:  # enrichment is best-effort, never fatal
                    logger.warning(
                        "work enrichment failed (kept unenriched): {exc}", exc=exc
                    )
            return result
        return None


__all__ = [
    "ResolvedWork",
    "WorkResolver",
    "WorkEnricher",
    "ResolverCascade",
    "title_overlap",
    "title_author_confidence",
    "passes_title_author_guard",
    "extract_arxiv_id",
    "looks_like_nl_policy",
    "looks_like_econ_wp",
    "DOI_MATCH_CONFIDENCE",
    "MIN_TITLE_OVERLAP",
    "MIN_MATCH_CONFIDENCE",
]
