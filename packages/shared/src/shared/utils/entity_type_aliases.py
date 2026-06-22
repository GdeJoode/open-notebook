"""Curated EN+NL residual entity-type alias map (Track L.2).

This module is the **single, language-specific surface** of Track L. The L.1
``canonical_bridge`` is language-agnostic: it resolves a label only when that
label is a type declared in an applied ontology (walking the ontology's own
``parent_type`` chain). This module catches the **residual** — labels the LLM
free-forms that are NOT in any applied ontology (``Persoon``, ``Organisatie``,
``Technologie``, ``ABBREVIATION``) — with a curated EN+NL alias map. The user
constraint is explicit: EN+NL covers ~98% of needs, and there is **no
LLM/embedding fallback** for the residual 2%.

Resolution order at the persist boundary (``entity_persistence_service``):

    1. L.1 ontology bridge (``resolve_ontology_type``) — language-agnostic.
    2. L.2 EN+NL alias map (:data:`ENTITY_TYPE_ALIASES`, here).
    3. L.2 non-silent unknown-label fallback (:func:`resolve_residual_type`) —
       an unmapped label coarses to a sensible default BUT the original label is
       ALWAYS preserved (``primary_type`` + ``type_tags``); it is never silently
       flattened to ``other`` (the historical bug that lost the rich type).

**Language-containment guarantee (AC5):** every Dutch literal in Track L lives
HERE and nowhere else — ``canonical_bridge.py`` (L.1) must stay free of Dutch
strings. A structure test asserts this. Adding a language = extending this one
map (and/or adding an ontology), never editing the bridge.

**technology / programme decision (L.2, option (a)):** the aliases
``technologie``/``technology`` → ``technology`` and ``programma`` /
``regiodeal`` / ``deal`` / ``project`` → ``programme`` are the REAL canonical
targets. But ``technology`` and ``programme`` are not yet members of the
persistence enum ``_ALLOWED_ENTITY_TYPES`` (that is L.3). Until L.3 lands, the
runtime enum-guard in ``entity_persistence_service`` re-pins these to ``other``
(the original label is still preserved in ``primary_type`` — never lost). This
keeps L.3 a pure enum addition: no alias rewrites needed when it lands. The
alternative (interim-mapping ``technologie``→``concept`` / ``programma``→
``creative_work`` and re-pointing in L.3) was rejected to avoid a churned alias
map and a misleading coarse type in the interim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Common short forms / NER tags + the curated EN+NL residual map. Keyed on the
# normalized form (lower-cased, spaces/hyphens -> underscores). Values are
# canonical ``entity_type`` targets. Overlap with the L.1 bridge is harmless:
# the bridge runs FIRST and wins; this map is the residual catcher for labels
# the bridge returned ``None`` for.
ENTITY_TYPE_ALIASES = {
    # --- English short forms / spaCy NER tags (pre-L.2 set, kept) ---
    "org": "organization",
    "organisation": "organization",  # British spelling
    "norp": "organization",  # nationalities / religious / political groups
    "company": "organization",
    "per": "person",
    "people": "person",
    "loc": "location",
    "gpe": "location",  # geopolitical entity (spaCy NER)
    "geo": "location",
    "fac": "location",  # facility
    "place": "location",
    "gov": "government_organization",
    "govt": "government_organization",
    "law": "legislation",
    "misc": "other",
    # --- Dutch (NL) residual terms the LLM free-forms (L.2) ---
    # people
    "persoon": "person",
    "wethouder": "person",  # alderman (municipal office holder)
    "bestuurder": "person",  # administrator / board member
    "ambtenaar": "person",  # civil servant
    # organizations
    "organisatie": "organization",
    "ministerie": "government_organization",
    "uitvoeringsorganisatie": "government_organization",  # executive agency
    # places / administrative areas
    "gemeente": "administrative_area",  # municipality
    "provincie": "administrative_area",  # province
    "regio": "administrative_area",  # region
    # topics / themes
    "thema": "topic",
    "beleidsthema": "topic",  # policy theme
    "sector": "topic",  # economic / policy sector (curated as topic; see risks)
    # legislation
    "wet": "legislation",  # statute / act
    # creative works / publications
    "document": "creative_work",
    "publicatie": "creative_work",
    "publication": "creative_work",
    # technology / programme — REAL targets; the enum-guard re-pins them to
    # ``other`` until L.3 adds these canonical types (option (a), see docstring).
    "technologie": "technology",
    "technology": "technology",
    "programma": "programme",
    "regiodeal": "programme",
    "deal": "programme",
    "project": "programme",
}


# Labels that are extraction NOISE, not entity types. ``ABBREVIATION`` is the
# *form* of a mention (the LLM tagged HOW the name was written, not WHAT it is);
# ``Amount`` is a number. These must NOT silently become ``other`` and lose the
# signal that the extractor mis-tagged: the fallback coarses them to a sensible
# default while PRESERVING the raw label in ``primary_type`` so L.6 metrics and
# L.5 retro-typing can see them. Kept distinct from genuine-unknown labels
# (``Frobnicator``) so a reviewer can tell noise from an unmodeled real type.
NON_TYPE_LABELS = frozenset({"abbreviation", "amount"})


def _normalize(raw: object) -> str:
    """Normalize a raw label to the alias-map key form (lower + underscores)."""
    return str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")


@dataclass
class ResidualResolution:
    """Outcome of the L.2 residual (alias + fallback) resolution.

    Attributes:
        entity_type: The canonical ``entity_type`` to attempt to persist. May be
            a not-yet-in-enum value (``technology`` / ``programme``) when the
            alias map points there; the persistence enum-guard re-pins it. For an
            unmapped label this is a sensible coarse default, NEVER a silent
            ``other`` that discards the label.
        primary_type: The original raw label, preserved whenever it carries a
            non-trivial signal (i.e. always, except an empty/whitespace label).
            This is the L.2 anti-flattening guarantee — never null when a real
            label exists.
        type_tags: ``[raw_label]`` when a label is preserved, else empty.
        is_noise: ``True`` when the label is in :data:`NON_TYPE_LABELS`
            (extraction noise), so the caller may flag it. ``False`` for a
            genuine-unknown label (an unmodeled real type).
        mapped: ``True`` when the alias map resolved the label; ``False`` when
            the fallback default was used.
    """

    entity_type: str
    primary_type: Optional[str] = None
    type_tags: List[str] = field(default_factory=list)
    is_noise: bool = False
    mapped: bool = False


# Coarse defaults for the non-silent fallback. A genuine-unknown label and
# extraction noise both default to ``concept`` (the softest non-``other`` bucket
# for an abstract/unknown signal) while the raw label is preserved separately —
# so the rich label survives even though the coarse projection is generic.
_UNKNOWN_DEFAULT = "concept"
_NOISE_DEFAULT = "concept"


def resolve_residual_type(raw_label: object) -> ResidualResolution:
    """Resolve a residual label (not handled by the L.1 bridge) to a type.

    This is the L.2 path. Order:

    1. **Alias map** (:data:`ENTITY_TYPE_ALIASES`) — a curated EN+NL hit returns
       the canonical target and (because the raw label still carried a rich
       signal, e.g. ``Technologie``) preserves the original in ``primary_type``.
    2. **Non-silent fallback** — an unmapped label coarses to a sensible default
       but ALWAYS preserves the raw label in ``primary_type`` / ``type_tags``.
       Extraction noise (:data:`NON_TYPE_LABELS`) is flagged via ``is_noise``;
       a genuine-unknown label is not.

    The empty/whitespace label is the only case with no preserved
    ``primary_type`` (there is no signal to preserve) — it returns the unknown
    default with ``primary_type=None``.

    Note: the returned ``entity_type`` is NOT enum-guarded here (the alias map
    may legitimately point at ``technology`` / ``programme`` pre-L.3). The
    persistence boundary applies the ``_ALLOWED_ENTITY_TYPES`` guard, re-pinning
    a not-yet-valid value to ``other`` while keeping the preserved label intact.
    """
    raw_str = str(raw_label or "").strip()
    norm = _normalize(raw_label)

    # (1) Curated EN+NL alias hit.
    mapped = ENTITY_TYPE_ALIASES.get(norm)
    if mapped is not None:
        return ResidualResolution(
            entity_type=mapped,
            primary_type=raw_str or None,
            type_tags=[raw_str] if raw_str else [],
            is_noise=False,
            mapped=True,
        )

    # (2a) No signal to preserve — empty label.
    if not norm:
        return ResidualResolution(
            entity_type=_UNKNOWN_DEFAULT,
            primary_type=None,
            type_tags=[],
            is_noise=False,
            mapped=False,
        )

    # (2b) Extraction noise — preserve the label, flag it, coarse to a default.
    if norm in NON_TYPE_LABELS:
        return ResidualResolution(
            entity_type=_NOISE_DEFAULT,
            primary_type=raw_str,
            type_tags=[raw_str],
            is_noise=True,
            mapped=False,
        )

    # (2c) Genuine-unknown label — preserve it (rich label survives), coarse to
    # the unknown default. NEVER a silent ``other`` that discards the label.
    return ResidualResolution(
        entity_type=_UNKNOWN_DEFAULT,
        primary_type=raw_str,
        type_tags=[raw_str],
        is_noise=False,
        mapped=False,
    )
