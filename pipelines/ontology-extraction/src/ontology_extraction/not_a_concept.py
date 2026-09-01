"""Extraction-time not-a-concept gate (Track N.3).

A stricter, EXTRACTION-time cousin of entity-filtering's post-hoc ``noise_filter``:
it drops surface forms the LLM emitted that are not genuine domain concepts —
UI/navigation labels, table/figure references, page furniture, boilerplate — so
the graph is not polluted with plausible-but-empty "concepts" (the article's core
lesson: selection/abstention over generation).

Two tiers (Decision N-D4):

1. A **deterministic pre-pass** (:func:`classify_deterministic`) that HIGH-
   PRECISION rejects the obvious non-concepts and fast-ACCEPTS the obvious
   concepts. It is intentionally conservative on the reject side — the AC is that
   no real domain entity is dropped — so anything it is unsure about it returns as
   AMBIGUOUS (``None``) rather than guessing.
2. A **batched LLM-judge** (default ON, D4) that arbitrates only the ambiguous
   middle. The judge's prompt/parse are pure functions here so they unit-test
   without an LLM; the async call itself is orchestrated at the run_pass2 seam and
   mocked in tests. With no judge available the ambiguous set is KEPT (never
   dropped on a guess).

Everything here is pure + deterministic given the same input, so the whole gate is
fast to unit-test without the model.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from loguru import logger

# Generic / catch-all labels. An entity the LLM gave a SPECIFIC schema type is
# trusted (it committed to a type — "map aggressively"); only the generically-
# typed ones are candidates for the not-a-concept gate. Tunable.
_GENERIC_LABELS = frozenset(
    {"", "other", "others", "unknown", "misc", "miscellaneous", "none", "n/a", "concept"}
)

# UNCONDITIONAL reject — UI/nav/boilerplate that is never a domain concept no
# matter what label the LLM guessed (a mis-typed "Click here" is still furniture).
# These are multi-word phrases or UI action verbs with no plausible entity reading.
# EN + NL. Tunable via ``extra_reject_exact``.
_REJECT_ALWAYS = frozenset(
    {
        # navigation / UI phrases + actions
        "click here", "read more", "learn more", "see more", "show more",
        "back to top", "continue reading", "view all", "see all",
        "table of contents", "log in", "log out", "sign up", "sign in",
        "download", "downloads", "print", "share", "subscribe",
        "lees meer", "meer informatie", "inhoudsopgave",
        "downloaden", "afdrukken", "delen", "inloggen", "aanmelden",
        # boilerplate / legal
        "all rights reserved", "copyright", "confidential",
        "terms of service", "terms and conditions", "privacy policy",
        "cookie policy", "disclaimer", "colophon", "colofon",
        "alle rechten voorbehouden", "auteursrecht", "privacybeleid",
    }
)

# Generic table / form / reference FIELD words. Each is a HOMOGRAPH of a possible
# real entity ("Total" = TotalEnergies, "Page" a surname, "Index" a stock index),
# so — unlike ``_REJECT_ALWAYS`` — these reject ONLY under a GENERIC label. A
# SPECIFIC schema label is trusted and keeps them (the fast-accept fires first),
# honouring the AC that a specifically-typed real entity is never dropped. EN + NL.
_FIELD_WORDS = frozenset(
    {
        "total", "subtotal", "sum", "average", "totaal", "subtotaal",
        "date", "name", "amount", "number", "description", "type", "category",
        "status", "datum", "naam", "bedrag", "nummer", "omschrijving", "categorie",
        "yes", "no", "ja", "nee", "true", "false", "n/a", "tbd", "tba",
        "page", "figure", "table", "chapter", "section", "appendix", "note",
        "pagina", "figuur", "tabel", "hoofdstuk", "sectie", "bijlage", "noot",
    }
)

# A bare reference like "Figure 3", "Table 2a", "Page 12", "Hoofdstuk 4" — a
# pointer, not a concept.
_REF_RE = re.compile(
    r"^(?:figure|fig|table|page|chapter|section|appendix|annex|footnote|note|"
    r"exhibit|figuur|tabel|pagina|hoofdstuk|sectie|bijlage|noot)\s*\.?\s*"
    r"\d+[a-z]?$",
    re.IGNORECASE,
)

# Pure number / date / percentage / currency / ordinal / punctuation.
_NUMERIC_RE = re.compile(r"^[\W\d]*\d[\W\d]*$")  # contains a digit, no letters
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")

_MIN_LEN = 2


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _is_titlecase_proper(text: str) -> bool:
    """A multi-word proper-noun-ish phrase (2+ tokens, most starting upper)."""
    words = text.split()
    if len(words) < 2:
        return False
    capped = sum(1 for w in words if w[:1].isupper())
    return capped >= max(2, len(words) - 1)


def classify_deterministic(
    text: str,
    label: str = "",
    *,
    extra_reject_exact: Optional[frozenset] = None,
) -> Optional[bool]:
    """Deterministic tier of the not-a-concept gate.

    Returns:
        ``True``  — NOT a concept (reject, high precision).
        ``False`` — a concept (keep).
        ``None``  — ambiguous; defer to the LLM-judge (or keep if none).
    """
    t = _norm(text)
    low = t.lower()

    # -- UNCONDITIONAL reject (structural non-concepts + unambiguous furniture).
    # These fire regardless of label — a specific label cannot rescue "12345" or
    # "Click here" (the LLM mis-typed furniture).
    if len(t) < _MIN_LEN or _PUNCT_ONLY_RE.match(t):
        return True
    if _NUMERIC_RE.match(t):  # pure number / date / percentage / amount
        return True
    if _REF_RE.match(t):  # "Figure 3", "Hoofdstuk 4"
        return True
    if low in _REJECT_ALWAYS:
        return True
    if extra_reject_exact and low in extra_reject_exact:
        return True

    # -- fast ACCEPT: a SPECIFIC schema label is trusted, BEFORE the homograph
    # field-word reject — so a specifically-typed "Total"/"Index"/"Page" is KEPT
    # (the AC: a real, specifically-typed entity is never dropped).
    if label.strip().lower() not in _GENERIC_LABELS:
        return False

    # -- generic label from here on --------------------------------------------
    if low in _FIELD_WORDS:
        # A bare table/form field word under a generic label → furniture.
        return True
    if _is_titlecase_proper(t):
        # Multi-word proper name even under a generic label → a real entity.
        return False

    # -- AMBIGUOUS: generic label + a plain word/phrase (incl. UI homographs
    # like "Next"/"Home"/"Index") → defer to the judge, never a hard guess.
    return None


def partition_deterministic(
    entities: List[dict],
    *,
    extra_reject_exact: Optional[frozenset] = None,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split entities into (kept, rejected, ambiguous) by the deterministic tier.

    ``entities`` are plain dicts or objects exposing ``text``/``label`` — accessed
    via :func:`_get`. Order within each bucket is preserved.
    """
    kept: List[dict] = []
    rejected: List[dict] = []
    ambiguous: List[dict] = []
    for e in entities:
        verdict = classify_deterministic(
            _get(e, "text"), _get(e, "label"), extra_reject_exact=extra_reject_exact
        )
        if verdict is True:
            rejected.append(e)
        elif verdict is False:
            kept.append(e)
        else:
            ambiguous.append(e)
    return kept, rejected, ambiguous


def _get(entity, attr: str) -> str:
    """Read ``attr`` from a dict OR a pydantic/objects entity, as a string."""
    if isinstance(entity, dict):
        return str(entity.get(attr, "") or "")
    return str(getattr(entity, attr, "") or "")


# ---------------------------------------------------------------------------
# LLM-judge — pure prompt/parse (the async call lives at the run_pass2 seam)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "You are a strict knowledge-graph curator. For each candidate string decide "
    "whether it is a GENUINE domain concept or named entity worth storing in a "
    "knowledge graph, versus page-furniture: a UI/navigation label, a table or "
    "form field header, a figure/page reference, boilerplate, or a vague non-"
    "concept. When unsure, keep it (answer true). Judge the STRING, not its "
    "popularity."
)


def build_judge_prompt(items: List[Tuple[str, str]]) -> str:
    """Render the batched judge user prompt for ``(text, label)`` candidates."""
    lines = [
        "Classify each candidate below. Return ONLY this JSON (no prose):",
        "",
        '{"verdicts": [{"text": "<verbatim>", "is_concept": true}]}',
        "",
        "is_concept = true means a real domain concept/entity; false means page-"
        "furniture / UI label / table field / reference / boilerplate.",
        "",
        "Candidates:",
    ]
    for text, label in items:
        lbl = (label or "").strip() or "other"
        lines.append(f'- "{text}" (type: {lbl})')
    return "\n".join(lines)


def parse_judge_response(
    raw: str, items: List[Tuple[str, str]]
) -> Dict[str, bool]:
    """Parse the judge response into ``{text: is_concept}`` for the items the judge
    EXPLICITLY ruled on (restricted to known candidate texts).

    Robust to markdown fences and junk. A candidate the judge stayed silent on is
    simply absent from the result — the CALLER defaults missing candidates to KEEP
    (never drop on a missing/garbled verdict), and can count the explicit verdicts
    as the number actually arbitrated. Garbage / empty ``raw`` → ``{}``.
    """
    known = {text for text, _ in items}
    verdicts: Dict[str, bool] = {}
    if not raw:
        return verdicts
    try:
        blob = raw.strip()
        # tolerate ```json fences / surrounding prose: grab the first {...} span
        start, end = blob.find("{"), blob.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return verdicts
        data = json.loads(blob[start : end + 1])
    except (ValueError, TypeError) as exc:
        logger.warning("not_a_concept: judge parse failed ({e}); keeping all", e=exc)
        return verdicts
    for v in data.get("verdicts", []) or []:
        if not isinstance(v, dict):
            continue
        text = str(v.get("text", "") or "")
        if text in known:
            verdicts[text] = bool(v.get("is_concept", True))
    return verdicts


__all__ = [
    "classify_deterministic",
    "partition_deterministic",
    "build_judge_prompt",
    "parse_judge_response",
    "JUDGE_SYSTEM_PROMPT",
]
