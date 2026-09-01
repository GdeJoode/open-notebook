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

# Exact (whole-string, lowercased) UI / navigation / boilerplate labels — never a
# domain concept. EN + NL. Tunable via ``extra_reject_exact``.
_REJECT_EXACT = frozenset(
    {
        # navigation / UI
        "click here", "read more", "learn more", "see more", "show more",
        "back to top", "next", "previous", "prev", "home", "menu", "search",
        "login", "log in", "logout", "log out", "sign up", "sign in",
        "download", "downloads", "print", "share", "subscribe", "submit",
        "contact", "contact us", "about", "about us", "overview", "index",
        "table of contents", "continue reading", "view all", "see all",
        "lees meer", "meer informatie", "inhoudsopgave", "vorige", "volgende",
        "downloaden", "afdrukken", "delen", "zoeken", "inloggen", "aanmelden",
        # boilerplate / legal
        "all rights reserved", "copyright", "confidential", "draft",
        "terms of service", "terms and conditions", "privacy policy",
        "cookie policy", "disclaimer", "colophon", "colofon", "voorwoord",
        "alle rechten voorbehouden", "auteursrecht", "privacybeleid",
        # generic table / form furniture
        "total", "subtotal", "totaal", "subtotaal", "sum", "average", "n/a",
        "tbd", "tba", "yes", "no", "ja", "nee", "true", "false",
        "date", "datum", "name", "naam", "amount", "bedrag", "number", "nummer",
        "description", "omschrijving", "type", "category", "categorie", "status",
        "page", "pagina", "figure", "figuur", "table", "tabel", "chapter",
        "hoofdstuk", "section", "sectie", "appendix", "bijlage", "note", "noot",
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

    # -- high-precision REJECT (obvious non-concepts) -----------------------
    if len(t) < _MIN_LEN or _PUNCT_ONLY_RE.match(t):
        return True
    if _NUMERIC_RE.match(t):  # pure number / date / percentage / amount
        return True
    if _REF_RE.match(t):  # "Figure 3", "Hoofdstuk 4"
        return True
    if low in _REJECT_EXACT:
        return True
    if extra_reject_exact and low in extra_reject_exact:
        return True

    # -- fast ACCEPT (clearly a concept) ------------------------------------
    if label.strip().lower() not in _GENERIC_LABELS:
        # The LLM committed to a specific schema type — trust it (do not spend a
        # judge call second-guessing an aggressively-mapped entity).
        return False
    if _is_titlecase_proper(t):
        # Multi-word proper name even under a generic label → a real entity.
        return False

    # -- AMBIGUOUS: generic label + a plain word/phrase ---------------------
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
    """Parse the judge response into ``{text: is_concept}``.

    Robust to markdown fences and junk. Any candidate the judge did not rule on
    defaults to ``True`` (KEEP — never drop on a missing/garbled verdict).
    """
    verdicts: Dict[str, bool] = {text: True for text, _ in items}
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
        if text in verdicts:
            verdicts[text] = bool(v.get("is_concept", True))
    return verdicts


__all__ = [
    "classify_deterministic",
    "partition_deterministic",
    "build_judge_prompt",
    "parse_judge_response",
    "JUDGE_SYSTEM_PROMPT",
]
