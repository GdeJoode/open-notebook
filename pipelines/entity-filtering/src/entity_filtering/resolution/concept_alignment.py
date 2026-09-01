"""Concept alignment — is a NEW concept related to the graph, or unprecedented?

``KGResolver`` answers a binary question — does this entity already EXIST in the
graph, or is it NEW? That leaves every unmatched entity floating: a concept that
sits near a whole family of known ones and a genuinely unprecedented one look
identical (both just ``is_new=True``). This module separates them:

* ``RELATED_TO`` — near in meaning to something the graph already holds (the
  common case: a sibling under the same type).
* ``NOVEL``      — nothing comparable was found. This is what N.4d's gap loop
  records as an ontology gap, which is why the ``reason_code`` below has to be
  trustworthy.

Why there is no subsumption here (D-N4-12)
==========================================
Earlier versions of this module also produced ``NARROWER_THAN``/``BROADER_THAN``
and seeded ``is_a`` edges. That is gone, and the reason is worth keeping: three
attempts failed identically — lexical containment, then name matching, then the
candidate's declared type — because **subsumption is a relation between TYPES,
while this table stores MENTIONS.** Nothing in this codebase creates a node
representing an ontology type; every ``entity`` row is written from a text mention
by ``EntityPersistenceService``. So "this entity is narrower than that entity" is
not a claim the data can support, however it is dressed up.

Subsumption now lives where a TYPE enters the system — an extension proposal, an
evolution ``SchemaProposal``, or the curator accepting one — where both sides of
the question are types and the parent slot is currently an unvalidated guess. An
accepted placement is applied as a SCHEMA re-parent, so every existing and future
entity of the re-parented type inherits the new ancestor through
``canonical_bridge`` instead of needing an edge each. See
``docs/tracks/N-evidence-first-extraction/plan.md`` §N.4d.

The tier was REMOVED rather than disabled: there is no story in which it becomes
correct, and dead machinery invites a fourth attempt. If a path ever marks concept
nodes explicitly (the vault-sync importer creates ``entity`` rows from notes and
could flag a concept page), the identification problem would be solved there and
the tier rebuilt against that flag — not resurrected from here.

Where the stage runs (D-N4-4)
=============================
Still AFTER ontology validation and graph centrality. It no longer emits relations
at all, so those constraints are currently moot — but they are what the N.4b
review established and the workflow tests still assert them, so a future producer
cannot silently reintroduce the two blockers that phase fixed: the ontology filter
discards a relation whose endpoint is off-batch, and the graph analyser turns an
unknown endpoint into a phantom node that shifts every PageRank score.

Evidence discipline (D-N4-7)
============================
Every negative verdict states **what was observed**, never what that observation
would imply. "The type query returned no rows" is a fact; "the graph holds no such
concepts" is an inference that can be false — the repository reports a *failed*
query as an empty result, so the two are genuinely indistinguishable from here.
The ``EV_*`` reason codes therefore name observations, and each has exactly one
cause:

* ``EV_NO_REPO``               — no repository was supplied; nothing was queried.
* ``EV_EMPTY_TEXT``            — the entity has no surface form.
* ``EV_NO_TYPE``               — the label did not resolve to a canonical type, so
  no query could be formed.
* ``EV_FETCH_FAILED``          — the fetch call itself raised.
* ``EV_NO_ROWS``               — the query returned zero rows (which does NOT
  prove the graph is empty: see above).
* ``EV_NO_QUERY_VECTOR``       — THIS entity carries no embedding, so nothing
  could be compared. A fact about the input, never about the graph.
* ``EV_NO_CANDIDATE_VECTORS``  — rows exist but none carry an embedding.
* ``EV_INCOMPARABLE_VECTORS``  — vectors exist on both sides but none could be
  compared (dimension mismatch or zero norm).
* ``EV_NONE_CLOSE``            — vectors were genuinely compared and none reached
  the floor. The only code that licenses a claim about similarity.
* ``EV_ERROR``                 — classification raised; nothing was established.

The lexical signal (D-N4-1 / D-N4-9)
====================================
Name containment means *alias* / *part_of* / *named_after* at least as often as
*subtype* — on real Dutch names ``Tweede Kamer der Staten-Generaal`` ⊃ ``Tweede
Kamer`` is an alias. It survives as ``lexical_alias_candidates``: review
candidates in the report, never a verdict and never auto-registered.

Candidates are fetched by CANONICAL type (D-N4-3): ``find_by_type`` filters the
canonical ``entity_type`` column, so passing the rich Track-L label returns ``[]``
by construction.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

# -- verdicts ---------------------------------------------------------------

RELATED_TO = "RELATED_TO"
NOVEL = "NOVEL"

VERDICTS = (RELATED_TO, NOVEL)

# -- methods (provenance for the verdict) -----------------------------------

METHOD_EMBEDDING = "embedding"
METHOD_JUDGE = "llm_judge"
METHOD_NONE = "none"

METHODS = (METHOD_EMBEDDING, METHOD_JUDGE, METHOD_NONE)

# -- evidence codes: each names an OBSERVATION with exactly one cause -------

EV_NO_REPO = "no_repo"
EV_EMPTY_TEXT = "empty_surface_form"
EV_NO_TYPE = "no_resolvable_type"
EV_FETCH_FAILED = "candidate_fetch_failed"
EV_NO_ROWS = "type_query_returned_no_rows"
EV_NO_QUERY_VECTOR = "entity_has_no_embedding"
EV_NO_CANDIDATE_VECTORS = "no_candidate_embeddings"
EV_INCOMPARABLE_VECTORS = "vectors_incomparable"
EV_NONE_CLOSE = "compared_none_close"
EV_ERROR = "classification_error"

REASON_CODES = (
    EV_NO_REPO,
    EV_EMPTY_TEXT,
    EV_NO_TYPE,
    EV_FETCH_FAILED,
    EV_NO_ROWS,
    EV_NO_QUERY_VECTOR,
    EV_NO_CANDIDATE_VECTORS,
    EV_INCOMPARABLE_VECTORS,
    EV_NONE_CLOSE,
    EV_ERROR,
)

# Fixed per-method confidences. A raw cosine is NEVER written here: mixing a
# similarity score with an ontological confidence makes the two incomparable (and
# would let the embedding tier outrank the ontology tier). The cosine is reported
# separately in ``Alignment.similarity`` and in the evidence text.
_CONF_EMBEDDING = 0.55
_CONF_JUDGE = 0.60
_CONF_NOVEL = 0.50


@dataclass(frozen=True)
class Alignment:
    """One concept's placement relative to the existing graph.

    ``reason_code`` is the machine-checkable half of the evidence (see the module
    docstring); ``evidence`` is its human-readable expansion. Both are always
    populated — an operator must be able to audit and reverse any verdict, and
    N.4c filters gap-recording on ``reason_code`` so it must never be a guess.

    ``canonical_type`` is carried so N.4b can stamp ``source_type``/``target_type``
    on a seeded edge (D-N4-5) without re-resolving the ontology.
    """

    verdict: str
    method: str
    confidence: float
    evidence: str
    reason_code: Optional[str] = None
    target_id: Optional[str] = None
    target_name: Optional[str] = None
    canonical_type: Optional[str] = None
    similarity: Optional[float] = None


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase, NFKC-fold, collapse whitespace (matches KGResolver._normalize)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text.lower().strip())


def _candidate_name(candidate: Dict[str, Any]) -> str:
    """A ``find_by_type`` row exposes ``name``; in-batch dicts use ``text``."""
    return str(candidate.get("name") or candidate.get("text", "") or "")


def _props(entity: Dict[str, Any]) -> Dict[str, Any]:
    """Entity properties, tolerating a missing OR explicitly-null ``properties``."""
    value = entity.get("properties")
    return value if isinstance(value, dict) else {}


# ---------------------------------------------------------------------------
# Ontology type resolution (canonical fetch type + rich ancestor chain)
# ---------------------------------------------------------------------------


def resolve_canonical_type(label: str, schemas: Optional[List[Any]]) -> Optional[str]:
    """``(canonical_type, rich_ancestors)`` for an extraction ``label``.

    * ``canonical_type`` — the coarse ``entity_type`` enum value the graph is
      indexed on; this is what ``find_by_type`` must be given (D-N4-3).
    * ``rich_ancestors`` — the declared ancestor TYPE names above ``label``
      (nearest first), the only sound deterministic subsumption evidence (D-N4-2).

    ontology-manager is an OPTIONAL extra of this pipeline, so the import is lazy
    and guarded: without it (or without applied schemas, or for an unresolvable
    label) this returns ``(None, [])`` and the caller degrades honestly.

    The ancestor list is filtered against the RESOLVED type name rather than the
    input ``label``, so a label that matched an ontology *alias* does not leave the
    entity's own type sitting in ``ancestors`` (which would let the tier "prove"
    that a concept is narrower than itself).
    """
    if not label or not schemas:
        return None
    try:
        from ontology_manager.canonical_bridge import (  # type: ignore[import-not-found]
            resolve_ontology_type,
        )
    except Exception as exc:  # noqa: BLE001 — optional extra not installed
        logger.debug("concept_alignment: ontology-manager unavailable ({e})", e=exc)
        return None
    try:
        resolution = resolve_ontology_type(label, schemas)
    except Exception as exc:  # noqa: BLE001 — malformed ontology → degrade
        logger.debug("concept_alignment: type resolution failed ({e})", e=exc)
        return None
    if resolution is None:
        return None
    return resolution.canonical


# ---------------------------------------------------------------------------
# Embedding neighbourhood (similarity only — never direction)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NeighbourProbe:
    """Outcome of comparing one entity against its candidate rows.

    Separating the causes is the whole point: ``nearest is None`` is ambiguous on
    its own (no query vector? no candidate vectors? all incomparable?), and
    collapsing them is how a confident falsehood gets written. Each counter below
    is an observation, and :meth:`reason_code` maps them to exactly one code.
    """

    nearest: Optional[Dict[str, Any]]
    score: float
    compared: int
    skipped_no_vector: int
    skipped_incomparable: int
    has_query_vector: bool

    def reason_code(self) -> Optional[str]:
        """The single code explaining why nothing was compared (None if it was)."""
        if not self.has_query_vector:
            return EV_NO_QUERY_VECTOR
        if self.compared:
            return None
        if self.skipped_incomparable:
            return EV_INCOMPARABLE_VECTORS
        return EV_NO_CANDIDATE_VECTORS


def _cosine(vec1: Sequence[float], vec2: Sequence[float]) -> Optional[float]:
    """Cosine similarity, or ``None`` when the pair is INCOMPARABLE.

    Deliberately not ``EmbeddingResolver._cosine_similarity``: that returns ``0.0``
    for mismatched lengths and zero-norm vectors, an out-of-band sentinel
    indistinguishable from a genuinely orthogonal pair. This repo has documented
    768/1024 embedding-dimension drift, so treating a dimension mismatch as
    "compared, not similar" would report a comparison that never happened.
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return None
    dot = 0.0
    n1 = 0.0
    n2 = 0.0
    for a, b in zip(vec1, vec2):
        dot += a * b
        n1 += a * a
        n2 += b * b
    if n1 <= 0.0 or n2 <= 0.0:
        return None
    return dot / ((n1 ** 0.5) * (n2 ** 0.5))


def probe_neighbours(
    embedding: Optional[Sequence[float]], candidates: List[Dict[str, Any]]
) -> NeighbourProbe:
    """Compare ``embedding`` against every candidate, counting WHY each was skipped.

    The first genuinely comparable candidate is taken unconditionally and only then
    is the score compared — never via a numeric sentinel, which would silently drop
    a boundary case (a 0.0 seed loses to an orthogonal pair, a -1.0 seed to an
    opposed one).
    """
    if not embedding:
        return NeighbourProbe(None, 0.0, 0, 0, 0, has_query_vector=False)
    best: Optional[Dict[str, Any]] = None
    best_score = 0.0
    compared = 0
    no_vector = 0
    incomparable = 0
    for cand in candidates:
        cand_emb = cand.get("embedding") or _props(cand).get("embedding")
        if not cand_emb:
            no_vector += 1
            continue
        try:
            score = _cosine(list(embedding), list(cand_emb))
        except (TypeError, ValueError):  # non-numeric vector contents
            score = None
        if score is None:
            incomparable += 1
            continue
        compared += 1
        if best is None or score > best_score:
            best_score = score
            best = cand
    return NeighbourProbe(
        nearest=best,
        score=best_score if best is not None else 0.0,
        compared=compared,
        skipped_no_vector=no_vector,
        skipped_incomparable=incomparable,
        has_query_vector=True,
    )


# ---------------------------------------------------------------------------
# LLM-judge — pure prompt/parse (the call lives in the orchestrator)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "You are a knowledge-graph ontologist. For each NEW concept you are given a "
    "short list of the nearest existing concepts in the graph. Decide whether the "
    "new concept is RELATED_TO one of them (same domain, meaningfully connected, "
    "but NOT a sub- or super-type) or genuinely NOVEL (nothing listed is close "
    "enough to link). Do NOT answer NARROWER_THAN or BROADER_THAN — subsumption "
    "is decided deterministically elsewhere. When in doubt answer NOVEL: a wrong "
    "link is worse than a missing one."
)

#: A judge item: ``(item_id, text, neighbour names)``. The id — not the text — is
#: the key. Two novel entities can share a surface form ("Den Haag" as a Gemeente
#: and as a Locatie) with DIFFERENT neighbours; keying on text let one ruling
#: satisfy both items and let item A borrow item B's link target.
JudgeItem = Tuple[str, str, List[str]]


def build_judge_prompt(items: Sequence[JudgeItem]) -> str:
    """Render the batched judge prompt; each concept carries its own id."""
    lines = [
        "Classify each NEW concept. Return ONLY this JSON (no prose):",
        "",
        '{"alignments": [{"id": "<id>", "verdict": "RELATED_TO", '
        '"target": "<one neighbour, or null>"}]}',
        "",
        f'verdict must be "{RELATED_TO}" or "{NOVEL}". '
        f'"target" is required for {RELATED_TO} and must be copied verbatim from '
        "THAT id's own neighbour list; use null for NOVEL. Echo the id exactly.",
        "",
        "New concepts:",
    ]
    for item_id, text, neighbours in items:
        shown = ", ".join(f'"{n}"' for n in neighbours) if neighbours else "(none)"
        lines.append(f'- id={item_id}: "{text}" — nearest existing: {shown}')
    return "\n".join(lines)


def parse_judge_response(
    raw: str, items: Sequence[JudgeItem]
) -> Dict[str, Tuple[str, Optional[str]]]:
    """Parse the judge reply into ``{item_id: (verdict, target)}`` for EXPLICIT
    rulings.

    Fences the judge four ways: an unknown id is ignored; only
    ``RELATED_TO``/``NOVEL`` are accepted (subsumption is not its call); a
    ``RELATED_TO`` whose target is not in THAT id's own neighbour list is
    downgraded to ``NOVEL`` (it may neither invent a target nor borrow another
    item's); and anything it stayed silent on is ABSENT, so the caller can both
    default it to ``NOVEL`` and count only what was truly arbitrated.
    Garbage/empty → ``{}``.
    """
    allowed = {item_id: set(neighbours) for item_id, _, neighbours in items}
    out: Dict[str, Tuple[str, Optional[str]]] = {}
    if not raw:
        return out
    try:
        blob = raw.strip()
        start, end = blob.find("{"), blob.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return out
        data = json.loads(blob[start : end + 1])
    except (ValueError, TypeError) as exc:
        logger.warning("concept_alignment: judge parse failed ({e}); all NOVEL", e=exc)
        return out
    for item in data.get("alignments", []) or []:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id", "") or "")
        if item_id not in allowed:
            continue
        verdict = str(item.get("verdict", "") or "").strip().upper()
        if verdict not in (RELATED_TO, NOVEL):
            continue
        target = item.get("target")
        target = str(target) if target else None
        if verdict == RELATED_TO and (not target or target not in allowed[item_id]):
            out[item_id] = (NOVEL, None)  # invented or borrowed target is not a link
            continue
        out[item_id] = (verdict, target if verdict == RELATED_TO else None)
    return out


# ---------------------------------------------------------------------------
# Lexical signal — ALIAS candidates, never subsumption (D-N4-1 / D-N4-9)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AliasCandidate:
    """A long-form/short-form name pair worth reviewing as an alias.

    Attempt 1 read name containment as ``is_a``; on real data it means *alias*,
    *part_of* or *named_after* far more often. The signal is genuinely useful
    though — ``KGResolver``'s fuzzy tier STRUCTURALLY misses these pairs because a
    large length delta tanks Levenshtein — so it is surfaced as a REVIEW candidate.
    It never becomes a relation and is never auto-registered: writing an alias
    merges two identities, which must be an explicit decision.
    """

    text: str
    candidate_name: str
    candidate_id: Optional[str]
    evidence: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "candidate_name": self.candidate_name,
            "candidate_id": self.candidate_id,
            "evidence": self.evidence,
        }


def _tokens(text: str) -> List[str]:
    return [t for t in _normalize(text).split(" ") if t]


def _is_token_subsequence(outer: Sequence[str], inner: Sequence[str]) -> bool:
    """True when ``inner`` is a CONTIGUOUS token run inside ``outer``.

    Token-boundary matching (not substring), so "deal" never matches "dealer".
    """
    n, m = len(outer), len(inner)
    if m == 0 or m > n:
        return False
    return any(list(outer[i : i + m]) == list(inner) for i in range(n - m + 1))


def lexical_alias_candidates(
    text: str,
    candidates: List[Dict[str, Any]],
    *,
    min_inner_tokens: int = 2,
) -> List[AliasCandidate]:
    """Name-containment pairs, in EITHER direction, as alias review candidates.

    ``min_inner_tokens`` keeps a single shared common word from pairing unrelated
    entities. Direction is deliberately NOT interpreted: which of the two is
    canonical is exactly what a reviewer decides.
    """
    tokens = _tokens(text)
    if not tokens:
        return []
    out: List[AliasCandidate] = []
    for cand in candidates:
        cand_name = _candidate_name(cand)
        cand_tokens = _tokens(cand_name)
        if not cand_tokens or cand_tokens == tokens:
            continue
        if (
            len(cand_tokens) >= min_inner_tokens
            and _is_token_subsequence(tokens, cand_tokens)
        ):
            evidence = f"{text!r} contains the existing name {cand_name!r}"
        elif (
            len(tokens) >= min_inner_tokens
            and _is_token_subsequence(cand_tokens, tokens)
        ):
            evidence = f"the existing name {cand_name!r} contains {text!r}"
        else:
            continue
        out.append(
            AliasCandidate(
                text=text,
                candidate_name=cand_name,
                candidate_id=str(cand.get("id", "") or "") or None,
                evidence=evidence + " — review as a possible alias, NOT a subtype",
            )
        )
    return out


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Orchestrator (verdicts + evidence; seeding is the caller's step, above)
# ---------------------------------------------------------------------------


@dataclass
class _Fetch:
    """A candidate fetch outcome: the rows, and whether the call itself succeeded."""

    rows: List[Dict[str, Any]]
    ok: bool


class ConceptAligner:
    """Classify the concepts KG resolution marked ``is_new`` (Track N.4a).

    Runs after ``KGResolver`` and only on entities it flagged ``is_new``.
    Enrichment is NON-destructive (``properties`` only): nothing is merged,
    removed, or re-typed, and N.4a emits no relations at all.

    Args:
        entity_repo: object exposing ``find_by_type`` (the ONLY repo method used —
            no new repository surface, no migration). ``None`` → every concept is
            NOVEL with the honest ``no_repo`` reason.
        schemas: applied ontologies, for the canonical fetch type and the ancestor
            chain. ``None`` → no type resolves, so nothing is queried.
        llm_caller: ``(system, user, model) -> str`` (sync or async) for the judge.
        judge_enabled: master switch for the judge tier (default ON, D4).
        related_floor / match_ceiling: the embedding band bounds.
        max_candidates: rows per type fetch. The underlying query is ``LIMIT n``
            with no ordering, so this is an ARBITRARY sample — every NOVEL verdict
            discloses the cap rather than implying it saw the whole graph.
        min_inner_tokens: precision guard for the alias-candidate signal.
    """

    def __init__(
        self,
        entity_repo: Optional[Any] = None,
        *,
        schemas: Optional[List[Any]] = None,
        llm_caller: Optional[Any] = None,
        model: str = "",
        judge_enabled: bool = True,
        related_floor: float = 0.75,
        match_ceiling: float = 0.90,
        max_candidates: int = 100,
        min_inner_tokens: int = 2,
    ) -> None:
        self._repo = entity_repo
        self._schemas = schemas
        self._llm_caller = llm_caller
        self._model = model
        self._judge_enabled = judge_enabled
        self._related_floor = related_floor
        self._match_ceiling = match_ceiling
        self._max_candidates = max_candidates
        self._min_inner_tokens = min_inner_tokens

    async def align(
        self, entities: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Classify every ``is_new`` entity. Returns ``(entities, report)``."""
        report: Dict[str, Any] = {
            "aligned_count": 0,
            "judged_count": 0,
            "verdict_counts": {v: 0 for v in VERDICTS},
            "method_counts": {m: 0 for m in METHODS},
            "reason_counts": {},
            "alias_candidates": [],
            "candidate_cap": self._max_candidates,
            "capped_type_fetches": [],
        }
        novel = [e for e in entities if _props(e).get("is_new")]
        if not novel:
            return entities, report

        cache: Dict[str, _Fetch] = {}
        decided: List[Tuple[Dict[str, Any], Alignment]] = []
        pending: List[Tuple[Dict[str, Any], Dict[str, Any], float, str, str]] = []

        for entity in novel:
            try:
                alignment, ambiguous, aliases = await self._classify(entity, cache)
            except Exception:
                logger.warning(
                    "ConceptAligner: classification failed for '{}', keeping NOVEL",
                    entity.get("text", "<unknown>"),
                    exc_info=True,
                )
                alignment, ambiguous, aliases = (
                    self._novel(
                        "classification raised before anything could be "
                        "established about the graph",
                        EV_ERROR,
                    ),
                    None,
                    [],
                )
            report["alias_candidates"].extend(a.to_dict() for a in aliases)
            if alignment is not None:
                decided.append((entity, alignment))
            elif ambiguous is not None:
                pending.append((entity, *ambiguous))

        if pending:
            decided.extend(await self._judge(pending, report))

        report["capped_type_fetches"] = sorted(
            t for t, f in cache.items() if len(f.rows) >= self._max_candidates
        )
        for entity, alignment in decided:
            self._enrich(entity, alignment)
            report["aligned_count"] += 1
            report["verdict_counts"][alignment.verdict] += 1
            report["method_counts"][alignment.method] += 1
            if alignment.reason_code:
                report["reason_counts"][alignment.reason_code] = (
                    report["reason_counts"].get(alignment.reason_code, 0) + 1
                )

        logger.info(
            "ConceptAligner: {} aligned ({} related, {} novel), {} judged, "
            "{} alias candidates",
            report["aligned_count"],
            report["verdict_counts"][RELATED_TO],
            report["verdict_counts"][NOVEL],
            report["judged_count"],
            len(report["alias_candidates"]),
        )
        return entities, report

    # -- deterministic tiers -------------------------------------------------

    async def _classify(
        self, entity: Dict[str, Any], cache: Dict[str, _Fetch]
    ) -> Tuple[
        Optional[Alignment],
        Optional[Tuple[Dict[str, Any], float, str, str]],
        List[AliasCandidate],
    ]:
        """Tiers 1-2. ``(alignment, ambiguous_band, alias_candidates)`` — exactly
        one of the first two is set. The band tuple is
        ``(nearest, score, canonical_type, sampling_note)``."""
        text = str(entity.get("text", "") or "").strip()
        if not text:
            return self._novel("the entity has no surface form", EV_EMPTY_TEXT), None, []

        label = str(entity.get("label", "") or "")
        canonical = resolve_canonical_type(label, self._schemas)

        if self._repo is None:
            return (
                self._novel(
                    "no repository was supplied — the graph was never queried",
                    EV_NO_REPO,
                    # C4: the type resolved fine; only the repo was absent. Keep it.
                    canonical_type=canonical,
                ),
                None,
                [],
            )
        if not canonical:
            return (
                self._novel(
                    f"label {label!r} did not resolve to a canonical type, so no "
                    "query could be formed",
                    EV_NO_TYPE,
                ),
                None,
                [],
            )

        fetch = await self._candidates(canonical, cache)
        if not fetch.ok:
            return (
                self._novel(
                    f"the candidate fetch for canonical type {canonical!r} raised; "
                    "nothing was established about the graph",
                    EV_FETCH_FAILED,
                    canonical_type=canonical,
                ),
                None,
                [],
            )
        candidates = fetch.rows
        if not candidates:
            return (
                self._novel(
                    f"the type query for {canonical!r} returned no rows (note: the "
                    "repository reports a failed query as an empty result, so this "
                    "does not by itself prove the graph holds none)",
                    EV_NO_ROWS,
                    canonical_type=canonical,
                ),
                None,
                [],
            )

        aliases = lexical_alias_candidates(
            text, candidates, min_inner_tokens=self._min_inner_tokens
        )

        probe = probe_neighbours(_props(entity).get("embedding"), candidates)
        sampled = self._sample_note(canonical, len(candidates))
        reason = probe.reason_code()
        if reason is not None:
            return (
                self._novel(
                    # C2: a "nothing was compared" outcome is just as much a claim
                    # about a capped sample as a distance is — disclose it here too.
                    self._probe_evidence(reason, probe, canonical, len(candidates))
                    + (sampled if reason != EV_NO_QUERY_VECTOR else ""),
                    reason,
                    canonical_type=canonical,
                ),
                None,
                aliases,
            )

        nearest, score = probe.nearest, probe.score
        if nearest is None:
            # C5: unreachable by construction (reason_code() is None ⇒ compared > 0
            # ⇒ nearest is set), but expressed as a branch rather than an `assert`,
            # which vanishes under `python -O` — a control guard must not be
            # optimised away on a path that writes evidence.
            return (
                self._novel(
                    "the neighbour probe reported a comparison but produced no "
                    "candidate; nothing was established",
                    EV_ERROR,
                    canonical_type=canonical,
                ),
                None,
                aliases,
            )
        if score >= self._match_ceiling:
            return (
                Alignment(
                    verdict=RELATED_TO,
                    method=METHOD_EMBEDDING,
                    confidence=_CONF_EMBEDDING,
                    similarity=round(score, 6),
                    evidence=(
                        f"cosine {score:.3f} ≥ {self._match_ceiling} to "
                        f"{_candidate_name(nearest)!r}{sampled}"
                    ),
                    target_id=str(nearest.get("id", "") or "") or None,
                    target_name=_candidate_name(nearest),
                    canonical_type=canonical,
                ),
                None,
                aliases,
            )
        if score < self._related_floor:
            return (
                Alignment(
                    verdict=NOVEL,
                    method=METHOD_NONE,
                    confidence=_CONF_NOVEL,
                    similarity=round(score, 6),
                    evidence=(
                        f"nearest of {probe.compared} compared concepts is "
                        f"{_candidate_name(nearest)!r} at cosine {score:.3f} < "
                        f"{self._related_floor}{sampled}"
                    ),
                    reason_code=EV_NONE_CLOSE,
                    canonical_type=canonical,
                ),
                None,
                aliases,
            )
        return None, (nearest, score, canonical, sampled), aliases

    def _sample_note(self, canonical: str, fetched: int) -> str:
        """Disclose that the candidate set is a capped, unordered sample (M4)."""
        if fetched < self._max_candidates:
            return ""
        return (
            f" — NOTE this compared an arbitrary sample of {self._max_candidates} "
            f"{canonical!r} rows (the query is LIMIT-capped and unordered), so the "
            "graph may hold closer concepts that were not fetched"
        )

    def _probe_evidence(
        self, reason: str, probe: NeighbourProbe, canonical: str, fetched: int
    ) -> str:
        """Human-readable expansion for a probe that compared nothing."""
        if reason == EV_NO_QUERY_VECTOR:
            return (
                "this entity carries no embedding, so nothing could be compared — "
                "this says nothing about the graph"
            )
        if reason == EV_INCOMPARABLE_VECTORS:
            return (
                f"{probe.skipped_incomparable} of {fetched} {canonical!r} rows "
                "carried an embedding that could not be compared (dimension "
                "mismatch or zero norm); no comparison was performed"
            )
        return (
            f"{fetched} {canonical!r} rows were fetched but none carried an "
            "embedding, so no comparison was performed"
        )

    async def _candidates(
        self, canonical: str, cache: Dict[str, _Fetch]
    ) -> _Fetch:
        """Fetch by CANONICAL type (D-N4-3), cached per batch.

        A raised fetch is cached as ``ok=False`` so the batch stays consistent, and
        the caller reports ``EV_FETCH_FAILED`` rather than claiming the graph is
        empty.
        """
        if canonical in cache:
            return cache[canonical]
        try:
            found = await self._repo.find_by_type(
                canonical, limit=self._max_candidates
            )
            fetch = _Fetch(rows=list(found or []), ok=True)
        except Exception:
            logger.debug(
                "ConceptAligner: candidate fetch failed for type '{}'",
                canonical,
                exc_info=True,
            )
            fetch = _Fetch(rows=[], ok=False)
        cache[canonical] = fetch
        return fetch

    # -- judge ---------------------------------------------------------------

    async def _judge(
        self,
        pending: List[Tuple[Dict[str, Any], Dict[str, Any], float, str, str]],
        report: Dict[str, Any],
    ) -> List[Tuple[Dict[str, Any], Alignment]]:
        """One batched call over the ambiguous band; silence/failure → NOVEL.

        Items are keyed by INDEX, not surface form: two novel entities can share a
        name with different neighbours, and a text key would let one ruling satisfy
        both and let one borrow the other's link target.
        """
        items: List[JudgeItem] = [
            (str(i), str(e.get("text", "") or ""), [_candidate_name(near)])
            for i, (e, near, _, _, _) in enumerate(pending)
        ]
        verdicts: Dict[str, Tuple[str, Optional[str]]] = {}
        if self._judge_enabled and self._llm_caller is not None:
            try:
                raw = self._llm_caller(
                    JUDGE_SYSTEM_PROMPT, build_judge_prompt(items), self._model
                )
                if hasattr(raw, "__await__"):
                    raw = await raw
                verdicts = parse_judge_response(str(raw), items)
            except Exception as exc:  # noqa: BLE001 — judge is best-effort
                logger.warning(
                    "ConceptAligner: judge failed ({e}); ambiguous band → NOVEL", e=exc
                )
                verdicts = {}
        report["judged_count"] = len(verdicts)

        out: List[Tuple[Dict[str, Any], Alignment]] = []
        for i, (entity, nearest, score, canonical, sampled) in enumerate(pending):
            item_id = str(i)
            ruled = item_id in verdicts  # only THIS item's own ruling counts
            verdict, target = verdicts.get(item_id, (NOVEL, None))
            if ruled and verdict == RELATED_TO:
                out.append((entity, Alignment(
                    verdict=RELATED_TO,
                    method=METHOD_JUDGE,
                    confidence=_CONF_JUDGE,
                    similarity=round(score, 6),
                    evidence=(
                        f"judge linked it to {target!r} (nearest cosine "
                        f"{score:.3f}){sampled}"
                    ),
                    target_id=str(nearest.get("id", "") or "") or None,
                    target_name=target or _candidate_name(nearest),
                    canonical_type=canonical,
                )))
            elif ruled:
                out.append((entity, Alignment(
                    verdict=NOVEL,
                    method=METHOD_JUDGE,
                    confidence=_CONF_NOVEL,
                    similarity=round(score, 6),
                    evidence=(
                        f"judge found no link (nearest cosine {score:.3f} to "
                        f"{_candidate_name(nearest)!r}){sampled}"
                    ),
                    reason_code=EV_NONE_CLOSE,
                    canonical_type=canonical,
                )))
            else:
                # NOT judged: no caller, judge disabled, or silent on THIS item.
                out.append((entity, Alignment(
                    verdict=NOVEL,
                    method=METHOD_NONE,
                    confidence=_CONF_NOVEL,
                    similarity=round(score, 6),
                    evidence=(
                        f"nearest cosine {score:.3f} to "
                        f"{_candidate_name(nearest)!r} is inconclusive and no judge "
                        f"verdict was obtained for this concept{sampled}"
                    ),
                    reason_code=EV_NONE_CLOSE,
                    canonical_type=canonical,
                )))
        return out

    # -- enrichment ----------------------------------------------------------

    @staticmethod
    def _novel(
        evidence: str, reason_code: str, *, canonical_type: Optional[str] = None
    ) -> Alignment:
        return Alignment(
            verdict=NOVEL,
            method=METHOD_NONE,
            confidence=_CONF_NOVEL,
            evidence=evidence,
            reason_code=reason_code,
            canonical_type=canonical_type,
        )

    @staticmethod
    def _enrich(entity: Dict[str, Any], alignment: Alignment) -> None:
        """Write the verdict + evidence into properties (non-destructive)."""
        props = entity.setdefault("properties", {})
        props["concept_alignment"] = alignment.verdict
        props["alignment_method"] = alignment.method
        props["alignment_confidence"] = alignment.confidence
        props["alignment_evidence"] = alignment.evidence
        props["alignment_reason_code"] = alignment.reason_code
        props["alignment_target_id"] = alignment.target_id
        props["alignment_target_name"] = alignment.target_name
        props["alignment_canonical_type"] = alignment.canonical_type
        props["alignment_similarity"] = alignment.similarity


__all__ = [
    "Alignment",
    "AliasCandidate",
    "ConceptAligner",
    "JudgeItem",
    "NeighbourProbe",
    "RELATED_TO",
    "NOVEL",
    "VERDICTS",
    "METHOD_EMBEDDING",
    "METHOD_JUDGE",
    "METHOD_NONE",
    "METHODS",
    "EV_NO_REPO",
    "EV_EMPTY_TEXT",
    "EV_NO_TYPE",
    "EV_FETCH_FAILED",
    "EV_NO_ROWS",
    "EV_NO_QUERY_VECTOR",
    "EV_NO_CANDIDATE_VECTORS",
    "EV_INCOMPARABLE_VECTORS",
    "EV_NONE_CLOSE",
    "EV_ERROR",
    "REASON_CODES",
    "resolve_canonical_type",
    "probe_neighbours",
    "lexical_alias_candidates",
    "build_judge_prompt",
    "parse_judge_response",
    "JUDGE_SYSTEM_PROMPT",
]
