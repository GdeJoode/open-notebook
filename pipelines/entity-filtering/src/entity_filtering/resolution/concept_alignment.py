"""Concept-level alignment taxonomy — verdicts (Track N.4a).

``KGResolver`` answers a binary question — does this entity already EXIST in the
graph, or is it NEW? That leaves every unmatched entity floating: a concept that
sits near a whole family of known ones and a genuinely unprecedented one look
identical (both just ``is_new=True``).

This module classifies a NOVEL concept RELATIVE to the existing graph:

* ``NARROWER_THAN`` — a specialisation of an existing concept.
* ``BROADER_THAN``  — a generalisation of one.
* ``RELATED_TO``    — near in meaning, not a subsumption (the common case: a
  sibling under the same type).
* ``NOVEL``         — nothing comparable in the graph. In N.4c this becomes an
  ontology GAP for ``OntologyEvolutionAgent``.

Scope of N.4a: **verdicts + evidence only**. No relation seeding and no workflow
stage — placement and seeding are N.4b, the gap loop and DI/env reachability are
N.4c. See ``docs/tracks/N-evidence-first-extraction/plan.md`` §N.4 (v2).

What attempt 1 got wrong, and what this module does instead
==========================================================

* **Lexical containment is NOT subsumption** (D-N4-1). "A contains B" means
  *alias* / *part_of* / *named_after* at least as often as *subtype* — on real
  Dutch names ``Tweede Kamer der Staten-Generaal`` ⊃ ``Tweede Kamer`` is an alias,
  ``Den Haag Zuidwest`` ⊃ ``Den Haag`` is part-of. Worse, ``KGResolver``'s fuzzy
  tier *rejects* long/short alias pairs (the length delta tanks Levenshtein), so
  aliases are exactly what reaches this module. There is therefore NO lexical tier
  here; what that signal should become instead is an open plan decision.
* **Subsumption comes only from the ontology** (D-N4-2): the label's own declared
  ``parent_type`` chain via ``canonical_bridge``. Embeddings inform RELATED/NOVEL
  ONLY — cosine measures similarity, never direction.
* **Candidates are fetched by CANONICAL type** (D-N4-3). ``find_by_type`` filters
  the canonical ``entity_type`` column; passing the rich Track-L label
  (``Gemeente``, ``RegioDeal``) returns ``[]`` by construction, which silently
  turned the whole stage into a no-op in attempt 1.
* **Evidence must be falsifiable** (D-N4-7): the module distinguishes "no
  candidates were fetched" (a fact about the query) from "candidates were fetched
  and none were close" (a fact about the graph). It never asserts the second when
  only the first is known.

The tiers, deterministic-first (the Track-N house style — the LLM proposes, the
system decides):

1. **Type-chain subsumption** (pure, via ``canonical_bridge``). NOTE this fires
   only when an ancestor TYPE is materialised as a graph node — a candidate row
   from ``find_by_type`` carries ``id``/``name``/``embedding``/``weight`` and no
   type column, so the ancestor is matched on NAME. For an ordinary instance whose
   ancestor type is not a node, the chain restates the entity's own type and adds
   nothing, so no verdict is emitted.
2. **Embedding band**: at/above the match ceiling similarity alone implies
   ``RELATED_TO``; below the related floor nothing is close → ``NOVEL``
   deterministically. Only the band between them is genuinely ambiguous.
3. **Batched LLM-judge** (default ON, D4) arbitrates exactly that band:
   ``RELATED_TO`` (to which neighbour) vs ``NOVEL``. It may NOT answer subsumption,
   and a ``RELATED_TO`` whose target is not in that concept's OWN neighbour list is
   downgraded to ``NOVEL`` — the judge cannot invent a link. No judge / failure /
   silence → ``NOVEL``, with the embedding evidence retained.

Enrichment is NON-destructive: verdicts are written to the entity's ``properties``
and nothing is merged, removed, or re-typed.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

from entity_filtering.resolution.embedding_resolver import EmbeddingResolver

# -- verdicts ---------------------------------------------------------------

NARROWER_THAN = "NARROWER_THAN"
BROADER_THAN = "BROADER_THAN"
RELATED_TO = "RELATED_TO"
NOVEL = "NOVEL"

VERDICTS = (NARROWER_THAN, BROADER_THAN, RELATED_TO, NOVEL)

# -- methods (provenance for the verdict) -----------------------------------

METHOD_TYPE_CHAIN = "type_chain"
METHOD_EMBEDDING = "embedding"
METHOD_JUDGE = "llm_judge"
METHOD_NONE = "none"

METHODS = (METHOD_TYPE_CHAIN, METHOD_EMBEDDING, METHOD_JUDGE, METHOD_NONE)

# -- evidence codes (D-N4-7: falsifiable, machine-checkable reasons) ---------

#: no repository was supplied — nothing about the graph is known
EV_NO_REPO = "no_repo"
#: the label could not be resolved to a canonical type, so nothing was queried
EV_NO_TYPE = "no_resolvable_type"
#: a query ran and returned no rows of a comparable type
EV_NO_CANDIDATES = "no_candidates_fetched"
#: candidates exist but none carried an embedding to compare against
EV_NO_VECTORS = "no_comparable_vectors"
#: candidates were compared and none were close enough
EV_NONE_CLOSE = "candidates_fetched_none_close"

_CONF_TYPE_CHAIN = 0.80
_CONF_JUDGE = 0.60
_CONF_NOVEL = 0.50


@dataclass(frozen=True)
class Alignment:
    """One concept's placement relative to the existing graph.

    ``reason_code`` is the machine-checkable half of the evidence (one of the
    ``EV_*`` constants for the negative paths); ``evidence`` is its human-readable
    expansion. Both are always populated — an operator must be able to audit and
    reverse any verdict.
    """

    verdict: str
    method: str
    confidence: float
    evidence: str
    reason_code: Optional[str] = None
    target_id: Optional[str] = None
    target_name: Optional[str] = None


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
    """Entity properties, tolerating a missing OR explicitly-null ``properties``.

    A DB/LLM row can carry ``properties: None``; ``.get("properties", {})`` returns
    that ``None`` and every downstream ``.get`` would raise.
    """
    value = entity.get("properties")
    return value if isinstance(value, dict) else {}


# ---------------------------------------------------------------------------
# Ontology type resolution (canonical fetch type + rich ancestor chain)
# ---------------------------------------------------------------------------


def resolve_types(
    label: str, schemas: Optional[List[Any]]
) -> Tuple[Optional[str], List[str]]:
    """``(canonical_type, rich_ancestors)`` for an extraction ``label``.

    * ``canonical_type`` — the coarse ``entity_type`` enum value the graph is
      actually indexed on; this is what ``find_by_type`` must be given (D-N4-3).
    * ``rich_ancestors`` — the declared ancestor TYPE names above ``label``
      (nearest first), the only sound deterministic subsumption evidence (D-N4-2).

    ontology-manager is an OPTIONAL extra of this pipeline, so the import is lazy
    and guarded: without it (or without applied schemas, or for an unresolvable
    label) this returns ``(None, [])`` and the caller degrades honestly rather than
    querying with a label the column does not hold.
    """
    if not label or not schemas:
        return None, []
    try:
        from ontology_manager.canonical_bridge import (  # type: ignore[import-not-found]
            resolve_ontology_type,
        )
    except Exception as exc:  # noqa: BLE001 — optional extra not installed
        logger.debug("concept_alignment: ontology-manager unavailable ({e})", e=exc)
        return None, []
    try:
        resolution = resolve_ontology_type(label, schemas)
    except Exception as exc:  # noqa: BLE001 — malformed ontology → degrade
        logger.debug("concept_alignment: type resolution failed ({e})", e=exc)
        return None, []
    if resolution is None:
        return None, []
    ancestors = [
        t for t in resolution.type_tags if _normalize(t) != _normalize(label)
    ]
    return resolution.canonical, ancestors


def type_chain_subsumption(
    ancestors: Sequence[str], candidates: List[Dict[str, Any]]
) -> Optional[Alignment]:
    """NARROWER_THAN when an ancestor TYPE is MATERIALISED as a graph node.

    A ``find_by_type`` row has no type column, so the ancestor is matched on the
    candidate's NAME. When no ancestor is materialised this returns ``None``
    DELIBERATELY rather than a type-level verdict: for an ordinary instance the
    chain merely restates the entity's own declared type, which the label already
    carries — asserting it as an alignment would be an empty claim.
    """
    if not ancestors or not candidates:
        return None
    by_name = {_normalize(_candidate_name(c)): c for c in candidates}
    for ancestor in ancestors:  # nearest ancestor first
        cand = by_name.get(_normalize(ancestor))
        if cand is None:
            continue
        return Alignment(
            verdict=NARROWER_THAN,
            method=METHOD_TYPE_CHAIN,
            confidence=_CONF_TYPE_CHAIN,
            evidence=(
                f"the ontology declares this type under {ancestor!r}, which exists "
                f"in the graph as {_candidate_name(cand)!r}"
            ),
            target_id=str(cand.get("id", "") or "") or None,
            target_name=_candidate_name(cand),
        )
    return None


# ---------------------------------------------------------------------------
# Embedding neighbourhood (similarity only — never direction)
# ---------------------------------------------------------------------------


def nearest_by_embedding(
    embedding: Optional[Sequence[float]], candidates: List[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], float]:
    """``(nearest candidate, cosine)``; ``(None, 0.0)`` when nothing is comparable.

    The first comparable candidate is taken unconditionally, and only then is the
    score compared — NOT via a numeric sentinel. Any sentinel inside the cosine
    range silently drops a boundary case (0.0 seeds lose to an orthogonal pair,
    -1.0 seeds lose to an opposed pair), and the caller would then report "no
    comparable vectors" — a falsehood about the graph (D-N4-7). ``best is None``
    must mean exactly one thing: no candidate carried a vector.
    """
    if not embedding:
        return None, 0.0
    best: Optional[Dict[str, Any]] = None
    best_score = 0.0
    for cand in candidates:
        cand_emb = cand.get("embedding") or _props(cand).get("embedding")
        if not cand_emb:
            continue
        score = EmbeddingResolver._cosine_similarity(list(embedding), list(cand_emb))
        if best is None or score > best_score:
            best_score = score
            best = cand
    return (best, best_score) if best is not None else (None, 0.0)


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


def build_judge_prompt(items: List[Tuple[str, List[str]]]) -> str:
    """Render the batched judge prompt for ``(novel_text, [neighbour names])``."""
    lines = [
        "Classify each NEW concept. Return ONLY this JSON (no prose):",
        "",
        '{"alignments": [{"text": "<verbatim>", "verdict": "RELATED_TO", '
        '"target": "<one neighbour, or null>"}]}',
        "",
        f'verdict must be "{RELATED_TO}" or "{NOVEL}". '
        f'"target" is required for {RELATED_TO} and must be copied verbatim from '
        "that concept's own neighbour list; use null for NOVEL.",
        "",
        "New concepts:",
    ]
    for text, neighbours in items:
        shown = ", ".join(f'"{n}"' for n in neighbours) if neighbours else "(none)"
        lines.append(f'- "{text}" — nearest existing: {shown}')
    return "\n".join(lines)


def parse_judge_response(
    raw: str, items: List[Tuple[str, List[str]]]
) -> Dict[str, Tuple[str, Optional[str]]]:
    """Parse the judge reply into ``{text: (verdict, target)}`` for EXPLICIT rulings.

    Fences the judge three ways: only ``RELATED_TO``/``NOVEL`` are accepted
    (subsumption is not its call); a ``RELATED_TO`` whose target is not in that
    concept's OWN neighbour list is downgraded to ``NOVEL`` (it may not invent a
    link target, nor borrow another item's); and anything it stayed silent on is
    ABSENT from the result so the caller can both default it to ``NOVEL`` and count
    only what was truly arbitrated. Garbage/empty → ``{}``.
    """
    allowed = {text: set(neighbours) for text, neighbours in items}
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
        text = str(item.get("text", "") or "")
        if text not in allowed:
            continue
        verdict = str(item.get("verdict", "") or "").strip().upper()
        if verdict not in (RELATED_TO, NOVEL):
            continue
        target = item.get("target")
        target = str(target) if target else None
        if verdict == RELATED_TO and (not target or target not in allowed[text]):
            out[text] = (NOVEL, None)  # fabricated / borrowed target is not a link
            continue
        out[text] = (verdict, target if verdict == RELATED_TO else None)
    return out


# ---------------------------------------------------------------------------
# Lexical signal — ALIAS candidates, never subsumption (D-N4-1, resolved)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AliasCandidate:
    """A long-form/short-form name pair worth reviewing as an alias.

    Attempt 1 read name containment as ``is_a``; on real data it means *alias*,
    *part_of* or *named_after* far more often. The signal itself is genuinely
    useful though — ``KGResolver``'s fuzzy tier STRUCTURALLY misses these pairs
    because a large length delta tanks Levenshtein similarity — so it is surfaced
    here as a REVIEW candidate. It never becomes a relation and is never
    auto-registered: writing an alias merges two identities in the graph, which is
    an explicit decision, not a side effect of classification.
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
    """True when ``inner`` appears as a CONTIGUOUS token run inside ``outer``.

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
    entities ("Gemeente Den Haag" vs "Gemeente Den Bosch" yields nothing — neither
    contains the other). Direction is deliberately NOT interpreted: which of the
    two is canonical is exactly what a reviewer decides.
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
# Orchestrator (N.4a: verdicts + evidence only — no seeding, no workflow stage)
# ---------------------------------------------------------------------------


class ConceptAligner:
    """Classify the concepts KG resolution marked ``is_new`` (Track N.4a).

    Runs after ``KGResolver`` and only on entities it flagged ``is_new`` — an
    entity that already matched a KG node needs no placement. Enrichment is
    NON-destructive (properties only): nothing is merged, removed, or re-typed, and
    N.4a emits no relations at all.

    Args:
        entity_repo: object exposing ``find_by_type`` (the ONLY repo method used —
            no new repository surface, no migration). ``None`` → every concept is
            NOVEL with the honest ``no_repo`` reason.
        schemas: applied ontologies, for the canonical fetch type and the ancestor
            chain. ``None`` → no type resolves, so nothing is queried.
        llm_caller: ``(system, user, model) -> str`` (sync or async) for the judge.
        judge_enabled: master switch for the judge tier (default ON, D4).
        related_floor / match_ceiling: the embedding band bounds.
        max_candidates: upper bound per type fetch.
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
        }
        novel = [e for e in entities if _props(e).get("is_new")]
        if not novel:
            return entities, report

        cache: Dict[str, List[Dict[str, Any]]] = {}
        decided: List[Tuple[Dict[str, Any], Alignment]] = []
        pending: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = []

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
                    self._novel("classification raised", EV_NO_CANDIDATES),
                    None,
                    [],
                )
            report["alias_candidates"].extend(a.to_dict() for a in aliases)
            if alignment is not None:
                decided.append((entity, alignment))
            elif ambiguous is not None:
                pending.append((entity, ambiguous[0], ambiguous[1]))

        if pending:
            decided.extend(await self._judge(pending, report))

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
            "ConceptAligner: {} aligned ({} narrower, {} broader, {} related, "
            "{} novel), {} judged, {} alias candidates",
            report["aligned_count"],
            report["verdict_counts"][NARROWER_THAN],
            report["verdict_counts"][BROADER_THAN],
            report["verdict_counts"][RELATED_TO],
            report["verdict_counts"][NOVEL],
            report["judged_count"],
            len(report["alias_candidates"]),
        )
        return entities, report

    # -- deterministic tiers -------------------------------------------------

    async def _classify(
        self, entity: Dict[str, Any], cache: Dict[str, List[Dict[str, Any]]]
    ) -> Tuple[
        Optional[Alignment],
        Optional[Tuple[Dict[str, Any], float]],
        List[AliasCandidate],
    ]:
        """Tiers 1-2. ``(alignment, ambiguous_band, alias_candidates)`` — exactly
        one of the first two is set."""
        text = str(entity.get("text", "") or "").strip()
        if not text:
            return self._novel("empty surface form", EV_NO_TYPE), None, []

        label = str(entity.get("label", "") or "")
        canonical, ancestors = resolve_types(label, self._schemas)

        if self._repo is None:
            return (
                self._novel(
                    "no repository available — the graph was never queried",
                    EV_NO_REPO,
                ),
                None,
                [],
            )
        if not canonical:
            return (
                self._novel(
                    f"label {label!r} does not resolve to a canonical type, so no "
                    "comparable concepts could be queried",
                    EV_NO_TYPE,
                ),
                None,
                [],
            )

        candidates = await self._candidates(canonical, cache)
        if not candidates:
            return (
                self._novel(
                    f"the graph holds no concepts of canonical type {canonical!r}",
                    EV_NO_CANDIDATES,
                ),
                None,
                [],
            )

        aliases = lexical_alias_candidates(
            text, candidates, min_inner_tokens=self._min_inner_tokens
        )

        hit = type_chain_subsumption(ancestors, candidates)
        if hit is not None:
            return hit, None, aliases

        embedding = _props(entity).get("embedding")
        nearest, score = nearest_by_embedding(embedding, candidates)
        if nearest is None:
            return (
                self._novel(
                    f"{len(candidates)} concepts of type {canonical!r} exist but "
                    "none could be compared (no embedding)",
                    EV_NO_VECTORS,
                ),
                None,
                aliases,
            )
        if score >= self._match_ceiling:
            return (
                Alignment(
                    verdict=RELATED_TO,
                    method=METHOD_EMBEDDING,
                    confidence=round(score, 6),
                    evidence=(
                        f"cosine {score:.3f} ≥ {self._match_ceiling} to "
                        f"{_candidate_name(nearest)!r}"
                    ),
                    target_id=str(nearest.get("id", "") or "") or None,
                    target_name=_candidate_name(nearest),
                ),
                None,
                aliases,
            )
        if score < self._related_floor:
            return (
                self._novel(
                    f"nearest of {len(candidates)} compared concepts is "
                    f"{_candidate_name(nearest)!r} at cosine {score:.3f} < "
                    f"{self._related_floor}",
                    EV_NONE_CLOSE,
                ),
                None,
                aliases,
            )
        return None, (nearest, score), aliases

    async def _candidates(
        self, canonical: str, cache: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Fetch by CANONICAL type (D-N4-3), cached per batch."""
        if canonical in cache:
            return cache[canonical]
        try:
            found = await self._repo.find_by_type(
                canonical, limit=self._max_candidates
            )
        except Exception:
            logger.debug(
                "ConceptAligner: candidate fetch failed for type '{}'",
                canonical,
                exc_info=True,
            )
            found = []
        cache[canonical] = list(found or [])
        return cache[canonical]

    # -- judge ---------------------------------------------------------------

    async def _judge(
        self,
        pending: List[Tuple[Dict[str, Any], Dict[str, Any], float]],
        report: Dict[str, Any],
    ) -> List[Tuple[Dict[str, Any], Alignment]]:
        """One batched call over the ambiguous band; silence/failure → NOVEL."""
        items = [
            (str(e.get("text", "") or ""), [_candidate_name(near)])
            for e, near, _ in pending
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
        for entity, nearest, score in pending:
            text = str(entity.get("text", "") or "")
            ruled = text in verdicts  # only THIS item's own ruling counts
            verdict, target = verdicts.get(text, (NOVEL, None))
            if ruled and verdict == RELATED_TO:
                out.append((entity, Alignment(
                    verdict=RELATED_TO,
                    method=METHOD_JUDGE,
                    confidence=_CONF_JUDGE,
                    evidence=(
                        f"judge linked it to {target!r} (nearest cosine {score:.3f})"
                    ),
                    target_id=str(nearest.get("id", "") or "") or None,
                    target_name=target or _candidate_name(nearest),
                )))
            elif ruled:
                out.append((entity, Alignment(
                    verdict=NOVEL,
                    method=METHOD_JUDGE,
                    confidence=_CONF_NOVEL,
                    evidence=(
                        f"judge found no link (nearest cosine {score:.3f} to "
                        f"{_candidate_name(nearest)!r})"
                    ),
                    reason_code=EV_NONE_CLOSE,
                )))
            else:
                # NOT judged: no caller, judge disabled, or it stayed silent on
                # THIS item. Never claim a judge verdict we did not get.
                out.append((entity, Alignment(
                    verdict=NOVEL,
                    method=METHOD_NONE,
                    confidence=_CONF_NOVEL,
                    evidence=(
                        f"nearest cosine {score:.3f} to "
                        f"{_candidate_name(nearest)!r} is inconclusive and no judge "
                        "verdict was obtained"
                    ),
                    reason_code=EV_NONE_CLOSE,
                )))
        return out

    # -- enrichment ----------------------------------------------------------

    @staticmethod
    def _novel(evidence: str, reason_code: str) -> Alignment:
        return Alignment(
            verdict=NOVEL,
            method=METHOD_NONE,
            confidence=_CONF_NOVEL,
            evidence=evidence,
            reason_code=reason_code,
        )

    @staticmethod
    def _enrich(entity: Dict[str, Any], alignment: Alignment) -> None:
        """Write the verdict + evidence into properties (non-destructive)."""
        props = entity.setdefault("properties", {})
        if not isinstance(props, dict):  # properties: None on the incoming row
            props = {}
            entity["properties"] = props
        props["concept_alignment"] = alignment.verdict
        props["alignment_method"] = alignment.method
        props["alignment_confidence"] = alignment.confidence
        props["alignment_evidence"] = alignment.evidence
        props["alignment_reason_code"] = alignment.reason_code
        props["alignment_target_id"] = alignment.target_id
        props["alignment_target_name"] = alignment.target_name


__all__ = [
    "Alignment",
    "AliasCandidate",
    "ConceptAligner",
    "lexical_alias_candidates",
    "NARROWER_THAN",
    "BROADER_THAN",
    "RELATED_TO",
    "NOVEL",
    "VERDICTS",
    "METHOD_TYPE_CHAIN",
    "METHOD_EMBEDDING",
    "METHOD_JUDGE",
    "METHOD_NONE",
    "METHODS",
    "EV_NO_REPO",
    "EV_NO_TYPE",
    "EV_NO_CANDIDATES",
    "EV_NO_VECTORS",
    "EV_NONE_CLOSE",
    "resolve_types",
    "type_chain_subsumption",
    "nearest_by_embedding",
    "build_judge_prompt",
    "parse_judge_response",
    "JUDGE_SYSTEM_PROMPT",
]
