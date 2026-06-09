"""Orphan-connector module (B.5a).

Find entities of a given source that ended up with no incoming/outgoing
relations after the main extraction + filtering pipeline, propose new
relations via chunk-co-occurrence with non-orphan partners, and ask the
injected LLM to confirm or deny each proposal.

Algorithm (3 stages, async)
===========================

1. :func:`find_orphans` — query the entity repository for entities of
   a given source that participate in zero relations. The repository is
   injected via :class:`OrphanEntityRepoProtocol` so unit tests can
   supply an in-memory mock (mirrors the
   ``EntityRepositoryProtocol`` pattern used by the KG resolver).

2. :func:`propose_connections` — for each chunk that contains an
   orphan, every other entity in that chunk is a candidate partner.
   The output is a list of :class:`OrphanProposal` records carrying
   the orphan, the partner and the evidence chunk.

3. :func:`confirm_connections` — call the LLM once per proposal with
   :data:`orphan_prompts.ORPHAN_CONFIRM_SYSTEM_PROMPT`. The LLM
   returns ``{relation_type: str | null, confidence: float}``;
   confirmed proposals (non-null ``relation_type``) become
   :class:`shared.models.extraction.ExtractedRelation` records.

Decisions / non-goals
=====================

- **No clustering.** myKG's original ``orphan_connector.py`` pre-
  clusters orphans by embedding similarity before proposing partners.
  The B.5a plan defers that step to V2 (revisit if precision is low).
- **Per-call token budget = 1500.** Per Q-B-2, the heuristic stays
  ``len(text) // 4``; the cap is **lower** than Pass-1/Pass-2's 2400
  because orphan-connector fires many times per source — see the
  module docstring on :mod:`orphan_prompts`.
- **Malformed-JSON degrades gracefully.** Mirrors the B.1c / B.1d
  policy: log a WARNING, drop the proposal, never raise from content
  parsing.
- **DI for LLM caller.** Same callable signature as Pass-1 / Pass-2 so
  the production wiring in B.1f reuses the existing seam without
  ceremony.
- **Name normalisation via ``shared.utils.name_normalizer``** so the
  orphan-and-partner pair stays comparable across upstream variants
  (e.g. ``"Alice"`` vs ``"alice "``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from loguru import logger
from shared.models.extraction import ExtractedEntity, ExtractedRelation
from shared.utils.name_normalizer import normalize_entity_name

from entity_filtering.resolution.orphan_prompts import (
    ORPHAN_CONFIRM_SYSTEM_PROMPT,
    build_orphan_confirm_prompt,
)

# Per-call token budget — lower than Pass-2's 2400 because orphan-
# connector fires many times per source. See ``orphan_prompts``
# docstring for the budget rationale.
TOKEN_BUDGET_TARGET = 1500

# Trim malformed-LLM-output excerpts to this many characters in the
# WARNING log. Same value as Pass-2 — long enough to diagnose, short
# enough not to swamp the log line.
_MALFORMED_LOG_EXCERPT_CHARS = 240


# ---------------------------------------------------------------------------
# Public type aliases (mirrors Pass-2)
# ---------------------------------------------------------------------------


SyncLLMCaller = Callable[[str, str, str], str]
AsyncLLMCaller = Callable[[str, str, str], Awaitable[str]]
LLMCaller = Union[SyncLLMCaller, AsyncLLMCaller]


# ---------------------------------------------------------------------------
# Repository protocol — DI seam for the entity store
# ---------------------------------------------------------------------------


@runtime_checkable
class OrphanEntityRepoProtocol(Protocol):
    """Minimum repository surface the orphan-connector relies on.

    The concrete implementation lives in
    ``surrealdb-service.repositories.entity.EntityRepository`` (see
    :meth:`EntityRepository.list_orphans_for_source` added in this
    phase). Keeping the contract narrow here lets unit tests pass an
    in-memory mock without standing up SurrealDB.
    """

    async def list_orphans_for_source(
        self, source_id: str
    ) -> List[Dict[str, Any]]:
        """Return entity rows for the source that participate in zero
        relations (no incoming and no outgoing edges).

        Each row carries at minimum the ``id`` and ``canonical_name``
        fields; richer rows are passed through unchanged so future
        consumers (e.g. B.5b prune-lifecycle) can read additional
        columns without a second round-trip.
        """
        ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OrphanTokenBudgetExceeded(RuntimeError):
    """Raised when an orphan-confirm prompt exceeds the 1500-token cap.

    Fires *before* the LLM call so the caller sees an immediate, cheap
    failure rather than a downstream truncation or LLM over-spend.
    Mirrors :class:`pass2_typed_extraction.Pass2TokenBudgetExceeded` so
    the workflow can handle both with the same ``except`` clause.
    """

    def __init__(self, estimated: int, budget: int = TOKEN_BUDGET_TARGET):
        super().__init__(
            f"Orphan-confirm prompt exceeds token budget: "
            f"~{estimated} tokens estimated (budget: {budget})"
        )
        self.estimated = estimated
        self.budget = budget


# ---------------------------------------------------------------------------
# Proposal payload (DTO)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrphanProposal:
    """A candidate (orphan, partner, evidence-chunk) triple.

    ``orphan`` is the surface form of the entity with no relations,
    ``candidate_partner`` is a co-occurring entity from the same chunk
    (orphan or non-orphan), ``evidence_chunk_id`` is the chunk where
    they co-occur, and ``evidence_text`` is the chunk body that
    ``confirm_connections`` will feed back to the LLM.
    """

    orphan: str
    candidate_partner: str
    evidence_chunk_id: Optional[str]
    evidence_text: str


# ---------------------------------------------------------------------------
# Helpers (parsing + LLM transport)
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Q-B-2 heuristic: ``len(text) // 4`` chars-per-token."""
    return len(text) // 4


_BRACED_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """Unwrap ``` ```json ... ``` ``` blocks.

    Mirrors the Pass-1/Pass-2 helper so the cascade of graceful-
    degradation steps is identical across modules.
    """
    s = text.strip()
    if "```json" in s:
        s = s.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in s:
        s = s.split("```", 1)[1].split("```", 1)[0].strip()
    return s


def _salvage_json_object(text: str) -> str:
    """Extract the first ``{...}`` block from arbitrary prose.

    Returns the input unchanged when no object is found so the caller's
    ``json.loads`` triggers the graceful-degradation path with a
    well-formed JSONDecodeError.
    """
    match = _BRACED_OBJECT_RE.search(text)
    return match.group(0) if match else text


def _clamp_confidence(value: Any) -> float:
    """Coerce confidence to a float in ``[0.0, 1.0]``.

    Same shape-coercion as Pass-2: ``None`` and unparseable strings
    fall back to ``0.0``; values > 1.5 are assumed to be percentages
    and divided by 100; final result is clamped to the
    ``ExtractedRelation.confidence`` ``ge=0.0, le=1.0`` window.
    """
    if value is None:
        return 0.0
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if f > 1.5:
        f = f / 100.0
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


def _log_malformed(reason: str, response: str) -> None:
    """Structured WARNING log for a malformed-LLM-output case."""
    excerpt = response[:_MALFORMED_LOG_EXCERPT_CHARS]
    suffix = "..." if len(response) > _MALFORMED_LOG_EXCERPT_CHARS else ""
    logger.warning(
        "Orphan-connector returned empty result: {reason} | "
        "excerpt={excerpt!r}{suffix}",
        reason=reason,
        excerpt=excerpt,
        suffix=suffix,
    )


def _parse_confirm_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse one LLM confirmation response.

    Returns ``None`` when the response is empty, when it isn't valid
    JSON, when the top-level value isn't an object, or when ``relation_type``
    is ``None`` / empty. Otherwise returns a dict with keys
    ``relation_type`` (string, upper-snake) and ``confidence`` (float
    in ``[0, 1]``).

    Malformed responses **never raise** — they degrade gracefully to
    ``None`` with a WARNING log, matching the B.1c/B.1d policy.
    """
    if not response or not response.strip():
        _log_malformed("empty LLM response", response or "")
        return None

    text = _strip_code_fence(response)
    text = _salvage_json_object(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        _log_malformed(f"invalid JSON ({e})", response)
        return None

    if not isinstance(data, dict):
        _log_malformed(
            f"top-level JSON is {type(data).__name__}, expected object",
            response,
        )
        return None

    rel_type = data.get("relation_type")
    confidence = _clamp_confidence(data.get("confidence"))

    if rel_type is None or (isinstance(rel_type, str) and not rel_type.strip()):
        # Negative confirmation — not malformed, just a "no relation"
        # answer. Returning ``None`` causes the caller to skip the
        # proposal without an additional warning log.
        return None

    if not isinstance(rel_type, str):
        _log_malformed(
            f"relation_type is {type(rel_type).__name__}, expected string",
            response,
        )
        return None

    return {"relation_type": rel_type.strip(), "confidence": confidence}


async def _invoke_llm(
    caller: LLMCaller,
    system_prompt: str,
    user_prompt: str,
    model: str,
) -> str:
    """Call the LLM, supporting both sync and async callers."""
    try:
        raw = caller(system_prompt, user_prompt, model)
        if hasattr(raw, "__await__"):
            raw = await raw  # type: ignore[misc]
    except Exception as e:
        logger.exception(f"Orphan-connector LLM call failed: {e}")
        # Re-raise — transport-level failures are NOT silently swallowed.
        # The malformed-JSON path is reserved for content parsing.
        raise
    return str(raw)


# ---------------------------------------------------------------------------
# Stage 1: find_orphans
# ---------------------------------------------------------------------------


async def find_orphans(
    source_id: str, entity_repo: OrphanEntityRepoProtocol
) -> List[Dict[str, Any]]:
    """Return the entities of *source_id* that have zero relations.

    Args:
        source_id: Record ID of the source (e.g. ``"source:abc123"``)
            whose orphans we want.
        entity_repo: Repository implementing
            :class:`OrphanEntityRepoProtocol`.

    Returns:
        A list of entity-row dicts. Empty when the source has no orphans
        (or no entities at all). Always shaped — never ``None``.

    Note:
        The plan signature uses ``List[Entity]`` semantically; the repo
        returns rows as dicts (the canonical surface form across the
        rest of the resolution pipeline) so we return them unchanged.
        Callers that need a typed ``Entity`` can call
        ``Entity(**row)`` themselves.
    """
    if not source_id:
        logger.warning("find_orphans called with empty source_id; returning []")
        return []

    rows = await entity_repo.list_orphans_for_source(source_id)
    logger.info(
        "find_orphans: source={source_id} orphan_count={n}",
        source_id=source_id,
        n=len(rows),
    )
    return list(rows)


# ---------------------------------------------------------------------------
# Stage 2: propose_connections
# ---------------------------------------------------------------------------


def _entity_text(entity: Any) -> str:
    """Return the surface form of an entity in either dict or model form."""
    if isinstance(entity, ExtractedEntity):
        return entity.text
    if isinstance(entity, dict):
        # Accept both extraction-shape (``text``) and persistence-shape
        # (``canonical_name``) so callers don't have to pre-translate.
        return entity.get("text") or entity.get("canonical_name") or ""
    return str(entity)


def _chunk_text(chunk: Dict[str, Any]) -> str:
    """Read the chunk body, accepting common field-name variants."""
    return str(chunk.get("text") or chunk.get("body") or "")


def _chunk_id(chunk: Dict[str, Any]) -> Optional[str]:
    """Read the chunk identifier (best-effort across naming variants)."""
    return chunk.get("id") or chunk.get("chunk_id")


def _chunk_entities(chunk: Dict[str, Any]) -> List[str]:
    """Return surface forms of entities pre-attached to the chunk.

    The chunk shape supported by the orphan-connector is::

        {"id": "c-1", "text": "...", "entities": [<text|dict|ExtractedEntity>]}

    Where the list items can be raw strings, plain dicts, or
    ``ExtractedEntity`` instances. This loose shape mirrors the rest of
    the resolution pipeline so the workflow can hand chunks straight in.
    """
    raw = chunk.get("entities") or []
    out: List[str] = []
    for ent in raw:
        text = _entity_text(ent)
        if text:
            out.append(text)
    return out


async def propose_connections(
    orphans: Iterable[Any],
    chunks: Iterable[Dict[str, Any]],
    *,
    max_proposals_per_orphan: int = 3,
) -> List[OrphanProposal]:
    """Generate (orphan, partner, evidence-chunk) proposals.

    Heuristic: for each chunk that contains an orphan, every other
    entity in the same chunk is a candidate partner. Partners can be
    other orphans (we may pair two orphans together) or non-orphans.

    The function is ``async`` only for symmetry with the rest of the
    module — no I/O is performed.

    Args:
        orphans: Iterable of orphan entities (dicts, ``ExtractedEntity``,
            or raw strings — the surface form is extracted via
            :func:`_entity_text`).
        chunks: Iterable of chunk dicts. Each must carry ``text`` and
            an ``entities`` list (see :func:`_chunk_entities` for the
            shape).
        max_proposals_per_orphan: Cap on proposals per orphan. Used to
            bound the LLM-call count on busy sources. Defaults to 3 —
            tunable via :class:`OrphanConnectorConfig`.

    Returns:
        A list of :class:`OrphanProposal` records. Empty when no orphan
        co-occurs with another entity.
    """
    orphan_list = list(orphans)
    if not orphan_list:
        return []

    orphan_norm_to_text: Dict[str, str] = {}
    for ent in orphan_list:
        text = _entity_text(ent)
        norm = normalize_entity_name(text)
        if norm and norm not in orphan_norm_to_text:
            orphan_norm_to_text[norm] = text

    if not orphan_norm_to_text:
        return []

    # Track proposals per orphan so we can enforce the cap. The seen
    # set is keyed on (orphan_norm, partner_norm) so we don't emit
    # duplicate proposals for the same pair from multiple chunks —
    # the LLM-confirm step only needs one evidence chunk per pair.
    seen_pairs: set[tuple[str, str]] = set()
    proposals_per_orphan: Dict[str, int] = {n: 0 for n in orphan_norm_to_text}
    proposals: List[OrphanProposal] = []

    for chunk in chunks:
        body = _chunk_text(chunk)
        chunk_id = _chunk_id(chunk)
        chunk_entity_texts = _chunk_entities(chunk)
        if not chunk_entity_texts:
            continue

        # Map each chunk entity to its normalised form so orphan lookup
        # is whitespace/case-insensitive.
        chunk_normed: List[tuple[str, str]] = []
        for text in chunk_entity_texts:
            norm = normalize_entity_name(text)
            if norm:
                chunk_normed.append((text, norm))

        # Which orphans appear in this chunk?
        orphans_here = [
            (text, norm)
            for (text, norm) in chunk_normed
            if norm in orphan_norm_to_text
        ]
        if not orphans_here:
            continue

        for orphan_text, orphan_norm in orphans_here:
            if proposals_per_orphan[orphan_norm] >= max_proposals_per_orphan:
                continue
            for partner_text, partner_norm in chunk_normed:
                if partner_norm == orphan_norm:
                    continue
                pair_key = (orphan_norm, partner_norm)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                proposals.append(
                    OrphanProposal(
                        orphan=orphan_text,
                        candidate_partner=partner_text,
                        evidence_chunk_id=chunk_id,
                        evidence_text=body,
                    )
                )
                proposals_per_orphan[orphan_norm] += 1
                if (
                    proposals_per_orphan[orphan_norm]
                    >= max_proposals_per_orphan
                ):
                    break

    logger.info(
        "propose_connections: orphans={no} chunks={nc} proposals={np}",
        no=len(orphan_norm_to_text),
        nc=sum(1 for _ in chunks) if isinstance(chunks, list) else -1,
        np=len(proposals),
    )
    return proposals


# ---------------------------------------------------------------------------
# Stage 3: confirm_connections
# ---------------------------------------------------------------------------


async def confirm_connections(
    proposals: Iterable[OrphanProposal],
    llm_caller: LLMCaller,
    *,
    model: str = "default",
    min_confidence: float = 0.6,
) -> List[ExtractedRelation]:
    """Ask the LLM to confirm each proposal.

    For every proposal:

    1. Build the orphan-confirm prompt
       (:func:`orphan_prompts.build_orphan_confirm_prompt`).
    2. Estimate the prompt tokens; raise
       :class:`OrphanTokenBudgetExceeded` if the prompt exceeds
       :data:`TOKEN_BUDGET_TARGET`. The check fires **before** the LLM
       call so over-budget evidence is a loud, cheap failure.
    3. Invoke the LLM and parse the response with
       :func:`_parse_confirm_response`. Malformed responses degrade to
       a dropped proposal + WARNING log — they do not interrupt the
       run.
    4. Drop confirmations with ``confidence < min_confidence``.
    5. Emit one :class:`ExtractedRelation` per surviving confirmation.

    Args:
        proposals: Iterable of :class:`OrphanProposal` records (the
            output of :func:`propose_connections`).
        llm_caller: Sync or async LLM caller — same callable shape as
            Pass-1 / Pass-2 so the same DI seam works.
        model: Model identifier passed to the LLM caller.
        min_confidence: Discard confirmations below this confidence.
            Default 0.6 matches the conservative threshold set by
            :class:`OrphanConnectorConfig`.

    Returns:
        A list of confirmed :class:`ExtractedRelation` records. The
        relations carry ``source_chunk_id`` (so downstream provenance
        stays intact) and ``extraction_method='orphan_connector'``
        baked into ``properties`` for telemetry.

    Raises:
        OrphanTokenBudgetExceeded: Some prompt exceeded the budget.
            The exception fires before the LLM call so callers see an
            immediate, deterministic failure.
    """
    relations: List[ExtractedRelation] = []
    confirmed_count = 0
    denied_count = 0
    malformed_count = 0
    low_conf_count = 0

    for proposal in proposals:
        user_prompt = build_orphan_confirm_prompt(
            proposal.orphan,
            proposal.candidate_partner,
            proposal.evidence_text,
        )
        # Token budget guard — count the system prompt + the user
        # prompt together so the estimate matches what the LLM actually
        # receives. Mirrors the Pass-2 envelope (assembled prompt).
        estimated = _estimate_tokens(
            ORPHAN_CONFIRM_SYSTEM_PROMPT + "\n" + user_prompt
        )
        if estimated > TOKEN_BUDGET_TARGET:
            logger.warning(
                "Orphan-confirm token budget exceeded for orphan="
                "{orphan!r} partner={partner!r} chunk={chunk_id}: "
                "~{est} tokens > {budget}.",
                orphan=proposal.orphan,
                partner=proposal.candidate_partner,
                chunk_id=proposal.evidence_chunk_id,
                est=estimated,
                budget=TOKEN_BUDGET_TARGET,
            )
            raise OrphanTokenBudgetExceeded(estimated, TOKEN_BUDGET_TARGET)

        raw = await _invoke_llm(
            llm_caller, ORPHAN_CONFIRM_SYSTEM_PROMPT, user_prompt, model
        )
        parsed = _parse_confirm_response(raw)
        if parsed is None:
            # Either a negative answer (relation_type=null) or a
            # malformed response. Distinguish only for the running
            # tally — both outcomes drop the proposal.
            if raw and raw.strip() and "null" in raw:
                denied_count += 1
            else:
                malformed_count += 1
            continue

        if parsed["confidence"] < min_confidence:
            low_conf_count += 1
            logger.debug(
                "Orphan-confirm dropped low-confidence relation: "
                "{orphan!r} -[{rt}]-> {partner!r} conf={c}",
                orphan=proposal.orphan,
                rt=parsed["relation_type"],
                partner=proposal.candidate_partner,
                c=parsed["confidence"],
            )
            continue

        relations.append(
            ExtractedRelation(
                source_entity=proposal.orphan,
                target_entity=proposal.candidate_partner,
                relation_type=parsed["relation_type"],
                properties={"extraction_method": "orphan_connector"},
                confidence=parsed["confidence"],
                source_chunk_id=proposal.evidence_chunk_id,
            )
        )
        confirmed_count += 1

    logger.info(
        "confirm_connections: confirmed={c} denied={d} malformed={m} "
        "low_conf={lc} kept={k}",
        c=confirmed_count,
        d=denied_count,
        m=malformed_count,
        lc=low_conf_count,
        k=len(relations),
    )
    return relations


# ---------------------------------------------------------------------------
# Convenience: run() — single entry point for the workflow integration
# ---------------------------------------------------------------------------


async def run(
    source_id: str,
    chunks: Iterable[Dict[str, Any]],
    entity_repo: OrphanEntityRepoProtocol,
    llm_caller: LLMCaller,
    *,
    model: str = "default",
    max_proposals_per_orphan: int = 3,
    min_confidence: float = 0.6,
) -> List[ExtractedRelation]:
    """One-shot orchestration: find → propose → confirm.

    Convenience wrapper for callers that want a single coroutine to
    spit out the confirmed relations. The workflow uses this; tests
    typically call the three stage functions directly so they can
    assert per-stage shape.

    Returns:
        List of confirmed :class:`ExtractedRelation` records. Empty
        when the source has no orphans, no chunks, or every proposal
        fails confirmation.
    """
    orphans = await find_orphans(source_id, entity_repo)
    if not orphans:
        return []

    chunk_list = list(chunks)
    proposals = await propose_connections(
        orphans,
        chunk_list,
        max_proposals_per_orphan=max_proposals_per_orphan,
    )
    if not proposals:
        return []

    return await confirm_connections(
        proposals,
        llm_caller,
        model=model,
        min_confidence=min_confidence,
    )
