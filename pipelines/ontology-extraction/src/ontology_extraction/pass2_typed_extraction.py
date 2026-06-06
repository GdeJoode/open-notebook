"""Pass-2 typed extraction module (B.1d, single-schema).

Given a list of chunks, a target ontology, the list of accepted
extension types (from B.3c, conceptually), and an injected LLM
caller, produce an ``ExtractionResult`` populated with typed
entities and relations.

This is the module where **extraction-time confidence first
appears** in the pipeline: every entity and every relation in the
output carries a non-null ``confidence`` populated by the LLM. The
B4 "confidence everywhere" requirement is enforced by:

1. Asking the LLM for confidence in the system prompt
   (``PASS2_SYSTEM_PROMPT``) and the user prompt rules.
2. Defaulting missing/non-numeric confidences to ``0.0`` at parse
   time so a downstream filter still sees a number.
3. Tests asserting per-element that ``confidence is not None``.

Scope
=====
**Single-schema only.** The multi-schema orchestrator that runs
Pass-2 against several schemas and merges results lands in B.1e
(``multi_schema_orchestrator``). This module deliberately stays
schema-agnostic and stateless so the orchestrator can call it
multiple times per source without any setup cost.

Integration
===========
The LLM caller is injected the same way Pass-1 injects: a callable
``(system_prompt, user_prompt, model) -> str`` or its async sibling.
B.1f wires the production caller via DI inside
``EntityExtractionService``. The default lazy ``_default_llm_caller``
exists for ad-hoc CLI use only — production must not exercise that
path (it logs a WARNING canary).

Note about the existing ``LLMExtractor``: that class currently
imports ``llm_manager.manager`` which is the pre-rename API. The
``LLMManager → ModelManager`` migration is tracked separately and is
B.1f scope. This module does **not** touch ``llm-manager`` directly —
all LLM access flows through the injected caller — so the existing
TODO is unaffected by B.1d.

Telemetry (Q-B-6: always on)
============================
Every call emits structured INFO logs at start + end:

- ``pass2_chunk_start`` per chunk (chunk_id, estimated_tokens)
- ``pass2_chunk_complete`` per chunk (entity_count, relation_count)
- ``pass2_run_complete`` at the end (total_entities, total_relations)

These are intentionally cheap: pure loguru output. When the metrics
table lands (B.4), the values surface as counters/histograms. Until
then, the logs are the ledger.

Malformed-LLM-output policy
===========================
Parsing failures degrade gracefully — return an empty
``ExtractionResult`` for the affected chunk with a ``parse_error``
metadata entry plus a WARNING log. The orchestrator (B.1e) decides
whether to retry or skip. Transport errors (network / auth / timeout)
escape as ``Pass2ParseError`` so callers can branch on the
budget-exceeded path explicitly.
"""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from loguru import logger
from ontology_manager.schema import Ontology
from shared.models.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)

from ontology_extraction.prompts.pass2 import (
    PASS2_SYSTEM_PROMPT,
    build_pass2_prompt,
)

# Coarse token estimator: ``len(text) // 4`` matches Q-B-2's
# heuristic. Same envelope as Pass-1: a 20% safety margin against the
# 3000-token plan cap → enforce ≤ 2400 estimated tokens.
TOKEN_BUDGET_TARGET = 2400

# How many characters of a malformed LLM response to include in the
# WARNING log. Long enough to diagnose, short enough not to swamp the
# log line.
_MALFORMED_LOG_EXCERPT_CHARS = 240

# Async or sync LLM caller signatures. Tests inject plain sync
# callables, production injects async (matches the Pass-1 contract).
SyncLLMCaller = Callable[[str, str, str], str]
AsyncLLMCaller = Callable[[str, str, str], Awaitable[str]]
LLMCaller = Union[SyncLLMCaller, AsyncLLMCaller]


class Pass2TokenBudgetExceeded(RuntimeError):
    """Raised when a chunk's assembled prompt exceeds the Pass-2 budget.

    Fires *before* the LLM call so the caller sees an immediate,
    cheap failure rather than a downstream truncation or LLM
    over-spend. Mirrors ``pass1_schema_validation.TokenBudgetExceeded``
    so the orchestrator can handle both with the same except branch.
    """

    def __init__(self, estimated: int, budget: int = TOKEN_BUDGET_TARGET):
        super().__init__(
            f"Pass-2 prompt exceeds token budget: "
            f"~{estimated} tokens estimated (budget: {budget})"
        )
        self.estimated = estimated
        self.budget = budget


class Pass2ParseError(RuntimeError):
    """Raised when the LLM caller itself fails (network / auth / timeout).

    Distinct from ``Pass2TokenBudgetExceeded``. Content-parsing
    failures (malformed JSON, missing fields) do NOT raise this —
    they produce an empty ``ExtractionResult`` for the chunk with a
    WARNING log so the orchestrator can decide between retry / skip
    / surface.
    """


def _estimate_tokens(text: str) -> int:
    """Coarse token count via ``len(text) // 4`` (Q-B-2 heuristic).

    Module-level so tests can patch it for boundary checks without
    faking the entire prompt builder.
    """
    return len(text) // 4


_BRACED_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """Unwrap ```json ... ``` or plain ``` ... ``` blocks.

    Mirrors the Pass-1 helper so behaviour stays consistent.
    """
    s = text.strip()
    if "```json" in s:
        s = s.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in s:
        s = s.split("```", 1)[1].split("```", 1)[0].strip()
    return s


def _salvage_json_object(text: str) -> str:
    """Extract the first ``{...}`` block from arbitrary prose.

    Returns the original text unchanged when no object is found, so
    the caller's ``json.loads`` triggers the graceful-degradation
    path.
    """
    match = _BRACED_OBJECT_RE.search(text)
    return match.group(0) if match else text


def _clamp_confidence(value: Any) -> float:
    """Coerce a value to a float in ``[0.0, 1.0]``.

    The LLM occasionally returns ``"high"`` or ``87`` (percent) for
    confidence. We:

    - Treat ``None`` / unparseable strings as ``0.0`` (defensive — the
      B4 contract is satisfied as long as the field is present).
    - Divide by 100 when the value is > 1.5 (so ``87`` becomes
      ``0.87``).
    - Clamp the final value into ``[0, 1]`` since
      ``ExtractedEntity.confidence`` has a ``ge=0.0, le=1.0`` field
      constraint that would otherwise raise.
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
    """Structured WARNING log for a malformed-LLM-output case.

    The orchestrator (B.1e) and the future telemetry layer (B.4)
    look for these to count parse failures. Body is truncated to
    keep log lines bounded.
    """
    excerpt = response[:_MALFORMED_LOG_EXCERPT_CHARS]
    suffix = "..." if len(response) > _MALFORMED_LOG_EXCERPT_CHARS else ""
    logger.warning(
        "Pass-2 returned empty result: {reason} | excerpt={excerpt!r}{suffix}",
        reason=reason,
        excerpt=excerpt,
        suffix=suffix,
    )


def _parse_chunk_response(response: str) -> ExtractionResult:
    """Parse one chunk's LLM response into an ``ExtractionResult``.

    Per AC #4: malformed JSON / missing fields / non-object payloads
    degrade gracefully — return an empty ``ExtractionResult`` and emit
    a WARNING log. Never raise from content-parsing failures.

    Per AC #3: every entity and every relation in the returned result
    has a non-null ``confidence`` (defaults to ``0.0`` when the LLM
    omits it, so the field invariant holds even on partial output).
    """
    if not response or not response.strip():
        _log_malformed("empty LLM response", response or "")
        return ExtractionResult(metadata={"parse_error": "empty_response"})

    text = _strip_code_fence(response)
    text = _salvage_json_object(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        _log_malformed(f"invalid JSON ({e})", response)
        return ExtractionResult(metadata={"parse_error": "invalid_json"})

    if not isinstance(data, dict):
        _log_malformed(
            f"top-level JSON is {type(data).__name__}, expected object",
            response,
        )
        return ExtractionResult(metadata={"parse_error": "non_object_json"})

    entities: List[ExtractedEntity] = []
    raw_entities = data.get("entities", []) or []
    if not isinstance(raw_entities, list):
        raw_entities = []
    for e in raw_entities:
        if not isinstance(e, dict):
            # Defensive: silently drop non-object entries rather than
            # crashing the whole chunk.
            continue
        text_val = e.get("text") or e.get("name") or ""
        label_val = e.get("label") or e.get("entity_type") or "UNKNOWN"
        if not text_val:
            continue
        try:
            entities.append(
                ExtractedEntity(
                    text=str(text_val),
                    label=str(label_val),
                    properties=e.get("properties") or {},
                    confidence=_clamp_confidence(e.get("confidence")),
                )
            )
        except Exception as exc:  # pragma: no cover - belt-and-braces
            logger.warning(
                "Pass-2: dropping malformed entity payload: {exc} | {payload!r}",
                exc=exc,
                payload=e,
            )

    relations: List[ExtractedRelation] = []
    raw_relations = data.get("relations", []) or []
    if not isinstance(raw_relations, list):
        raw_relations = []
    for r in raw_relations:
        if not isinstance(r, dict):
            continue
        # Accept both Pass-2 canonical shape (source/target/type) and
        # the legacy LLMExtractor shape (subject/object/predicate) so
        # an LLM trained on the older prompt still parses.
        src = r.get("source") or r.get("subject") or ""
        tgt = r.get("target") or r.get("object") or ""
        rel_type = r.get("type") or r.get("predicate") or "RELATED_TO"
        if not src or not tgt:
            continue
        try:
            relations.append(
                ExtractedRelation(
                    source_entity=str(src),
                    target_entity=str(tgt),
                    relation_type=str(rel_type),
                    properties=r.get("properties") or {},
                    confidence=_clamp_confidence(r.get("confidence")),
                )
            )
        except Exception as exc:  # pragma: no cover - belt-and-braces
            logger.warning(
                "Pass-2: dropping malformed relation payload: {exc} | {payload!r}",
                exc=exc,
                payload=r,
            )

    return ExtractionResult(entities=entities, relations=relations)


async def _invoke_llm(
    caller: LLMCaller,
    system_prompt: str,
    user_prompt: str,
    model: str,
) -> str:
    """Call the LLM, supporting both sync and async callers.

    Wraps transport-level failures in ``Pass2ParseError`` so callers
    can branch on a single exception class. Mirrors the Pass-1
    contract.
    """
    try:
        raw = caller(system_prompt, user_prompt, model)
        # Coroutines are awaitable; strings are not — hasattr is
        # sufficient and avoids an ``inspect`` import.
        if hasattr(raw, "__await__"):
            raw = await raw  # type: ignore[misc]
    except Exception as e:
        logger.exception(f"Pass-2 LLM call failed: {e}")
        raise Pass2ParseError(f"LLM caller failed: {e}") from e
    return str(raw)


def _default_llm_caller() -> LLMCaller:
    """Lazy-import the production LLM caller.

    Kept as a function (not module-level) so unit tests with injected
    callers never trigger the import. Mirrors the lazy-import pattern
    in Pass-1 + ``LLMExtractor``.
    """
    # B.1f wires ``EntityExtractionService`` and supplies an LLM
    # caller via DI. This default is a safety net for ad-hoc CLI use
    # only. The ``llm_manager.manager`` symbol stays the same as
    # Pass-1's lazy import; the upcoming ``ModelManager`` rename is
    # B.1f scope and out of B.1d's blast radius.
    import llm_manager.manager  # noqa: F401  type: ignore[import-not-found]

    async def _call(system_prompt: str, user_prompt: str, model: str) -> str:
        # Stub: returns empty JSON and logs WARNING. Production must
        # not exercise this path — the WARNING is the canary.
        logger.warning(
            "Pass-2 typed-extraction using lazy default LLM caller — "
            "B.1f integration not yet wired; returning empty JSON."
        )
        return "{}"

    return _call


async def run_pass2(
    chunks: List[Dict[str, Any]],
    ontology: Ontology,
    accepted_extensions: Optional[List[Dict[str, Any]]] = None,
    llm_caller: Optional[LLMCaller] = None,
    model: str = "default",
) -> ExtractionResult:
    """Run Pass-2 typed extraction across a batch of chunks.

    For each chunk:

    1. Build a Pass-2 user prompt (ontology + accepted extensions +
       chunk text).
    2. Estimate the prompt token count; raise
       ``Pass2TokenBudgetExceeded`` if it exceeds the budget. (The
       caller is responsible for chunk sizing — failure here is
       loud and stops the whole run, not just this chunk, because a
       systematic oversize means downstream chunks will also fail.)
    3. Call the injected LLM caller and parse the response.
    4. Aggregate entities and relations into a single
       ``ExtractionResult``.

    Args:
        chunks: List of dicts. Each must carry ``text`` (the chunk
            content); ``id`` is optional and used to tag entities/
            relations with ``source_chunk_id``.
        ontology: Target ontology (``ontology_manager.schema.Ontology``).
        accepted_extensions: Optional list of extension dicts approved
            by the curator (B.3c). Empty or ``None`` triggers the
            back-compat path that mirrors the current ``LLMExtractor``
            behaviour for callers that don't yet wire extensions.
        llm_caller: Injected sync or async caller. ``None`` falls back
            to the lazy default (CLI / dev only; logs a canary).
        model: Model identifier passed to the LLM caller.

    Returns:
        A combined ``ExtractionResult``. Per AC #3 every entity and
        every relation has a non-null ``confidence``. Per AC #4
        chunks whose LLM response cannot be parsed contribute zero
        elements but do not interrupt the run.

    Raises:
        Pass2TokenBudgetExceeded: Prompt exceeds the 2400-token cap
            for some chunk.
        Pass2ParseError: LLM caller itself failed (transport).
    """
    extensions = accepted_extensions or []
    caller = llm_caller if llm_caller is not None else _default_llm_caller()

    # Empty chunks list short-circuits — no LLM calls, no errors,
    # empty result. AC #6.
    if not chunks:
        logger.info(
            "Pass-2 run skipped: no chunks provided (ontology={ontology}, "
            "extensions={ext_count})",
            ontology=ontology.metadata.name if ontology.metadata else "unknown",
            ext_count=len(extensions),
        )
        return ExtractionResult(
            metadata={
                "ontology_name": (
                    ontology.metadata.name if ontology.metadata else "unknown"
                ),
                "chunk_count": 0,
                "extensions_count": len(extensions),
                "total_entities": 0,
                "total_relations": 0,
            }
        )

    all_entities: List[ExtractedEntity] = []
    all_relations: List[ExtractedRelation] = []
    parse_failures = 0

    logger.info(
        "Pass-2 run start: chunks={n}, ontology={o}, extensions={e}",
        n=len(chunks),
        o=ontology.metadata.name if ontology.metadata else "unknown",
        e=len(extensions),
    )

    for chunk in chunks:
        chunk_text = str(chunk.get("text", "") or "")
        chunk_id = chunk.get("id")
        if not chunk_text.strip():
            # Silent skip — empty chunks aren't an error; matches the
            # current workflow.py behaviour.
            continue

        user_prompt = build_pass2_prompt(ontology, chunk_text, extensions)
        estimated = _estimate_tokens(user_prompt)
        if estimated > TOKEN_BUDGET_TARGET:
            # Per Q-B-6 telemetry policy, log before raising so the
            # ledger captures the breach even if a higher-level
            # handler swallows the exception.
            logger.warning(
                "Pass-2 token budget exceeded for chunk {chunk_id}: "
                "~{est} tokens > {budget}. Caller should shrink the "
                "chunk or compress the ontology.",
                chunk_id=chunk_id,
                est=estimated,
                budget=TOKEN_BUDGET_TARGET,
            )
            raise Pass2TokenBudgetExceeded(estimated, TOKEN_BUDGET_TARGET)

        logger.info(
            "pass2_chunk_start chunk_id={chunk_id} estimated_tokens={est}",
            chunk_id=chunk_id,
            est=estimated,
        )

        raw_response = await _invoke_llm(
            caller, PASS2_SYSTEM_PROMPT, user_prompt, model
        )
        chunk_result = _parse_chunk_response(raw_response)

        if chunk_result.metadata.get("parse_error"):
            parse_failures += 1

        # Tag with chunk_id so downstream pipelines can group entities
        # back to their originating chunk. Matches the convention in
        # ``ExtractionWorkflow.extract``.
        for entity in chunk_result.entities:
            entity.source_chunk_id = chunk_id
            all_entities.append(entity)
        for relation in chunk_result.relations:
            relation.source_chunk_id = chunk_id
            all_relations.append(relation)

        logger.info(
            "pass2_chunk_complete chunk_id={chunk_id} entities={ne} relations={nr}",
            chunk_id=chunk_id,
            ne=len(chunk_result.entities),
            nr=len(chunk_result.relations),
        )

    logger.info(
        "pass2_run_complete entities={ne} relations={nr} parse_failures={pf}",
        ne=len(all_entities),
        nr=len(all_relations),
        pf=parse_failures,
    )

    return ExtractionResult(
        entities=all_entities,
        relations=all_relations,
        metadata={
            "ontology_name": (
                ontology.metadata.name if ontology.metadata else "unknown"
            ),
            "chunk_count": len(chunks),
            "extensions_count": len(extensions),
            "total_entities": len(all_entities),
            "total_relations": len(all_relations),
            "parse_failures": parse_failures,
        },
    )
