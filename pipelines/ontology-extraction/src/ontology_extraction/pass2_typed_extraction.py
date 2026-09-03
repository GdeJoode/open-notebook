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
import os
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from loguru import logger
from ontology_manager.schema import Ontology
from shared.models.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)

from ontology_extraction.candidates import (
    _detect_lang,
    _load_spacy,
    extract_candidates,
    mine_hearst_isa,
)
from ontology_extraction.not_a_concept import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt,
    parse_judge_response,
    partition_deterministic,
)
from ontology_extraction.prompts.pass2 import (
    PASS2_SYSTEM_PROMPT,
    build_pass2_prompt,
)

# Coarse token estimator: ``len(text) // 4`` matches Q-B-2's
# heuristic. This is the LEGACY DEFAULT budget, used only on the
# fallback path where no ``token_budget`` is threaded into ``run_pass2``
# (CLI / dev). Production threads ``pass2_token_cap`` (a large,
# context-derived ceiling), so this constant never gates the proven
# recipe. Raised from 2400 → 2800 in N.1 to reserve room for the
# exhaustive-extraction prompt overhead (~+290 tokens), so a realistic
# single-chunk default-path call still fits without falsely tripping
# the guard. Headroom remains against the 3000-token plan cap.
TOKEN_BUDGET_TARGET = 2800


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _candidate_anchors_enabled() -> bool:
    """Track N.1: thread pre-LLM candidate anchors into the prompt (default ON)."""
    return _env_bool("EXTRACTION_CANDIDATE_ANCHORS", True)


def _candidate_top_k() -> int:
    raw = os.getenv("EXTRACTION_CANDIDATE_TOP_K")
    try:
        return max(0, int(raw)) if raw else 20
    except ValueError:
        return 20


def _domain_ner_enabled() -> bool:
    """Track N.1 stub gate: the EntityRuler domain gazetteer (default OFF)."""
    return _env_bool("EXTRACTION_DOMAIN_NER_ENABLED", False)


def _hearst_isa_enabled() -> bool:
    """Track N.2: seed deterministic Hearst is-a relations (default OFF, N.5b).

    Measured on the project's eight-document corpus while this defaulted ON: 220
    raw Hearst pairs across 3823 chunks, 138 of them distinct, and **zero**
    `is_a` edges in the graph — of 1895 relations in 100 types, not one. The
    precision gate is why: both endpoints must be entities the LLM extracted for
    the SAME chunk. Under the much looser reading "both endpoints exist anywhere
    in this notebook" only 15 distinct pairs survive, and their quality is mixed
    — `agrariers is_a ondernemers` is right, `banken is_a voedselketen` is wrong,
    `PD is_a Control variables` is a table artefact, and
    `projecten is_a uitvoeringsactiviteiten` repeats fifteen times.

    So this ships explicitly off rather than nominally on. The review finding it
    answers (I3) was not "the miner is bad" but "a producer whose output survives
    by accident": `is_a` was declared in no ontology, and the mined edges lived
    only because `OntologyValidator` downgrades an unknown predicate to a WARNING
    outside strict mode. One flag flip deleted them silently.

    Both halves of N.5b matter and neither works alone. `is_a` is now declared in
    the three root ontologies, so a run that turns this back on produces a
    predicate the validator recognises under strict mode too; and this default
    means nothing flows until somebody decides it should.
    """
    return _env_bool("EXTRACTION_HEARST_ISA", False)


def _not_a_concept_enabled() -> bool:
    """Track N.3: drop LLM-emitted page-furniture before the graph (default ON)."""
    return _env_bool("EXTRACTION_NOT_A_CONCEPT", True)


def _not_a_concept_judge_enabled() -> bool:
    """Track N.3: LLM-judge arbitrates the ambiguous not-a-concept middle (D4, ON)."""
    return _env_bool("EXTRACTION_NOT_A_CONCEPT_JUDGE", True)


# Conservative confidence for a Hearst-seeded relation — below a typical LLM
# extraction so downstream max-confidence merges prefer the LLM's own judgement
# where it also found the relation, and the audit's low-confidence checks can see it.
_HEARST_CONFIDENCE = 0.5


def _seed_hearst_relations(
    chunk_text: str,
    chunk_id: Any,
    entities: List[ExtractedEntity],
    existing: List[ExtractedRelation],
    *,
    nlp: Optional[Any] = None,
) -> List[ExtractedRelation]:
    """Return Hearst-mined ``is_a`` relations between ALREADY-extracted entities.

    Precision gate (N.2): a mined ``(narrow, broad)`` pair only becomes a relation
    when BOTH endpoints match an entity the LLM extracted for this chunk (exact
    normalized-lowercase). Skips pairs already present (any relation between the
    same endpoints). Each carries ``relation_source="hearst"`` provenance + a
    conservative confidence, so the post-filter/audit still govern it.
    """
    from ontology_extraction.candidates import _normalize

    ent_by_norm = {_normalize(e.text).lower(): e.text for e in entities if e.text}
    if not ent_by_norm:
        return []
    # Suppress only a DUPLICATE is_a on the same ordered pair — a pre-existing
    # relation of a DIFFERENT type between the endpoints must not block a
    # legitimate Hearst is_a (dedup key includes relation_type).
    have = {
        (
            r.source_entity.strip().lower(),
            r.target_entity.strip().lower(),
            (r.relation_type or "").strip().lower(),
        )
        for r in existing
    }
    seeded: List[ExtractedRelation] = []
    for narrow, broad in mine_hearst_isa(chunk_text, nlp=nlp):
        nk, bk = narrow.lower(), broad.lower()
        if nk not in ent_by_norm or bk not in ent_by_norm:
            continue  # precision gate: both endpoints must be extracted entities
        src, tgt = ent_by_norm[nk], ent_by_norm[bk]
        if (src.strip().lower(), tgt.strip().lower(), "is_a") in have:
            continue
        rel = ExtractedRelation(
            source_entity=src,
            target_entity=tgt,
            relation_type="is_a",
            confidence=_HEARST_CONFIDENCE,
            properties={"relation_source": "hearst"},
        )
        rel.source_chunk_id = chunk_id
        seeded.append(rel)
        have.add((src.strip().lower(), tgt.strip().lower(), "is_a"))
    return seeded


async def _apply_not_a_concept(
    entities: List[ExtractedEntity],
    relations: List[ExtractedRelation],
    *,
    caller: "LLMCaller",
    model: str,
    judge_enabled: bool,
) -> tuple[List[ExtractedEntity], List[ExtractedRelation], int, int]:
    """Track N.3: drop page-furniture entities (+ their relations) from a chunk.

    Deterministic tier first (high-precision reject / fast accept); the ambiguous
    middle goes to a single BATCHED LLM-judge call when ``judge_enabled`` and a
    caller is available, else it is KEPT (never dropped on a guess). A judge
    transport/parse failure also keeps the ambiguous set. Relations whose endpoints
    reference a removed entity are dropped (mirrors ``noise_filter``).

    Returns ``(kept_entities, kept_relations, removed_count, judged_count)``.
    """
    kept, rejected, ambiguous = partition_deterministic(entities)
    judged = 0
    if ambiguous:
        verdicts: Dict[str, bool] = {}
        if judge_enabled and caller is not None:
            items = [(e.text, e.label) for e in ambiguous]
            try:
                raw = await _invoke_llm(
                    caller, JUDGE_SYSTEM_PROMPT, build_judge_prompt(items), model
                )
                verdicts = parse_judge_response(raw, items)
            except Exception as exc:  # noqa: BLE001 — judge is best-effort; keep all
                logger.warning(
                    "pass2: not-a-concept judge failed ({e}); keeping ambiguous", e=exc
                )
                verdicts = {}
        # judged = entities the judge EXPLICITLY ruled on (not the batch size — a
        # partial/garbled response leaves the silent ones defaulted to keep below).
        judged = len(verdicts)
        for e in ambiguous:
            # default True → keep (no judge, or judge silent on this item)
            if verdicts.get(e.text, True):
                kept.append(e)
            else:
                rejected.append(e)
    # Drop only relations that reference an entity the gate REMOVED — a relation
    # whose endpoint was never an extracted entity (the LLM sometimes emits those)
    # is out of this gate's scope and passes through unchanged (pre-N.3 behaviour).
    # Compare on stripped text so a whitespace variant can't leave a dangling edge.
    removed_texts = {e.text.strip() for e in rejected}
    kept_relations = [
        r
        for r in relations
        if r.source_entity.strip() not in removed_texts
        and r.target_entity.strip() not in removed_texts
    ]
    return kept, kept_relations, len(rejected), judged

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
    token_budget: Optional[int] = None,
    candidate_anchors_enabled: Optional[bool] = None,
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
        token_budget: Per-prompt token ceiling. ``None`` keeps the legacy
            fixed :data:`TOKEN_BUDGET_TARGET` (2800) so existing callers are
            unchanged. Track M threads the ACTIVE model's context-derived
            input budget here so a context-packed window (which can legitimately
            run far past 2800 tokens on a big-context model) is not falsely
            rejected — while a window that genuinely overflows the model still
            raises loudly.

    Returns:
        A combined ``ExtractionResult``. Per AC #3 every entity and
        every relation has a non-null ``confidence``. Per AC #4
        chunks whose LLM response cannot be parsed contribute zero
        elements but do not interrupt the run.

    Raises:
        Pass2TokenBudgetExceeded: Prompt exceeds the legacy default
            2800-token cap (or the threaded ``token_budget``) for some chunk.
        Pass2ParseError: LLM caller itself failed (transport).
    """
    extensions = accepted_extensions or []
    caller = llm_caller if llm_caller is not None else _default_llm_caller()
    budget = token_budget if token_budget is not None else TOKEN_BUDGET_TARGET

    # Track N.1: pre-LLM candidate anchors. Enabled by default (env-overridable);
    # the corpus for TF-IDF salience is the source's own chunk texts, computed once.
    anchors_on = (
        candidate_anchors_enabled
        if candidate_anchors_enabled is not None
        else _candidate_anchors_enabled()
    )
    corpus_texts = (
        [str(c.get("text", "") or "") for c in chunks] if anchors_on else []
    )

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

    # Track N.3 telemetry counters (feed extraction_metrics): entities the LLM
    # emitted before the not-a-concept gate, how many it rejected, how many the
    # judge arbitrated, and chunks the LLM abstained on (returned no entity).
    nac_on = _not_a_concept_enabled()
    nac_judge = _not_a_concept_judge_enabled()
    entities_extracted = 0
    not_a_concept_removed = 0
    not_a_concept_judged = 0
    abstained_chunks = 0

    # Track N.1/N.2: detect the corpus language ONCE and load the spaCy model
    # once, so every chunk's candidate extraction (N.1) and Hearst is-a mining
    # (N.2) share the SAME model — consistent noun-chunk boundaries + a single
    # load, not a per-chunk detect/lookup. None when spaCy or the model is
    # unavailable; both layers degrade gracefully (candidates → regex fallback,
    # Hearst → no pairs).
    hearst_on = _hearst_isa_enabled()
    shared_nlp: Optional[Any] = None
    if anchors_on or hearst_on:
        # Detect over the SAME scope extract_candidates uses (whole corpus, capped
        # at 20k chars) so the injected model never disagrees with the model that
        # layer would have picked itself. corpus_texts is only populated when
        # anchors are on; fall back to the chunk texts for a Hearst-only run.
        lang_source = corpus_texts or [
            str(c.get("text", "") or "") for c in chunks
        ]
        shared_nlp = _load_spacy(_detect_lang(" ".join(lang_source)[:20000]))

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

        anchors: Optional[List[str]] = None
        if anchors_on:
            try:
                cands = extract_candidates(
                    chunk_text,
                    corpus_chunks=corpus_texts,
                    top_k=_candidate_top_k(),
                    domain_ner_enabled=_domain_ner_enabled(),
                    nlp=shared_nlp,
                )
                anchors = [c.text for c in cands]
            except Exception as exc:  # noqa: BLE001 — candidates are best-effort
                logger.warning(
                    "pass2: candidate extraction failed for chunk {cid} ({e}); "
                    "proceeding without anchors",
                    cid=chunk_id,
                    e=exc,
                )
                anchors = None

        user_prompt = build_pass2_prompt(
            ontology, chunk_text, extensions, candidate_anchors=anchors
        )
        estimated = _estimate_tokens(user_prompt)
        if estimated > budget and anchors:
            # Anchors are a best-effort recall nudge — they must NEVER turn a chunk
            # that would otherwise fit into a budget failure that stops the whole
            # run. Drop them and re-check before raising (N.1 safety).
            user_prompt = build_pass2_prompt(
                ontology, chunk_text, extensions, candidate_anchors=None
            )
            estimated = _estimate_tokens(user_prompt)
            if estimated <= budget:
                logger.info(
                    "pass2: dropped candidate anchors for chunk {cid} to stay "
                    "within the {b}-token budget",
                    cid=chunk_id,
                    b=budget,
                )
        if estimated > budget:
            # Per Q-B-6 telemetry policy, log before raising so the
            # ledger captures the breach even if a higher-level
            # handler swallows the exception.
            logger.warning(
                "Pass-2 token budget exceeded for chunk {chunk_id}: "
                "~{est} tokens > {budget}. Caller should shrink the "
                "chunk or compress the ontology.",
                chunk_id=chunk_id,
                est=estimated,
                budget=budget,
            )
            raise Pass2TokenBudgetExceeded(estimated, budget)

        logger.info(
            "pass2_chunk_start chunk_id={chunk_id} estimated_tokens={est}",
            chunk_id=chunk_id,
            est=estimated,
        )

        raw_response = await _invoke_llm(
            caller, PASS2_SYSTEM_PROMPT, user_prompt, model
        )
        chunk_result = _parse_chunk_response(raw_response)

        parse_error = bool(chunk_result.metadata.get("parse_error"))
        if parse_error:
            parse_failures += 1

        # Track N.3: abstention — a non-error chunk the LLM returned NO entity for
        # is a genuine abstention (page-furniture / no domain content), not a
        # failure. Count it on the RAW (pre-filter) result.
        entities_extracted += len(chunk_result.entities)
        if not parse_error and not chunk_result.entities:
            abstained_chunks += 1

        # Track N.3: the not-a-concept gate — drop page-furniture entities (+ the
        # relations referencing them) BEFORE they are appended or Hearst-seeded, so
        # only survivors reach the graph and the precision gate. Best-effort: a
        # failure here must never lose a chunk's real entities.
        if nac_on and chunk_result.entities:
            try:
                kept_e, kept_r, removed_ct, judged_ct = await _apply_not_a_concept(
                    chunk_result.entities,
                    chunk_result.relations,
                    caller=caller,
                    model=model,
                    judge_enabled=nac_judge,
                )
                chunk_result.entities = kept_e
                chunk_result.relations = kept_r
                not_a_concept_removed += removed_ct
                not_a_concept_judged += judged_ct
            except Exception as exc:  # noqa: BLE001 — gate is best-effort
                logger.warning(
                    "pass2: not-a-concept gate failed for chunk {cid} ({e}); "
                    "keeping all entities",
                    cid=chunk_id,
                    e=exc,
                )

        # Tag with chunk_id so downstream pipelines can group entities
        # back to their originating chunk. Matches the convention in
        # ``ExtractionWorkflow.extract``.
        for entity in chunk_result.entities:
            entity.source_chunk_id = chunk_id
            all_entities.append(entity)
        for relation in chunk_result.relations:
            relation.source_chunk_id = chunk_id
            all_relations.append(relation)

        # Track N.2: seed deterministic Hearst is-a relations between entities the
        # LLM already extracted (precision gate + provenance inside the helper).
        if hearst_on:
            try:
                seeded = _seed_hearst_relations(
                    chunk_text,
                    chunk_id,
                    chunk_result.entities,
                    chunk_result.relations,
                    nlp=shared_nlp,
                )
                all_relations.extend(seeded)
            except Exception as exc:  # noqa: BLE001 — best-effort, never fail the run
                logger.warning(
                    "pass2: Hearst is-a seeding failed for chunk {c} ({e})",
                    c=chunk_id,
                    e=exc,
                )

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
            # Track N.3: raw counts for extraction_metrics (over-generation +
            # abstain rate). entities_kept mirrors total_entities — kept for a
            # self-describing metadata contract the metric module reads directly.
            "entities_extracted": entities_extracted,
            "entities_kept": len(all_entities),
            "not_a_concept_removed": not_a_concept_removed,
            "not_a_concept_judged": not_a_concept_judged,
            "abstained_chunks": abstained_chunks,
        },
    )
