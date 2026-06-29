"""Pairwise contradiction judge over RELATED sources (Track Z.2).

Given a source, take its top RELATED sources (Track R substrate — NOT O(n²)),
form ``(a, b)`` candidate pairs, and ask an LLM to judge whether the two sources
**reinforce**, **contradict**, or are **neutral**. Only a *confident*
contradicts/reinforces verdict is persisted as a ``source_verdict`` edge (Z.1's
idempotent ``relate_verdict``); a neutral or below-threshold verdict writes
NOTHING.

Precision is the whole point. A fabricated ``contradicts`` edge pollutes the
graph and is worse than a missed one, so the gate is deliberately conservative:
the prompt biases the model toward ``neutral`` when unsure, a malformed/missing
response is treated as ``neutral`` (never a crash), and persistence requires
``confidence >= min_confidence`` (default :data:`DEFAULT_MIN_CONFIDENCE`).

Layering mirrors :class:`NoteAutoLinkService` (Y.2): the candidate substrate +
the LLM live in app-main, so this orchestrator lives here too. The LLM is
injected as an ``(system, user) -> str`` async callable (the J.4
``RoutedLLMCaller`` shape) so the service is fully testable without a live model.
Persistence goes through Z.1's ``relate_verdict`` (idempotent + injection-safe);
canonical ``source`` rows are never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from loguru import logger

# --- Precision-first defaults (Z.2 decision) -------------------------------
#
# ``min_confidence`` gates persistence. 0.7 is deliberately conservative: the
# judge must be clearly confident before a contradicts/reinforces edge is
# written, so weak/uncertain judgments (the bulk, given the prompt biases toward
# neutral) never reach the graph. Lower it only once the judge is trusted.
DEFAULT_MIN_CONFIDENCE: float = 0.7

# ``k`` bounds the related fan-out per source so the LLM cost stays O(top-k),
# not O(corpus). A source links to at most ``DEFAULT_K`` related neighbours.
DEFAULT_K: int = 5

# Validation bounds (defense-in-depth — also enforced at the route layer in Z.3).
MAX_K: int = 50
MIN_CONFIDENCE_FLOOR: float = 0.0
MIN_CONFIDENCE_CEIL: float = 1.0

# The verdicts that may ever be persisted. ``neutral`` is intentionally absent:
# a neutral judgment is the "no edge" outcome (precision-first).
PERSISTABLE_VERDICTS: frozenset[str] = frozenset({"contradicts", "reinforces"})
ALL_VERDICTS: frozenset[str] = PERSISTABLE_VERDICTS | {"neutral"}

# How much of each source's text the judge sees. Capping keeps the prompt within
# the chat model's context and bounds cost; titles + a bounded snippet are enough
# to judge reinforce/contradict for the source-pair grain.
_SNIPPET_CHARS: int = 2000


# --- Judge prompt (precision-biased) ---------------------------------------
#
# System prompt: a careful fact-checking judge. The bias toward ``neutral`` is
# load-bearing — it is repeated and made the explicit default so the model does
# NOT manufacture a contradiction from mere topical difference or partial
# overlap. Output is strict JSON (json_mode is also requested at the provider).
JUDGE_SYSTEM_PROMPT = (
    "You are a careful, conservative fact-checking judge. You are given two "
    "documents (A and B) that are already known to be topically related. Your "
    "ONLY job is to decide how their factual claims relate:\n"
    '  - "contradicts": A and B make claims that cannot both be true — a direct '
    "factual conflict (e.g. opposite numbers, mutually exclusive statements, "
    "explicit disagreement).\n"
    '  - "reinforces": A and B make claims that clearly support or corroborate '
    "each other — the same conclusion or mutually strengthening evidence.\n"
    '  - "neutral": anything else — they are merely on the same topic, discuss '
    "different aspects, partially overlap, or you are not sure.\n"
    "\n"
    "Precision over recall. A false contradiction is far worse than a missed "
    'one. DEFAULT TO "neutral". Only choose "contradicts" or "reinforces" when '
    "there is a CLEAR, SPECIFIC factual relationship you can point to in the "
    "text — never from topic similarity alone, never from a guess. When in any "
    'doubt, answer "neutral" with low confidence.\n'
    "\n"
    "Respond with ONLY a JSON object, no prose before or after:\n"
    '{"verdict": "contradicts|reinforces|neutral", '
    '"confidence": 0.0-1.0, '
    '"reasoning": "one short sentence citing the specific claims, or why neutral"}\n'
    "\n"
    '"confidence" is your certainty in the verdict in [0, 1]. Be honest and '
    "low when uncertain."
)


@dataclass
class JudgeVerdict:
    """A parsed (and normalised) judge verdict for one pair.

    Always well-formed regardless of the raw LLM output: a malformed/missing
    response is coerced to ``neutral`` / ``confidence=0.0`` (see
    :func:`parse_verdict`), so downstream code never has to defend against bad
    shapes — it can only ever fail the precision gate, never crash.
    """

    verdict: str
    confidence: float
    reasoning: str

    def is_persistable(self, min_confidence: float) -> bool:
        """True iff this verdict should be written as an edge (precision gate).

        Requires BOTH a persistable verdict (``contradicts``/``reinforces`` —
        never ``neutral``) AND ``confidence >= min_confidence``.
        """
        return (
            self.verdict in PERSISTABLE_VERDICTS
            and self.confidence >= min_confidence
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


@dataclass
class JudgeRunResult:
    """Summary of judging a source's related pairs (or a single pair)."""

    source_id: str
    status: str = "judged"
    judged: int = 0
    contradicts: int = 0
    reinforces: int = 0
    neutral: int = 0
    below_threshold: int = 0
    edges_written: int = 0
    candidates_considered: int = 0
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    k: int = DEFAULT_K
    verdict_pairs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "status": self.status,
            "judged": self.judged,
            "contradicts": self.contradicts,
            "reinforces": self.reinforces,
            "neutral": self.neutral,
            "below_threshold": self.below_threshold,
            "edges_written": self.edges_written,
            "candidates_considered": self.candidates_considered,
            "min_confidence": self.min_confidence,
            "k": self.k,
            "verdict_pairs": self.verdict_pairs,
        }


# --- Candidate generation (pure / testable) --------------------------------


def build_candidate_pairs(
    source_id: str,
    related_ids: List[str],
    *,
    k: int = DEFAULT_K,
) -> List[Tuple[str, str]]:
    """Form ``(source_id, related)`` pairs from a related-source list.

    Pure and deterministic so it is unit-testable without a DB or LLM. The
    related list comes from the Track R substrate (already topically related,
    already top-k ranked) — this never enumerates the whole corpus.

    Guarantees:
      * **self excluded** — a related id equal to ``source_id`` is dropped (a
        source never judged against itself);
      * **deduped** — a related id appearing twice yields ONE pair. Because every
        pair shares ``source_id`` as its first element, ``(a, b)`` and ``(b, a)``
        cannot both arise here, but a repeated ``b`` is still collapsed;
      * **order-preserving** — pairs keep the related list's ranking order (most
        related first), so a downstream cap is meaningful;
      * **bounded** — at most ``k`` pairs are returned (defense-in-depth on top of
        the substrate's own LIMIT). ``k <= 0`` yields no pairs.

    Args:
        source_id: the source to judge from (the first element of every pair).
        related_ids: candidate related source ids, ranked most-related first.
        k: max pairs to return.

    Returns:
        Up to ``k`` ``(source_id, related_id)`` tuples, deduped and self-free.
    """
    if k <= 0:
        return []

    pairs: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for rid in related_ids:
        if not rid or rid == source_id:
            continue
        if rid in seen:
            continue
        seen.add(rid)
        pairs.append((source_id, rid))
        if len(pairs) >= k:
            break
    return pairs


# --- Robust verdict parsing ------------------------------------------------


def parse_verdict(raw: Optional[str]) -> JudgeVerdict:
    """Parse an LLM judge response into a well-formed :class:`JudgeVerdict`.

    Precision-first robustness: ANY problem (empty output, non-JSON, missing
    keys, an unknown verdict label, a non-numeric/out-of-range confidence)
    degrades to ``neutral`` / ``confidence=0.0`` rather than raising. A
    ``neutral`` (or low-confidence) verdict never produces an edge, so a parse
    failure can only ever SUPPRESS an edge — it can never fabricate one.

    Accepted shape::

        {"verdict": "contradicts"|"reinforces"|"neutral",
         "confidence": <float in [0,1]>,
         "reasoning": "<str>"}

    Tolerances:
      * code fences (```json … ```) are stripped;
      * the verdict label is lower-cased and trimmed; an unknown label -> neutral;
      * a confidence outside [0, 1] is clamped; a non-numeric one -> 0.0;
      * a missing ``reasoning`` -> "".

    Hard rule (the no-false-edge invariant): ONLY a top-level JSON **object** is
    accepted. A top-level JSON ARRAY — even ``[{"verdict": "contradicts", ...}]``,
    a common "respond with JSON" failure mode — degrades to ``neutral``; the
    parser must NEVER descend into an array element and lift a verdict out of it,
    because that would fabricate a confident edge from a malformed response. Any
    non-object top-level value (array, scalar, etc.) -> neutral.
    """
    if not raw or not raw.strip():
        return JudgeVerdict("neutral", 0.0, "")

    obj = _parse_top_level_object(raw)
    if obj is None:
        logger.debug(
            "judge parse: non-object / unparseable response -> neutral; raw={!r}",
            raw[:200],
        )
        return JudgeVerdict("neutral", 0.0, "")

    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in ALL_VERDICTS:
        # Unknown/absent label is NOT a contradiction — never assert from noise.
        verdict = "neutral"

    confidence = _coerce_confidence(obj.get("confidence"))

    reasoning_val = obj.get("reasoning", "")
    reasoning = str(reasoning_val) if reasoning_val is not None else ""

    return JudgeVerdict(verdict=verdict, confidence=confidence, reasoning=reasoning)


def _strip_code_fence(text: str) -> str:
    """Strip a single surrounding ```/```json code fence, if present.

    Many models wrap JSON in a fence even in json_mode. We remove a leading
    ```` ```json ```` / ```` ``` ```` line and a trailing ```` ``` ```` so the
    remainder can be parsed as bare JSON. Leaves un-fenced text unchanged.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    # Drop the opening fence line (``` or ```json) and an optional closing fence.
    after_open = stripped[3:]
    newline = after_open.find("\n")
    if newline == -1:
        return stripped  # malformed single-line fence — leave as-is
    body = after_open[newline + 1 :]
    close = body.rfind("```")
    if close != -1:
        body = body[:close]
    return body.strip()


def _parse_top_level_object(text: str) -> Optional[dict]:
    """Parse ``text`` to a JSON object, accepting ONLY a top-level ``dict``.

    The no-false-edge invariant lives here. We first try to parse the cleaned
    (fence-stripped) text as JSON and accept the result ONLY if it is a ``dict``.
    A top-level ARRAY (``[...]``) or any scalar -> ``None`` (neutral); we NEVER
    descend into an array element to lift out a verdict object.

    A prose-wrapped object (``"Here is my judgment: {…}"``) is still recovered by
    a guarded balanced-``{...}`` scan — BUT only when the cleaned text is not a
    top-level array (the first meaningful char is not ``[``), so a ``{`` nested
    inside ``[{…}]`` is never extracted.
    """
    import json

    cleaned = _strip_code_fence(text)
    if not cleaned:
        return None

    # Direct parse: the json_mode happy path. Accept ONLY a dict; a list/scalar
    # (e.g. ``[{"verdict": "contradicts", ...}]``) is rejected outright.
    try:
        parsed = json.loads(cleaned)
    except Exception:  # noqa: BLE001 — fall through to the prose scan
        parsed = None
    if parsed is not None:
        return parsed if isinstance(parsed, dict) else None

    # Prose fallback: a balanced ``{...}`` embedded in surrounding text
    # (``"Here is my judgment: {…}"``). Guard against descending into an array:
    # if the FIRST JSON-structural character (``[`` or ``{``) is ``[`` — whether
    # the array is bare (``[{…}]``) or prose-wrapped (``"Result: [{…}]"``) — the
    # ``{`` that follows is nested inside that array, so we must NOT extract it.
    # Refuse unless a ``{`` appears at top level before any ``[``.
    brace = cleaned.find("{")
    bracket = cleaned.find("[")
    if brace == -1:
        return None
    if bracket != -1 and bracket < brace:
        return None
    return _first_balanced_object(cleaned)


def _first_balanced_object(text: str) -> Optional[dict]:
    """Return the first balanced ``{...}`` slice in ``text`` as a dict, or None.

    String-aware brace matching (a ``}`` inside a JSON string does not close the
    object). Caller guarantees ``text`` is not a top-level array, so the first
    ``{`` is the object itself, not one nested inside ``[...]``.
    """
    import json

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                except Exception:  # noqa: BLE001
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _coerce_confidence(value: Any) -> float:
    """Coerce a raw confidence to a clamped [0, 1] float; else -> 0.0.

    Strict-numeric: only a real ``int``/``float`` is accepted. A confidence that
    arrives as a STRING (``"0.9"``) or any other type signals the model is not
    honoring json_mode, so it degrades to ``0.0`` (neutral / no edge) rather than
    being string-parsed into a gate-clearing score — precision-first. ``bool`` is
    excluded (it is an ``int`` subclass but never a real confidence).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    conf = float(value)
    if conf != conf:  # NaN
        return 0.0
    return max(MIN_CONFIDENCE_FLOOR, min(conf, MIN_CONFIDENCE_CEIL))


# --- Compact judge context -------------------------------------------------


def _snippet(text: Optional[str], limit: int = _SNIPPET_CHARS) -> str:
    """A bounded, single-spaced snippet of a source's text for the prompt."""
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + " …"


def build_judge_user_prompt(
    a_title: Optional[str],
    a_text: Optional[str],
    b_title: Optional[str],
    b_text: Optional[str],
) -> str:
    """Build the COMPACT user prompt: both sources' titles + bounded snippets."""
    return (
        "Document A\n"
        f"Title: {a_title or '(untitled)'}\n"
        f"Text: {_snippet(a_text)}\n"
        "\n"
        "Document B\n"
        f"Title: {b_title or '(untitled)'}\n"
        f"Text: {_snippet(b_text)}\n"
        "\n"
        "Judge how A and B relate per your instructions. Respond with ONLY the "
        "JSON object."
    )


# An async ``(system, user) -> str`` callable — the J.4 ``RoutedLLMCaller`` shape.
LLMCaller = Callable[[str, str], Awaitable[str]]


class ContradictionJudgeService:
    """Orchestrate source → related pairs → judged ``source_verdict`` edges (Z.2)."""

    def __init__(
        self,
        source_service: Any,
        related_service: Any,
        source_repo: Any,
        llm_caller: LLMCaller,
        *,
        judge_model: str = "",
    ):
        """
        Args:
            source_service: a ``SourceService`` (``get`` for source text/title).
            related_service: the Track R related substrate. Either a
                ``HybridRetrievalService`` (``find_related_hybrid``) or anything
                exposing ``find_related(source_id, k)`` — the service uses whichever
                is present (hybrid preferred), so candidates always come from the
                related substrate, never an all-pairs scan.
            source_repo: a ``SourceRepository`` for Z.1's idempotent, injection-safe
                ``relate_verdict``.
            llm_caller: an async ``(system, user) -> str`` callable (the J.4
                ``RoutedLLMCaller``). Injected so tests mock the model entirely.
            judge_model: provenance label stamped onto each persisted edge.
        """
        self.source_service = source_service
        self.related_service = related_service
        self.source_repo = source_repo
        self.llm_caller = llm_caller
        self.judge_model = judge_model

    @staticmethod
    def _clamp_params(
        k: Optional[int], min_confidence: Optional[float]
    ) -> Tuple[int, float]:
        """Normalise/clamp params to safe bounds (defense-in-depth)."""
        eff_k = DEFAULT_K if k is None else int(k)
        eff_k = max(1, min(eff_k, MAX_K))

        eff_min = (
            DEFAULT_MIN_CONFIDENCE
            if min_confidence is None
            else float(min_confidence)
        )
        eff_min = max(MIN_CONFIDENCE_FLOOR, min(eff_min, MIN_CONFIDENCE_CEIL))
        return eff_k, eff_min

    async def _fetch_related_ids(self, source_id: str, k: int) -> List[str]:
        """Top-``k`` related source ids via the Track R substrate (hybrid first).

        Prefers ``find_related_hybrid`` (dense + KG fusion); falls back to a
        ``find_related`` (dense-only) if the injected service only offers that.
        Returns ids in ranking order. Never raises — a substrate error degrades
        to no candidates (nothing judged) rather than a 500.
        """
        try:
            if hasattr(self.related_service, "find_related_hybrid"):
                rows = await self.related_service.find_related_hybrid(source_id, k)
            else:
                rows = await self.related_service.find_related(source_id, k)
        except Exception as e:  # noqa: BLE001 — degrade to no candidates
            logger.warning(
                f"judge: related lookup failed for {source_id}: {e}"
            )
            return []
        return [str(r["id"]) for r in (rows or []) if r.get("id")]

    async def judge_pair(
        self,
        a_id: str,
        b_id: str,
        *,
        min_confidence: Optional[float] = None,
    ) -> Tuple[JudgeVerdict, bool]:
        """Judge ONE ``(a, b)`` source pair; persist only on a confident verdict.

        Flow: load both sources → build a compact context → call the LLM →
        :func:`parse_verdict` (robust) → precision gate. An edge is RELATEd (via
        Z.1's idempotent ``relate_verdict``) ONLY when the verdict is
        ``contradicts``/``reinforces`` AND ``confidence >= min_confidence``.

        Returns ``(verdict, edge_written)``. Never raises for the ordinary cases
        (missing source, malformed LLM output) — those yield a neutral/no-edge
        outcome so a batch run is never aborted by one bad pair.
        """
        _, eff_min = self._clamp_params(None, min_confidence)

        a = await self.source_service.get(a_id)
        b = await self.source_service.get(b_id)
        if a is None or b is None:
            logger.info(
                f"judge: skipping pair ({a_id}, {b_id}) — source missing"
            )
            return JudgeVerdict("neutral", 0.0, ""), False

        user_prompt = build_judge_user_prompt(
            a.title, a.full_text, b.title, b.full_text
        )

        try:
            raw = await self.llm_caller(JUDGE_SYSTEM_PROMPT, user_prompt)
        except Exception as e:  # noqa: BLE001 — a model failure is neutral/no-edge
            logger.warning(f"judge: LLM call failed for ({a_id}, {b_id}): {e}")
            return JudgeVerdict("neutral", 0.0, ""), False

        verdict = parse_verdict(raw)

        if not verdict.is_persistable(eff_min):
            # neutral OR below-threshold contradicts/reinforces → NO edge.
            return verdict, False

        ok = await self.source_repo.relate_verdict(
            a_id,
            b_id,
            verdict=verdict.verdict,
            confidence=verdict.confidence,
            reasoning=verdict.reasoning,
            judge_model=self.judge_model,
        )
        if not ok:
            # relate_verdict refused (self/wrong-table/unsafe id) or hit a DB
            # error. Report the verdict but no edge — never crash the run.
            logger.warning(
                f"judge: relate_verdict declined for ({a_id}, {b_id})"
            )
        return verdict, ok

    async def judge_source(
        self,
        source_id: str,
        *,
        k: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> JudgeRunResult:
        """Judge ``source_id`` against its top related sources, precision-first.

        Flow: existence check → fetch top-``k`` related (Track R substrate) →
        :func:`build_candidate_pairs` (dedup/self-free/bounded) → judge each pair
        → tally. Only confident contradicts/reinforces verdicts become edges;
        neutral/below-threshold are counted but written nowhere.

        Idempotent: each edge goes through Z.1's clear-before-relate, so
        re-running yields the identical edge set.

        Returns a :class:`JudgeRunResult` summary; never raises for the ordinary
        not-found / no-related cases (reported via ``status``).
        """
        eff_k, eff_min = self._clamp_params(k, min_confidence)

        source = await self.source_service.get(source_id)
        if source is None:
            return JudgeRunResult(
                source_id=source_id,
                status="not_found",
                min_confidence=eff_min,
                k=eff_k,
            )

        related_ids = await self._fetch_related_ids(source_id, eff_k)
        pairs = build_candidate_pairs(source_id, related_ids, k=eff_k)

        result = JudgeRunResult(
            source_id=source_id,
            status="judged",
            candidates_considered=len(pairs),
            min_confidence=eff_min,
            k=eff_k,
        )

        for a_id, b_id in pairs:
            verdict, edge_written = await self.judge_pair(
                a_id, b_id, min_confidence=eff_min
            )
            result.judged += 1
            if verdict.verdict == "contradicts":
                result.contradicts += 1
            elif verdict.verdict == "reinforces":
                result.reinforces += 1
            else:
                result.neutral += 1

            if verdict.verdict in PERSISTABLE_VERDICTS and not edge_written:
                # A contradicts/reinforces verdict that did NOT clear the gate
                # (below threshold) — distinct from a neutral.
                if not verdict.is_persistable(eff_min):
                    result.below_threshold += 1

            if edge_written:
                result.edges_written += 1

            result.verdict_pairs.append(
                {
                    "from": a_id,
                    "to": b_id,
                    "verdict": verdict.verdict,
                    "confidence": verdict.confidence,
                    "edge_written": edge_written,
                }
            )

        logger.info(
            "judge {}: judged={} contradicts={} reinforces={} neutral={} "
            "below_threshold={} edges={} (k={}, min_conf={})",
            source_id,
            result.judged,
            result.contradicts,
            result.reinforces,
            result.neutral,
            result.below_threshold,
            result.edges_written,
            eff_k,
            eff_min,
        )
        return result
