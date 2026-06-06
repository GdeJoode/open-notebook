"""Pass-1 schema validation module (B.1c, single-schema only).

Given a text sample and a base ontology, ask an LLM whether the
ontology fits, what fraction of source concepts it covers, and what
extensions would close the gap. The output mirrors
``shared.models.notebook_schema.Pass1Result`` exactly so callers can
hand the dump to ``Pass1ResultRepository.record(...)`` without
re-shaping.

Scope
=====
**Single schema only** in this phase. The multi-schema orchestrator
that runs Pass-1 against several candidates and picks the best lands
in B.1e. The ``alternative_schemas`` field here records the top-3
fallback {name, confidence} dicts the LLM noticed — actually
re-running against them is B.1e's job.

Token budget
============
The plan caps the assembled prompt at 3000 tokens with a 20% safety
margin (target ≤ 2400 tokens). Per Q-B-2, we use the coarse
``len(text) // 4`` heuristic — no ``tiktoken`` dependency. The
``TokenBudgetExceeded`` exception fires *before* the LLM call, so a
caller that produced an oversized sample gets a loud failure rather
than a quietly-truncated extraction. For very large ontologies the
schema summary auto-compresses (see ``COMPACT_SUMMARY_THRESHOLD``)
so a 100-type ontology + 1500-token sample still fits.

Integration
===========
The actual LLM call is injected as an ``llm_caller`` argument: a
callable ``(system_prompt: str, user_prompt: str, model: str) -> str``
returning raw response text. This keeps the module testable without
``llm-manager`` running (tests pass a canned callable) and matches
the lazy-import pattern in ``LLMExtractor``. The production wiring
into ``EntityExtractionService`` is B.1f — see the TODO marker there.

Malformed-LLM-output policy
===========================
The plan (AC #3) requires graceful degradation when the LLM returns
malformed JSON: ``_parse_response`` returns an empty ``Pass1Output``
(``detected_schema=""``, ``coverage_pct=0.0``, ``confidence_in_choice=0.0``,
empty lists) and emits a structured WARNING log. The orchestrator
(B.1e) decides whether to retry, skip, or surface in the UI. The
distinct ``Pass1ParseError`` is reserved for *transport* failures
(network, auth, timeout) signalled by the LLM caller itself.
"""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ontology_extraction.prompts.pass1 import (
    PASS1_SYSTEM_PROMPT,
    build_pass1_prompt,
)

# Coarse token estimator: ``len(text) // 4`` matches Q-B-2's
# heuristic. With a 20% safety margin against the 3000-token plan
# cap, we enforce ≤ 2400 estimated tokens for the assembled prompt.
TOKEN_BUDGET_TARGET = 2400

# How many characters of a malformed LLM response to include in the
# WARNING log. Long enough to diagnose, short enough not to swamp the
# log line. Truncation is centralised so the format stays consistent.
_MALFORMED_LOG_EXCERPT_CHARS = 240

# Sync callable: ``(system_prompt, user_prompt, model) -> str``
SyncLLMCaller = Callable[[str, str, str], str]
# Async callable: ``(system_prompt, user_prompt, model) -> Awaitable[str]``
AsyncLLMCaller = Callable[[str, str, str], Awaitable[str]]
# Accept either flavour — tests can pass plain sync callables, the
# real LLM client returns awaitables.
LLMCaller = Union[SyncLLMCaller, AsyncLLMCaller]


class TokenBudgetExceeded(RuntimeError):
    """Raised when ``len(prompt) // 4`` exceeds the Pass-1 budget.

    Fires *before* the LLM call so the caller sees an immediate,
    cheap failure rather than a downstream truncation or LLM
    over-spend. The 2400-token target comes from the 3000-token plan
    cap minus a 20% safety margin (Q-B-2).
    """

    def __init__(self, estimated: int, budget: int = TOKEN_BUDGET_TARGET):
        super().__init__(
            f"Pass-1 prompt exceeds token budget: "
            f"~{estimated} tokens estimated (budget: {budget})"
        )
        self.estimated = estimated
        self.budget = budget


def _estimate_tokens(text: str) -> int:
    """Coarse token count via ``len(text) // 4`` (Q-B-2 heuristic).

    Lives as a module-level helper so tests can patch it for boundary
    checks without faking the entire prompt builder.
    """
    return len(text) // 4


class Pass1Output(BaseModel):
    """Validated output of one Pass-1 run.

    Field shapes match ``shared.models.notebook_schema.Pass1Result``
    exactly so a caller can do::

        pass1 = await validator.run(text, ontology)
        await repo.record(
            Pass1Result(
                source=src_id,
                notebook=nb_id,
                schema_attempted="scholarly",
                **pass1.model_dump(),
            )
        )

    The two-model split keeps DB plumbing (``source``, ``notebook``,
    ``schema_attempted``, persistence metadata) out of the LLM
    contract.
    """

    model_config = ConfigDict(extra="ignore")

    detected_schema: str = Field(
        default="",
        description=(
            "Schema name the LLM judged best-fit for the text. Empty "
            "string is the sentinel for 'unknown' — emitted by the "
            "graceful malformed-JSON path so callers can branch on it."
        ),
    )
    confidence_in_choice: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LLM confidence in ``detected_schema`` (0.0–1.0).",
    )
    coverage_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of source concepts the ATTEMPTED schema covered "
            "(0.0–1.0). Drives the notebook-level rolling average."
        ),
    )
    uncovered_concepts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Surface forms not placed under any attempted-schema type. "
            "Each dict is roughly {surface_form, suggested_type}."
        ),
    )
    proposed_extensions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Extension proposals derived from ``uncovered_concepts``.",
    )
    alternative_schemas: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Top-3 fallback schemas the LLM noticed, each a dict "
            "shaped roughly ``{'name': str, 'confidence': float}``. "
            "Shape matches ``Pass1Result.alternative_schemas`` so the "
            "model_dump feeds the repo directly without lifting "
            "strings into dicts. B.1e re-runs Pass-1 against the "
            "named schemas."
        ),
    )

    @field_validator("coverage_pct", "confidence_in_choice", mode="before")
    @classmethod
    def clamp_to_unit_interval(cls, v: Any) -> float:
        """Accept ints / strings / percentages; clamp into [0, 1].

        The LLM sometimes returns ``87`` or ``"87"`` (treating
        ``coverage_pct`` as a 0-100 percentage) instead of ``0.87``.
        Be defensive: anything ≥ 1.5 we treat as a percentage and
        divide by 100. Anything outside [0, 1] after that gets
        clamped — the field constraint would otherwise reject
        sensible-but-mis-scaled output.
        """
        if v is None:
            return 0.0
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return 0.0
        # Percentage-style: 87 → 0.87. Use 1.5 as the threshold so a
        # legit "1.0" passes through unchanged.
        if fv > 1.5:
            fv = fv / 100.0
        # Clamp to be safe — the Field bounds would raise otherwise.
        if fv < 0.0:
            return 0.0
        if fv > 1.0:
            return 1.0
        return fv

    @field_validator("uncovered_concepts", "proposed_extensions", mode="before")
    @classmethod
    def ensure_list_of_dicts(cls, v: Any) -> List[Dict[str, Any]]:
        """Coerce None / non-list values to an empty list.

        LLMs occasionally emit ``null`` for empty arrays. The DB
        column is FLEXIBLE on these fields, so we accept anything
        list-shaped and drop the rest defensively.
        """
        if v is None:
            return []
        if not isinstance(v, list):
            return []
        return [item for item in v if isinstance(item, dict)]

    @field_validator("alternative_schemas", mode="before")
    @classmethod
    def normalise_alternative_schemas(cls, v: Any) -> List[Dict[str, Any]]:
        """Accept list-of-dicts; coerce list-of-strings for back-compat.

        The canonical LLM contract emits
        ``[{"name": "general", "confidence": 0.4}, ...]`` so the dump
        feeds ``Pass1Result.alternative_schemas`` (also a
        ``List[Dict[str, Any]]``) without re-shaping.

        We still accept a bare ``List[str]`` defensively because an
        LLM may regress to the simpler format. Bare strings are
        lifted to ``{"name": s}`` so downstream readers find ``name``
        in the expected place.
        """
        if v is None:
            return []
        if not isinstance(v, list):
            return []
        normalised: List[Dict[str, Any]] = []
        for item in v:
            if isinstance(item, dict):
                # Drop entries missing ``name`` — they would have no
                # meaningful interpretation for B.1e.
                name = item.get("name")
                if name is None:
                    continue
                normalised.append({**item, "name": str(name)})
            elif isinstance(item, str):
                normalised.append({"name": item})
            # Silently drop other types (None, ints, etc.) — defensive
            # against malformed LLM output.
        return normalised[:3]


# Regex for the "salvage a JSON object from arbitrary prose" minor.
# Greedy match so we capture nested objects rather than stopping at
# the first inner ``}``. Anchored to braces because that's the only
# structural shape Pass-1 accepts (arrays are rejected upstream).
_BRACED_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """Unwrap ```json ... ``` or plain ``` ... ``` blocks.

    Mirrors ``LLMExtractor._parse_response`` so behaviour is
    consistent across pipeline modules.
    """
    s = text.strip()
    if "```json" in s:
        s = s.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in s:
        s = s.split("```", 1)[1].split("```", 1)[0].strip()
    return s


def _salvage_json_object(text: str) -> str:
    """Extract the first ``{...}`` block from arbitrary prose.

    LLMs sometimes wrap JSON in commentary ("Here is the result: {...}
    Hope that helps."). After ``_strip_code_fence`` removes markdown
    fences, this regex pulls out the brace-delimited JSON object so
    ``json.loads`` has a fighting chance. Returns the original text
    unchanged if no object is found — the caller will fail and the
    graceful malformed-JSON path takes over.
    """
    match = _BRACED_OBJECT_RE.search(text)
    return match.group(0) if match else text


class Pass1ParseError(RuntimeError):
    """Raised when the LLM caller itself fails (network, auth, timeout).

    Distinct from ``TokenBudgetExceeded``. Content-parsing failures
    (malformed JSON, missing fields) **do NOT** raise this — they
    return an empty ``Pass1Output`` with a WARNING log so the
    orchestrator (B.1e) can decide between retry / skip / surface.
    """


def _empty_pass1_output() -> "Pass1Output":
    """Build the sentinel ``Pass1Output`` for malformed-LLM-output paths.

    Centralised so the graceful-degradation contract is documented in
    one place and every degraded return looks identical from the
    caller's perspective.
    """
    return Pass1Output(
        detected_schema="",
        confidence_in_choice=0.0,
        coverage_pct=0.0,
        uncovered_concepts=[],
        proposed_extensions=[],
        alternative_schemas=[],
    )


def _log_malformed(reason: str, response: str) -> None:
    """Emit a structured WARNING log for a malformed-LLM-output case.

    The orchestrator looks for these log lines to count parse
    failures. Body is truncated to keep log lines bounded.
    """
    excerpt = response[:_MALFORMED_LOG_EXCERPT_CHARS]
    suffix = "..." if len(response) > _MALFORMED_LOG_EXCERPT_CHARS else ""
    logger.warning(
        "Pass-1 returned empty Pass1Output: {reason} | excerpt={excerpt!r}{suffix}",
        reason=reason,
        excerpt=excerpt,
        suffix=suffix,
    )


class Pass1SchemaValidator:
    """Run a single-schema Pass-1 validation against a text sample.

    Stateless wrapper around the prompt builder + LLM call + parser.
    A new instance per run is fine — the only ``__init__`` argument
    is the LLM caller, which is cheap to capture.

    Token-budget note
    -----------------
    The prompt-assembly pipeline auto-compresses the schema summary
    for ontologies with more than ``COMPACT_SUMMARY_THRESHOLD`` (30)
    types. With that compression in place, a synthetic 100-type
    ontology + 1500-token sample fits comfortably under the
    2400-token cap (see ``TestStressTokenBudget`` in the test
    module). The orchestrator (B.1e) is still responsible for
    selecting a sample size that fits — this module fails loudly
    via ``TokenBudgetExceeded`` rather than silently truncating.
    """

    def __init__(
        self,
        llm_caller: Optional[LLMCaller] = None,
        default_model: str = "default",
    ):
        """Construct a validator.

        Args:
            llm_caller: Callable executing the LLM round-trip. If
                ``None``, the validator will lazy-import
                ``llm_manager.manager`` at call time (mirroring
                ``LLMExtractor``). Pass an explicit callable in
                tests so the import is not attempted.
            default_model: Default model id passed to ``run`` if the
                caller does not override.
        """
        self._llm_caller = llm_caller
        self._default_model = default_model

    async def run(
        self,
        text_sample: str,
        ontology: Dict[str, Any],
        model: Optional[str] = None,
    ) -> Pass1Output:
        """Validate ``ontology`` against ``text_sample``.

        Args:
            text_sample: ~1500-token excerpt of the source. The
                caller is responsible for chunk selection; we do not
                truncate (so a budget miss is loud, not silent).
            ontology: Dict-shaped ontology — either the raw YAML
                load or ``Ontology.model_dump()``. The schema name
                is read from ``ontology["metadata"]["name"]``.
            model: Override the default model id.

        Returns:
            A populated ``Pass1Output``. On malformed LLM output the
            return is the empty sentinel (``detected_schema=""``,
            ``coverage_pct=0.0``, ``confidence_in_choice=0.0``) plus a
            WARNING log — no exception.

        Raises:
            TokenBudgetExceeded: Prompt exceeds the 2400-token cap.
            Pass1ParseError: LLM caller itself failed (transport).
        """
        prompt = build_pass1_prompt(ontology, text_sample)
        estimated = _estimate_tokens(prompt)
        if estimated > TOKEN_BUDGET_TARGET:
            logger.warning(
                f"Pass-1 token budget exceeded: ~{estimated} tokens > "
                f"{TOKEN_BUDGET_TARGET}. Caller should shrink the sample "
                "or the schema summary."
            )
            raise TokenBudgetExceeded(estimated, TOKEN_BUDGET_TARGET)

        # Resolve LLM caller. Lazy-import the real client if none was
        # injected; tests always inject so this branch is dev-only.
        caller = self._llm_caller
        if caller is None:
            caller = self._default_llm_caller()

        chosen_model = model or self._default_model
        try:
            raw_response = caller(PASS1_SYSTEM_PROMPT, prompt, chosen_model)
            # Both sync and async callers are supported. ``inspect``
            # isn't needed — coroutines are awaitable, strings are not.
            if hasattr(raw_response, "__await__"):
                raw_response = await raw_response  # type: ignore[misc]
        except Exception as e:
            # Transport-level failure (network, timeout, auth). Wrap
            # in Pass1ParseError so the orchestrator can branch on it
            # the same way it branches on TokenBudgetExceeded.
            logger.exception(f"Pass-1 LLM call failed: {e}")
            raise Pass1ParseError(f"LLM caller failed: {e}") from e

        return self._parse_response(str(raw_response))

    @staticmethod
    def _parse_response(response: str) -> Pass1Output:
        """Parse and validate an LLM response into a ``Pass1Output``.

        Per plan AC #3, content-parsing failures degrade gracefully:
        return the empty sentinel ``Pass1Output`` and emit a WARNING
        log. The orchestrator decides whether to retry / skip /
        surface — we don't raise.

        Tolerates markdown code fences and brace-delimited JSON
        embedded in arbitrary prose (minor #3). Accepts percentage-
        style scalars (the field validators handle scaling).
        """
        if not response or not response.strip():
            _log_malformed("empty LLM response", response or "")
            return _empty_pass1_output()

        # Unwrap markdown fence, then salvage a brace-delimited block
        # from any surrounding prose (covers "Here is the JSON: {...}"
        # patterns). Salvage is a no-op if the input is already pure
        # JSON.
        text = _strip_code_fence(response)
        text = _salvage_json_object(text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            _log_malformed(f"invalid JSON ({e})", response)
            return _empty_pass1_output()

        if not isinstance(data, dict):
            _log_malformed(
                f"top-level JSON is {type(data).__name__}, expected object",
                response,
            )
            return _empty_pass1_output()

        try:
            return Pass1Output(**data)
        except ValidationError as e:
            _log_malformed(f"output failed Pass1Output validation ({e})", response)
            return _empty_pass1_output()
        except Exception as e:
            # Belt-and-braces: a Pydantic upgrade could surface a new
            # exception class. Treat any constructor failure as a
            # graceful-degradation case rather than crashing the
            # pipeline.
            _log_malformed(f"unexpected Pass1Output construction error ({e})", response)
            return _empty_pass1_output()

    @staticmethod
    def _default_llm_caller() -> LLMCaller:
        """Lazy-import the production LLM caller.

        Kept as a method (not module-level) so unit tests with
        injected callers never trigger the import. Mirrors the
        lazy-import pattern in ``LLMExtractor.extract``.
        """
        # NOTE B.1f will wire ``EntityExtractionService`` into
        # ``Pass1SchemaValidator`` and supply an LLM caller via DI.
        # This default is a safety net for ad-hoc CLI use only.
        # The import is kept so the dependency surfaces if missing —
        # but the manager itself is not instantiated, since B.1f
        # owns the wiring (the previous attempt-1 code allocated a
        # ModelManager and threw it away, which was the minor-#4
        # dead-instantiation finding).
        import llm_manager.manager  # noqa: F401  type: ignore[import-not-found]

        async def _call(system_prompt: str, user_prompt: str, model: str) -> str:
            # This stub returns empty JSON and logs WARNING. B.1f
            # replaces it with the real ModelManager round-trip. The
            # WARNING is the canary — production must not exercise
            # this path.
            logger.warning(
                "Pass1SchemaValidator using lazy default LLM caller — "
                "B.1f integration not yet wired; returning empty JSON."
            )
            return "{}"

        return _call
