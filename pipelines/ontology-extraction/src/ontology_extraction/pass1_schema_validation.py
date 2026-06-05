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
in B.1e. The ``alternative_schemas`` field here just records the
top-3 fallback NAMES the LLM noticed — actually re-running against
them is B.1e's job.

Token budget
============
The plan caps the assembled prompt at 3000 tokens with a 20% safety
margin (target ≤ 2400 tokens). Per Q-B-2, we use the coarse
``len(text) // 4`` heuristic — no ``tiktoken`` dependency. The
``TokenBudgetExceeded`` exception fires *before* the LLM call, so a
caller that produced an oversized sample gets a loud failure rather
than a quietly-truncated extraction.

Integration
===========
The actual LLM call is injected as an ``llm_caller`` argument: a
callable ``(system_prompt: str, user_prompt: str, model: str) -> str``
returning raw response text. This keeps the module testable without
``llm-manager`` running (tests pass a canned callable) and matches
the lazy-import pattern in ``LLMExtractor``. The production wiring
into ``EntityExtractionService`` is B.1f — see the TODO marker there.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ontology_extraction.prompts.pass1 import build_pass1_prompt

# Coarse token estimator: ``len(text) // 4`` matches Q-B-2's
# heuristic. With a 20% safety margin against the 3000-token plan
# cap, we enforce ≤ 2400 estimated tokens for the assembled prompt.
TOKEN_BUDGET_TARGET = 2400

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
        description="Schema name the LLM judged best-fit for the text."
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
    alternative_schemas: List[str] = Field(
        default_factory=list,
        description=(
            "Top-3 fallback schema NAMES the LLM noticed. B.1e re-runs "
            "Pass-1 against these to pick the best fit."
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
    def ensure_list_of_strings(cls, v: Any) -> List[str]:
        """Coerce None / non-list to an empty list, drop non-strings."""
        if v is None:
            return []
        if not isinstance(v, list):
            return []
        return [str(item) for item in v if item is not None][:3]


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


class Pass1ParseError(RuntimeError):
    """Raised when the LLM response cannot be parsed into ``Pass1Output``.

    Distinct from ``TokenBudgetExceeded`` so callers can branch on
    transient (retry) vs structural (fail-loud) failures.
    """


class Pass1SchemaValidator:
    """Run a single-schema Pass-1 validation against a text sample.

    Stateless wrapper around the prompt builder + LLM call + parser.
    A new instance per run is fine — the only ``__init__`` argument
    is the LLM caller, which is cheap to capture.
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
            A populated ``Pass1Output``.

        Raises:
            TokenBudgetExceeded: Prompt exceeds the 2400-token cap.
            Pass1ParseError: LLM response could not be parsed.
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

        system_prompt = (
            "You are a meticulous ontology curator. Judge schema fit "
            "honestly — under-confidence is better than over-confidence."
        )

        # Resolve LLM caller. Lazy-import the real client if none was
        # injected; tests always inject so this branch is dev-only.
        caller = self._llm_caller
        if caller is None:
            caller = self._default_llm_caller()

        chosen_model = model or self._default_model
        try:
            raw_response = caller(system_prompt, prompt, chosen_model)
            # Both sync and async callers are supported. ``inspect``
            # isn't needed — coroutines are awaitable, strings are not.
            if hasattr(raw_response, "__await__"):
                raw_response = await raw_response  # type: ignore[misc]
        except Exception as e:
            # Don't swallow — Pass-1 failure should bubble so the
            # caller can decide between retry / fallback / abort.
            logger.exception(f"Pass-1 LLM call failed: {e}")
            raise

        return self._parse_response(str(raw_response))

    @staticmethod
    def _parse_response(response: str) -> Pass1Output:
        """Parse and validate an LLM response into a ``Pass1Output``.

        Tolerates markdown code fences and accepts percentage-style
        scalars (the field validators handle scaling). Raises
        ``Pass1ParseError`` on anything that can't be coerced.
        """
        if not response or not response.strip():
            raise Pass1ParseError("Empty LLM response")

        text = _strip_code_fence(response)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Pass-1 response is not valid JSON: {e}")
            raise Pass1ParseError(f"Invalid JSON: {e}") from e

        if not isinstance(data, dict):
            raise Pass1ParseError(
                f"Pass-1 response must be a JSON object, got {type(data).__name__}"
            )

        try:
            return Pass1Output(**data)
        except Exception as e:
            logger.warning(f"Pass-1 response failed validation: {e}")
            raise Pass1ParseError(f"Output validation failed: {e}") from e

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
        from llm_manager.manager import ModelManager  # type: ignore[import-not-found]

        manager = ModelManager()

        async def _call(system_prompt: str, user_prompt: str, model: str) -> str:
            # ModelManager exposes ``get_model_from_config(...).complete``
            # but the surface differs by provider; this stub is a
            # placeholder that B.1f will replace with the real
            # wiring. Returning empty JSON keeps the parser failure
            # path testable in production until then.
            logger.warning(
                "Pass1SchemaValidator using lazy default LLM caller — "
                "B.1f integration not yet wired."
            )
            return "{}"

        return _call
