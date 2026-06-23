"""Pack persisted ingestion chunks into model-context-sized windows (Track M.3).

The single public entry point is :func:`pack_chunks_for_model`. It takes the
list of chunk dicts the extraction service already builds from the ``chunk``
table (``{"text", "id", "section_path", ...}``) and greedily concatenates them
into as few windows as fit the active model's ``context_window``, reserving
headroom for the prompt template + ``max_output_tokens`` + a safety margin.

A 256K-context model packs ~all of a typical document's chunks into one or two
windows (so extraction issues two or three LLM calls instead of ~28); an 8K
local fallback packs ~one chunk per window. The packed window carries the
concatenated text plus a ``constituent_chunk_ids`` provenance list and the
structural metadata of its FIRST chunk, so downstream section-aware tagging and
B.8 entity→chunk attribution still have something to bind to (Decision M-D4:
provenance becomes a window-of-chunks list, not a single id).

Token sizing is deliberately conservative — a ``chars / 4`` heuristic rounded
up, times a 0.85 safety margin — because under-packing only costs a few extra
calls while over-packing risks a context overflow that silently truncates
extraction output. No tokenizer dependency is introduced.

M.3 adds a tunable max-window-size CAP on top of the context-derived budget.
Packing a whole document into ONE full-context window was measured to COLLAPSE
extraction recall (the model summarizes a handful of themes — ~8 entities —
instead of exhaustively enumerating ~307) and is no faster (a single
100K-token call is slow). The packing budget therefore becomes
``min(input_budget_tokens(...), max_window_tokens)``, where ``max_window_tokens``
is resolved from the ``EXTRACTION_MAX_WINDOW_TOKENS`` env var (default 6000) via
:func:`resolve_max_window_tokens`. A moderate window preserves recall while
still cutting the per-2000-char-chunk call count. A cap of 0 (or an
unset-but-large value) disables the cap, restoring full-context packing. The
Pass-2 guard budget (:func:`pass2_token_cap`) is UNCHANGED — it stays based on
the full context, and a capped small window passes it trivially.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from loguru import logger

# Conservative fallback context window for a model row that has no
# ``context_window`` populated. Small enough to be safe on any provider; the
# seed (M.1) gives real chain models their true windows so this only bites an
# unseeded/operator-misconfigured model.
DEFAULT_CONTEXT_WINDOW = 8192

# Tokens reserved for the system + user prompt scaffolding (ontology block,
# extraction instructions, JSON format spec) around the packed chunk text.
# Generous on purpose — the Pass-2 ontology block alone can run ~1-1.5K tokens.
DEFAULT_PROMPT_OVERHEAD_TOKENS = 2000

# Fraction of the computed input budget we actually fill. Leaves slack for the
# char→token heuristic under-estimating token-dense or non-English text.
DEFAULT_SAFETY_MARGIN = 0.85

# Floor so a pathological config (tiny context, huge overhead) can never yield a
# zero/negative budget that would drop every chunk. One small chunk always fits.
_MIN_INPUT_BUDGET_TOKENS = 256

# Separator inserted between concatenated chunk texts in a packed window.
_CHUNK_JOIN = "\n\n"

# Env var that tunes the PACKING window-size cap (M.3). Set an int; the packer
# fills each window to ``min(input_budget, EXTRACTION_MAX_WINDOW_TOKENS)`` so a
# big-context model produces several MODERATE windows instead of one full-context
# window. Sweepable without a rebuild (DI/env config, no registry/DB change),
# mirroring the M.2 ``resolve_per_provider_rpm`` pattern.
_MAX_WINDOW_TOKENS_ENV_VAR = "EXTRACTION_MAX_WINDOW_TOKENS"

# Default window-size cap. Measured live: packing a whole document into one
# full-context window collapses extraction recall (the model summarizes the key
# themes — ~8 entities — instead of exhaustively enumerating ~307) AND is no
# faster (one 100K-token call is slow). A moderate ~6K window preserves
# exhaustive recall while still cutting the per-2000-char-chunk call count by
# packing several chunks per call. This is the recall-preserving default.
DEFAULT_MAX_WINDOW_TOKENS = 6000

# Sentinel: any resolved cap >= this is treated as "no cap" (fall back to the
# full ``input_budget`` — the pre-M.3 full-context packing behaviour). A value
# at or above a large model's context window can never bind, so naming it makes
# the back-compat escape hatch explicit rather than relying on a magic int.
_NO_CAP_THRESHOLD = 1_000_000_000

# Sentinel distinguishing "caller passed no max_window_tokens" (resolve from
# env) from "caller explicitly passed None" (no cap). A plain ``None`` default
# could not tell those apart. A single-member Enum is a mypy-friendly sentinel.
class _Unresolved(Enum):
    token = 0


_UNRESOLVED = _Unresolved.token


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for ``text`` via ``ceil(chars / 4)``.

    Rounds UP so the packer never under-counts a window against its budget. The
    same heuristic the extraction pipeline already uses (``len // 4`` in
    pass2), made conservative by the ceiling so boundary windows stay safe.
    """
    if not text:
        return 0
    return math.ceil(len(text) / 4)


def input_budget_tokens(
    *,
    context_window: Optional[int],
    max_output_tokens: Optional[int],
    prompt_overhead_tokens: int = DEFAULT_PROMPT_OVERHEAD_TOKENS,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
) -> int:
    """Compute the per-window INPUT token budget for a model.

    ``(context_window - max_output_tokens - prompt_overhead) * safety_margin``,
    floored at :data:`_MIN_INPUT_BUDGET_TOKENS` so a single chunk always packs.
    A null ``context_window`` degrades to :data:`DEFAULT_CONTEXT_WINDOW` (M's
    graceful-degrade contract for unseeded models).
    """
    ctx = context_window or DEFAULT_CONTEXT_WINDOW
    out = max_output_tokens or 0
    raw = (ctx - out - prompt_overhead_tokens) * safety_margin
    return max(_MIN_INPUT_BUDGET_TOKENS, int(raw))


def resolve_max_window_tokens(
    *,
    environ: Dict[str, str] | None = None,
) -> Optional[int]:
    """Resolve the PACKING window-size cap from the env (M.3).

    Reads :data:`_MAX_WINDOW_TOKENS_ENV_VAR` (``EXTRACTION_MAX_WINDOW_TOKENS``)
    as an int and returns the cap to apply to the per-window packing budget.
    Mirrors the M.2 ``resolve_per_provider_rpm`` env-resolver pattern so the cap
    is sweepable without a rebuild (DI/env config, no registry/DB change).

    Semantics:

    * **Unset / unparseable** → :data:`DEFAULT_MAX_WINDOW_TOKENS` (6000), the
      recall-preserving default.
    * **0 or negative** → ``None`` ("no cap": packing falls back to the full
      ``input_budget`` — the pre-M.3 full-context behaviour). 0 is the explicit
      opt-out.
    * **>= :data:`_NO_CAP_THRESHOLD`** → ``None`` ("no cap"): an
      unset-but-deliberately-large value also disables the cap.
    * Any other positive int → that value (the cap, in tokens).

    A returned ``None`` means "do not cap"; an int means "cap the window to this
    many input tokens (intersected with ``input_budget`` at packing time)".
    """
    env = environ if environ is not None else os.environ
    raw = env.get(_MAX_WINDOW_TOKENS_ENV_VAR)
    if raw is None:
        return DEFAULT_MAX_WINDOW_TOKENS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            f"resolve_max_window_tokens: {_MAX_WINDOW_TOKENS_ENV_VAR}={raw!r} "
            f"is not an int; using default {DEFAULT_MAX_WINDOW_TOKENS}."
        )
        return DEFAULT_MAX_WINDOW_TOKENS
    if value <= 0:
        # Explicit opt-out: 0/negative disables the cap (full-context packing).
        return None
    if value >= _NO_CAP_THRESHOLD:
        return None
    return value


def pass2_token_cap(
    *,
    context_window: Optional[int],
    max_output_tokens: Optional[int],
) -> int:
    """Per-prompt token cap to thread into Pass-2's whole-prompt guard.

    Pass-2's guard measures ``ontology_block + chunk_text`` TOGETHER, whereas
    :func:`input_budget_tokens` (used for PACKING) already subtracts the
    prompt-overhead the ontology block occupies. Threading the packing budget
    into the Pass-2 guard would therefore double-count that overhead and falsely
    abort a window packed right up to ``input_budget`` (the ontology gets re-added
    on top of a budget that already removed it).

    The cap that matches what Pass-2 actually MEASURES is the full input space
    minus reserved output — ``context_window - max_output_tokens``. The packer
    guarantees ``chunk_text <= input_budget = (ctx - max_output - overhead) * 0.85``
    and the ontology block ~= the reserved ``overhead``, so
    ``ontology + chunk_text <= overhead + input_budget < ctx - max_output`` fits.
    A window whose prompt genuinely exceeds ``ctx - max_output`` still raises, so
    the real overflow guard is preserved. Floored at
    :data:`_MIN_INPUT_BUDGET_TOKENS` for the same pathological-config reason.
    """
    ctx = context_window or DEFAULT_CONTEXT_WINDOW
    out = max_output_tokens or 0
    return max(_MIN_INPUT_BUDGET_TOKENS, ctx - out)


@dataclass
class PackedWindow:
    """One model-context-sized window of concatenated ingestion chunks.

    ``text`` is the joined chunk content (one LLM call's worth). ``id`` is the
    first constituent chunk's id (kept so the existing per-chunk tagging that
    reads ``chunk["id"]`` still has a stable value), while
    ``constituent_chunk_ids`` preserves the full provenance list (M-D4). The
    structural metadata fields mirror the first chunk so section-aware
    extraction context still resolves.
    """

    text: str
    id: Optional[str]
    constituent_chunk_ids: List[str] = field(default_factory=list)
    estimated_tokens: int = 0
    section_path: List[str] = field(default_factory=list)
    section_level: int = 0
    physical_page: Optional[int] = None
    element_type: Optional[str] = None
    source_id: Optional[str] = None
    section_heading: Optional[str] = None

    def to_chunk_dict(self) -> Dict[str, Any]:
        """Render as the chunk-dict shape the extraction workflow consumes.

        Output is interchangeable with the un-packed chunk dicts the service
        builds today, so the workflow / pass2 loops need no shape change — they
        just see fewer, larger chunks.
        """
        d: Dict[str, Any] = {
            "text": self.text,
            "id": self.id,
            "section_path": list(self.section_path),
            "section_level": self.section_level,
            "physical_page": self.physical_page,
            "element_type": self.element_type,
            "source_id": self.source_id,
            "constituent_chunk_ids": list(self.constituent_chunk_ids),
        }
        if self.section_heading is not None:
            d["section_heading"] = self.section_heading
        return d


def _split_oversized_text(text: str, budget_tokens: int) -> List[str]:
    """Re-split a single text whose estimated tokens exceed ``budget_tokens``.

    Slices on a character boundary derived from the token budget (``budget * 4``
    chars per slice, the inverse of the ``chars/4`` estimator). This is the M.4
    oversized-chunk GUARD seed: a 40K-char table chunk handed to an 8K-context
    candidate is broken into budget-safe pieces rather than silently
    overflowing. Splitting on chars (not sentence boundaries) is acceptable for
    an extraction window — the LLM still reads coherent prose, and the dedup
    layer reconciles entities that straddle a split.
    """
    if estimate_tokens(text) <= budget_tokens:
        return [text]
    # Per-slice char budget; keep a small margin under the token budget so the
    # ceiling estimator on the slice stays within budget.
    chars_per_slice = max(1, int(budget_tokens * 4 * 0.95))
    return [
        text[i : i + chars_per_slice]
        for i in range(0, len(text), chars_per_slice)
    ]


def pack_chunks_for_model(
    chunks: List[Dict[str, Any]],
    *,
    context_window: Optional[int],
    max_output_tokens: Optional[int],
    prompt_overhead_tokens: int = DEFAULT_PROMPT_OVERHEAD_TOKENS,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    max_window_tokens: Union[int, None, _Unresolved] = _UNRESOLVED,
) -> List[Dict[str, Any]]:
    """Re-pack persisted ingestion chunks into model-context-sized windows.

    Greedily concatenates ``chunks`` (in order) into windows, starting a new
    window whenever appending the next chunk would push the running token
    estimate over the PACKING budget. That budget is::

        effective_budget = min(input_budget_tokens(...), max_window_tokens)

    where ``max_window_tokens`` is the M.3 tunable window-size CAP. When the
    caller leaves it unresolved (the default) it is read from the env via
    :func:`resolve_max_window_tokens` (default 6000); a ``None`` cap means
    "no cap" and the budget reduces to ``input_budget`` (pre-M.3 full-context
    packing). Capping to a moderate window produces MORE, smaller windows — the
    measured sweet spot that preserves exhaustive extraction recall (packing a
    whole doc into one full-context window collapses recall) without reverting
    to one-LLM-call-per-2000-char-chunk.

    A single chunk that alone exceeds the effective budget is re-split (the M.4
    oversized guard) so no emitted window ever exceeds it — the falsifiable
    no-overflow invariant, preserved under a cap.

    Order is preserved; each window carries its constituent chunk-id list and
    the structural metadata of its first chunk. Empty/whitespace chunks are
    dropped (matching the workflow's existing skip).

    Returns a list of chunk dicts (via :meth:`PackedWindow.to_chunk_dict`)
    interchangeable with the un-packed input, so callers swap the list in front
    of ``workflow.extract`` / ``run_multi_schema`` with no other change.
    """
    input_budget = input_budget_tokens(
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        prompt_overhead_tokens=prompt_overhead_tokens,
        safety_margin=safety_margin,
    )

    # M.3 cap: an unresolved sentinel reads the env-tunable default; an explicit
    # value (including ``None`` = no cap) is honoured as passed. The cap only
    # ever SHRINKS the packing window — ``min`` with ``input_budget`` keeps the
    # no-overflow invariant and floors at one chunk per window.
    cap: Optional[int]
    if isinstance(max_window_tokens, _Unresolved):
        cap = resolve_max_window_tokens()
    else:
        cap = max_window_tokens
    if cap is None:
        budget = input_budget
    else:
        budget = max(_MIN_INPUT_BUDGET_TOKENS, min(input_budget, cap))

    windows: List[PackedWindow] = []
    current: Optional[PackedWindow] = None

    def _flush() -> None:
        nonlocal current
        if current is not None and current.text:
            windows.append(current)
        current = None

    def _seed_window(chunk: Dict[str, Any], text: str) -> PackedWindow:
        cid = chunk.get("id")
        return PackedWindow(
            text=text,
            id=str(cid) if cid is not None else None,
            constituent_chunk_ids=[str(cid)] if cid is not None else [],
            estimated_tokens=estimate_tokens(text),
            section_path=list(chunk.get("section_path") or []),
            section_level=chunk.get("section_level", 0) or 0,
            physical_page=chunk.get("physical_page"),
            element_type=chunk.get("element_type"),
            source_id=chunk.get("source_id"),
            section_heading=chunk.get("section_heading"),
        )

    for chunk in chunks:
        text = str(chunk.get("text", "") or "").strip()
        if not text:
            continue

        # Oversized single chunk → re-split into budget-safe pieces, each its
        # own window (M.4 guard). Flush any in-progress window first so order
        # is preserved.
        if estimate_tokens(text) > budget:
            _flush()
            for piece in _split_oversized_text(text, budget):
                windows.append(_seed_window(chunk, piece))
            continue

        if current is None:
            current = _seed_window(chunk, text)
            continue

        # Would appending overflow the budget? +1 token for the join newline.
        prospective = current.estimated_tokens + estimate_tokens(text) + 1
        if prospective > budget:
            _flush()
            current = _seed_window(chunk, text)
        else:
            current.text = f"{current.text}{_CHUNK_JOIN}{text}"
            current.estimated_tokens = estimate_tokens(current.text)
            cid = chunk.get("id")
            if cid is not None:
                current.constituent_chunk_ids.append(str(cid))

    _flush()

    packed = [w.to_chunk_dict() for w in windows]
    logger.info(
        "context_packer: packed {n_in} ingestion chunks into {n_out} windows "
        "(budget={budget} tok = min(input_budget={ib}, cap={cap}), "
        "ctx={ctx}, max_out={out})",
        n_in=len(chunks),
        n_out=len(packed),
        budget=budget,
        ib=input_budget,
        cap=cap,
        ctx=context_window,
        out=max_output_tokens,
    )
    return packed
