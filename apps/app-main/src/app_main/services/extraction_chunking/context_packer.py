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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
) -> List[Dict[str, Any]]:
    """Re-pack persisted ingestion chunks into model-context-sized windows.

    Greedily concatenates ``chunks`` (in order) into windows, starting a new
    window whenever appending the next chunk would push the running token
    estimate over the model's input budget
    (:func:`input_budget_tokens`). A single chunk that alone exceeds the budget
    is re-split (the M.4 oversized guard) so no emitted window ever exceeds the
    budget — the falsifiable no-overflow invariant.

    Order is preserved; each window carries its constituent chunk-id list and
    the structural metadata of its first chunk. Empty/whitespace chunks are
    dropped (matching the workflow's existing skip).

    Returns a list of chunk dicts (via :meth:`PackedWindow.to_chunk_dict`)
    interchangeable with the un-packed input, so callers swap the list in front
    of ``workflow.extract`` / ``run_multi_schema`` with no other change.
    """
    budget = input_budget_tokens(
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        prompt_overhead_tokens=prompt_overhead_tokens,
        safety_margin=safety_margin,
    )

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
        "(budget={budget} tok, ctx={ctx}, max_out={out})",
        n_in=len(chunks),
        n_out=len(packed),
        budget=budget,
        ctx=context_window,
        out=max_output_tokens,
    )
    return packed
