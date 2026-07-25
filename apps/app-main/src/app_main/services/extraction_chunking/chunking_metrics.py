"""Measurement over context-packed windows (Track M.5).

The falsifiable gate for the M.3/M.4 heterogeneous re-chunking: given the windows
:func:`context_packer.pack_chunks_for_model` produced for a candidate model,
compute how many LLM calls they cost (``est_calls`` = one call per window) and —
the safety invariant — how many would OVERFLOW that model's usable input space
(``overflow_count``, which must be 0).

The overflow threshold is the TRUE input space a window's text may occupy without
the assembled prompt (ontology block + text) exceeding ``context_window -
max_output_tokens``: ``context_window - max_output_tokens - prompt_overhead``.
The packer fills each window only up to ``input_budget = 0.85 * that`` (or a
smaller M.3 cap) and re-splits any single oversized chunk, so a correctly packed
chain always measures ``overflow_count == 0`` — that is what the M.5 integration
test asserts across a Gemini→Ollama→llama chain.

Pure functions over the packed dicts — no LLM call, no DB — so it doubles as a CI
gate the heterogeneous-chain test can assert on directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app_main.services.extraction_chunking.context_packer import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_PROMPT_OVERHEAD_TOKENS,
    estimate_tokens,
)


@dataclass(frozen=True)
class ChunkingMetrics:
    """Per-candidate packing measurement.

    * ``est_calls`` — number of packed windows = extraction LLM calls for this
      candidate (the cost the M.3 packing reduces vs. one-call-per-2000-chars).
    * ``overflow_count`` — windows whose text would overflow the candidate's
      usable input space (MUST be 0 — the M.4 no-overflow invariant).
    * ``max_window_tokens`` — largest single window's estimated tokens.
    * ``total_input_tokens`` — sum across windows (roughly constant across
      candidates for the same document; only the packing granularity changes).
    """

    est_calls: int
    overflow_count: int
    max_window_tokens: int
    total_input_tokens: int


def input_space_tokens(
    *,
    context_window: Optional[int],
    max_output_tokens: Optional[int],
    prompt_overhead_tokens: int = DEFAULT_PROMPT_OVERHEAD_TOKENS,
) -> int:
    """The TRUE per-window input space (no safety margin) — the overflow ceiling.

    ``context_window - max_output_tokens - prompt_overhead``. A window whose text
    estimate exceeds this would push ``ontology + text`` past ``ctx - max_output``
    and overflow. Null context degrades to :data:`DEFAULT_CONTEXT_WINDOW`.
    """
    ctx = context_window or DEFAULT_CONTEXT_WINDOW
    out = max_output_tokens or 0
    return ctx - out - prompt_overhead_tokens


def measure_packing(
    windows: List[Dict[str, Any]],
    *,
    context_window: Optional[int],
    max_output_tokens: Optional[int],
    prompt_overhead_tokens: int = DEFAULT_PROMPT_OVERHEAD_TOKENS,
) -> ChunkingMetrics:
    """Measure the packed ``windows`` against a candidate model's input space."""
    ceiling = input_space_tokens(
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        prompt_overhead_tokens=prompt_overhead_tokens,
    )
    ests = [estimate_tokens(str(w.get("text", "") or "")) for w in windows]
    return ChunkingMetrics(
        est_calls=len(windows),
        overflow_count=sum(1 for e in ests if e > ceiling),
        max_window_tokens=max(ests, default=0),
        total_input_tokens=sum(ests),
    )


__all__ = ["ChunkingMetrics", "input_space_tokens", "measure_packing"]
