"""Context-aware extraction chunk packing (Track M).

The ingestion pipeline chunks every document once at a fixed ~2000-char size
(``pipelines/ingestion/.../chunking/extractor.py``) and persists the chunks to
the ``chunk`` table. Extraction then loops **one LLM call per persisted chunk**
(``ExtractionWorkflow.extract`` single-mode and ``run_pass2`` both iterate the
chunk list), so a 256K-context model issues ~28 tiny calls for a document it
could read in two or three — the huge context is wasted and the per-call
overhead distorts any cross-model speed/accuracy comparison.

Track M makes the *effective* chunks-per-LLM-call derive from the ACTIVE
model's ``context_window``. The persisted 2000-char chunk stays the storage
unit; :func:`pack_chunks_for_model` RE-PACKS those chunks into model-sized
windows (concatenated text + merged provenance) before they reach the
per-chunk extraction loop, so each packed window becomes one LLM call and the
call count drops in proportion to the context window.

The packer also carries the M.4 oversized-chunk GUARD: a single ingestion chunk
that alone exceeds a (small) candidate's input budget is re-split, and an
already-packed batch can be re-packed for a smaller fallback candidate before
the call so a window sized for a 256K primary never overflows a local
Ollama fallback's effective context.
"""

from app_main.services.extraction_chunking.context_packer import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_WINDOW_TOKENS,
    DEFAULT_PROMPT_OVERHEAD_TOKENS,
    PackedWindow,
    estimate_tokens,
    input_budget_tokens,
    pack_chunks_for_model,
    pass2_token_cap,
    resolve_max_window_tokens,
)

__all__ = [
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_MAX_WINDOW_TOKENS",
    "DEFAULT_PROMPT_OVERHEAD_TOKENS",
    "PackedWindow",
    "estimate_tokens",
    "input_budget_tokens",
    "pack_chunks_for_model",
    "pass2_token_cap",
    "resolve_max_window_tokens",
]
