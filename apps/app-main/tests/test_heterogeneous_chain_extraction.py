"""Track M.5 — heterogeneous failover-chain packing gate.

The crux of Track M: each model in the extraction failover chain has a different
context window (Gemini ~1M, Ollama Cloud ~32K, llama3.1:8b ~8K). M.3 packs the
persisted ingestion chunks into windows sized to the ACTIVE candidate, and M.4
re-splits any single chunk that would overflow the smallest candidate. This test
packs the SAME document for every candidate in the chain and asserts the
falsifiable invariants:

1. **No overflow, ever** — for every candidate, no packed window exceeds that
   candidate's usable input space (`overflow_count == 0`), INCLUDING an oversized
   table-sized chunk that alone exceeds the llama window (the M.4 guard).
2. **Heterogeneous granularity** — a big-context candidate packs the doc into
   FEWER calls than a small-context one (`est_calls` strictly decreases as the
   context window grows), which is what keeps the big models from wasting calls
   and the small models from overflowing.
3. **The M.3 window cap trades calls for recall uniformly** — under the default
   cap, a big-context candidate uses MORE (moderate) windows than uncapped, so no
   candidate issues one giant recall-collapsing call.

Pure packing + measurement (no LLM, no DB): this is the regression gate over the
shipped M.3/M.4, not a live extraction.
"""

from __future__ import annotations

from typing import List

from app_main.services.extraction_chunking.chunking_metrics import measure_packing
from app_main.services.extraction_chunking.context_packer import (
    estimate_tokens,
    pack_chunks_for_model,
)

# The live failover chain, widest → narrowest context. (name, context_window,
# max_output_tokens) — the M.1-seeded shape of the chain models.
CHAIN = [
    ("gemini-2.5-flash", 1_000_000, 8192),
    ("ollama-cloud", 32_768, 4096),
    ("llama3.1:8b", 8192, 2048),
]


def _doc(n_chunks: int, chunk_chars: int = 2000) -> List[dict]:
    """A synthetic persisted document: ``n_chunks`` ingestion chunks (~2000 chars,
    the ingestion default). Text is distinct per chunk so packing is realistic."""
    return [
        {
            "id": f"chunk:{i}",
            "text": (f"Section {i}. " + "lorem ipsum dolor sit amet ")
            * (chunk_chars // 27),
            "section_path": [f"S{i}"],
            "source_id": "source:m5",
        }
        for i in range(n_chunks)
    ]


def test_no_candidate_overflows_and_calls_shrink_with_context():
    # ~30K tokens of document (60 * 2000 chars) — big enough that the narrow
    # candidates need several windows while Gemini needs one.
    doc = _doc(60)

    metrics = {}
    for name, ctx, out in CHAIN:
        # Uncapped (max_window_tokens=None) → PURE context-derived packing, so the
        # heterogeneous divergence is visible (the cap would equalize windows).
        windows = pack_chunks_for_model(
            doc, context_window=ctx, max_output_tokens=out, max_window_tokens=None
        )
        m = measure_packing(windows, context_window=ctx, max_output_tokens=out)
        metrics[name] = m
        # Invariant 1: NOTHING overflows this candidate's input space.
        assert m.overflow_count == 0, f"{name} overflowed: {m}"

    # Invariant 2: fewer calls as the context window grows.
    assert (
        metrics["gemini-2.5-flash"].est_calls
        < metrics["ollama-cloud"].est_calls
        < metrics["llama3.1:8b"].est_calls
    ), {k: v.est_calls for k, v in metrics.items()}
    # Gemini swallows the whole doc in a single call; llama needs many.
    assert metrics["gemini-2.5-flash"].est_calls == 1
    assert metrics["llama3.1:8b"].est_calls >= 5


def test_oversized_chunk_is_split_for_the_narrow_candidate():
    # One table-sized chunk (~10K tokens) that alone blows past llama's ~4K input
    # space, among normal chunks. The M.4 guard must re-split it so no llama
    # window overflows — the no-silent-truncation guarantee.
    big_text = "x" * 40_000  # ~10K tokens
    doc = _doc(3) + [
        {"id": "chunk:big", "text": big_text, "source_id": "source:m5"}
    ]
    assert estimate_tokens(big_text) > 4000  # would overflow llama unguarded

    name, ctx, out = CHAIN[-1]  # llama
    windows = pack_chunks_for_model(
        doc, context_window=ctx, max_output_tokens=out, max_window_tokens=None
    )
    m = measure_packing(windows, context_window=ctx, max_output_tokens=out)
    assert m.overflow_count == 0  # the oversized chunk was re-split, not overflowed
    # The big chunk contributed multiple windows (it was split into pieces).
    assert m.est_calls > 3

    # Gemini fits the same big chunk without splitting.
    g_name, g_ctx, g_out = CHAIN[0]
    g_windows = pack_chunks_for_model(
        doc, context_window=g_ctx, max_output_tokens=g_out, max_window_tokens=None
    )
    g = measure_packing(g_windows, context_window=g_ctx, max_output_tokens=g_out)
    assert g.overflow_count == 0
    assert g.est_calls == 1  # whole doc + big chunk in one window


def test_default_window_cap_trades_calls_for_recall(monkeypatch):
    # Under the default M.3 cap (~6000), a big-context candidate uses MORE,
    # moderate windows than uncapped — no single recall-collapsing giant call —
    # while still never overflowing. Pin the env so the DEFAULT (6000) is what
    # resolves, independent of any ambient EXTRACTION_MAX_WINDOW_TOKENS.
    monkeypatch.delenv("EXTRACTION_MAX_WINDOW_TOKENS", raising=False)
    doc = _doc(60)
    name, ctx, out = CHAIN[0]  # gemini

    uncapped = pack_chunks_for_model(
        doc, context_window=ctx, max_output_tokens=out, max_window_tokens=None
    )
    capped = pack_chunks_for_model(  # omit max_window_tokens → env default (6000)
        doc, context_window=ctx, max_output_tokens=out
    )

    m_uncapped = measure_packing(uncapped, context_window=ctx, max_output_tokens=out)
    m_capped = measure_packing(capped, context_window=ctx, max_output_tokens=out)

    assert m_uncapped.est_calls == 1
    assert m_capped.est_calls > m_uncapped.est_calls  # cap → several moderate calls
    assert m_capped.overflow_count == 0
    # Every capped window is a MODERATE size (well under the full context).
    assert m_capped.max_window_tokens <= 6000 * 1.1
