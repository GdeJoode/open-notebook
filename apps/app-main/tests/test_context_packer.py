"""Unit tests for the Track M context-aware chunk packer (M.3 + M.4 guard)."""

import math

from app_main.services.extraction_chunking.context_packer import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_WINDOW_TOKENS,
    estimate_tokens,
    input_budget_tokens,
    pack_chunks_for_model,
    pass2_token_cap,
    resolve_max_window_tokens,
)


def _chunks(n: int, *, chars: int = 2000, source_id: str = "source:doc") -> list:
    """n ingestion-sized chunks of ~``chars`` each with stable ids/metadata."""
    return [
        {
            "text": "word " * (chars // 5),  # ~chars characters
            "id": f"chunk:{i}",
            "section_path": [f"section {i}"],
            "section_level": 1,
            "physical_page": i,
            "element_type": "paragraph",
            "source_id": source_id,
            "section_heading": f"section {i}",
        }
        for i in range(n)
    ]


def _max_window_tokens(packed: list) -> int:
    return max(estimate_tokens(w["text"]) for w in packed)


class TestEstimateTokens:
    def test_empty_is_zero(self):
        assert estimate_tokens("") == 0

    def test_rounds_up(self):
        # 5 chars -> ceil(5/4) == 2, never truncates downward.
        assert estimate_tokens("abcde") == 2


class TestInputBudget:
    def test_big_context_big_budget(self):
        budget = input_budget_tokens(
            context_window=256_000, max_output_tokens=4096, prompt_overhead_tokens=2000
        )
        # ~(256000 - 4096 - 2000) * 0.85
        assert budget > 200_000

    def test_null_context_degrades_to_default(self):
        budget = input_budget_tokens(
            context_window=None, max_output_tokens=None, prompt_overhead_tokens=2000
        )
        # Uses DEFAULT_CONTEXT_WINDOW, stays positive and bounded.
        assert 0 < budget < DEFAULT_CONTEXT_WINDOW

    def test_pathological_config_floors(self):
        # overhead > context would go negative without the floor.
        budget = input_budget_tokens(
            context_window=1000, max_output_tokens=2000, prompt_overhead_tokens=5000
        )
        assert budget >= 256


class TestPass2TokenCap:
    """The Pass-2 cap (M rev2) must NOT re-subtract the packer's overhead.

    Pass-2's guard measures ``ontology + chunk_text`` together, so the cap
    threaded into it is the full input space minus reserved output —
    ``context_window - max_output_tokens`` — not the already-overhead-subtracted
    ``input_budget``. The cap must therefore sit ABOVE ``input_budget`` by at
    least the reserved overhead so a window packed to ``input_budget`` plus an
    ontology block (~overhead) still fits.
    """

    def test_cap_is_context_minus_output(self):
        assert pass2_token_cap(context_window=128_000, max_output_tokens=4096) == (
            128_000 - 4096
        )

    def test_cap_exceeds_packing_budget_by_at_least_overhead(self):
        cap = pass2_token_cap(context_window=128_000, max_output_tokens=4096)
        pack = input_budget_tokens(context_window=128_000, max_output_tokens=4096)
        # The cap must leave room for the ontology block the packer reserved as
        # overhead — otherwise threading it would double-subtract that overhead.
        assert cap > pack
        assert cap - pack >= 2000  # >= DEFAULT_PROMPT_OVERHEAD_TOKENS

    def test_null_context_degrades_to_default(self):
        cap = pass2_token_cap(context_window=None, max_output_tokens=None)
        assert cap == DEFAULT_CONTEXT_WINDOW

    def test_pathological_config_floors(self):
        cap = pass2_token_cap(context_window=100, max_output_tokens=500)
        assert cap >= 256


class TestPackCountDropsForBigContext:
    """The full-context (uncapped) packing behaviour. M.3 caps the window by
    default, so these back-compat cases pass ``max_window_tokens=None`` to
    exercise the pre-M.3 path explicitly."""

    def test_big_context_few_windows(self):
        chunks = _chunks(60)
        packed = pack_chunks_for_model(
            chunks,
            context_window=128_000,
            max_output_tokens=4096,
            prompt_overhead_tokens=2000,
            max_window_tokens=None,  # no cap → full-context packing
        )
        # 60 tiny chunks collapse into a handful of windows for a 128K model.
        assert len(packed) <= 5
        assert len(packed) < len(chunks)

    def test_pack_count_strictly_drops_vs_per_chunk(self):
        chunks = _chunks(28)  # the live ~28-batch case
        packed = pack_chunks_for_model(
            chunks,
            context_window=256_000,
            max_output_tokens=4096,
            prompt_overhead_tokens=2000,
            max_window_tokens=None,  # no cap → full-context packing
        )
        # 256K model reads the whole document in a single call.
        assert len(packed) <= 2
        assert len(packed) < 28


class TestCallCountDropMeasurement:
    """The headline Track M improvement: a big-context model's effective
    LLM-call count drops from the live ~28 (one per 2000-char chunk) to a
    handful, while a small fallback re-packs without overflow."""

    def test_28_chunk_doc_drops_from_28_calls_to_few_for_128k(self):
        chunks = _chunks(28)  # the live case: 28 persisted 2000-char chunks
        old_call_count = len(chunks)  # pre-M: one LLM call per chunk
        packed = pack_chunks_for_model(
            chunks, context_window=128_000, max_output_tokens=4096
        )
        new_call_count = len(packed)
        assert old_call_count == 28
        assert new_call_count <= 3
        # The whole point: a >5x reduction in LLM calls for a big-context model.
        assert new_call_count < old_call_count / 5

    def test_same_doc_small_fallback_repacks_without_overflow(self):
        chunks = _chunks(28)
        fb_budget = input_budget_tokens(context_window=8192, max_output_tokens=512)
        primary = pack_chunks_for_model(
            chunks, context_window=128_000, max_output_tokens=4096
        )
        fallback = pack_chunks_for_model(
            chunks, context_window=8192, max_output_tokens=512
        )
        # Re-pack happened (more, smaller windows than the primary) and none
        # overflow the 8K fallback budget.
        assert len(fallback) > len(primary)
        assert _max_window_tokens(fallback) <= fb_budget


class TestManyBoundedWindowsForSmallContext:
    def test_small_context_many_windows_none_overflow(self):
        chunks = _chunks(40)
        budget = input_budget_tokens(
            context_window=8192, max_output_tokens=512, prompt_overhead_tokens=2000
        )
        packed = pack_chunks_for_model(
            chunks,
            context_window=8192,
            max_output_tokens=512,
            prompt_overhead_tokens=2000,
        )
        assert len(packed) >= 5
        assert _max_window_tokens(packed) <= budget


class TestOversizedChunkGuard:
    def test_single_oversized_chunk_is_resplit(self):
        # One 40K-char table chunk vs an 8K context: must re-split, no overflow.
        big = [
            {
                "text": "x" * 40_000,
                "id": "chunk:big",
                "section_path": [],
                "section_level": 0,
                "physical_page": None,
                "element_type": "table",
                "source_id": "source:doc",
            }
        ]
        budget = input_budget_tokens(
            context_window=8192, max_output_tokens=512, prompt_overhead_tokens=2000
        )
        packed = pack_chunks_for_model(
            big,
            context_window=8192,
            max_output_tokens=512,
            prompt_overhead_tokens=2000,
        )
        assert len(packed) >= 2
        assert _max_window_tokens(packed) <= budget

    def test_resplit_for_small_fallback_vs_packed_for_primary(self):
        # The M.4 invariant in miniature: the SAME chunks packed for a 256K
        # primary make few windows; re-packed for an 8K fallback make many,
        # none overflowing the fallback.
        chunks = _chunks(30)
        primary = pack_chunks_for_model(
            chunks, context_window=256_000, max_output_tokens=4096
        )
        fallback = pack_chunks_for_model(
            chunks, context_window=8192, max_output_tokens=512
        )
        assert len(primary) < len(fallback)
        fb_budget = input_budget_tokens(
            context_window=8192, max_output_tokens=512
        )
        assert _max_window_tokens(fallback) <= fb_budget


class TestOrderAndProvenancePreserved:
    def test_constituent_ids_cover_all_chunks_in_order(self):
        chunks = _chunks(10)
        packed = pack_chunks_for_model(
            chunks, context_window=128_000, max_output_tokens=4096
        )
        flattened = [
            cid for w in packed for cid in w["constituent_chunk_ids"]
        ]
        assert flattened == [f"chunk:{i}" for i in range(10)]

    def test_window_carries_first_chunk_metadata(self):
        chunks = _chunks(4)
        packed = pack_chunks_for_model(
            chunks, context_window=128_000, max_output_tokens=4096
        )
        first = packed[0]
        assert first["source_id"] == "source:doc"
        assert first["id"] == "chunk:0"
        assert first["section_path"] == ["section 0"]

    def test_empty_and_whitespace_chunks_dropped(self):
        chunks = [
            {"text": "real content here", "id": "chunk:0"},
            {"text": "   ", "id": "chunk:1"},
            {"text": "", "id": "chunk:2"},
        ]
        packed = pack_chunks_for_model(
            chunks, context_window=128_000, max_output_tokens=4096
        )
        flattened = [cid for w in packed for cid in w["constituent_chunk_ids"]]
        assert flattened == ["chunk:0"]


class TestResolveMaxWindowTokens:
    """The M.3 env resolver: EXTRACTION_MAX_WINDOW_TOKENS overrides; default
    6000; 0/negative = no cap; unparseable falls back to default; large = no
    cap (back-compat escape hatch)."""

    def test_unset_defaults_to_6000(self):
        assert resolve_max_window_tokens(environ={}) == 6000
        assert DEFAULT_MAX_WINDOW_TOKENS == 6000

    def test_env_override(self):
        assert (
            resolve_max_window_tokens(
                environ={"EXTRACTION_MAX_WINDOW_TOKENS": "12000"}
            )
            == 12000
        )

    def test_zero_means_no_cap(self):
        assert (
            resolve_max_window_tokens(
                environ={"EXTRACTION_MAX_WINDOW_TOKENS": "0"}
            )
            is None
        )

    def test_negative_means_no_cap(self):
        assert (
            resolve_max_window_tokens(
                environ={"EXTRACTION_MAX_WINDOW_TOKENS": "-1"}
            )
            is None
        )

    def test_unparseable_falls_back_to_default(self):
        assert (
            resolve_max_window_tokens(
                environ={"EXTRACTION_MAX_WINDOW_TOKENS": "not-an-int"}
            )
            == 6000
        )

    def test_very_large_means_no_cap(self):
        # An unset-but-deliberately-large value disables the cap.
        assert (
            resolve_max_window_tokens(
                environ={"EXTRACTION_MAX_WINDOW_TOKENS": "2000000000"}
            )
            is None
        )


class TestMaxWindowCap:
    """M.3 packing cap: the budget becomes min(input_budget, max_window_tokens).
    A small cap yields MANY windows (recall-preserving); a large cap restores
    full-context packing (back-compat)."""

    def test_small_cap_yields_many_windows_matching_budget_math(self):
        # 280 chunks of ~500 tok each, capped at 6000 → the live recall case.
        # Expected windows ~= ceil(total_tokens / cap), give or take join
        # overhead. Assert WELL above 1 and close to the budget math.
        n = 280
        chunks = _chunks(n, chars=2000)  # ~500 tok each
        per_chunk_tok = estimate_tokens(chunks[0]["text"])  # 500
        cap = 6000
        packed = pack_chunks_for_model(
            chunks,
            context_window=131_000,
            max_output_tokens=4096,
            max_window_tokens=cap,
        )
        # Chunks that fit per window (each adds +1 join token after the first).
        per_window = 1 + (cap - per_chunk_tok) // (per_chunk_tok + 1)
        expected = math.ceil(n / per_window)
        assert len(packed) > 10  # many windows, emphatically not one
        assert len(packed) == expected
        # No window exceeds the cap.
        assert _max_window_tokens(packed) <= cap

    def test_large_cap_restores_full_context_packing(self):
        # A cap above input_budget can never bind → identical to no-cap.
        chunks = _chunks(60, chars=2000)
        capped = pack_chunks_for_model(
            chunks,
            context_window=128_000,
            max_output_tokens=4096,
            max_window_tokens=200_000,
        )
        uncapped = pack_chunks_for_model(
            chunks,
            context_window=128_000,
            max_output_tokens=4096,
            max_window_tokens=None,
        )
        assert [w["text"] for w in capped] == [w["text"] for w in uncapped]
        assert len(capped) <= 5  # full-context packing: a handful of windows

    def test_cap_never_exceeds_min_input_budget_and_cap(self):
        chunks = _chunks(50, chars=2000)
        cap = 4000
        ib = input_budget_tokens(context_window=128_000, max_output_tokens=4096)
        packed = pack_chunks_for_model(
            chunks,
            context_window=128_000,
            max_output_tokens=4096,
            max_window_tokens=cap,
        )
        assert _max_window_tokens(packed) <= min(ib, cap)

    def test_default_cap_applied_when_unresolved(self, monkeypatch):
        # No explicit max_window_tokens + no env → resolver default (6000).
        monkeypatch.delenv("EXTRACTION_MAX_WINDOW_TOKENS", raising=False)
        chunks = _chunks(60, chars=2000)
        packed = pack_chunks_for_model(
            chunks, context_window=128_000, max_output_tokens=4096
        )
        # 60 * ~500 tok / 6000 cap → ~6 windows, NOT 1 — the recall-preserving
        # default binds even on a huge-context model.
        assert len(packed) > 1
        assert _max_window_tokens(packed) <= DEFAULT_MAX_WINDOW_TOKENS

    def test_env_override_changes_window_count(self, monkeypatch):
        chunks = _chunks(60, chars=2000)
        monkeypatch.setenv("EXTRACTION_MAX_WINDOW_TOKENS", "3000")
        tight = pack_chunks_for_model(
            chunks, context_window=128_000, max_output_tokens=4096
        )
        monkeypatch.setenv("EXTRACTION_MAX_WINDOW_TOKENS", "12000")
        loose = pack_chunks_for_model(
            chunks, context_window=128_000, max_output_tokens=4096
        )
        # A tighter cap produces strictly more (smaller) windows.
        assert len(tight) > len(loose)

    def test_oversized_chunk_resplit_under_cap_no_overflow(self):
        # A single chunk larger than the CAP must still be re-split against the
        # cap (not the full input_budget) — no overflow, no drop, no infinite
        # loop. 40K chars ≈ 10000 tok > 6000 cap.
        big = [
            {
                "text": "x" * 40_000,
                "id": "chunk:big",
                "section_path": [],
                "section_level": 0,
                "physical_page": None,
                "element_type": "table",
                "source_id": "source:doc",
            }
        ]
        cap = 6000
        packed = pack_chunks_for_model(
            big,
            context_window=128_000,
            max_output_tokens=4096,
            max_window_tokens=cap,
        )
        assert len(packed) >= 2  # re-split into multiple budget-safe pieces
        assert _max_window_tokens(packed) <= cap
        # No content dropped: total chars across pieces == original.
        assert sum(len(w["text"]) for w in packed) == 40_000

    def test_capped_small_window_passes_pass2_guard_trivially(self):
        # The Pass-2 guard budget stays based on the FULL context; a window
        # capped to 6000 tok is far under ctx - max_output, so the guard never
        # trips. Proves the cap doesn't interact with pass2_token_cap.
        cap = 6000
        ctx, out = 128_000, 4096
        chunks = _chunks(40, chars=2000)
        packed = pack_chunks_for_model(
            chunks,
            context_window=ctx,
            max_output_tokens=out,
            max_window_tokens=cap,
        )
        guard = pass2_token_cap(context_window=ctx, max_output_tokens=out)
        assert guard == ctx - out  # unchanged by the cap
        # Every capped window (+ a generous ontology block) clears the guard.
        assert _max_window_tokens(packed) <= cap
        assert cap + 2000 < guard


class TestDeterminism:
    def test_same_input_same_windows(self):
        chunks = _chunks(20)
        a = pack_chunks_for_model(chunks, context_window=32_000, max_output_tokens=2048)
        b = pack_chunks_for_model(chunks, context_window=32_000, max_output_tokens=2048)
        assert [w["text"] for w in a] == [w["text"] for w in b]
        assert [w["constituent_chunk_ids"] for w in a] == [
            w["constituent_chunk_ids"] for w in b
        ]
