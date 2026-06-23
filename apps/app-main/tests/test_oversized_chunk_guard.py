"""Track M.4 — oversized-chunk guard for the small-context failover fallback.

Two layers of the interim guard (Decision M-D3 (b)):

  * The packer re-splits / re-packs a document for a SMALL candidate so no
    window exceeds that candidate's input budget — the same document packed for
    a 256K primary makes few big windows, re-packed for an 8K fallback makes
    many bounded ones.
  * ``call_candidate`` requests ``num_ctx`` = the local candidate's
    (conservative) context_window so an Ollama fallback allocates the window the
    packer sized against instead of its ~2-4K default.
"""

from app_main.services.extraction_chunking.context_packer import (
    estimate_tokens,
    input_budget_tokens,
    pack_chunks_for_model,
)
from app_main.services.model_routing.llm_call import _ollama_num_ctx_for
from app_main.services.model_routing.route_resolver import ModelCandidate
from shared.models.llm import Model


def _doc(n: int, chars: int = 2000) -> list:
    return [
        {"text": "word " * (chars // 5), "id": f"chunk:{i}", "source_id": "source:d"}
        for i in range(n)
    ]


class TestPackerResplitsForSmallFallback:
    def test_small_candidate_no_window_overflows(self):
        chunks = _doc(40)
        # The pack a 256K primary would build.
        primary = pack_chunks_for_model(
            chunks, context_window=256_000, max_output_tokens=4096
        )
        # Re-packed for the 8K local fallback.
        fallback = pack_chunks_for_model(
            chunks, context_window=8192, max_output_tokens=512
        )
        fb_budget = input_budget_tokens(
            context_window=8192, max_output_tokens=512
        )
        # The crux invariant: no fallback window exceeds the 8K budget.
        assert max(estimate_tokens(w["text"]) for w in fallback) <= fb_budget
        # And the re-pack genuinely happened — many more, smaller windows.
        assert len(fallback) > len(primary)

    def test_head_sized_window_would_overflow_small_fallback(self):
        # A single primary-sized window's token estimate vs the fallback budget
        # — this is the overflow the live bug hit before re-packing.
        chunks = _doc(40)
        primary = pack_chunks_for_model(
            chunks, context_window=256_000, max_output_tokens=4096
        )
        fb_budget = input_budget_tokens(context_window=8192, max_output_tokens=512)
        assert estimate_tokens(primary[0]["text"]) > fb_budget


class TestOllamaNumCtxGuard:
    def test_local_candidate_requests_its_context_window(self):
        model = Model(
            name="llama3.1:8b",
            provider="ollama",
            type="language",
            is_local=True,
            context_window=8192,
        )
        cand = ModelCandidate(
            provider="ollama", model_id="llama3.1:8b", is_local=True, model=model
        )
        assert _ollama_num_ctx_for(cand) == 8192

    def test_cloud_candidate_returns_none(self):
        model = Model(
            name="mistral", provider="nvidia", type="language", context_window=128000
        )
        cand = ModelCandidate(
            provider="nvidia", model_id="mistral", is_local=False, model=model
        )
        assert _ollama_num_ctx_for(cand) is None

    def test_local_without_context_window_returns_none(self):
        model = Model(name="llama", provider="ollama", type="language", is_local=True)
        cand = ModelCandidate(
            provider="ollama", model_id="llama", is_local=True, model=model
        )
        assert _ollama_num_ctx_for(cand) is None
