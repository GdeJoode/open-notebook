"""Tests for the orphan-connector module (B.5a).

The tests cover every public surface of
``entity_filtering.resolution.orphan_connector``:

1. :func:`find_orphans` returns only entities with zero relations.
2. :func:`propose_connections` emits one proposal per orphan-partner
   pair, with multi-chunk dedup so the same pair only appears once.
3. :func:`confirm_connections` upgrades LLM-confirmed proposals to
   :class:`shared.models.extraction.ExtractedRelation` records, drops
   denials and respects ``min_confidence``.
4. Token-budget violations raise :class:`OrphanTokenBudgetExceeded`
   *before* the LLM call (no LLM invocation on the over-budget path).
5. Empty inputs short-circuit cleanly with no LLM calls.
6. Malformed LLM JSON degrades to ``None`` per proposal with a
   WARNING log — never raises.

All tests are async because the public API is async. The LLM caller
is a thin in-memory stub (mirrors the Pass-1/Pass-2 DI seam).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from entity_filtering.resolution.orphan_connector import (
    TOKEN_BUDGET_TARGET,
    OrphanProposal,
    OrphanTokenBudgetExceeded,
    _clamp_confidence,
    _estimate_tokens,
    _parse_confirm_response,
    confirm_connections,
    find_orphans,
    propose_connections,
    run,
)
from entity_filtering.resolution.orphan_prompts import (
    ORPHAN_CONFIRM_SYSTEM_PROMPT,
    build_orphan_confirm_prompt,
)


# ---------------------------------------------------------------------------
# Mocks / fixtures
# ---------------------------------------------------------------------------


class MockOrphanRepo:
    """In-memory mock implementing :class:`OrphanEntityRepoProtocol`."""

    def __init__(self, orphans_by_source: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        self.orphans_by_source = orphans_by_source or {}
        self.call_count = 0

    async def list_orphans_for_source(
        self, source_id: str
    ) -> List[Dict[str, Any]]:
        self.call_count += 1
        return list(self.orphans_by_source.get(source_id, []))


class RecordingLLM:
    """Async LLM stub that records every invocation and returns a queued reply."""

    def __init__(self, responses: List[str] | str):
        if isinstance(responses, str):
            self._responses: List[str] = [responses]
            self._sticky = True
        else:
            self._responses = list(responses)
            self._sticky = False
        self.calls: List[Dict[str, str]] = []

    async def __call__(self, system_prompt: str, user_prompt: str, model: str) -> str:
        self.calls.append(
            {
                "system": system_prompt,
                "user": user_prompt,
                "model": model,
            }
        )
        if self._sticky:
            return self._responses[0]
        if not self._responses:
            return "{}"
        return self._responses.pop(0)


def _entity_row(
    id_: str,
    name: str,
    entity_type: str = "MISC",
) -> Dict[str, Any]:
    """Compact entity row mimicking the repository surface."""
    return {
        "id": id_,
        "canonical_name": name,
        "entity_type": entity_type,
        "source_documents": ["source:demo"],
        "properties": {},
    }


def _chunk(
    chunk_id: str,
    text: str,
    entity_texts: List[str],
) -> Dict[str, Any]:
    """Compact chunk dict with entity-text list."""
    return {
        "id": chunk_id,
        "text": text,
        "entities": list(entity_texts),
    }


def _ok_response(rel_type: str = "WORKS_AT", confidence: float = 0.82) -> str:
    return json.dumps(
        {"relation_type": rel_type, "confidence": confidence}
    )


def _deny_response() -> str:
    return json.dumps({"relation_type": None, "confidence": 0.9})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    """Coverage for the small private helpers — keeps the main flow tests focused."""

    def test_estimate_tokens_div_by_four(self):
        assert _estimate_tokens("abcd" * 100) == 100

    def test_clamp_confidence_handles_percent(self):
        assert _clamp_confidence(87) == pytest.approx(0.87)

    def test_clamp_confidence_handles_string(self):
        assert _clamp_confidence("not a number") == 0.0

    def test_clamp_confidence_below_zero(self):
        assert _clamp_confidence(-0.5) == 0.0

    def test_clamp_confidence_above_one(self):
        assert _clamp_confidence(1.4) == 1.0

    def test_clamp_confidence_none(self):
        assert _clamp_confidence(None) == 0.0


# ---------------------------------------------------------------------------
# Stage 1: find_orphans
# ---------------------------------------------------------------------------


class TestFindOrphans:
    """Acceptance criterion #1."""

    async def test_returns_only_orphans(self):
        """5 entities, 2 orphans → find_orphans returns exactly 2."""
        repo = MockOrphanRepo(
            {
                "source:demo": [
                    _entity_row("entity:o1", "Alice"),
                    _entity_row("entity:o2", "Bob"),
                ]
            }
        )
        orphans = await find_orphans("source:demo", repo)
        assert len(orphans) == 2
        names = sorted(e["canonical_name"] for e in orphans)
        assert names == ["Alice", "Bob"]

    async def test_empty_source_id_returns_empty(self):
        repo = MockOrphanRepo()
        orphans = await find_orphans("", repo)
        assert orphans == []
        assert repo.call_count == 0

    async def test_unknown_source_returns_empty(self):
        repo = MockOrphanRepo()
        orphans = await find_orphans("source:nonexistent", repo)
        assert orphans == []


# ---------------------------------------------------------------------------
# Stage 2: propose_connections
# ---------------------------------------------------------------------------


class TestProposeConnections:
    """Acceptance criterion #2 + multi-chunk dedup behaviour."""

    async def test_simple_cooccurrence(self):
        """2 orphans co-occurring with another entity in 1 chunk → 1 proposal each.

        Each orphan emits a proposal for every other chunk entity. Since
        relations are directed, Alice→Bob and Bob→Alice are distinct
        proposals (the LLM may legitimately confirm a relation in one
        direction and deny it in the other). The (orphan_norm,
        partner_norm) dedup only stops the same orphan from generating
        the same partner twice — not from generating the inverse pair.
        """
        orphans = [
            {"text": "Alice"},
            {"text": "Bob"},
        ]
        chunks = [
            _chunk(
                "c-1",
                "Alice and Bob both work for MIT.",
                ["Alice", "Bob", "MIT"],
            ),
        ]
        proposals = await propose_connections(orphans, chunks)
        # Expect: Alice→Bob, Alice→MIT, Bob→Alice, Bob→MIT.
        by_orphan: Dict[str, List[OrphanProposal]] = {}
        for p in proposals:
            by_orphan.setdefault(p.orphan, []).append(p)
        assert "Alice" in by_orphan
        assert "Bob" in by_orphan
        # Each orphan emits one proposal per *other* entity (no self
        # pair, no duplicate partners for the same orphan).
        assert len(by_orphan["Alice"]) == 2
        assert len(by_orphan["Bob"]) == 2
        # All directed pairs are unique (no duplicates per orphan).
        directed_pairs = {
            (p.orphan, p.candidate_partner) for p in proposals
        }
        assert len(directed_pairs) == len(proposals)

    async def test_multi_chunk_same_pair_dedups(self):
        """A pair that co-occurs in two chunks is proposed only once."""
        orphans = [{"text": "Alice"}]
        chunks = [
            _chunk("c-1", "Alice and Bob met.", ["Alice", "Bob"]),
            _chunk(
                "c-2", "Alice met Bob again later.", ["Alice", "Bob"]
            ),
        ]
        proposals = await propose_connections(orphans, chunks)
        assert len(proposals) == 1
        assert proposals[0].orphan == "Alice"
        assert proposals[0].candidate_partner == "Bob"
        # The first chunk wins as the evidence carrier — deterministic
        # for downstream test fixtures.
        assert proposals[0].evidence_chunk_id == "c-1"

    async def test_max_proposals_per_orphan_respected(self):
        """The cap stops emission once the budget is exhausted."""
        orphans = [{"text": "Alice"}]
        partners = [f"Partner{i}" for i in range(10)]
        chunks = [_chunk("c-1", "Alice met many partners.", ["Alice"] + partners)]
        proposals = await propose_connections(
            orphans, chunks, max_proposals_per_orphan=3
        )
        assert len(proposals) == 3
        for p in proposals:
            assert p.orphan == "Alice"
            assert p.candidate_partner.startswith("Partner")

    async def test_orphan_not_in_any_chunk(self):
        """Orphans with no chunk co-occurrence yield no proposals."""
        orphans = [{"text": "Carol"}]
        chunks = [_chunk("c-1", "Alice met Bob.", ["Alice", "Bob"])]
        proposals = await propose_connections(orphans, chunks)
        assert proposals == []

    async def test_normalization_matches_case_variants(self):
        """The name normaliser lets ``alice`` match ``Alice``.

        When an orphan matches a chunk entity by normalised form, the
        chunk's surface form becomes the orphan label in the proposal —
        that way the evidence text and the orphan label agree, which
        helps the LLM ground the answer.
        """
        orphans = [{"text": "alice "}]  # whitespace + lowercase
        chunks = [_chunk("c-1", "Alice met Bob.", ["Alice", "Bob"])]
        proposals = await propose_connections(orphans, chunks)
        assert len(proposals) == 1
        assert proposals[0].orphan == "Alice"
        assert proposals[0].candidate_partner == "Bob"

    async def test_accepts_extractedentity_orphans(self):
        """Orphans can be ExtractedEntity instances, not just dicts."""
        from shared.models.extraction import ExtractedEntity

        orphans = [ExtractedEntity(text="Alice", label="PERSON")]
        chunks = [_chunk("c-1", "Alice met Bob.", ["Alice", "Bob"])]
        proposals = await propose_connections(orphans, chunks)
        assert len(proposals) == 1
        assert proposals[0].orphan == "Alice"

    async def test_empty_orphan_list_returns_empty(self):
        """Acceptance criterion #6 — empty orphans short-circuit."""
        proposals = await propose_connections([], [])
        assert proposals == []

    async def test_empty_chunk_entities_skipped(self):
        orphans = [{"text": "Alice"}]
        chunks = [_chunk("c-1", "Alice met someone.", [])]
        proposals = await propose_connections(orphans, chunks)
        assert proposals == []

    async def test_self_pair_never_proposed(self):
        """Minor-6 fix (review attempt 1): an orphan must never be paired
        with itself even when the upstream chunk lists the same surface
        form twice (e.g. anaphora-resolution duplication).

        The dedup guard inside ``propose_connections`` keys on the
        normalised name, so case- and whitespace-variants of the same
        entity should also be rejected as self-pairs.
        """
        orphans = [{"text": "Alice"}]
        chunks = [
            _chunk(
                "c-1",
                "Alice spoke about Alice and met Bob.",
                # Duplicate Alice + a whitespace variant + Bob — only
                # Alice→Bob should survive.
                ["Alice", "Alice", "alice ", "Bob"],
            )
        ]
        proposals = await propose_connections(orphans, chunks)
        # Exactly one proposal (Alice -> Bob). No Alice -> Alice and no
        # Alice -> "alice " (whitespace variant).
        assert len(proposals) == 1
        assert proposals[0].orphan == "Alice"
        assert proposals[0].candidate_partner == "Bob"


# ---------------------------------------------------------------------------
# Stage 3: confirm_connections
# ---------------------------------------------------------------------------


class TestConfirmConnections:
    """Acceptance criteria #3, #4, #5, #7."""

    async def test_llm_confirms_all_proposals(self):
        """AC #3 — LLM confirms all → ExtractedRelations with confidence ≥ threshold."""
        proposals = [
            OrphanProposal("Alice", "Bob", "c-1", "Alice and Bob met."),
            OrphanProposal("Carol", "Dan", "c-2", "Carol works with Dan."),
        ]
        llm = RecordingLLM(_ok_response("KNOWS", 0.85))
        relations = await confirm_connections(
            proposals, llm, min_confidence=0.6
        )
        assert len(relations) == 2
        for rel in relations:
            assert rel.relation_type == "KNOWS"
            assert rel.confidence >= 0.6
            assert rel.properties.get("extraction_method") == "orphan_connector"
        assert len(llm.calls) == 2

    async def test_llm_denies_all_yields_no_relations(self):
        """AC #4 — relation_type=null → 0 relations."""
        proposals = [
            OrphanProposal("Alice", "Bob", "c-1", "Alice ate. Bob slept."),
        ]
        llm = RecordingLLM(_deny_response())
        relations = await confirm_connections(proposals, llm)
        assert relations == []
        # The LLM was still called — denial only happens after parsing.
        assert len(llm.calls) == 1

    async def test_low_confidence_filtered_out(self):
        proposals = [
            OrphanProposal("Alice", "Bob", "c-1", "Alice and Bob met."),
        ]
        llm = RecordingLLM(_ok_response("KNOWS", confidence=0.4))
        relations = await confirm_connections(
            proposals, llm, min_confidence=0.6
        )
        assert relations == []

    async def test_token_budget_exceeded_raises_before_call(self):
        """AC #5 — over-budget prompt raises BEFORE LLM call."""
        # 1500 tokens × 4 chars/token ≈ 6000 chars. Use 8000 to stay
        # well beyond the cap regardless of the small prompt scaffold.
        huge_text = "x" * 8000
        proposals = [
            OrphanProposal("Alice", "Bob", "c-1", huge_text),
        ]
        llm = RecordingLLM(_ok_response())
        with pytest.raises(OrphanTokenBudgetExceeded) as exc:
            await confirm_connections(proposals, llm)
        assert exc.value.budget == TOKEN_BUDGET_TARGET
        assert exc.value.estimated > TOKEN_BUDGET_TARGET
        # LLM was never invoked.
        assert llm.calls == []

    async def test_malformed_json_degrades_gracefully(self):
        """AC #7 — malformed JSON drops proposal, never raises."""
        proposals = [
            OrphanProposal("Alice", "Bob", "c-1", "Alice met Bob."),
            OrphanProposal("Carol", "Dan", "c-2", "Carol joined Dan."),
        ]
        llm = RecordingLLM(
            [
                "{not valid json,",  # malformed
                _ok_response("KNOWS", 0.85),  # valid
            ]
        )
        relations = await confirm_connections(proposals, llm)
        assert len(relations) == 1
        assert relations[0].source_entity == "Carol"
        # Both proposals went to the LLM — the malformed one wasn't
        # short-circuited.
        assert len(llm.calls) == 2

    async def test_non_object_top_level_dropped(self):
        proposals = [
            OrphanProposal("Alice", "Bob", "c-1", "Alice met Bob."),
        ]
        llm = RecordingLLM("[]")
        relations = await confirm_connections(proposals, llm)
        assert relations == []

    async def test_relation_type_non_string_dropped(self):
        proposals = [
            OrphanProposal("Alice", "Bob", "c-1", "Alice met Bob."),
        ]
        llm = RecordingLLM(json.dumps({"relation_type": 123, "confidence": 0.9}))
        relations = await confirm_connections(proposals, llm)
        assert relations == []

    async def test_empty_string_relation_type_treated_as_deny(self):
        proposals = [
            OrphanProposal("Alice", "Bob", "c-1", "Alice met Bob."),
        ]
        llm = RecordingLLM(json.dumps({"relation_type": "  ", "confidence": 0.9}))
        relations = await confirm_connections(proposals, llm)
        assert relations == []

    async def test_code_fence_unwrapped(self):
        proposals = [
            OrphanProposal("Alice", "Bob", "c-1", "Alice met Bob."),
        ]
        wrapped = (
            "```json\n"
            + json.dumps({"relation_type": "MET", "confidence": 0.7})
            + "\n```"
        )
        llm = RecordingLLM(wrapped)
        relations = await confirm_connections(proposals, llm)
        assert len(relations) == 1
        assert relations[0].relation_type == "MET"

    async def test_prose_with_embedded_json_recovered(self):
        proposals = [
            OrphanProposal("Alice", "Bob", "c-1", "Alice met Bob."),
        ]
        prose = (
            'Sure, here is my answer: {"relation_type": "MET", '
            '"confidence": 0.66}. Hope that helps!'
        )
        llm = RecordingLLM(prose)
        relations = await confirm_connections(proposals, llm)
        assert len(relations) == 1
        assert relations[0].relation_type == "MET"

    async def test_empty_response_dropped(self):
        proposals = [
            OrphanProposal("Alice", "Bob", "c-1", "Alice met Bob."),
        ]
        llm = RecordingLLM("")
        relations = await confirm_connections(proposals, llm)
        assert relations == []

    async def test_sync_callable_supported(self):
        """The LLM caller may be sync (matches Pass-2's contract)."""
        proposals = [
            OrphanProposal("Alice", "Bob", "c-1", "Alice met Bob."),
        ]
        calls: List[str] = []

        def sync_caller(system: str, user: str, model: str) -> str:
            calls.append(user)
            return _ok_response("MET", 0.9)

        relations = await confirm_connections(proposals, sync_caller)
        assert len(relations) == 1
        assert calls  # synchronous path executed

    async def test_llm_transport_exception_propagates(self):
        """Transport-level failures escape so the caller can branch."""
        proposals = [
            OrphanProposal("Alice", "Bob", "c-1", "Alice met Bob."),
        ]

        async def boom(system: str, user: str, model: str) -> str:
            raise RuntimeError("connection refused")

        with pytest.raises(RuntimeError, match="connection refused"):
            await confirm_connections(proposals, boom)


# ---------------------------------------------------------------------------
# Stage 6: empty orphans short-circuit
# ---------------------------------------------------------------------------


class TestEmptyOrphansShortCircuit:
    """AC #6 — empty orphans list → empty proposals, no LLM calls."""

    async def test_empty_orphans_no_llm_calls(self):
        llm = RecordingLLM("UNREACHED")
        proposals = await propose_connections([], [])
        assert proposals == []
        relations = await confirm_connections(proposals, llm)
        assert relations == []
        assert llm.calls == []


# ---------------------------------------------------------------------------
# Top-level run() orchestration
# ---------------------------------------------------------------------------


class TestRun:
    """End-to-end happy/sad paths for the one-shot ``run`` helper."""

    async def test_happy_path(self):
        repo = MockOrphanRepo(
            {"source:demo": [_entity_row("entity:o1", "Alice")]}
        )
        chunks = [_chunk("c-1", "Alice and Bob met.", ["Alice", "Bob"])]
        llm = RecordingLLM(_ok_response("MET", 0.9))

        relations = await run(
            "source:demo", chunks, repo, llm, min_confidence=0.6
        )
        assert len(relations) == 1
        assert relations[0].source_entity == "Alice"
        assert relations[0].target_entity == "Bob"
        assert relations[0].relation_type == "MET"
        assert relations[0].source_chunk_id == "c-1"

    async def test_no_orphans_short_circuits(self):
        repo = MockOrphanRepo({})
        chunks = [_chunk("c-1", "x", ["A", "B"])]
        llm = RecordingLLM("UNREACHED")
        relations = await run("source:demo", chunks, repo, llm)
        assert relations == []
        assert llm.calls == []

    async def test_no_proposals_short_circuits(self):
        repo = MockOrphanRepo(
            {"source:demo": [_entity_row("entity:o1", "Alice")]}
        )
        # Alice doesn't co-occur with anyone.
        chunks = [_chunk("c-1", "Bob met Carol.", ["Bob", "Carol"])]
        llm = RecordingLLM("UNREACHED")
        relations = await run("source:demo", chunks, repo, llm)
        assert relations == []
        assert llm.calls == []


# ---------------------------------------------------------------------------
# B.5b production wiring: mark_pending_reconnect after run() (attempt 2 B1)
# ---------------------------------------------------------------------------


class _PruneRepoMock:
    """Mock implementing BOTH the connector and prune surfaces.

    Captures every ``update_orphan_status`` call so the integration test
    can assert the lifecycle write happened with the right shape. Models
    just enough state to make the assertions exact.
    """

    def __init__(self, orphans_by_source):
        self._orphans_by_source = dict(orphans_by_source)
        # Mirror the entity rows back as a mutable map so the recorded
        # writes can be checked against the resulting state.
        self.entities = {
            row["id"]: dict(row)
            for rows in orphans_by_source.values()
            for row in rows
        }
        self.update_calls = []

    async def list_orphans_for_source(self, source_id):
        return [dict(row) for row in self._orphans_by_source.get(source_id, [])]

    async def update_orphan_status(
        self,
        entity_id,
        status,
        *,
        increment_attempts=False,
        set_first_orphaned_at=False,
        set_last_reconnect_attempt_at=False,
    ):
        self.update_calls.append(
            {
                "entity_id": entity_id,
                "status": status,
                "increment_attempts": increment_attempts,
                "set_first_orphaned_at": set_first_orphaned_at,
                "set_last_reconnect_attempt_at": set_last_reconnect_attempt_at,
            }
        )
        row = self.entities.get(entity_id)
        if row is None:
            return False
        row["orphan_status"] = status
        if increment_attempts:
            row["reconnect_attempts"] = (
                int(row.get("reconnect_attempts") or 0) + 1
            )
        return True


class TestRunMarksPendingReconnect:
    """B1 (attempt 2): wire ``mark_pending_reconnect`` into ``run``.

    The reviewer rejected attempt 1 because the lifecycle had no
    production caller. These tests pin the new contract:

    1. When the connector confirms ZERO relations for an orphan, the
       orphan is flipped to ``pending_reconnect`` with attempts=1.
    2. When the connector confirms a relation that references the
       orphan, the orphan is NOT marked pending (it has been
       reconciled).
    3. When the repository does not implement the prune surface (i.e.
       a stripped-down B.5a-only mock), the mark step is a silent
       no-op so existing tests stay green.
    """

    async def test_marks_orphan_pending_when_no_relations_confirmed(self):
        """Orphan with NO co-occurring partner -> pending_reconnect."""
        repo = _PruneRepoMock(
            {"source:demo": [_entity_row("entity:o1", "Alice")]}
        )
        # Alice doesn't co-occur with anyone -> 0 proposals -> 0 relations.
        chunks = [_chunk("c-1", "Bob met Carol.", ["Bob", "Carol"])]
        llm = RecordingLLM("UNREACHED")

        relations = await run("source:demo", chunks, repo, llm)

        assert relations == []
        # Exactly one update call for entity:o1, flipping to pending.
        assert len(repo.update_calls) == 1
        call = repo.update_calls[0]
        assert call["entity_id"] == "entity:o1"
        assert call["status"] == "pending_reconnect"
        assert call["increment_attempts"] is True
        assert call["set_first_orphaned_at"] is True
        assert call["set_last_reconnect_attempt_at"] is True
        # Counter advanced to 1 on the row itself.
        assert repo.entities["entity:o1"]["reconnect_attempts"] == 1
        assert (
            repo.entities["entity:o1"]["orphan_status"] == "pending_reconnect"
        )

    async def test_does_not_mark_when_orphan_reconciled(self):
        """Orphan that DID gain a relation stays unflagged.

        ``confirm_connections`` returns a relation referencing Alice;
        the mark step must skip Alice and leave the lifecycle alone.
        """
        repo = _PruneRepoMock(
            {"source:demo": [_entity_row("entity:o1", "Alice")]}
        )
        chunks = [_chunk("c-1", "Alice and Bob met.", ["Alice", "Bob"])]
        llm = RecordingLLM(_ok_response("MET", 0.9))

        relations = await run("source:demo", chunks, repo, llm)

        assert len(relations) == 1
        # NO update_orphan_status call for the reconciled orphan.
        assert repo.update_calls == []
        # Lifecycle untouched.
        assert (
            repo.entities["entity:o1"].get("orphan_status")
            in (None, "")
        )

    async def test_marks_unreconciled_subset_when_mixed(self):
        """Two orphans, one reconciled, one not -> exactly one mark.

        Confirms the mark step targets only the orphan(s) that did NOT
        gain any relation, not the whole orphan set indiscriminately.
        """
        repo = _PruneRepoMock(
            {
                "source:demo": [
                    _entity_row("entity:o1", "Alice"),
                    _entity_row("entity:o2", "Carol"),
                ]
            }
        )
        # Alice co-occurs with Bob; Carol has no partner in the chunk.
        chunks = [
            _chunk("c-1", "Alice and Bob met.", ["Alice", "Bob"]),
        ]
        llm = RecordingLLM(_ok_response("KNOWS", 0.9))

        relations = await run("source:demo", chunks, repo, llm)

        assert len(relations) == 1
        # Only Carol gets marked pending.
        assert len(repo.update_calls) == 1
        assert repo.update_calls[0]["entity_id"] == "entity:o2"
        assert repo.update_calls[0]["status"] == "pending_reconnect"

    async def test_no_pending_mark_when_repo_lacks_prune_surface(self):
        """Connector-only mocks (MockOrphanRepo) silently skip the mark.

        This is the backward-compat hatch: the B.5a contract on
        ``OrphanEntityRepoProtocol`` does not require
        ``update_orphan_status``, and many older tests inject mocks
        that don't have it. The wire step must duck-type and skip.
        """
        repo = MockOrphanRepo(
            {"source:demo": [_entity_row("entity:o1", "Alice")]}
        )
        chunks = [_chunk("c-1", "Bob met Carol.", ["Bob", "Carol"])]
        llm = RecordingLLM("UNREACHED")

        # Should not raise even though the repo lacks update_orphan_status.
        relations = await run("source:demo", chunks, repo, llm)

        assert relations == []
        # The duck-type gate fires; no AttributeError.
        assert not hasattr(repo, "update_orphan_status")

    async def test_mark_step_failure_does_not_lose_confirmed_relations(self):
        """A failing update_orphan_status must not drop the run's output.

        The relations we DID confirm are valuable; an orphan-prune
        hiccup is not a reason to lose them.
        """

        class _BrokenPruneRepo(_PruneRepoMock):
            async def update_orphan_status(self, *args, **kwargs):
                raise RuntimeError("simulated DB blip")

        repo = _BrokenPruneRepo(
            {
                "source:demo": [
                    _entity_row("entity:o1", "Alice"),
                    _entity_row("entity:o2", "Carol"),  # un-reconciled
                ]
            }
        )
        chunks = [_chunk("c-1", "Alice and Bob met.", ["Alice", "Bob"])]
        llm = RecordingLLM(_ok_response("KNOWS", 0.9))

        relations = await run("source:demo", chunks, repo, llm)

        # The confirmed relation survives the mark-step failure.
        assert len(relations) == 1
        assert relations[0].source_entity == "Alice"


# ---------------------------------------------------------------------------
# Prompt + response shape
# ---------------------------------------------------------------------------


class TestPromptShape:
    """Sanity checks on the prompt builder + response parser pair."""

    def test_system_prompt_pins_json_contract(self):
        assert "strict JSON" in ORPHAN_CONFIRM_SYSTEM_PROMPT
        assert "null" in ORPHAN_CONFIRM_SYSTEM_PROMPT
        assert "confidence" in ORPHAN_CONFIRM_SYSTEM_PROMPT

    def test_user_prompt_renders_entities_and_chunk(self):
        prompt = build_orphan_confirm_prompt("Alice", "Bob", "Alice and Bob met.")
        assert "Entity A: Alice" in prompt
        assert "Entity B: Bob" in prompt
        assert "Alice and Bob met." in prompt
        assert "Output Format" in prompt
        # The example JSON shape is part of the contract.
        assert '"relation_type"' in prompt
        assert '"confidence"' in prompt

    def test_parse_confirm_valid(self):
        out = _parse_confirm_response(json.dumps(
            {"relation_type": "WORKS_AT", "confidence": 0.9}
        ))
        assert out == {"relation_type": "WORKS_AT", "confidence": 0.9}

    def test_parse_confirm_denial_returns_none(self):
        out = _parse_confirm_response(json.dumps(
            {"relation_type": None, "confidence": 0.9}
        ))
        assert out is None
