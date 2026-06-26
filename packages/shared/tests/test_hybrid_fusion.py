"""Unit tests for the pure hybrid-fusion ranker (Track R.3).

No DB. These assert the RRF fusion contract directly on constructed inputs:
the RRF math, missing-signal handling (a one-signal source still ranks),
weight tuning provably reordering results, the per-result provenance, and the
**ablation** proof — on a case where the two signals DISAGREE, KG-only,
dense-only, and fused produce demonstrably different orders, so neither signal
is dead weight.
"""

from __future__ import annotations

from shared.retrieval.hybrid_fusion import (
    BALANCED,
    DEFAULT_RRF_K,
    KG_HEAVY,
    PRESETS,
    FusionWeights,
    fuse_rankings,
    get_preset,
)


def _dense(*pairs):
    """Build a dense signal list ``[(id, cosine), ...]`` best-first."""
    return list(pairs)


def _kg(*triples):
    """Build a KG signal list ``[(id, score, explanation), ...]`` best-first."""
    return [(sid, score, expl) for sid, score, expl in triples]


def _expl(name, etype="programme", weight=2.0, df=2, eid="entity:x", via=False):
    return [
        {
            "entity_id": eid,
            "name": name,
            "type": etype,
            "document_frequency": df,
            "weight": weight,
            "via_relation": via,
        }
    ]


def _order(results):
    return [r.source_id for r in results]


# ---------------------------------------------------------------------------
# RRF math
# ---------------------------------------------------------------------------
class TestRRFMath:
    def test_single_signal_contribution_is_weight_over_k_plus_rank(self):
        """A source present only in dense scores exactly w/(k+rank)."""
        w = FusionWeights(dense=2.0, kg=1.0, rrf_k=60)
        results = fuse_rankings(_dense(("source:a", 0.9)), _kg(), weights=w)
        assert len(results) == 1
        r = results[0]
        # rank 1, dense weight 2.0, k 60 -> 2/(60+1)
        assert abs(r.fused_score - 2.0 / 61.0) < 1e-12
        assert abs(r.dense.contribution - 2.0 / 61.0) < 1e-12
        assert r.dense.rank == 1
        assert r.kg.contribution == 0.0
        assert r.kg.present is False

    def test_two_signal_source_sums_both_terms(self):
        """A source in both signals sums the two RRF terms."""
        w = FusionWeights(dense=1.0, kg=1.0, rrf_k=60)
        results = fuse_rankings(
            _dense(("source:a", 0.8)),
            _kg(("source:a", 3.0, _expl("Regio Deal"))),
            weights=w,
        )
        r = results[0]
        expected = 1.0 / 61.0 + 1.0 / 61.0
        assert abs(r.fused_score - expected) < 1e-12
        assert r.dense.present and r.kg.present

    def test_default_weights_are_kg_heavy(self):
        """The default preset is KG-prominent (kg weight >= dense)."""
        assert KG_HEAVY.kg_prominent
        assert BALANCED.kg_prominent
        assert KG_HEAVY.kg > KG_HEAVY.dense
        assert DEFAULT_RRF_K == 60

    def test_deterministic_tiebreak_on_equal_fused_score(self):
        """Equal fused score -> ascending source_id tie-break."""
        # Both sources rank 1 in exactly one signal with equal weight -> tie.
        results = fuse_rankings(
            _dense(("source:b", 0.9)),
            _kg(("source:a", 3.0, [])),
            weights=BALANCED,
        )
        # source:a and source:b both have fused = 1/(60+1); a < b
        assert _order(results) == ["source:a", "source:b"]


# ---------------------------------------------------------------------------
# Missing-signal handling (AC4)
# ---------------------------------------------------------------------------
class TestMissingSignal:
    def test_source_in_only_one_signal_still_appears(self):
        """AC4: a source present in only one signal is NOT dropped."""
        results = fuse_rankings(
            _dense(("source:dense_only", 0.95)),
            _kg(("source:kg_only", 4.0, _expl("X"))),
            weights=KG_HEAVY,
        )
        ids = set(_order(results))
        assert "source:dense_only" in ids
        assert "source:kg_only" in ids

    def test_union_of_both_signals_returned(self):
        results = fuse_rankings(
            _dense(("source:a", 0.9), ("source:b", 0.5)),
            _kg(("source:b", 3.0, []), ("source:c", 2.0, [])),
            weights=BALANCED,
        )
        assert set(_order(results)) == {"source:a", "source:b", "source:c"}

    def test_two_signal_source_beats_equal_rank_one_signal_source(self):
        """Corroboration bias: in both > in one, at the same rank."""
        # source:both is rank 1 in BOTH; source:dense is rank 1 in dense only.
        results = fuse_rankings(
            _dense(("source:both", 0.9), ("source:dense", 0.8)),
            _kg(("source:both", 3.0, [])),
            weights=BALANCED,
        )
        assert _order(results)[0] == "source:both"

    def test_absent_signal_contributes_zero_not_phantom_rank(self):
        results = fuse_rankings(
            _dense(("source:a", 0.9)),
            _kg(),
            weights=KG_HEAVY,
        )
        assert results[0].kg.contribution == 0.0
        assert results[0].kg.score is None
        assert results[0].kg.rank is None


# ---------------------------------------------------------------------------
# Weight tuning provably reorders (AC3)
# ---------------------------------------------------------------------------
class TestWeightTuning:
    def _disagreement_inputs(self):
        """Dense favours A; KG favours B (the signals disagree)."""
        dense = _dense(("source:A", 0.95), ("source:B", 0.40))
        kg = _kg(
            ("source:B", 4.5, _expl("Regio Deal Noord")),
            ("source:A", 2.6, _expl("gezondheid", etype="topic", weight=0.3)),
        )
        return dense, kg

    def test_kg_heavy_promotes_kg_favoured_source(self):
        """AC3: bumping the KG weight provably reorders the result."""
        dense, kg = self._disagreement_inputs()

        balanced = fuse_rankings(dense, kg, weights=BALANCED)
        kg_heavy = fuse_rankings(dense, kg, weights=KG_HEAVY)

        # Balanced: A is rank1-dense + rank2-kg; B is rank2-dense + rank1-kg.
        # With equal weights these tie on fused score (symmetric ranks) -> A
        # wins the id tie-break. Heavy KG must flip B above A.
        assert _order(balanced)[0] == "source:A"
        assert _order(kg_heavy)[0] == "source:B"
        assert _order(balanced) != _order(kg_heavy)

    def test_extreme_dense_weight_promotes_dense_favoured_source(self):
        """Symmetric proof: a dense-heavy weight reorders the other way."""
        dense, kg = self._disagreement_inputs()
        dense_heavy = FusionWeights(dense=5.0, kg=1.0)
        results = fuse_rankings(dense, kg, weights=dense_heavy)
        assert _order(results)[0] == "source:A"

    def test_monotonic_kg_weight_sweep_changes_winner(self):
        """Sweeping kg weight up flips the winner from A to B at some point."""
        dense, kg = self._disagreement_inputs()
        winners = []
        for kg_w in (0.5, 1.0, 2.0, 4.0):
            res = fuse_rankings(
                dense, kg, weights=FusionWeights(dense=1.0, kg=kg_w)
            )
            winners.append(_order(res)[0])
        assert "source:A" in winners and "source:B" in winners


# ---------------------------------------------------------------------------
# Ablation proof (AC2) — each signal independently changes the ranking
# ---------------------------------------------------------------------------
class TestAblation:
    """Constructed disagreement case: KG-only, dense-only, fused all differ.

    Four sources, signals deliberately in tension. The orderings are an
    ASYMMETRIC permutation of each other (not a clean reversal — a clean
    reversal under equal weights collapses to a symmetric tie), chosen so the
    BALANCED fusion lands on a genuine BLEND distinct from BOTH single signals:
      * dense order:  W > X > Y > Z
      * KG order:     Y > Z > W > X
    A source's fused rank then depends on BOTH its dense and KG positions, so
    neither single signal reproduces the fused order — each one demonstrably
    moves the ranking (neither is dead weight). The KG-heavy default
    additionally lets the KG winner lead.
    """

    DENSE = [
        ("source:W", 0.95),
        ("source:X", 0.80),
        ("source:Y", 0.60),
        ("source:Z", 0.40),
    ]
    KG = [
        ("source:Y", 4.8, [{"entity_id": "e:1", "name": "Programme Y",
                            "type": "programme", "document_frequency": 2,
                            "weight": 4.8, "via_relation": False}]),
        ("source:Z", 3.0, [{"entity_id": "e:2", "name": "Org Z",
                            "type": "organization", "document_frequency": 3,
                            "weight": 3.0, "via_relation": False}]),
        ("source:W", 1.5, [{"entity_id": "e:3", "name": "topic w",
                            "type": "topic", "document_frequency": 5,
                            "weight": 1.5, "via_relation": False}]),
        ("source:X", 0.6, [{"entity_id": "e:4", "name": "other x",
                            "type": "other", "document_frequency": 6,
                            "weight": 0.6, "via_relation": False}]),
    ]

    def _dense_only_order(self):
        return [sid for sid, _ in self.DENSE]

    def _kg_only_order(self):
        return [sid for sid, _, _ in self.KG]

    def test_dense_only_and_kg_only_disagree(self):
        """Precondition: the two signals genuinely disagree on this case."""
        assert self._dense_only_order() != self._kg_only_order()

    def test_fused_differs_from_both_single_signals(self):
        """AC2: under BALANCED, fused order != dense-only AND != kg-only.

        The fused order is a genuine blend of the two disagreeing signals,
        copying neither — the cleanest demonstration that BOTH signals
        contribute to the fused order.
        """
        fused = _order(fuse_rankings(self.DENSE, self.KG, weights=BALANCED))
        assert fused != self._dense_only_order()
        assert fused != self._kg_only_order()

    def test_kg_heavy_fused_leads_with_kg_winner(self):
        """With the KG-heavy default, the KG-top source leads the fusion."""
        fused = _order(fuse_rankings(self.DENSE, self.KG, weights=KG_HEAVY))
        assert fused[0] == "source:Y"  # KG winner (rank-1 in KG)

    def test_removing_dense_changes_the_order(self):
        """Ablate dense -> order changes => dense is not dead weight."""
        with_dense = _order(
            fuse_rankings(self.DENSE, self.KG, weights=BALANCED)
        )
        without_dense = _order(
            fuse_rankings([], self.KG, weights=BALANCED)
        )
        assert with_dense != without_dense

    def test_removing_kg_changes_the_order(self):
        """Ablate KG -> order changes => KG is not dead weight."""
        with_kg = _order(fuse_rankings(self.DENSE, self.KG, weights=BALANCED))
        without_kg = _order(fuse_rankings(self.DENSE, [], weights=BALANCED))
        assert with_kg != without_kg


# ---------------------------------------------------------------------------
# Provenance (AC1)
# ---------------------------------------------------------------------------
class TestProvenance:
    def test_result_carries_per_signal_score_rank_contribution(self):
        """AC1: each result exposes dense vs KG score, rank, contribution."""
        results = fuse_rankings(
            _dense(("source:a", 0.88)),
            _kg(("source:a", 3.3, _expl("Regio Deal", weight=3.3))),
            weights=KG_HEAVY,
        )
        r = results[0]
        assert r.dense.score == 0.88
        assert r.dense.rank == 1
        assert r.dense.contribution > 0
        assert r.kg.score == 3.3
        assert r.kg.rank == 1
        assert r.kg.contribution > 0
        # contributions sum to fused score
        assert abs(
            r.fused_score - (r.dense.contribution + r.kg.contribution)
        ) < 1e-12

    def test_kg_entities_passed_through(self):
        """AC1: the KG driving-entities explanation is carried into the fusion."""
        results = fuse_rankings(
            _dense(),
            _kg(("source:a", 3.3, _expl("Regio Deal", etype="programme"))),
            weights=KG_HEAVY,
        )
        r = results[0]
        assert r.kg_entities
        assert r.kg_entities[0]["name"] == "Regio Deal"
        assert r.kg_entities[0]["type"] == "programme"

    def test_titles_hydrated_when_provided(self):
        results = fuse_rankings(
            _dense(("source:a", 0.9)),
            _kg(),
            weights=KG_HEAVY,
            titles={"source:a": "Convenant A"},
        )
        assert results[0].title == "Convenant A"

    def test_no_title_map_yields_none_titles(self):
        results = fuse_rankings(_dense(("source:a", 0.9)), _kg())
        assert results[0].title is None


# ---------------------------------------------------------------------------
# Presets + limit
# ---------------------------------------------------------------------------
class TestPresetsAndLimit:
    def test_get_preset_resolves_known_names(self):
        assert get_preset("kg-heavy") is KG_HEAVY
        assert get_preset("balanced") is BALANCED
        assert get_preset("KG-HEAVY") is KG_HEAVY  # case-insensitive

    def test_get_preset_defaults_to_kg_heavy(self):
        assert get_preset(None) is KG_HEAVY
        assert get_preset("nonsense") is KG_HEAVY
        assert get_preset("") is KG_HEAVY

    def test_presets_registry_exposes_both(self):
        assert set(PRESETS) == {"kg-heavy", "balanced"}

    def test_limit_caps_results(self):
        results = fuse_rankings(
            _dense(("source:a", 0.9), ("source:b", 0.8), ("source:c", 0.7)),
            _kg(),
            weights=KG_HEAVY,
            limit=2,
        )
        assert len(results) == 2

    def test_empty_inputs_yield_empty(self):
        assert fuse_rankings([], []) == []
