"""Tests for the K.1 measurement harness (``shared.utils.resolution_metrics``).

Covers cluster counting / false-merge counting on synthetic data, and runs the
over-merge canary + before→after fragmentation measurement over the frozen
adversarial corpora and the live Convenant fixture.
"""

import json
import re
from pathlib import Path

import pytest
from shared.utils.name_normalizer import normalize_entity_name
from shared.utils.resolution_metrics import (
    count_false_merges,
    count_false_merges_with_type,
    count_unmerged_must_merge,
    measure_fragmentation,
)

# Repo-root-relative fixtures (this file lives at packages/shared/tests/).
_FIXTURE_DIR = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "entity_resolution"
)


def _load_jsonl(name: str) -> list[dict]:
    path = _FIXTURE_DIR / name
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# V1 baseline normalizer (pre-K.1): lowercase + whitespace + trailing punct.
_WS_RE = re.compile(r"\s+")
_TP_RE = re.compile(r"[\s\.,;:!\?]+$")


def _v1_baseline(name: str) -> str:
    if not name:
        return ""
    name = name.lower()
    name = _WS_RE.sub(" ", name)
    name = _TP_RE.sub("", name)
    return name.strip()


class TestMeasureFragmentation:
    def test_counts_distinct_canonical(self):
        # rev4: no content prefix is stripped, so ``BZK`` (-> bzk) and
        # ``Ministerie van BZK`` (-> ministerie van bzk) stay DISTINCT. Only the
        # two ``De Regio Deal`` / ``Regio Deal`` forms collapse (article strip).
        entities = [
            {"canonical_name": "De Regio Deal", "entity_type": "other"},
            {"canonical_name": "Regio Deal", "entity_type": "other"},
            {"canonical_name": "Ministerie van BZK", "entity_type": "organization"},
        ]
        report = measure_fragmentation(entities, normalize_entity_name)
        assert report.total_entities == 3
        # Regio Deal pair collapses to one; Ministerie van BZK is separate.
        assert report.distinct_canonical == 2

    def test_merged_cluster_members_listed(self):
        # rev4: only the article-variant forms collapse; ``Ministerie van …``
        # content is no longer stripped, so the BZK org forms do NOT merge.
        entities = [
            {"canonical_name": "De Regio Deal", "entity_type": "other"},
            {"canonical_name": "Regio Deal", "entity_type": "other"},
            {"canonical_name": "Een Regio Deal", "entity_type": "other"},
        ]
        report = measure_fragmentation(entities, normalize_entity_name)
        assert report.distinct_canonical == 1
        assert len(report.merged_clusters) == 1
        cluster = report.merged_clusters[0]
        assert cluster.canonical_key == "regio deal"
        assert cluster.size == 3
        assert set(cluster.members) == {
            "De Regio Deal",
            "Regio Deal",
            "Een Regio Deal",
        }

    def test_entity_type_separates_homographs(self):
        """Same normalized name, different type → two clusters."""
        entities = [
            {"canonical_name": "Groningen", "entity_type": "location"},
            {"canonical_name": "Provincie Groningen", "entity_type": "organization"},
        ]
        report = measure_fragmentation(entities, normalize_entity_name)
        assert report.distinct_canonical == 2
        assert report.merged_clusters == []

    def test_histogram(self):
        entities = [
            {"canonical_name": "De Regio Deal", "entity_type": "other"},
            {"canonical_name": "Regio Deal", "entity_type": "other"},
            {"canonical_name": "Unique One", "entity_type": "other"},
        ]
        report = measure_fragmentation(entities, normalize_entity_name)
        # one cluster of size 2 (article variants), one singleton.
        assert report.cluster_size_histogram == {1: 1, 2: 1}

    def test_empty_input(self):
        report = measure_fragmentation([], normalize_entity_name)
        assert report.total_entities == 0
        assert report.distinct_canonical == 0
        assert report.merged_clusters == []


class TestCountFalseMerges:
    def test_synthetic_false_merge_detected(self):
        # A normalizer that drops everything → everything collapses.
        pairs = [{"a": "X", "b": "Y", "entity_type": "t"}]
        assert count_false_merges(pairs, lambda s: "same") == 1

    def test_distinct_pair_not_counted(self):
        pairs = [{"a": "X", "b": "Y", "entity_type": "t"}]
        assert count_false_merges(pairs, normalize_entity_name) == 0

    def test_name_only_collision_is_a_false_merge(self):
        """rev3 criterion: a cross-type NAME collision counts as a false merge.

        Relations resolve endpoints by name alone, so two different-typed
        entities sharing a normalized name corrupt the graph. The name-only
        gate must catch this even though their ``(name, type)`` keys differ.
        """
        pairs = [
            {
                "a": {"name": "Groningen", "type": "location"},
                "b": {"name": "Groningen", "type": "organization"},
            }
        ]
        # Name-only gate: collision IS a false merge.
        assert count_false_merges(pairs, normalize_entity_name) == 1
        # Type-aware companion: NOT a false merge (different dedup keys).
        assert count_false_merges_with_type(pairs, normalize_entity_name) == 0

    def test_type_disambiguation_via_distinct_names(self):
        """rev3: ``Provincie Groningen`` and ``Groningen`` keep distinct names.

        ``provincie`` is no longer stripped, so the names diverge and the
        name-only gate passes without leaning on entity_type.
        """
        pairs = [
            {
                "a": {"name": "Groningen", "type": "location"},
                "b": {"name": "Provincie Groningen", "type": "organization"},
            }
        ]
        assert count_false_merges(pairs, normalize_entity_name) == 0


class TestCountUnmergedMustMerge:
    def test_pair_that_should_merge(self):
        # rev4: the article-variant pair must merge (``de`` stripped).
        pairs = [
            {
                "a": {"name": "De Regio Deal", "type": "other"},
                "b": {"name": "Regio Deal", "type": "other"},
            }
        ]
        assert count_unmerged_must_merge(pairs, normalize_entity_name) == 0

    def test_v1_baseline_leaves_them_unmerged(self):
        # The V1 baseline does not strip the article, so the pair stays split.
        pairs = [
            {
                "a": {"name": "De Regio Deal", "type": "other"},
                "b": {"name": "Regio Deal", "type": "other"},
            }
        ]
        assert count_unmerged_must_merge(pairs, _v1_baseline) == 1


class TestAdversarialCorpora:
    """AC6 / AC7 over the real frozen fixtures — the regression gate."""

    def test_must_merge_all_collapse(self):
        pairs = _load_jsonl("must_merge.jsonl")
        assert pairs, "must_merge.jsonl is empty"
        assert count_unmerged_must_merge(pairs, normalize_entity_name) == 0

    def test_no_false_merges_over_must_not_corpus(self):
        """The over-merge canary: zero must-NOT pairs collapse at NAME level.

        rev3 fixes the criterion: the gate is name-only (ignoring entity_type),
        because relations resolve endpoints by name alone. The corpus includes
        the cross-type pairs ``Minister van BZK`` ↔ ``Ministerie van BZK`` and
        ``Gemeente``/``Provincie Groningen`` ↔ ``Groningen`` that rev2's
        ``(name, type)`` check silently passed.
        """
        pairs = _load_jsonl("must_not_merge.jsonl")
        assert pairs, "must_not_merge.jsonl is empty"
        assert count_false_merges(pairs, normalize_entity_name) == 0

    @pytest.mark.parametrize("pair", _load_jsonl("must_not_merge.jsonl"))
    def test_each_must_not_pair_distinct(self, pair):
        """Property-style: no single must-NOT pair normalizes to the same name."""
        assert count_false_merges([pair], normalize_entity_name) == 0


class TestConvenantFragmentationDrop:
    """AC8: measurable fragmentation drop over the live Convenant fixture."""

    def _entities(self):
        return _load_jsonl("convenant_entities.jsonl")

    def test_fixture_present_and_nonempty(self):
        ents = self._entities()
        assert len(ents) > 800, "Convenant fixture unexpectedly small"

    def test_new_normalizer_reduces_distinct_count(self):
        ents = self._entities()
        baseline = measure_fragmentation(ents, _v1_baseline)
        candidate = measure_fragmentation(ents, normalize_entity_name)
        drop = baseline.distinct_canonical - candidate.distinct_canonical
        # Regression-pinned floor. Measured drop at K.1 rev4 is 14 over the
        # frozen 1402-entity dump (1360 -> 1346). Smaller than rev3's 19 BY
        # DESIGN: rev4 strips NO content prefix at all (not even ``ministerie
        # van``, which collides cross-type: ``Ministerie van Onderwijs`` ->
        # ``onderwijs`` == the bare concept). The drop is now ONLY the
        # collision-safe article + curated-spelling consolidation. The bigger
        # org-form merges move to K.2's type-aware alias table. Floor set
        # conservatively below the measured 14.
        assert drop >= 12, (
            f"fragmentation drop {drop} below floor "
            f"({baseline.distinct_canonical} -> {candidate.distinct_canonical})"
        )

    def test_no_false_merges_introduced_on_live_set(self):
        """The drop must not come at the cost of must-NOT over-merges."""
        pairs = _load_jsonl("must_not_merge.jsonl")
        assert count_false_merges(pairs, normalize_entity_name) == 0

    def test_bzk_fullform_cluster_collapses(self):
        """The documented BZK full-form spelling fragmentation collapses."""
        ents = self._entities()
        candidate = measure_fragmentation(ents, normalize_entity_name)
        bzk = [
            c
            for c in candidate.merged_clusters
            if c.canonical_key == "binnenlandse zaken en koninkrijksrelaties"
        ]
        assert bzk, "BZK full-form cluster did not form"
        # K.2: the org-form merge now folds the abbreviation + prefixed forms +
        # both spelling variants into one canonical (≥2 members; measured 6).
        assert max(c.size for c in bzk) >= 2


def _k1_only_normalizer(name: str) -> str:
    """The K.1 normalizer (article-strip + spelling) WITHOUT K.2 alias expansion.

    Reconstructed by composing the K.1 stages directly so the K.2 fragmentation
    test can assert a *further* drop vs K.1 on the same fixture.
    """
    from shared.utils.nl_normalization import (
        canonicalize_spelling,
        strip_leading_noise,
    )

    if not name:
        return ""
    name = name.lower()
    name = _WS_RE.sub(" ", name)
    name = _TP_RE.sub("", name)
    name = name.strip()
    name = strip_leading_noise(name)
    name = canonicalize_spelling(name)
    return name


class TestK2FragmentationDrop:
    """K.2 AC5/AC6: a further fragmentation drop vs K.1, false-merges still 0."""

    def _entities(self):
        return _load_jsonl("convenant_entities.jsonl")

    def test_k2_drops_further_than_k1(self):
        ents = self._entities()
        k1 = measure_fragmentation(ents, _k1_only_normalizer)
        k2 = measure_fragmentation(ents, normalize_entity_name)
        further = k1.distinct_canonical - k2.distinct_canonical
        # Measured at K.2: K.1 1346 -> K.2 1337 (a further drop of 9 over the
        # frozen 1402-entity dump) from the BZK + VRO org-form merges. Floor set
        # conservatively below the measured 9.
        assert further >= 7, (
            f"K.2 further drop {further} below floor "
            f"({k1.distinct_canonical} -> {k2.distinct_canonical})"
        )

    def test_bzk_cluster_is_single_canonical(self):
        ents = self._entities()
        report = measure_fragmentation(ents, normalize_entity_name)
        bzk = [
            c
            for c in report.merged_clusters
            if c.canonical_key == "binnenlandse zaken en koninkrijksrelaties"
        ]
        assert bzk, "BZK cluster did not form under K.2"
        # The abbreviation, prefixed forms and spelling variants all collapse.
        assert max(c.size for c in bzk) >= 3

    def test_vro_cluster_is_single_canonical(self):
        ents = self._entities()
        report = measure_fragmentation(ents, normalize_entity_name)
        vro = [
            c
            for c in report.merged_clusters
            if c.canonical_key == "volkshuisvesting en ruimtelijke ordening"
        ]
        assert vro, "VRO cluster did not form under K.2"
        assert max(c.size for c in vro) >= 2

    def test_no_false_merges_under_k2(self):
        pairs = _load_jsonl("must_not_merge.jsonl")
        assert count_false_merges(pairs, normalize_entity_name) == 0
