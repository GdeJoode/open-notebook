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
        entities = [
            {"canonical_name": "BZK", "entity_type": "organization"},
            {"canonical_name": "Ministerie van BZK", "entity_type": "organization"},
            {"canonical_name": "Regio Deal", "entity_type": "other"},
        ]
        report = measure_fragmentation(entities, normalize_entity_name)
        # BZK + Ministerie van BZK collapse to one; Regio Deal is separate.
        assert report.total_entities == 3
        assert report.distinct_canonical == 2

    def test_merged_cluster_members_listed(self):
        entities = [
            {"canonical_name": "BZK", "entity_type": "organization"},
            {"canonical_name": "Ministerie van BZK", "entity_type": "organization"},
            {"canonical_name": "Minister van BZK", "entity_type": "organization"},
        ]
        report = measure_fragmentation(entities, normalize_entity_name)
        assert report.distinct_canonical == 1
        assert len(report.merged_clusters) == 1
        cluster = report.merged_clusters[0]
        assert cluster.canonical_key == "bzk"
        assert cluster.size == 3
        assert set(cluster.members) == {
            "BZK",
            "Ministerie van BZK",
            "Minister van BZK",
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
            {"canonical_name": "BZK", "entity_type": "organization"},
            {"canonical_name": "Ministerie van BZK", "entity_type": "organization"},
            {"canonical_name": "Unique One", "entity_type": "other"},
        ]
        report = measure_fragmentation(entities, normalize_entity_name)
        # one cluster of size 2, one singleton.
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

    def test_type_disambiguation(self):
        """Same name, different type → not a false merge."""
        pairs = [
            {
                "a": {"name": "Groningen", "type": "location"},
                "b": {"name": "Provincie Groningen", "type": "organization"},
            }
        ]
        assert count_false_merges(pairs, normalize_entity_name) == 0


class TestCountUnmergedMustMerge:
    def test_pair_that_should_merge(self):
        pairs = [
            {
                "a": {"name": "Ministerie van BZK", "type": "organization"},
                "b": {"name": "BZK", "type": "organization"},
            }
        ]
        assert count_unmerged_must_merge(pairs, normalize_entity_name) == 0

    def test_v1_baseline_leaves_them_unmerged(self):
        pairs = [
            {
                "a": {"name": "Ministerie van BZK", "type": "organization"},
                "b": {"name": "BZK", "type": "organization"},
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
        """The over-merge canary: zero must-NOT pairs ever collapse (AC7)."""
        pairs = _load_jsonl("must_not_merge.jsonl")
        assert pairs, "must_not_merge.jsonl is empty"
        assert count_false_merges(pairs, normalize_entity_name) == 0

    @pytest.mark.parametrize("pair", _load_jsonl("must_not_merge.jsonl"))
    def test_each_must_not_pair_distinct(self, pair):
        """Property-style: no single must-NOT pair normalizes equal."""
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
        # Regression-pinned floor. Measured drop at K.1 is 46 over the frozen
        # 1402-entity dump (1360 -> 1314). Floor set conservatively below that.
        assert drop >= 40, (
            f"fragmentation drop {drop} below floor "
            f"({baseline.distinct_canonical} -> {candidate.distinct_canonical})"
        )

    def test_no_false_merges_introduced_on_live_set(self):
        """The drop must not come at the cost of must-NOT over-merges."""
        pairs = _load_jsonl("must_not_merge.jsonl")
        assert count_false_merges(pairs, normalize_entity_name) == 0

    def test_bzk_fullform_cluster_collapses(self):
        """The documented BZK full-form fragmentation collapses to one key."""
        ents = self._entities()
        candidate = measure_fragmentation(ents, normalize_entity_name)
        bzk = [
            c
            for c in candidate.merged_clusters
            if c.canonical_key == "binnenlandse zaken en koninkrijksrelaties"
        ]
        assert bzk, "BZK full-form cluster did not form"
        # The 'other'-typed cluster collapses ≥5 surface forms (article/role/
        # spelling variants) into one canonical key.
        assert max(c.size for c in bzk) >= 5
