"""Tests for the curated EN+NL residual alias map + non-silent fallback (L.2).

These cover the L.2 acceptance criteria over the *residual* resolver in
``shared.utils.entity_type_aliases`` (the alias path that runs when the L.1
ontology bridge returns ``None``), plus the language-surface-isolation guard
that confines every Dutch literal in Track L to this one module.

Note: ``resolve_residual_type`` is intentionally NOT enum-guarded — it emits
``technology`` / ``programme`` (the real targets, option (a)). As of L.3 those
are valid members of the persistence ``_ALLOWED_ENTITY_TYPES`` enum, so the
boundary now passes them through and the labels land on the real canonical (no
longer re-pinned to ``other``). The enum behaviour is exercised in
``apps/app-main/tests/test_entity_persistence_service.py``.
"""

from pathlib import Path

import pytest
from shared.utils.entity_type_aliases import (
    ENTITY_TYPE_ALIASES,
    NON_TYPE_LABELS,
    resolve_residual_type,
)


class TestDutchAliases:
    """AC1 — the curated NL residual terms map to their canonical targets."""

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("Persoon", "person"),
            ("Organisatie", "organization"),
            ("Organisation", "organization"),
            ("Ministerie", "government_organization"),
            ("Uitvoeringsorganisatie", "government_organization"),
            ("Gemeente", "administrative_area"),
            ("Provincie", "administrative_area"),
            ("Regio", "administrative_area"),
            ("Wethouder", "person"),
            ("Bestuurder", "person"),
            ("Ambtenaar", "person"),
            ("Thema", "topic"),
            ("BeleidsThema", "topic"),
            ("Sector", "topic"),
            ("Wet", "legislation"),
            ("Document", "creative_work"),
            ("Publicatie", "creative_work"),
            ("Publication", "creative_work"),
        ],
    )
    def test_nl_residual_terms_map(self, label, expected):
        res = resolve_residual_type(label)
        assert res.entity_type == expected
        assert res.mapped is True
        # An alias hit preserves the raw label (rich signal survives).
        assert res.primary_type == label
        assert res.type_tags == [label]
        assert res.is_noise is False

    def test_case_and_separators_normalized(self):
        # AC1: matching is case-insensitive.
        assert resolve_residual_type("GEMEENTE").entity_type == "administrative_area"
        assert resolve_residual_type("BeleidsThema").entity_type == "topic"


class TestTechnologyProgrammeOptionA:
    """AC2 — technology/programme aliases ARE the real targets (option a).

    As of L.3 ``technology`` / ``programme`` are members of
    ``_ALLOWED_ENTITY_TYPES``, so the persistence enum-guard now passes them
    through and they land on the real canonical (previously re-pinned to
    ``other``). At THIS residual layer the alias map points at the real target
    and the rich label is preserved — exactly as it always has.
    """

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("Technologie", "technology"),
            ("Technology", "technology"),
            ("Programma", "programme"),
            ("RegioDeal", "programme"),
            ("Deal", "programme"),
            ("Project", "programme"),
        ],
    )
    def test_technology_programme_alias_exists(self, label, expected):
        res = resolve_residual_type(label)
        # The alias points at the real canonical target (option a); as of L.3 the
        # persistence enum-guard passes these through unchanged.
        assert res.entity_type == expected
        assert res.mapped is True
        # AC2: the rich label is preserved regardless of the enum-guard.
        assert res.primary_type == label

    def test_alias_targets_documented_in_map(self):
        # The map literally contains the real targets — proves option (a), not
        # an interim concept/creative_work mapping.
        assert ENTITY_TYPE_ALIASES["technologie"] == "technology"
        assert ENTITY_TYPE_ALIASES["programma"] == "programme"


class TestNonSilentFallback:
    """AC3 + AC4 — noise and genuine-unknown labels are never silently lost."""

    def test_abbreviation_noise_preserves_signal(self):
        # AC3: ABBREVIATION is extraction noise (the FORM not a type). It must
        # NOT silently become ``other`` — the original signal is retained and
        # the label is flagged as noise.
        res = resolve_residual_type("ABBREVIATION")
        assert res.primary_type is not None
        assert res.primary_type == "ABBREVIATION"
        assert res.is_noise is True
        assert res.mapped is False
        # coarses to a sensible default, not ``other``
        assert res.entity_type != "other"

    def test_amount_is_noise(self):
        res = resolve_residual_type("Amount")
        assert res.is_noise is True
        assert res.primary_type == "Amount"

    def test_genuinely_unknown_preserves_rich_label(self):
        # AC4: a genuinely-unknown label coarses to a default BUT the rich label
        # is preserved (NOT lost to ``other``), and it is NOT flagged as noise
        # (it is an unmodeled real type, distinct from mis-tagged noise).
        res = resolve_residual_type("Frobnicator")
        assert res.primary_type == "Frobnicator"
        assert res.type_tags == ["Frobnicator"]
        assert res.is_noise is False
        assert res.mapped is False
        assert res.entity_type != "other"

    def test_empty_label_has_no_signal_to_preserve(self):
        # The only case with no preserved primary_type: there is no label.
        res = resolve_residual_type("")
        assert res.primary_type is None
        res2 = resolve_residual_type("   ")
        assert res2.primary_type is None


class TestLanguageSurfaceIsolation:
    """AC5 — every Dutch literal in Track L lives ONLY in this module."""

    def _bridge_path(self) -> Path:
        # canonical_bridge.py is the L.1 (language-agnostic) module.
        here = Path(__file__).resolve()
        # repo root: .../packages/shared/tests/test_*.py -> up 3 to repo root
        repo_root = here.parents[3]
        return (
            repo_root
            / "packages"
            / "ontology-manager"
            / "src"
            / "ontology_manager"
            / "canonical_bridge.py"
        )

    def test_canonical_bridge_has_no_dutch_literals(self):
        bridge = self._bridge_path()
        assert bridge.exists(), f"bridge not found at {bridge}"
        text = bridge.read_text(encoding="utf-8").lower()
        # A sample of the Dutch terms that live in THIS module — none of them
        # may appear as a literal in the language-agnostic L.1 bridge. (The
        # bridge resolves ``Gemeente`` only because an ontology declares its
        # parent_type, never because the bridge knows the Dutch word.)
        dutch_terms = [
            "persoon",
            "organisatie",
            "ministerie",
            "gemeente",
            "provincie",
            "wethouder",
            "ambtenaar",
            "beleidsthema",
            "technologie",
            "programma",
            "publicatie",
            "uitvoeringsorganisatie",
        ]
        offenders = [t for t in dutch_terms if t in text]
        assert offenders == [], (
            f"Dutch literals leaked into canonical_bridge.py (L.1 must stay "
            f"language-agnostic): {offenders}"
        )

    def test_dutch_terms_present_in_this_module(self):
        # Positive half of the containment guarantee: the Dutch terms DO live
        # here (so the test above is meaningful, not vacuously green).
        for term in ["persoon", "gemeente", "ministerie", "wethouder"]:
            assert term in ENTITY_TYPE_ALIASES


class TestOtherBucketRecovery:
    """AC6 — measure recovery of the live ``other``-bucket labels.

    Over the labels the analysis found stuck in ``other``, the L.2 alias path
    recovers them to a non-``other`` canonical. As of L.3, technology/programme
    are valid enum members, so even the deal/programme labels land on a real
    canonical at the persistence boundary (no residual re-pin).
    """

    # The live ``other``-bucket labels enumerated in the plan (AC6).
    LIVE_OTHER_LABELS = [
        "RegioDeal",
        "Gemeente",
        "Persoon",
        "Organisatie",
        "Ministerie",
        "Programma",
        "Sector",
        "Wethouder",
        "Thema",
        "Regio",
        "Project",
    ]

    def test_alias_recovers_all_live_other_labels(self):
        unresolved = [
            label
            for label in self.LIVE_OTHER_LABELS
            if not resolve_residual_type(label).mapped
        ]
        # Every enumerated label must be recovered by the alias map.
        assert unresolved == [], f"unrecovered live-other labels: {unresolved}"

    def test_recovery_rate_recorded(self):
        # Record the measured recovery for the PR / status report. All 11 of the
        # enumerated live-other labels are mapped by the alias path; 3 of them
        # (RegioDeal/Programma/Project) point at ``programme`` — as of L.3 a valid
        # enum member, so they now land on the real canonical at the persistence
        # boundary instead of re-pinning to ``other``.
        results = {
            label: resolve_residual_type(label).entity_type
            for label in self.LIVE_OTHER_LABELS
        }
        mapped_count = sum(
            1 for label in self.LIVE_OTHER_LABELS if resolve_residual_type(label).mapped
        )
        assert mapped_count == len(self.LIVE_OTHER_LABELS)  # 11/11 recovered
        # The 3 deal/programme labels target ``programme`` (now a valid enum).
        programme_labels = [
            label for label, t in results.items() if t in {"programme", "technology"}
        ]
        assert set(programme_labels) == {"RegioDeal", "Programma", "Project"}


class TestNonTypeLabelsSet:
    def test_non_type_labels_membership(self):
        assert "abbreviation" in NON_TYPE_LABELS
        assert "amount" in NON_TYPE_LABELS
        # A real type is NOT in the noise set.
        assert "gemeente" not in NON_TYPE_LABELS
