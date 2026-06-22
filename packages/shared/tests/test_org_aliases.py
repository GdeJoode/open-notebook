"""Tests for the K.2 curated org-alias layer + extensible override config.

Covers the acceptance criteria from the K.2 plan:
- abbreviation / prefix / full-form all collapse to one canonical (AC1, AC2);
- distinct ministries stay distinct + unknown abbreviation passes through (AC3);
- an operator-supplied override is honoured with no code change (AC4);
- the org/concept non-collision and minister(person)/ministry(org) split that
  keep the name-only false-merge gate at zero (the type-safety gate).
"""

from __future__ import annotations

import json

import pytest
from shared.utils.org_aliases import _GOV_ORG_ALIASES, expand_org_alias


class TestExpandOrgAlias:
    def test_known_abbreviation_expands(self):
        assert (
            expand_org_alias("bzk")
            == "binnenlandse zaken en koninkrijksrelaties"
        )

    def test_prefixed_form_expands(self):
        assert (
            expand_org_alias("ministerie van bzk")
            == "binnenlandse zaken en koninkrijksrelaties"
        )

    def test_full_form_is_its_own_canonical(self):
        # The bare full-form is itself a key mapping onto the canonical, so the
        # abbreviation and the spelled-out name converge.
        assert (
            expand_org_alias("binnenlandse zaken en koninkrijksrelaties")
            == "binnenlandse zaken en koninkrijksrelaties"
        )

    def test_unknown_passes_through(self):
        assert expand_org_alias("xyz") == "xyz"

    def test_empty_passes_through(self):
        assert expand_org_alias("") == ""

    def test_exact_match_only_no_substring(self):
        # A longer surface form that merely CONTAINS an abbreviation must not be
        # expanded — exact-match only protects the noisy live ``vro (...)`` forms.
        assert (
            expand_org_alias("vro (vereniging van regio's)")
            == "vro (vereniging van regio's)"
        )

    def test_distinct_ministries_have_distinct_canonicals(self):
        assert expand_org_alias("bzk") != expand_org_alias("ezk")

    def test_custom_alias_map_argument(self):
        custom = {"foo": "bar"}
        assert expand_org_alias("foo", aliases=custom) == "bar"
        # built-in keys are absent from the custom map → pass through.
        assert expand_org_alias("bzk", aliases=custom) == "bzk"


class TestTypeSafety:
    """The canonicals must never be a bare different-typed concept token."""

    def test_ocw_canonical_is_org_not_bare_concept(self):
        # OCW must map to the distinct org canonical, NOT the bare ``onderwijs``.
        assert expand_org_alias("ocw") == "onderwijs cultuur en wetenschap"
        assert expand_org_alias("ocw") != "onderwijs"

    def test_no_canonical_is_a_single_bare_token(self):
        # Sanity guard for the whole table: every canonical is a multi-word org
        # name (so it cannot collide with a bare single-token concept). The
        # single-word ministries (financien, defensie) are full org names, not
        # ambiguous concept tokens, and are allow-listed here.
        single_word_orgs = {"financien", "defensie"}
        for canonical in set(_GOV_ORG_ALIASES.values()):
            if " " not in canonical:
                assert canonical in single_word_orgs, (
                    f"unexpected single-token canonical {canonical!r}"
                )

    def test_minister_role_forms_not_keyed(self):
        # ``minister van …`` (person) forms are deliberately absent so the role
        # never folds onto the ministry org canonical.
        for key in _GOV_ORG_ALIASES:
            assert not key.startswith("minister van "), (
                f"person-role key {key!r} must not be in the org-alias table"
            )


class TestOverrideConfig:
    """AC4: file-based operator override honoured by normalize_entity_name."""

    @pytest.fixture
    def _reload(self, monkeypatch, tmp_path):
        """Point the loader at a temp override file and reload around the test."""
        import shared.config.alias_overrides as overrides
        import shared.utils.name_normalizer as nn

        def _apply(payload: dict, *, raw_json: str | None = None):
            path = tmp_path / "alias_overrides.json"
            path.write_text(
                raw_json if raw_json is not None else json.dumps(payload),
                encoding="utf-8",
            )
            monkeypatch.setenv("ONB_ALIAS_OVERRIDES_PATH", str(path))
            overrides.reload_alias_overrides()
            return nn.normalize_entity_name

        yield _apply
        # Restore the built-in floor for subsequent tests.
        monkeypatch.delenv("ONB_ALIAS_OVERRIDES_PATH", raising=False)
        overrides.reload_alias_overrides()

    def test_operator_override_honoured(self, _reload):
        normalize = _reload({"Min AZ": "Algemene Zaken"})
        assert normalize("Min AZ") == "algemene zaken"
        # And it converges with the built-in AZ forms.
        assert normalize("Min AZ") == normalize("Ministerie van AZ")

    def test_empty_key_or_value_rejected(self, _reload):
        import shared.config.alias_overrides as overrides

        _reload({"": "bad", "ok key": "  ", "real": "Algemene Zaken"})
        resolved = overrides.get_resolved_aliases()
        assert "" not in resolved
        assert "ok key" not in resolved
        assert resolved["real"] == "algemene zaken"

    def test_malformed_file_is_ignored(self, _reload):
        # A non-object / unparseable file must not break normalization.
        normalize = _reload({}, raw_json="not json {")
        assert normalize("bzk") == "binnenlandse zaken en koninkrijksrelaties"

    def test_db_overlay_seam(self, monkeypatch):
        import shared.config.alias_overrides as overrides

        monkeypatch.delenv("ONB_ALIAS_OVERRIDES_PATH", raising=False)
        overrides.set_db_overlay({"My Org": "Canonical Org"})
        try:
            resolved = overrides.get_resolved_aliases()
            assert resolved["my org"] == "canonical org"
        finally:
            overrides.set_db_overlay({})
            overrides.reload_alias_overrides()


class TestNormalizeEntityNameIntegration:
    """AC1-3 + AC5 through the public ``normalize_entity_name`` surface."""

    def test_ac1_bzk_forms_collapse(self):
        from shared.utils.name_normalizer import normalize_entity_name as n

        forms = [
            n("BZK"),
            n("ministerie van BZK"),
            n("Binnenlandse Zaken en Koninkrijksrelaties"),
        ]
        assert len(set(forms)) == 1

    def test_ac2_vro_forms_collapse(self):
        from shared.utils.name_normalizer import normalize_entity_name as n

        assert n("VRO") == n("Volkshuisvesting en Ruimtelijke Ordening")

    def test_ac3_distinct_and_unknown(self):
        from shared.utils.name_normalizer import normalize_entity_name as n

        assert n("BZK") != n("EZK")
        assert n("XYZ") == "xyz"

    def test_org_concept_non_collision(self):
        from shared.utils.name_normalizer import normalize_entity_name as n

        assert n("Ministerie van Onderwijs") != n("Onderwijs")
        assert n("Ministerie van OCW") != n("Onderwijs")
