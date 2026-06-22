"""Tests for the K.1 NL normalization rules (``shared.utils.nl_normalization``).

These pin the leading-noise strip, the spelling canonicalization, and — most
importantly — the tail-preservation precision guards. The guards are the spine
of Track K: over-merging distinct entities is worse than fragmentation.
"""

import pytest
from shared.utils.nl_normalization import (
    _LEADING_ARTICLES,
    _ROLE_ORG_PREFIXES,
    canonicalize_spelling,
    strip_leading_noise,
)


class TestStripLeadingNoiseArticles:
    """Leading article (de/het/een) handling."""

    def test_strips_de(self):
        assert strip_leading_noise("de regio deal") == "regio deal"

    def test_strips_het(self):
        assert strip_leading_noise("het bedrijfsleven") == "bedrijfsleven"

    def test_strips_een(self):
        assert strip_leading_noise("een gebied") == "gebied"

    def test_only_strips_at_word_boundary(self):
        """The trailing space in the article guards against ``deal`` -> ``al``."""
        assert strip_leading_noise("deal") == "deal"
        assert strip_leading_noise("hethond") == "hethond"

    def test_at_most_one_article(self):
        """``de het x`` strips only the first article (then no prefix match)."""
        # 'het x' is not a role/org prefix, so it survives.
        assert strip_leading_noise("de het x") == "het x"

    def test_bare_article_not_collapsed_to_empty(self):
        """An article on its own keeps the original rather than emptying."""
        assert strip_leading_noise("de") == "de"


class TestStripLeadingNoiseRoleOrgPrefixes:
    """Org leader phrase handling (rev3: only ``ministerie van`` survives)."""

    def test_ministerie_van(self):
        assert strip_leading_noise("ministerie van bzk") == "bzk"

    def test_het_ministerie_van(self):
        assert strip_leading_noise("het ministerie van bzk") == "bzk"

    def test_article_then_ministerie_prefix(self):
        """Article strips first, then the ministerie prefix."""
        assert (
            strip_leading_noise("de ministerie van binnenlandse zaken")
            == "binnenlandse zaken"
        )

    def test_longest_match_first(self):
        """``het ministerie van`` wins over the bare ``ministerie van``."""
        assert strip_leading_noise("ministerie van onderwijs") == "onderwijs"
        assert strip_leading_noise("het ministerie van onderwijs") == "onderwijs"


class TestStripLeadingNoiseCrossTypeGuard:
    """rev3 cross-type collision guard: person/municipality leaders NOT stripped.

    Relations resolve endpoints by name alone, so stripping ``minister van`` or
    ``gemeente``/``provincie`` would collide a person/org onto another entity's
    name. These prefixes are deliberately absent from the strip set, so the
    surface form passes through unchanged.
    """

    def test_minister_van_not_stripped(self):
        """``Minister van BZK`` (person) must stay distinct from the org ``bzk``."""
        assert strip_leading_noise("minister van bzk") == "minister van bzk"

    def test_minister_van_financien_not_stripped(self):
        assert (
            strip_leading_noise("minister van financiën")
            == "minister van financiën"
        )

    def test_staatssecretaris_van_not_stripped(self):
        assert (
            strip_leading_noise("staatssecretaris van defensie")
            == "staatssecretaris van defensie"
        )

    def test_gemeente_not_stripped(self):
        """``Gemeente Groningen`` (org) must stay distinct from ``groningen``."""
        assert strip_leading_noise("gemeente groningen") == "gemeente groningen"

    def test_provincie_not_stripped(self):
        assert strip_leading_noise("provincie drenthe") == "provincie drenthe"

    def test_minister_vs_ministerie_distinct(self):
        """The headline cross-type case: person leader vs org leader diverge."""
        assert strip_leading_noise("minister van bzk") != strip_leading_noise(
            "ministerie van bzk"
        )


class TestStripLeadingNoiseGuard:
    """The precision guard: never strip into an empty/too-short tail."""

    def test_bare_ministerie_van_unchanged(self):
        assert strip_leading_noise("ministerie van") == "ministerie van"

    def test_too_short_tail_keeps_original(self):
        """A 1-char tail is below the floor → keep the prefixed form."""
        assert strip_leading_noise("ministerie van a") == "ministerie van a"

    def test_empty_passthrough(self):
        assert strip_leading_noise("") == ""


class TestTailPreservationCanary:
    """The must-NOT-merge invariant at the function level (NAME-only, rev3).

    With the municipality/person leaders removed from the strip set, every pair
    below stays distinct at the NORMALIZED-NAME level — including the cross-type
    cases that rev2 only kept apart via entity_type.
    """

    @pytest.mark.parametrize(
        "a,b",
        [
            ("minister van bzk", "minister van financiën"),
            ("gemeente groningen", "gemeente drenthe"),
            ("ministerie van onderwijs", "ministerie van onderwijs en arbeid"),
            # rev3: these now diverge at the name level (no prefix stripped).
            ("provincie groningen", "groningen"),
            ("gemeente groningen", "groningen"),
            ("minister van bzk", "ministerie van bzk"),
        ],
    )
    def test_distinct_names_stay_distinct(self, a, b):
        """Names that must not merge stay distinct at the name level."""
        assert strip_leading_noise(a) != strip_leading_noise(b)


class TestCanonicalizeSpelling:
    """The curated spelling-variant map."""

    def test_koninkrijk_variant(self):
        assert (
            canonicalize_spelling("binnenlandse zaken en koninkrijkrelaties")
            == "binnenlandse zaken en koninkrijksrelaties"
        )

    def test_already_canonical_unchanged(self):
        s = "binnenlandse zaken en koninkrijksrelaties"
        assert canonicalize_spelling(s) == s

    def test_unknown_passes_through(self):
        """An unrecognized spelling is left untouched (no fuzzy guessing)."""
        assert canonicalize_spelling("regio deal") == "regio deal"
        assert canonicalize_spelling("koninkrijkje") == "koninkrijkje"

    def test_empty_passthrough(self):
        assert canonicalize_spelling("") == ""


class TestRuleTables:
    """Sanity checks on the rule tables themselves."""

    def test_articles_lowercased_with_trailing_space(self):
        for art in _LEADING_ARTICLES:
            assert art == art.lower()
            assert art.endswith(" ")

    def test_prefixes_lowercased_with_trailing_space(self):
        for pre in _ROLE_ORG_PREFIXES:
            assert pre == pre.lower()
            assert pre.endswith(" ")

    def test_prefixes_longest_first_within_overlaps(self):
        """``het ministerie van`` precedes the bare ``ministerie van``."""
        idx = {p: i for i, p in enumerate(_ROLE_ORG_PREFIXES)}
        assert idx["het ministerie van "] < idx["ministerie van "]

    def test_cross_type_leaders_absent(self):
        """rev3: person-role + municipality leaders are NOT in the strip set.

        Stripping any of these would collide a person/org onto another entity's
        name, which corrupts name-only relation resolution.
        """
        forbidden = {
            "minister van ",
            "de minister van ",
            "staatssecretaris van ",
            "de staatssecretaris van ",
            "gemeente ",
            "provincie ",
        }
        assert forbidden.isdisjoint(_ROLE_ORG_PREFIXES)
