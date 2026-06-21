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
    """Role/org leader phrase handling."""

    def test_ministerie_van(self):
        assert strip_leading_noise("ministerie van bzk") == "bzk"

    def test_minister_van(self):
        assert strip_leading_noise("minister van financiën") == "financiën"

    def test_staatssecretaris_van(self):
        assert strip_leading_noise("staatssecretaris van defensie") == "defensie"

    def test_gemeente(self):
        assert strip_leading_noise("gemeente groningen") == "groningen"

    def test_provincie(self):
        assert strip_leading_noise("provincie drenthe") == "drenthe"

    def test_article_then_role_prefix(self):
        """Article strips first, then the role prefix."""
        assert (
            strip_leading_noise("de minister van binnenlandse zaken")
            == "binnenlandse zaken"
        )

    def test_longest_match_first(self):
        """``ministerie van`` wins over ``minister van`` on a shared head."""
        # 'ministerie van x' must strip the full 'ministerie van ', leaving 'x'
        # would be too short, so guard returns unchanged; use a real tail.
        assert strip_leading_noise("ministerie van onderwijs") == "onderwijs"
        # And 'minister van onderwijs' (person/role) -> same tail 'onderwijs'
        assert strip_leading_noise("minister van onderwijs") == "onderwijs"


class TestStripLeadingNoiseGuard:
    """The precision guard: never strip into an empty/too-short tail."""

    def test_bare_gemeente_unchanged(self):
        assert strip_leading_noise("gemeente") == "gemeente"

    def test_bare_provincie_unchanged(self):
        assert strip_leading_noise("provincie") == "provincie"

    def test_bare_ministerie_van_unchanged(self):
        assert strip_leading_noise("ministerie van") == "ministerie van"

    def test_too_short_tail_keeps_original(self):
        """A 1-char tail is below the floor → keep the prefixed form."""
        assert strip_leading_noise("gemeente a") == "gemeente a"

    def test_empty_passthrough(self):
        assert strip_leading_noise("") == ""


class TestTailPreservationCanary:
    """The must-NOT-merge invariant at the function level."""

    @pytest.mark.parametrize(
        "a,b",
        [
            ("minister van bzk", "minister van financiën"),
            ("gemeente groningen", "gemeente drenthe"),
            ("ministerie van onderwijs", "ministerie van onderwijs en arbeid"),
            ("provincie groningen", "gemeente groningen"),  # same tail, see note
        ],
    )
    def test_distinct_tails_stay_distinct(self, a, b):
        """Names whose tails differ must not collapse.

        ``provincie groningen``/``gemeente groningen`` DO collapse on the bare
        name (both -> ``groningen``); the type discriminator in the harness keeps
        them apart. Here we only assert the non-type cases differ, and document
        the same-tail case explicitly.
        """
        sa, sb = strip_leading_noise(a), strip_leading_noise(b)
        if (a, b) == ("provincie groningen", "gemeente groningen"):
            # Same tail by design — distinctness comes from entity_type.
            assert sa == sb == "groningen"
        else:
            assert sa != sb


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
        """``ministerie van`` precedes ``minister van`` (longest-overlap-first)."""
        idx = {p: i for i, p in enumerate(_ROLE_ORG_PREFIXES)}
        assert idx["ministerie van "] < idx["minister van "]
        assert idx["de minister van "] < idx["minister van "]
