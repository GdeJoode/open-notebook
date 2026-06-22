"""Tests for the K.1 NL normalization rules (``shared.utils.nl_normalization``).

These pin the leading-article strip, the spelling canonicalization, and — most
importantly — the collision-safety guarantee. That guarantee is the spine of
Track K: over-merging distinct entities is worse than fragmentation, and because
relations resolve endpoints by name alone, K.1 strips ONLY meaning-free articles
(no content-bearing prefix that could collapse one entity onto another's name).
"""

import pytest
from shared.utils.nl_normalization import (
    _LEADING_ARTICLES,
    canonicalize_spelling,
    strip_leading_noise,
)


class TestStripLeadingNoiseArticles:
    """Leading article (de/het/een) handling — the only strip K.1 performs."""

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
        """``de het x`` strips only the first article."""
        assert strip_leading_noise("de het x") == "het x"

    def test_bare_article_not_collapsed_to_empty(self):
        """An article on its own keeps the original rather than emptying."""
        assert strip_leading_noise("de") == "de"


class TestStripLeadingNoiseNoContentPrefix:
    """rev4 collision-safety: no content-bearing prefix is ever stripped.

    Stripping ``ministerie van`` / ``gemeente`` / ``minister van`` would
    collapse an org/role surface form onto a bare concept token another real
    entity owns (``Ministerie van Onderwijs`` -> ``onderwijs`` == the bare
    ``onderwijs``). Because relations resolve endpoints by name alone, that
    corrupts the graph. So these prefixes pass through unchanged; type-aware
    org-form merging is Track K.2's job.
    """

    def test_ministerie_van_not_stripped(self):
        assert (
            strip_leading_noise("ministerie van onderwijs")
            == "ministerie van onderwijs"
        )

    def test_ministerie_van_bzk_not_stripped(self):
        assert strip_leading_noise("ministerie van bzk") == "ministerie van bzk"

    def test_het_ministerie_van_not_stripped(self):
        """Only the leading article strips; the ministerie content stays."""
        assert (
            strip_leading_noise("het ministerie van bzk") == "ministerie van bzk"
        )

    def test_minister_van_not_stripped(self):
        assert strip_leading_noise("minister van bzk") == "minister van bzk"

    def test_staatssecretaris_van_not_stripped(self):
        assert (
            strip_leading_noise("staatssecretaris van defensie")
            == "staatssecretaris van defensie"
        )

    def test_gemeente_not_stripped(self):
        assert strip_leading_noise("gemeente groningen") == "gemeente groningen"

    def test_provincie_not_stripped(self):
        assert strip_leading_noise("provincie drenthe") == "provincie drenthe"


class TestStripLeadingNoiseCollisionGuard:
    """The headline rev4 collision: ministerie-org vs the bare concept."""

    def test_ministerie_onderwijs_vs_bare_onderwijs_distinct(self):
        """``Ministerie van Onderwijs`` must NOT collapse onto ``onderwijs``."""
        assert strip_leading_noise("ministerie van onderwijs") != strip_leading_noise(
            "onderwijs"
        )

    def test_minister_vs_ministerie_distinct(self):
        """Person leader vs org leader stay distinct (neither stripped)."""
        assert strip_leading_noise("minister van bzk") != strip_leading_noise(
            "ministerie van bzk"
        )

    def test_empty_passthrough(self):
        assert strip_leading_noise("") == ""


class TestCollisionSafetyCanary:
    """The must-NOT-merge invariant at the function level (NAME-only, rev4).

    With NO content prefix stripped, every pair below stays distinct at the
    normalized-name level — including the rev4 collision attempts.
    """

    @pytest.mark.parametrize(
        "a,b",
        [
            ("minister van bzk", "minister van financiën"),
            ("gemeente groningen", "gemeente drenthe"),
            ("ministerie van onderwijs", "ministerie van onderwijs en arbeid"),
            ("provincie groningen", "groningen"),
            ("gemeente groningen", "groningen"),
            ("minister van bzk", "ministerie van bzk"),
            # rev4 collision attempts: content-prefix strip would merge these.
            ("ministerie van onderwijs", "onderwijs"),
            ("ministerie van bzk", "bzk"),
        ],
    )
    def test_distinct_names_stay_distinct(self, a, b):
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

    def test_only_articles_are_in_the_strip_set(self):
        """rev4: the strip set is exactly the three articles — no content prefix.

        Any content-bearing leader (``ministerie van``, ``gemeente`` …) in the
        article set would re-introduce a name-only collision risk.
        """
        assert set(_LEADING_ARTICLES) == {"de ", "het ", "een "}
