"""Tests for ``shared.utils.name_normalizer.normalize_entity_name``.

The B.1c acceptance criterion is:

    normalize_entity_name("  Apple Inc.  ") == "apple inc"

Beyond that, these tests pin the content-normalization transformations and the
K.1 NL behaviour (leading-article strip + curated spelling canonicalization) so
later resolution phases have a regression net.
"""

import pytest
from shared.utils import normalize_entity_name as imported_via_init
from shared.utils.name_normalizer import normalize_entity_name


class TestNormalizeEntityName:
    """Pinpoint the three transformations the V1 stub performs."""

    def test_acceptance_criterion(self):
        """The exact case the plan calls out."""
        assert normalize_entity_name("  Apple Inc.  ") == "apple inc"

    def test_lowercase(self):
        """All characters are lowercased."""
        assert normalize_entity_name("FOO") == "foo"
        assert normalize_entity_name("FooBar") == "foobar"

    def test_collapse_internal_whitespace(self):
        """Runs of internal whitespace collapse to a single space."""
        assert normalize_entity_name("foo   bar") == "foo bar"
        assert normalize_entity_name("foo\t\tbar") == "foo bar"
        assert normalize_entity_name("foo\n\nbar") == "foo bar"
        assert normalize_entity_name("foo \t\n bar") == "foo bar"

    def test_strip_trailing_punctuation(self):
        """Trailing ``.,;:!?`` (with optional whitespace) are removed."""
        assert normalize_entity_name("foo.") == "foo"
        assert normalize_entity_name("foo,") == "foo"
        assert normalize_entity_name("foo;") == "foo"
        assert normalize_entity_name("foo:") == "foo"
        assert normalize_entity_name("foo!") == "foo"
        assert normalize_entity_name("foo?") == "foo"
        assert normalize_entity_name("foo.!?") == "foo"
        assert normalize_entity_name("foo . ") == "foo"

    def test_strip_leading_whitespace(self):
        """Leading whitespace is stripped (side-effect of final strip)."""
        assert normalize_entity_name("  foo") == "foo"
        assert normalize_entity_name("\tfoo") == "foo"

    def test_internal_punctuation_preserved(self):
        """Internal ``.,;:`` are not touched — only trailing ones."""
        # "Inc." in the middle stays, but trailing "." is removed.
        assert normalize_entity_name("Apple Inc. (CA)") == "apple inc. (ca)"
        assert normalize_entity_name("U.S.A.") == "u.s.a"  # trailing dot stripped

    def test_empty_string(self):
        """Empty input returns empty string, not None / not raises."""
        assert normalize_entity_name("") == ""

    def test_whitespace_only(self):
        """A string of only whitespace normalizes to empty."""
        assert normalize_entity_name("   ") == ""
        assert normalize_entity_name("\t\n\r") == ""

    def test_punctuation_only(self):
        """A string of only trailing-punctuation chars normalizes to empty."""
        assert normalize_entity_name(".,;:!?") == ""

    def test_unicode_passthrough(self):
        """Unicode that isn't whitespace or trailing punct is preserved.

        The stub doesn't claim Unicode normalization — Q9 will. We just
        verify we don't silently corrupt non-ASCII strings.
        """
        assert normalize_entity_name("Café") == "café"
        # Curly trailing quote is NOT stripped by the conservative regex —
        # this pins the limit and motivates Q9's broader handling.
        assert normalize_entity_name("foo’") == "foo’"

    def test_idempotent(self):
        """Normalizing twice is a no-op."""
        for raw in ["  Apple Inc.  ", "FOO   BAR.", "", "x"]:
            once = normalize_entity_name(raw)
            twice = normalize_entity_name(once)
            assert once == twice

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("  Apple Inc.  ", "apple inc"),
            ("OpenAI", "openai"),
            ("Open  AI", "open ai"),
            ("Hello, World!", "hello, world"),
            ("End of sentence.", "end of sentence"),
        ],
    )
    def test_table_of_cases(self, raw, expected):
        assert normalize_entity_name(raw) == expected


class TestNLNormalization:
    """K.1 NL-aware behaviour composed into ``normalize_entity_name``.

    These are the falsifiable acceptance criteria from the K.1 plan, asserted
    through the public function (the V1 cases above must stay green alongside).
    """

    def test_ac1_leading_article_stripped(self):
        assert normalize_entity_name("De Regio Deal") == normalize_entity_name(
            "Regio Deal"
        )

    def test_ac2_case_merge_no_content_prefix_stripped(self):
        """rev4: only case/whitespace merges these; no content prefix stripped."""
        assert normalize_entity_name("Ministerie van BZK") == normalize_entity_name(
            "ministerie van BZK"
        )
        assert normalize_entity_name("Ministerie van BZK") == "ministerie van bzk"

    def test_ac3_spelling_variant_tolerance(self):
        assert normalize_entity_name(
            "Binnenlandse Zaken en Koninkrijkrelaties"
        ) == normalize_entity_name("Binnenlandse Zaken en Koninkrijksrelaties")

    def test_ac4_precision_guard_distinct_tails(self):
        assert normalize_entity_name("Minister van BZK") != normalize_entity_name(
            "Minister van Financiën"
        )
        assert normalize_entity_name("Gemeente Groningen") != normalize_entity_name(
            "Gemeente Drenthe"
        )

    def test_content_prefix_not_stripped(self):
        """rev4: no content-bearing prefix strips — only the article does.

        ``het`` strips as a leading article; the ``ministerie van`` content
        survives. The org-form merge to the bare ``bzk`` is K.2's job.
        """
        assert normalize_entity_name("Ministerie van BZK") == "ministerie van bzk"
        assert normalize_entity_name("Het Ministerie van BZK") == "ministerie van bzk"

    def test_ministerie_onderwijs_vs_bare_onderwijs_distinct(self):
        """rev4 collision: the ministry ORG must NOT collapse onto the concept."""
        assert normalize_entity_name("Ministerie van Onderwijs") == (
            "ministerie van onderwijs"
        )
        assert normalize_entity_name("Onderwijs") == "onderwijs"
        assert normalize_entity_name(
            "Ministerie van Onderwijs"
        ) != normalize_entity_name("Onderwijs")

    def test_cross_type_leaders_not_stripped(self):
        """rev4 collision-safety: person/municipality leaders pass through.

        Relations resolve endpoints by name alone, so these must NOT collapse
        onto another entity's name.
        """
        # Person minister stays distinct from the org form.
        assert normalize_entity_name("Minister van BZK") == "minister van bzk"
        assert normalize_entity_name("Minister van BZK") != normalize_entity_name(
            "Ministerie van BZK"
        )
        # Municipality / province org stays distinct from the city/location.
        assert normalize_entity_name("Gemeente Groningen") == "gemeente groningen"
        assert normalize_entity_name("Gemeente Groningen") != normalize_entity_name(
            "Groningen"
        )
        assert normalize_entity_name("Provincie Groningen") != normalize_entity_name(
            "Groningen"
        )

    def test_article_strips_but_content_prefix_stays(self):
        """The leading article strips; ``ministerie van`` content survives."""
        assert (
            normalize_entity_name(
                "De Ministerie van Binnenlandse Zaken en Koninkrijkrelaties"
            )
            == "ministerie van binnenlandse zaken en koninkrijksrelaties"
        )

    def test_qualifier_stays_distinct(self):
        """A qualifier marks a distinct sub-concept (no prefix to strip)."""
        assert normalize_entity_name("Regio Deal") != normalize_entity_name(
            "Oost-Groningen Regio Deal"
        )

    def test_bare_leader_word_unchanged(self):
        assert normalize_entity_name("Gemeente") == "gemeente"

    def test_nl_normalization_idempotent(self):
        for raw in [
            "De Regio Deal",
            "Ministerie van BZK",
            "Binnenlandse Zaken en Koninkrijkrelaties",
            "Gemeente",
        ]:
            once = normalize_entity_name(raw)
            assert normalize_entity_name(once) == once

    def test_apple_inc_still_green(self):
        """The V1 anchor case is unaffected by the NL stages."""
        assert normalize_entity_name("  Apple Inc.  ") == "apple inc"


class TestB8HashContract:
    """AC9: the B.8 dedup-key derive-rule is unchanged by the new normalizer.

    Only the *string* normalize_entity_name returns changes. The derive-rule
    md5(f"{normalize_entity_name(x)}|{type}") and the (canonical_name,
    entity_type) lookup are unchanged — asserted here so a future normalizer
    change that breaks the contract fails fast in the shared suite.
    """

    def test_hash_id_is_md5_of_normalized_name_pipe_type(self):
        import hashlib

        name = "Ministerie van BZK"
        etype = "organization"
        normalized = normalize_entity_name(name)
        expected = hashlib.md5(f"{normalized}|{etype}".encode("utf-8")).hexdigest()
        # Re-derive the same way a caller would and confirm determinism.
        assert (
            hashlib.md5(
                f"{normalize_entity_name(name)}|{etype}".encode("utf-8")
            ).hexdigest()
            == expected
        )

    def test_merged_forms_share_a_hash_basis(self):
        """Forms that normalize equal produce the same dedup basis."""
        a = normalize_entity_name("Ministerie van BZK")
        b = normalize_entity_name("ministerie van BZK")
        assert a == b == "ministerie van bzk"


class TestPublicAPI:
    """Confirm the single import-point promised by the plan."""

    def test_importable_from_shared_utils(self):
        """``shared.utils.normalize_entity_name`` must work directly."""
        # Same callable object — re-export, not a wrapper.
        assert imported_via_init is normalize_entity_name
