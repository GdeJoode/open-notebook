"""Tests for ``shared.utils.name_normalizer.normalize_entity_name``.

V1 stub coverage. The acceptance criterion in B.1c is:

    normalize_entity_name("  Apple Inc.  ") == "apple inc"

Beyond that, these tests pin the three transformations the stub
guarantees so the eventual Q9 replacement has a regression net.
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


class TestPublicAPI:
    """Confirm the single import-point promised by the plan."""

    def test_importable_from_shared_utils(self):
        """``shared.utils.normalize_entity_name`` must work directly."""
        # Same callable object — re-export, not a wrapper.
        assert imported_via_init is normalize_entity_name
