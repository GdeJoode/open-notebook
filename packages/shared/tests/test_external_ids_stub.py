"""Tests for ``shared.utils.external_ids.resolve_external_ids``.

K.4 upgraded the D.0 stub: ``resolve_external_ids`` now returns the entity's
reconciled ``external_ids`` (populated by the vocabulary reconciler). These tests
hold the **two contracts Track D's exporters depend on**:

1. an entity with NO reconciled identifiers still returns ``[]`` (the original
   empty-input contract — exporters never break);
2. the signature is unchanged: ``Entity -> List[str]``, single import point.

The ``TestReconciledLookups`` class is the K.4 upgrade — it asserts a reconciled
entity emits its real URIs.
"""

from __future__ import annotations

import pytest

from shared.models.entity import Entity
from shared.utils import resolve_external_ids as imported_via_init
from shared.utils.external_ids import resolve_external_ids


@pytest.fixture
def sample_entity() -> Entity:
    """Minimal valid Entity for resolver smoke tests."""
    return Entity(
        canonical_name="Ada Lovelace",
        entity_type="Person",
    )


class TestResolveExternalIds:
    """Un-reconciled entities resolve to [] (the preserved empty-input contract)."""

    def test_acceptance_criterion(self, sample_entity: Entity):
        """The exact case the plan calls out."""
        assert resolve_external_ids(sample_entity) == []

    def test_returns_list_not_none(self, sample_entity: Entity):
        """Callers can safely iterate -- never get a None."""
        result = resolve_external_ids(sample_entity)
        assert isinstance(result, list)

    def test_organization_entity_returns_empty(self):
        """No special-case for Org/Concept/etc. types in V1."""
        org = Entity(canonical_name="MIT", entity_type="Organization")
        assert resolve_external_ids(org) == []

    def test_handles_entity_with_minimal_fields(self):
        """No KeyError when the entity has only the required fields."""
        bare = Entity(canonical_name="x", entity_type="Concept")
        assert resolve_external_ids(bare) == []

    def test_handles_entity_with_rich_metadata(self):
        """The stub does not inspect properties / type_tags / aliases.

        Confirms Q-D-10 (empty aliases for V1) -- the resolver does not
        try to derive identifiers from alias rows even if they exist on
        the entity.
        """
        rich = Entity(
            canonical_name="Albert Einstein",
            entity_type="Person",
            type_tags=["Person", "Researcher", "PhysicsLaureate"],
            primary_type="Person",
            description="20th-century physicist",
            properties={"orcid": "0000-0000-0000-0000"},
        )
        # Even with properties hinting at an ORCID, the V1 stub does
        # not extract them. Q9 will.
        assert resolve_external_ids(rich) == []


class TestPublicAPI:
    """Confirm the single import-point promised by the plan.

    Track D's three exporters (D.1/D.2/D.3) must all reach the resolver
    via ``shared.utils.resolve_external_ids`` -- when Q9 lands, the
    re-export here is what makes the swap a one-file change.
    """

    def test_importable_from_shared_utils(self):
        """``shared.utils.resolve_external_ids`` must work directly."""
        # Same callable object -- re-export, not a wrapper.
        assert imported_via_init is resolve_external_ids

    def test_function_signature_returns_list_of_str(self, sample_entity: Entity):
        """Signature contract: ``Entity -> List[str]``.

        Pin the return-type shape so the swap can't drift to a different
        container without an explicit caller update.
        """
        result = resolve_external_ids(sample_entity)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)


class TestReconciledLookups:
    """K.4 upgrade: a reconciled entity emits its real external URIs."""

    def test_reconciled_entity_returns_uris(self):
        """An entity the reconciler linked carries its URI(s) through."""
        e = Entity(
            canonical_name="Binnenlandse Zaken en Koninkrijksrelaties",
            entity_type="organization",
            external_ids=[
                "https://identifier.overheid.nl/tooi/id/ministerie/mnre1034"
            ],
        )
        assert resolve_external_ids(e) == [
            "https://identifier.overheid.nl/tooi/id/ministerie/mnre1034"
        ]

    def test_multiple_uris_preserved(self):
        e = Entity(
            canonical_name="Some Paper",
            entity_type="scholarly_article",
            external_ids=["https://doi.org/10.1/a", "https://doi.org/10.1/b"],
        )
        assert resolve_external_ids(e) == [
            "https://doi.org/10.1/a",
            "https://doi.org/10.1/b",
        ]

    def test_result_is_a_copy_not_the_field(self):
        """Mutating the result must not mutate the entity's field."""
        e = Entity(
            canonical_name="X",
            entity_type="organization",
            external_ids=["https://doi.org/10.1/x"],
        )
        result = resolve_external_ids(e)
        result.append("https://evil/injected")
        assert e.external_ids == ["https://doi.org/10.1/x"]

    def test_empty_canonical_name_still_empty(self):
        """AC5: an empty-canonical-name entity returns [] (stub contract kept)."""
        e = Entity(canonical_name="", entity_type="organization")
        assert resolve_external_ids(e) == []

    def test_dedups_and_drops_falsy(self):
        e = Entity(
            canonical_name="X",
            entity_type="organization",
            external_ids=["https://doi.org/10.1/x", "https://doi.org/10.1/x"],
        )
        assert resolve_external_ids(e) == ["https://doi.org/10.1/x"]
