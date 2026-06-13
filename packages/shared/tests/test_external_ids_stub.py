"""Tests for ``shared.utils.external_ids.resolve_external_ids``.

V1 stub coverage. The acceptance criterion for Phase D.0 is:

    resolve_external_ids(entity) == []

Beyond that, these tests pin the import surface so the Track M4 Q9 swap
(TOOI + Crossref resolution) only has to swap the function body and
nothing else in the rest of the codebase.

These are the regression-guard tests the Q9 implementation should keep
green for the "stub still resolves to []" call-site contract on entities
that have no external identifier mapping.
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
    """V1 stub returns empty for every input."""

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

        Pin the return-type shape so Q9 can't drift to a different
        container without an explicit caller update.
        """
        result = resolve_external_ids(sample_entity)
        assert isinstance(result, list)
        for item in result:  # vacuous today, but pins the element type
            assert isinstance(item, str)
