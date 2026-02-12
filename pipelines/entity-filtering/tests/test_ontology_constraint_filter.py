"""Tests for OntologyConstraintFilter ontology-based entity and relation validation."""

import pytest

from entity_filtering.validation.ontology_constraint_filter import (
    OntologyConstraintFilter,
)
from ontology_manager.schema import (
    DataType,
    EntityTypeDefinition,
    Ontology,
    OntologyMetadata,
    PropertyDefinition,
    RelationshipTypeDefinition,
    ValidationRules,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entity(text, label="MISC", properties=None):
    """Helper to build an entity dict in the pipeline's canonical format."""
    return {
        "text": text,
        "label": label,
        "properties": properties.copy() if properties else {},
    }


def _relation(source, target, rel_type="RELATED_TO", properties=None):
    """Helper to build a relation dict in the pipeline's canonical format."""
    return {
        "source_entity": source,
        "target_entity": target,
        "relation_type": rel_type,
        "properties": properties.copy() if properties else {},
    }


def _make_ontology(
    entity_types=None,
    relationship_types=None,
):
    """Helper to build a minimal Ontology for testing."""
    return Ontology(
        metadata=OntologyMetadata(name="test", version="1.0"),
        entity_types=entity_types or {},
        relationship_types=relationship_types or {},
    )


# ---------------------------------------------------------------------------
# Standard test ontology fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_ontology():
    """Ontology with PERSON, ORG entity types and WORKS_FOR relation."""
    return _make_ontology(
        entity_types={
            "PERSON": EntityTypeDefinition(name="PERSON", description="A person"),
            "ORG": EntityTypeDefinition(name="ORG", description="An organization"),
        },
        relationship_types={
            "WORKS_FOR": RelationshipTypeDefinition(
                name="WORKS_FOR",
                description="Employment relationship",
                domain=["PERSON"],
                range=["ORG"],
            ),
        },
    )


@pytest.fixture
def rich_ontology():
    """Ontology with multiple entity types, relation types, and property definitions."""
    return _make_ontology(
        entity_types={
            "PERSON": EntityTypeDefinition(
                name="PERSON",
                description="A person",
                properties=[
                    PropertyDefinition(
                        name="age",
                        data_type=DataType.INTEGER,
                        description="Age in years",
                    ),
                    PropertyDefinition(
                        name="email",
                        data_type=DataType.EMAIL,
                        description="Email address",
                    ),
                ],
            ),
            "ORG": EntityTypeDefinition(name="ORG", description="An organization"),
            "LOC": EntityTypeDefinition(name="LOC", description="A location"),
            "EVENT": EntityTypeDefinition(name="EVENT", description="An event"),
        },
        relationship_types={
            "WORKS_FOR": RelationshipTypeDefinition(
                name="WORKS_FOR",
                description="Employment",
                domain=["PERSON"],
                range=["ORG"],
            ),
            "LOCATED_IN": RelationshipTypeDefinition(
                name="LOCATED_IN",
                description="Location relationship",
                domain=["PERSON", "ORG"],
                range=["LOC"],
            ),
            "ATTENDED": RelationshipTypeDefinition(
                name="ATTENDED",
                description="Attended an event",
                domain=["PERSON"],
                range=["EVENT"],
            ),
        },
    )


# ---------------------------------------------------------------------------
# Report key expectations
# ---------------------------------------------------------------------------

EXPECTED_REPORT_KEYS = {
    "total_entities",
    "total_relations",
    "valid_entities",
    "invalid_entities",
    "valid_relations",
    "invalid_relations",
    "entity_issues",
    "relation_issues",
}


# ===================================================================
# No ontology provided (no-op mode)
# ===================================================================


class TestNoOntology:
    """When no ontology is provided, the filter should be a no-op."""

    def test_no_ontology_returns_all_entities(self):
        """Without an ontology, all entities pass through unchanged."""
        ocf = OntologyConstraintFilter(ontology=None)
        entities = [_entity("Alice", "PERSON"), _entity("Acme", "ORG")]
        relations = [_relation("Alice", "Acme", "WORKS_FOR")]

        valid_ents, valid_rels, report = ocf.filter(entities, relations)

        assert len(valid_ents) == 2, "All entities should pass through without ontology"
        assert len(valid_rels) == 1, "All relations should pass through without ontology"

    def test_no_ontology_report_counts_correct(self):
        """Report should show all entities/relations as valid when no ontology."""
        ocf = OntologyConstraintFilter(ontology=None)
        entities = [_entity("A"), _entity("B")]
        relations = [_relation("A", "B")]

        _, _, report = ocf.filter(entities, relations)

        assert report["total_entities"] == 2
        assert report["valid_entities"] == 2
        assert report["invalid_entities"] == 0
        assert report["total_relations"] == 1
        assert report["valid_relations"] == 1
        assert report["invalid_relations"] == 0

    def test_no_ontology_empty_issues(self):
        """No issues should be reported when no ontology."""
        ocf = OntologyConstraintFilter(ontology=None)
        _, _, report = ocf.filter([_entity("A")], [])
        assert report["entity_issues"] == []
        assert report["relation_issues"] == []


# ===================================================================
# Empty inputs
# ===================================================================


class TestEmptyInputs:
    """Tests for empty entity/relation lists."""

    def test_empty_entities_and_relations(self, basic_ontology):
        """Empty inputs return empty results with zeroed report."""
        ocf = OntologyConstraintFilter(ontology=basic_ontology)
        valid_ents, valid_rels, report = ocf.filter([], [])

        assert valid_ents == []
        assert valid_rels == []
        assert report["total_entities"] == 0
        assert report["total_relations"] == 0

    def test_report_structure_on_empty(self, basic_ontology):
        """Report contains all expected keys even on empty input."""
        ocf = OntologyConstraintFilter(ontology=basic_ontology)
        _, _, report = ocf.filter([], [])
        assert set(report.keys()) == EXPECTED_REPORT_KEYS


# ===================================================================
# Valid entities pass through
# ===================================================================


class TestValidEntities:
    """Tests for entities that match ontology definitions."""

    def test_valid_entity_passes(self, basic_ontology):
        """An entity with a known type should pass validation."""
        ocf = OntologyConstraintFilter(ontology=basic_ontology, strict=True)
        entities = [_entity("Alice", "PERSON")]
        valid_ents, _, report = ocf.filter(entities, [])

        assert len(valid_ents) == 1
        assert valid_ents[0]["text"] == "Alice"
        assert report["valid_entities"] == 1
        assert report["invalid_entities"] == 0

    def test_multiple_valid_entities_pass(self, basic_ontology):
        """Multiple entities with known types should all pass."""
        ocf = OntologyConstraintFilter(ontology=basic_ontology, strict=True)
        entities = [
            _entity("Alice", "PERSON"),
            _entity("Bob", "PERSON"),
            _entity("Acme", "ORG"),
        ]
        valid_ents, _, report = ocf.filter(entities, [])

        assert len(valid_ents) == 3
        assert report["valid_entities"] == 3


# ===================================================================
# Invalid entity type handling
# ===================================================================


class TestInvalidEntityType:
    """Tests for entities with types not defined in the ontology."""

    def test_unknown_type_removed_in_strict_mode(self, basic_ontology):
        """In strict mode, an entity with unknown type should be removed."""
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=True, filter_invalid_entities=True,
        )
        entities = [_entity("Volcano", "GEOLOGICAL_FEATURE")]
        valid_ents, _, report = ocf.filter(entities, [])

        assert len(valid_ents) == 0, "Unknown entity type should be removed in strict mode"
        assert report["invalid_entities"] == 1

    def test_unknown_type_kept_in_non_strict_mode(self, basic_ontology):
        """In non-strict mode, unknown types produce warnings, not errors.
        Since the filter only removes entities with errors, they are kept.
        """
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=False, filter_invalid_entities=True,
        )
        entities = [_entity("Volcano", "GEOLOGICAL_FEATURE")]
        valid_ents, _, report = ocf.filter(entities, [])

        assert len(valid_ents) == 1, (
            "Non-strict mode: unknown type produces warning, not error; entity kept"
        )

    def test_strict_unknown_entity_issue_recorded(self, basic_ontology):
        """Entity issues should contain details about the rejected entity."""
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=True, filter_invalid_entities=True,
        )
        entities = [_entity("Volcano", "GEOLOGICAL_FEATURE")]
        _, _, report = ocf.filter(entities, [])

        assert len(report["entity_issues"]) == 1
        issue = report["entity_issues"][0]
        assert issue["entity_text"] == "Volcano"
        assert issue["entity_label"] == "GEOLOGICAL_FEATURE"
        assert len(issue["issues"]) > 0
        assert issue["issues"][0]["severity"] == "error"

    def test_filter_invalid_entities_false_keeps_invalid(self, basic_ontology):
        """With filter_invalid_entities=False, invalid entities are kept."""
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=True, filter_invalid_entities=False,
        )
        entities = [_entity("Volcano", "GEOLOGICAL_FEATURE")]
        valid_ents, _, _ = ocf.filter(entities, [])

        assert len(valid_ents) == 1, (
            "filter_invalid_entities=False should keep invalid entities"
        )


# ===================================================================
# Valid relations pass through
# ===================================================================


class TestValidRelations:
    """Tests for relations that match ontology definitions."""

    def test_valid_relation_passes(self, basic_ontology):
        """A relation with known predicate and correct domain/range passes."""
        ocf = OntologyConstraintFilter(ontology=basic_ontology, strict=True)
        entities = [_entity("Alice", "PERSON"), _entity("Acme", "ORG")]
        relations = [_relation("Alice", "Acme", "WORKS_FOR")]

        _, valid_rels, report = ocf.filter(entities, relations)

        assert len(valid_rels) == 1
        assert report["valid_relations"] == 1
        assert report["invalid_relations"] == 0


# ===================================================================
# Invalid relation predicate handling
# ===================================================================


class TestInvalidRelationPredicate:
    """Tests for relations with unknown predicate types."""

    def test_unknown_predicate_removed_in_strict_mode(self, basic_ontology):
        """In strict mode, a relation with unknown predicate produces an error
        and is removed.
        """
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=True, filter_invalid_relations=True,
        )
        entities = [_entity("Alice", "PERSON"), _entity("Bob", "PERSON")]
        relations = [_relation("Alice", "Bob", "MENTORS")]

        _, valid_rels, report = ocf.filter(entities, relations)

        assert len(valid_rels) == 0, (
            "Unknown predicate should be removed in strict mode"
        )
        assert report["invalid_relations"] == 1

    def test_unknown_predicate_kept_in_non_strict_mode(self, basic_ontology):
        """In non-strict mode, unknown predicate is a warning, so relation is kept."""
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=False, filter_invalid_relations=True,
        )
        entities = [_entity("Alice", "PERSON"), _entity("Bob", "PERSON")]
        relations = [_relation("Alice", "Bob", "MENTORS")]

        _, valid_rels, _ = ocf.filter(entities, relations)

        assert len(valid_rels) == 1, (
            "Non-strict mode: unknown predicate produces warning, relation kept"
        )

    def test_strict_relation_issue_recorded(self, basic_ontology):
        """Relation issues should contain details about the rejected relation."""
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=True, filter_invalid_relations=True,
        )
        entities = [_entity("Alice", "PERSON"), _entity("Bob", "PERSON")]
        relations = [_relation("Alice", "Bob", "MENTORS")]

        _, _, report = ocf.filter(entities, relations)

        assert len(report["relation_issues"]) == 1
        issue = report["relation_issues"][0]
        assert issue["source"] == "Alice"
        assert issue["target"] == "Bob"
        assert issue["relation_type"] == "MENTORS"
        assert len(issue["issues"]) > 0

    def test_filter_invalid_relations_false_keeps_invalid(self, basic_ontology):
        """With filter_invalid_relations=False, invalid relations are kept."""
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=True, filter_invalid_relations=False,
        )
        entities = [_entity("Alice", "PERSON"), _entity("Bob", "PERSON")]
        relations = [_relation("Alice", "Bob", "MENTORS")]

        _, valid_rels, _ = ocf.filter(entities, relations)

        assert len(valid_rels) == 1, (
            "filter_invalid_relations=False should keep invalid relations"
        )


# ===================================================================
# Domain/range validation
# ===================================================================


class TestDomainRangeValidation:
    """Tests for domain and range constraint validation.

    Note: The OntologyValidator reports domain/range violations as
    WARNINGS, not ERRORS. Since OntologyConstraintFilter only removes
    entities/relations with has_errors() == True, domain/range violations
    alone do NOT cause removal. They appear only in the validation report.
    """

    def test_wrong_domain_produces_warning_not_removal(self, basic_ontology):
        """A relation with wrong source type (domain) stays because domain
        violations are warnings, not errors.
        """
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=False, filter_invalid_relations=True,
        )
        # WORKS_FOR domain is [PERSON], but we use ORG as source
        entities = [_entity("Acme", "ORG"), _entity("BigCorp", "ORG")]
        relations = [_relation("Acme", "BigCorp", "WORKS_FOR")]

        _, valid_rels, report = ocf.filter(entities, relations)

        # The relation has a valid predicate (WORKS_FOR is known), so no error.
        # Domain violation is a warning only.
        assert len(valid_rels) == 1, (
            "Domain violation is a warning; relation should NOT be removed"
        )

    def test_wrong_range_produces_warning_not_removal(self, basic_ontology):
        """A relation with wrong target type (range) stays because range
        violations are warnings, not errors.
        """
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=False, filter_invalid_relations=True,
        )
        # WORKS_FOR range is [ORG], but we use PERSON as target
        entities = [_entity("Alice", "PERSON"), _entity("Bob", "PERSON")]
        relations = [_relation("Alice", "Bob", "WORKS_FOR")]

        _, valid_rels, report = ocf.filter(entities, relations)

        assert len(valid_rels) == 1, (
            "Range violation is a warning; relation should NOT be removed"
        )

    def test_wrong_domain_removed_when_strict_unknown_predicate(self, basic_ontology):
        """In strict mode, a completely unknown predicate IS an error and
        gets the relation removed.
        """
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=True, filter_invalid_relations=True,
        )
        entities = [_entity("Acme", "ORG"), _entity("BigCorp", "ORG")]
        relations = [_relation("Acme", "BigCorp", "ACQUIRES")]

        _, valid_rels, report = ocf.filter(entities, relations)

        assert len(valid_rels) == 0, (
            "Strict mode: unknown predicate 'ACQUIRES' is an error, relation removed"
        )
        assert report["invalid_relations"] == 1


# ===================================================================
# Cascade removal: relations removed when endpoint entities removed
# ===================================================================


class TestCascadeRemoval:
    """Tests for cascade behavior when entities are removed."""

    def test_relation_removed_when_source_entity_removed(self, basic_ontology):
        """If the source entity is removed, relations from it are also removed."""
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=True, filter_invalid_entities=True,
        )
        entities = [
            _entity("Volcano", "GEOLOGICAL_FEATURE"),  # unknown type -> removed
            _entity("Acme", "ORG"),
        ]
        relations = [_relation("Volcano", "Acme", "WORKS_FOR")]

        valid_ents, valid_rels, report = ocf.filter(entities, relations)

        assert len(valid_ents) == 1, "Only ORG entity should remain"
        assert valid_ents[0]["text"] == "Acme"
        assert len(valid_rels) == 0, (
            "Relation should be cascade-removed because source entity was removed"
        )
        assert report["invalid_relations"] >= 1

    def test_relation_removed_when_target_entity_removed(self, basic_ontology):
        """If the target entity is removed, relations to it are also removed."""
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=True, filter_invalid_entities=True,
        )
        entities = [
            _entity("Alice", "PERSON"),
            _entity("UnknownThing", "WHATEVER"),  # unknown type -> removed
        ]
        relations = [_relation("Alice", "UnknownThing", "WORKS_FOR")]

        valid_ents, valid_rels, report = ocf.filter(entities, relations)

        assert len(valid_ents) == 1
        assert len(valid_rels) == 0, (
            "Relation should be cascade-removed because target entity was removed"
        )

    def test_cascade_does_not_affect_valid_relations(self, basic_ontology):
        """Relations between valid entities should not be affected by cascade."""
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=True, filter_invalid_entities=True,
        )
        entities = [
            _entity("Alice", "PERSON"),
            _entity("Acme", "ORG"),
            _entity("BadEntity", "UNKNOWN_TYPE"),  # will be removed
        ]
        relations = [
            _relation("Alice", "Acme", "WORKS_FOR"),           # valid
            _relation("BadEntity", "Acme", "WORKS_FOR"),       # cascade removed
        ]

        valid_ents, valid_rels, report = ocf.filter(entities, relations)

        assert len(valid_ents) == 2
        assert len(valid_rels) == 1
        assert valid_rels[0]["source_entity"] == "Alice"


# ===================================================================
# Report structure and counts
# ===================================================================


class TestReportStructure:
    """Tests for the validation report format and accuracy."""

    def test_report_keys(self, basic_ontology):
        """Report should contain all expected keys."""
        ocf = OntologyConstraintFilter(ontology=basic_ontology)
        _, _, report = ocf.filter([_entity("A", "PERSON")], [])
        assert set(report.keys()) == EXPECTED_REPORT_KEYS

    def test_report_total_counts(self, basic_ontology):
        """Total counts should reflect input sizes."""
        ocf = OntologyConstraintFilter(ontology=basic_ontology)
        entities = [_entity("A", "PERSON"), _entity("B", "ORG")]
        relations = [_relation("A", "B", "WORKS_FOR")]
        _, _, report = ocf.filter(entities, relations)

        assert report["total_entities"] == 2
        assert report["total_relations"] == 1

    def test_report_valid_counts_match_output(self, basic_ontology):
        """valid_entities and valid_relations counts should match output list lengths."""
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=True, filter_invalid_entities=True,
        )
        entities = [
            _entity("Alice", "PERSON"),
            _entity("Volcano", "GEOLOGICAL_FEATURE"),
        ]
        relations = []

        valid_ents, _, report = ocf.filter(entities, relations)

        assert report["valid_entities"] == len(valid_ents)
        assert report["invalid_entities"] == 1
        assert report["valid_entities"] + report["invalid_entities"] == report["total_entities"]

    def test_entity_issue_structure(self, basic_ontology):
        """Each entity issue dict should have the expected fields."""
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=True, filter_invalid_entities=True,
        )
        entities = [_entity("Volcano", "GEOLOGICAL_FEATURE")]
        _, _, report = ocf.filter(entities, [])

        assert len(report["entity_issues"]) == 1
        issue = report["entity_issues"][0]
        assert "entity_text" in issue
        assert "entity_label" in issue
        assert "issues" in issue
        assert isinstance(issue["issues"], list)
        # Each sub-issue should have severity, message, path
        sub = issue["issues"][0]
        assert "severity" in sub
        assert "message" in sub
        assert "path" in sub

    def test_relation_issue_structure(self, basic_ontology):
        """Each relation issue dict should have the expected fields."""
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=True, filter_invalid_relations=True,
        )
        entities = [_entity("Alice", "PERSON"), _entity("Bob", "PERSON")]
        relations = [_relation("Alice", "Bob", "MENTORS")]
        _, _, report = ocf.filter(entities, relations)

        assert len(report["relation_issues"]) == 1
        issue = report["relation_issues"][0]
        assert "source" in issue
        assert "target" in issue
        assert "relation_type" in issue
        assert "issues" in issue
        sub = issue["issues"][0]
        assert "severity" in sub
        assert "message" in sub
        assert "path" in sub


# ===================================================================
# Properties preservation
# ===================================================================


class TestPropertiesPreservation:
    """Tests that entity and relation properties survive filtering."""

    def test_valid_entity_properties_preserved(self, basic_ontology):
        """Properties on valid entities should not be modified."""
        ocf = OntologyConstraintFilter(ontology=basic_ontology)
        entities = [_entity("Alice", "PERSON", properties={"age": 30, "role": "dev"})]
        valid_ents, _, _ = ocf.filter(entities, [])

        assert valid_ents[0]["properties"]["age"] == 30
        assert valid_ents[0]["properties"]["role"] == "dev"

    def test_valid_relation_properties_preserved(self, basic_ontology):
        """Properties on valid relations should not be modified."""
        ocf = OntologyConstraintFilter(ontology=basic_ontology)
        entities = [_entity("Alice", "PERSON"), _entity("Acme", "ORG")]
        relations = [_relation("Alice", "Acme", "WORKS_FOR", properties={"since": 2020})]

        _, valid_rels, _ = ocf.filter(entities, relations)

        assert valid_rels[0]["properties"]["since"] == 2020


# ===================================================================
# Mixed valid/invalid batch
# ===================================================================


class TestMixedBatch:
    """Tests with a mix of valid and invalid entities/relations."""

    def test_mixed_entities_filtered_correctly(self, basic_ontology):
        """Only invalid entities should be removed; valid ones stay."""
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=True, filter_invalid_entities=True,
        )
        entities = [
            _entity("Alice", "PERSON"),      # valid
            _entity("Acme", "ORG"),           # valid
            _entity("Volcano", "GEO"),        # invalid
            _entity("Bob", "PERSON"),         # valid
        ]
        valid_ents, _, report = ocf.filter(entities, [])

        valid_texts = {e["text"] for e in valid_ents}
        assert valid_texts == {"Alice", "Acme", "Bob"}
        assert report["invalid_entities"] == 1
        assert report["valid_entities"] == 3

    def test_mixed_relations_filtered_correctly(self, basic_ontology):
        """Only invalid relations (or cascade-removed) should be removed."""
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=True,
            filter_invalid_entities=True, filter_invalid_relations=True,
        )
        entities = [
            _entity("Alice", "PERSON"),
            _entity("Acme", "ORG"),
            _entity("Bob", "PERSON"),
        ]
        relations = [
            _relation("Alice", "Acme", "WORKS_FOR"),   # valid predicate
            _relation("Alice", "Bob", "FRIENDS_WITH"),  # unknown predicate -> error in strict
        ]
        _, valid_rels, report = ocf.filter(entities, relations)

        assert len(valid_rels) == 1
        assert valid_rels[0]["relation_type"] == "WORKS_FOR"
        assert report["invalid_relations"] == 1


# ===================================================================
# Multiple entity types and relation types
# ===================================================================


class TestMultipleTypes:
    """Tests with richer ontologies containing multiple types."""

    def test_multiple_entity_types(self, rich_ontology):
        """All known entity types should be accepted."""
        ocf = OntologyConstraintFilter(ontology=rich_ontology, strict=True)
        entities = [
            _entity("Alice", "PERSON"),
            _entity("Acme", "ORG"),
            _entity("Paris", "LOC"),
            _entity("PyCon", "EVENT"),
        ]
        valid_ents, _, report = ocf.filter(entities, [])

        assert len(valid_ents) == 4
        assert report["invalid_entities"] == 0

    def test_multiple_relation_types(self, rich_ontology):
        """Multiple relation types should all be accepted when valid."""
        ocf = OntologyConstraintFilter(ontology=rich_ontology, strict=True)
        entities = [
            _entity("Alice", "PERSON"),
            _entity("Acme", "ORG"),
            _entity("Paris", "LOC"),
            _entity("PyCon", "EVENT"),
        ]
        relations = [
            _relation("Alice", "Acme", "WORKS_FOR"),
            _relation("Alice", "Paris", "LOCATED_IN"),
            _relation("Alice", "PyCon", "ATTENDED"),
        ]
        _, valid_rels, report = ocf.filter(entities, relations)

        assert len(valid_rels) == 3
        assert report["invalid_relations"] == 0


# ===================================================================
# Ontology with property definitions
# ===================================================================


class TestOntologyPropertyDefinitions:
    """Tests for entities with properties matching ontology property definitions."""

    def test_entity_with_matching_properties_passes(self, rich_ontology):
        """An entity whose properties match the ontology definitions should pass."""
        ocf = OntologyConstraintFilter(ontology=rich_ontology, strict=True)
        entities = [
            _entity(
                "Alice", "PERSON",
                properties={"age": 30, "email": "alice@example.com"},
            ),
        ]
        valid_ents, _, report = ocf.filter(entities, [])

        assert len(valid_ents) == 1
        assert report["invalid_entities"] == 0

    def test_entity_with_extra_properties_passes(self, rich_ontology):
        """Extra properties not in the ontology should produce INFO, not errors."""
        ocf = OntologyConstraintFilter(ontology=rich_ontology, strict=True)
        entities = [
            _entity(
                "Alice", "PERSON",
                properties={"age": 30, "favorite_color": "blue"},
            ),
        ]
        valid_ents, _, report = ocf.filter(entities, [])

        # Unknown properties produce INFO-level issues, not errors
        assert len(valid_ents) == 1


# ===================================================================
# Format translation
# ===================================================================


class TestFormatTranslation:
    """Tests that verify the pipeline-to-validator format translation works."""

    def test_entity_label_translated_to_type(self, basic_ontology):
        """entity['label'] should be translated to validator entity['type']."""
        ocf = OntologyConstraintFilter(ontology=basic_ontology, strict=True)
        # If translation fails, PERSON would not be recognized
        entities = [_entity("Alice", "PERSON")]
        valid_ents, _, _ = ocf.filter(entities, [])
        assert len(valid_ents) == 1

    def test_entity_text_translated_to_name(self, basic_ontology):
        """entity['text'] should be translated to validator entity['name']."""
        ocf = OntologyConstraintFilter(ontology=basic_ontology, strict=True)
        entities = [_entity("Alice", "PERSON")]
        valid_ents, _, _ = ocf.filter(entities, [])
        # If name translation failed, validator would report an error
        assert len(valid_ents) == 1

    def test_relation_type_translated_to_predicate(self, basic_ontology):
        """relation['relation_type'] -> validator rel['predicate']."""
        ocf = OntologyConstraintFilter(ontology=basic_ontology, strict=True)
        entities = [_entity("Alice", "PERSON"), _entity("Acme", "ORG")]
        relations = [_relation("Alice", "Acme", "WORKS_FOR")]
        _, valid_rels, _ = ocf.filter(entities, relations)
        assert len(valid_rels) == 1

    def test_source_target_translated_to_subject_object(self, basic_ontology):
        """source_entity/target_entity -> subject/object in validator."""
        ocf = OntologyConstraintFilter(ontology=basic_ontology, strict=True)
        entities = [_entity("Alice", "PERSON"), _entity("Acme", "ORG")]
        relations = [_relation("Alice", "Acme", "WORKS_FOR")]
        _, valid_rels, _ = ocf.filter(entities, relations)
        # If subject/object translation failed, validator would reject
        assert len(valid_rels) == 1


# ===================================================================
# Parametrized tests
# ===================================================================


class TestParametrized:
    """Parametrized tests covering various flag combinations."""

    @pytest.mark.parametrize(
        "strict,filter_ent,filter_rel",
        [
            (True, True, True),
            (True, False, True),
            (True, True, False),
            (False, True, True),
            (False, False, False),
        ],
        ids=[
            "strict-all-filtering",
            "strict-no-entity-filtering",
            "strict-no-relation-filtering",
            "non-strict-all-filtering",
            "non-strict-no-filtering",
        ],
    )
    def test_valid_data_passes_regardless_of_flags(
        self, basic_ontology, strict, filter_ent, filter_rel
    ):
        """Valid entities and relations should always pass regardless of flags."""
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology,
            strict=strict,
            filter_invalid_entities=filter_ent,
            filter_invalid_relations=filter_rel,
        )
        entities = [_entity("Alice", "PERSON"), _entity("Acme", "ORG")]
        relations = [_relation("Alice", "Acme", "WORKS_FOR")]

        valid_ents, valid_rels, _ = ocf.filter(entities, relations)

        assert len(valid_ents) == 2
        assert len(valid_rels) == 1

    @pytest.mark.parametrize(
        "entity_label,should_pass_strict",
        [
            ("PERSON", True),
            ("ORG", True),
            ("UNKNOWN", False),
            ("", False),
        ],
        ids=["known-person", "known-org", "unknown-type", "empty-type"],
    )
    def test_entity_type_strict_validation(
        self, basic_ontology, entity_label, should_pass_strict
    ):
        """Strict mode should only pass known entity types."""
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=True, filter_invalid_entities=True,
        )
        entities = [_entity("TestEntity", entity_label)]
        valid_ents, _, _ = ocf.filter(entities, [])

        if should_pass_strict:
            assert len(valid_ents) == 1, (
                f"Entity with label '{entity_label}' should pass strict validation"
            )
        else:
            assert len(valid_ents) == 0, (
                f"Entity with label '{entity_label}' should fail strict validation"
            )

    @pytest.mark.parametrize(
        "rel_type,should_pass_strict",
        [
            ("WORKS_FOR", True),
            ("UNKNOWN_REL", False),
        ],
        ids=["known-relation", "unknown-relation"],
    )
    def test_relation_type_strict_validation(
        self, basic_ontology, rel_type, should_pass_strict
    ):
        """Strict mode should only pass known relation predicates."""
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=True, filter_invalid_relations=True,
        )
        entities = [_entity("Alice", "PERSON"), _entity("Acme", "ORG")]
        relations = [_relation("Alice", "Acme", rel_type)]

        _, valid_rels, _ = ocf.filter(entities, relations)

        if should_pass_strict:
            assert len(valid_rels) == 1
        else:
            assert len(valid_rels) == 0


# ===================================================================
# Entity with missing text/label edge cases
# ===================================================================


class TestEdgeCases:
    """Edge case tests for unusual entity/relation data."""

    def test_entity_with_empty_text(self, basic_ontology):
        """An entity with empty text should fail (name required by validator)."""
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=True, filter_invalid_entities=True,
        )
        entities = [_entity("", "PERSON")]
        valid_ents, _, report = ocf.filter(entities, [])

        # The validator requires entity["name"] to be non-empty
        assert len(valid_ents) == 0
        assert report["invalid_entities"] == 1

    def test_entity_with_empty_label(self, basic_ontology):
        """An entity with empty label should fail (type required by validator)."""
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=True, filter_invalid_entities=True,
        )
        entities = [_entity("Alice", "")]
        valid_ents, _, report = ocf.filter(entities, [])

        assert len(valid_ents) == 0
        assert report["invalid_entities"] == 1

    def test_relation_with_missing_endpoint_entity(self, basic_ontology):
        """A relation whose endpoint was never in the entity list is cascade-removed."""
        ocf = OntologyConstraintFilter(
            ontology=basic_ontology, strict=True,
        )
        entities = [_entity("Alice", "PERSON")]
        # "NonExistent" is not in the entity list
        relations = [_relation("Alice", "NonExistent", "WORKS_FOR")]

        _, valid_rels, report = ocf.filter(entities, relations)

        assert len(valid_rels) == 0, (
            "Relation with missing target entity should be cascade-removed"
        )
        assert report["invalid_relations"] == 1
