"""Tests for SHACL-like ontology validation."""

import pytest

from ontology_manager.schema import (
    DataType,
    EntityTypeDefinition,
    Ontology,
    OntologyMetadata,
    PropertyDefinition,
    RelationshipTypeDefinition,
    ValidationRules,
)
from ontology_manager.validator import (
    OntologyValidator,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)


def make_test_ontology():
    """Create a test ontology for validation tests."""
    return Ontology(
        metadata=OntologyMetadata(name="test", version="1.0", description="Test ontology"),
        entity_types={
            "Person": EntityTypeDefinition(
                name="Person",
                description="A person",
                aliases=["Human", "Individual"],
                properties=[
                    PropertyDefinition(
                        name="name",
                        data_type=DataType.STRING,
                        validation=ValidationRules(required=True, min_length=1),
                    ),
                    PropertyDefinition(
                        name="age",
                        data_type=DataType.INTEGER,
                        validation=ValidationRules(min_value=0, max_value=200),
                    ),
                    PropertyDefinition(
                        name="email",
                        data_type=DataType.EMAIL,
                    ),
                    PropertyDefinition(
                        name="role",
                        data_type=DataType.STRING,
                        validation=ValidationRules(
                            allowed_values=["engineer", "manager", "designer"],
                        ),
                    ),
                    PropertyDefinition(
                        name="bio",
                        data_type=DataType.STRING,
                        validation=ValidationRules(max_length=500),
                    ),
                    PropertyDefinition(
                        name="username",
                        data_type=DataType.STRING,
                        validation=ValidationRules(pattern=r"^[a-z0-9_]+$"),
                    ),
                ],
            ),
            "Organization": EntityTypeDefinition(
                name="Organization",
                description="An organization",
                properties=[
                    PropertyDefinition(name="name", data_type=DataType.STRING),
                ],
            ),
        },
        relationship_types={
            "WORKS_AT": RelationshipTypeDefinition(
                name="WORKS_AT",
                description="Employment relationship",
                domain=["Person"],
                range=["Organization"],
                properties=[PropertyDefinition(name="role", data_type=DataType.STRING)],
            ),
        },
    )


class TestValidationReport:
    """Tests for the ValidationReport dataclass."""

    def test_initial_state(self):
        """Test initial report is valid with no issues."""
        report = ValidationReport(is_valid=True)
        assert report.is_valid is True
        assert report.issues == []
        assert report.error_count == 0
        assert report.warning_count == 0
        assert report.info_count == 0

    def test_add_error_invalidates_report(self):
        """Test adding an error sets is_valid to False."""
        report = ValidationReport(is_valid=True)
        report.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            path="entity.type",
            message="Missing type",
        ))
        assert report.is_valid is False
        assert report.error_count == 1
        assert report.has_errors() is True

    def test_add_warning_keeps_valid(self):
        """Test adding a warning does not invalidate the report."""
        report = ValidationReport(is_valid=True)
        report.add_issue(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            path="entity.name",
            message="Name too short",
        ))
        assert report.is_valid is True
        assert report.warning_count == 1
        assert report.has_warnings() is True
        assert report.has_errors() is False

    def test_add_info_keeps_valid(self):
        """Test adding an info issue does not affect validity."""
        report = ValidationReport(is_valid=True)
        report.add_issue(ValidationIssue(
            severity=ValidationSeverity.INFO,
            path="entity.props.extra",
            message="Unknown property",
        ))
        assert report.is_valid is True
        assert report.info_count == 1

    def test_to_dict(self):
        """Test converting report to dictionary."""
        report = ValidationReport(is_valid=True, entity_count=2, relationship_count=1)
        report.add_issue(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            path="entity.name",
            message="test",
            value="foo",
            expected="bar",
            rule="test_rule",
        ))
        result = report.to_dict()
        assert result["is_valid"] is True
        assert result["entity_count"] == 2
        assert result["relationship_count"] == 1
        assert len(result["issues"]) == 1
        assert result["issues"][0]["severity"] == "warning"
        assert result["issues"][0]["rule"] == "test_rule"


class TestValidationIssueStr:
    """Tests for ValidationIssue string representation."""

    def test_str_format(self):
        """Test __str__ returns formatted string."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            path="entity.type",
            message="Entity type is required",
        )
        assert str(issue) == "[ERROR] entity.type: Entity type is required"


class TestValidateEntity:
    """Tests for OntologyValidator.validate_entity."""

    def test_valid_entity(self):
        """Test validating a fully valid entity."""
        ontology = make_test_ontology()
        validator = OntologyValidator(ontology)

        entity = {
            "type": "Person",
            "name": "John Doe",
            "properties": {
                "name": "John Doe",
                "age": 30,
                "email": "john@example.com",
            },
        }
        report = validator.validate_entity(entity)
        assert report.is_valid is True
        assert report.error_count == 0

    def test_missing_type_is_error(self):
        """Test that entity without type produces an error."""
        ontology = make_test_ontology()
        validator = OntologyValidator(ontology)

        entity = {"name": "John Doe"}
        report = validator.validate_entity(entity)
        assert report.is_valid is False
        assert report.error_count == 1
        assert any("type is required" in i.message for i in report.issues)

    def test_unknown_type_strict_is_error(self):
        """Test that unknown type in strict mode produces an error."""
        ontology = make_test_ontology()
        validator = OntologyValidator(ontology, strict=True)

        entity = {"type": "UnknownType", "name": "Test"}
        report = validator.validate_entity(entity)
        assert report.is_valid is False
        assert any(
            i.severity == ValidationSeverity.ERROR and "Unknown entity type" in i.message
            for i in report.issues
        )

    def test_unknown_type_non_strict_is_warning(self):
        """Test that unknown type in non-strict mode produces a warning."""
        ontology = make_test_ontology()
        validator = OntologyValidator(ontology, strict=False)

        entity = {"type": "UnknownType", "name": "Test"}
        report = validator.validate_entity(entity)
        assert report.is_valid is True
        assert any(
            i.severity == ValidationSeverity.WARNING and "Unknown entity type" in i.message
            for i in report.issues
        )

    def test_missing_required_property_is_warning(self):
        """Test that missing required property produces a warning."""
        ontology = make_test_ontology()
        validator = OntologyValidator(ontology)

        entity = {
            "type": "Person",
            "name": "John Doe",
            "properties": {},
        }
        report = validator.validate_entity(entity)
        assert report.has_warnings() is True
        assert any(
            "Required property 'name' is missing" in i.message
            for i in report.issues
        )

    def test_missing_entity_name_is_error(self):
        """Test that entity without name produces an error."""
        ontology = make_test_ontology()
        validator = OntologyValidator(ontology)

        entity = {"type": "Person"}
        report = validator.validate_entity(entity)
        assert any(
            i.severity == ValidationSeverity.ERROR and "name is required" in i.message
            for i in report.issues
        )

    def test_entity_type_name_override(self):
        """Test passing entity_type_name parameter overrides entity dict."""
        ontology = make_test_ontology()
        validator = OntologyValidator(ontology)

        entity = {"name": "Acme Corp"}
        report = validator.validate_entity(entity, entity_type_name="Organization")
        assert report.is_valid is True

    def test_entity_type_from_entity_type_key(self):
        """Test entity_type key in entity dict is used as fallback."""
        ontology = make_test_ontology()
        validator = OntologyValidator(ontology)

        entity = {"entity_type": "Person", "name": "Jane"}
        report = validator.validate_entity(entity)
        assert report.is_valid is True

    def test_unknown_property_is_info(self):
        """Test that unknown properties produce info-level issues."""
        ontology = make_test_ontology()
        validator = OntologyValidator(ontology)

        entity = {
            "type": "Person",
            "name": "John",
            "properties": {
                "name": "John",
                "unknown_field": "value",
            },
        }
        report = validator.validate_entity(entity)
        assert any(
            i.severity == ValidationSeverity.INFO and "Unknown property" in i.message
            for i in report.issues
        )


class TestPropertyValidation:
    """Tests for property-level validation constraints."""

    def setup_method(self):
        """Set up validator for each test."""
        self.ontology = make_test_ontology()
        self.validator = OntologyValidator(self.ontology)

    def test_string_min_length_violation(self):
        """Test string shorter than min_length produces warning."""
        entity = {
            "type": "Person",
            "name": "",
            "properties": {"name": ""},
        }
        report = self.validator.validate_entity(entity)
        assert any(
            "too short" in i.message.lower() for i in report.issues
        )

    def test_string_max_length_violation(self):
        """Test string longer than max_length produces warning."""
        entity = {
            "type": "Person",
            "name": "John",
            "properties": {
                "name": "John",
                "bio": "X" * 501,
            },
        }
        report = self.validator.validate_entity(entity)
        assert any(
            "too long" in i.message.lower() for i in report.issues
        )

    def test_string_pattern_violation(self):
        """Test string not matching pattern produces warning."""
        entity = {
            "type": "Person",
            "name": "John",
            "properties": {
                "name": "John",
                "username": "Invalid Username!",
            },
        }
        report = self.validator.validate_entity(entity)
        assert any(
            "pattern" in i.message.lower() for i in report.issues
        )

    def test_string_pattern_valid(self):
        """Test string matching pattern produces no pattern issues."""
        entity = {
            "type": "Person",
            "name": "John",
            "properties": {
                "name": "John",
                "username": "john_doe_42",
            },
        }
        report = self.validator.validate_entity(entity)
        pattern_issues = [i for i in report.issues if i.rule == "pattern"]
        assert len(pattern_issues) == 0

    def test_numeric_min_value_violation(self):
        """Test numeric value below minimum produces warning."""
        entity = {
            "type": "Person",
            "name": "John",
            "properties": {
                "name": "John",
                "age": -1,
            },
        }
        report = self.validator.validate_entity(entity)
        assert any(
            i.rule == "min_value" for i in report.issues
        )

    def test_numeric_max_value_violation(self):
        """Test numeric value above maximum produces warning."""
        entity = {
            "type": "Person",
            "name": "John",
            "properties": {
                "name": "John",
                "age": 201,
            },
        }
        report = self.validator.validate_entity(entity)
        assert any(
            i.rule == "max_value" for i in report.issues
        )

    def test_numeric_value_in_range(self):
        """Test numeric value within range produces no min/max issues."""
        entity = {
            "type": "Person",
            "name": "John",
            "properties": {
                "name": "John",
                "age": 30,
            },
        }
        report = self.validator.validate_entity(entity)
        range_issues = [i for i in report.issues if i.rule in ("min_value", "max_value")]
        assert len(range_issues) == 0

    def test_enum_constraint_violation(self):
        """Test value not in allowed_values produces warning."""
        entity = {
            "type": "Person",
            "name": "John",
            "properties": {
                "name": "John",
                "role": "astronaut",
            },
        }
        report = self.validator.validate_entity(entity)
        assert any(
            i.rule == "allowed_values" for i in report.issues
        )

    def test_enum_constraint_valid(self):
        """Test value in allowed_values produces no enum issues."""
        entity = {
            "type": "Person",
            "name": "John",
            "properties": {
                "name": "John",
                "role": "engineer",
            },
        }
        report = self.validator.validate_entity(entity)
        enum_issues = [i for i in report.issues if i.rule == "allowed_values"]
        assert len(enum_issues) == 0

    def test_email_type_validation_valid(self):
        """Test valid email passes type check."""
        entity = {
            "type": "Person",
            "name": "John",
            "properties": {
                "name": "John",
                "email": "john@example.com",
            },
        }
        report = self.validator.validate_entity(entity)
        type_issues = [i for i in report.issues if i.rule == "type" and "email" in i.path]
        assert len(type_issues) == 0

    def test_email_type_validation_invalid(self):
        """Test invalid email fails type check."""
        entity = {
            "type": "Person",
            "name": "John",
            "properties": {
                "name": "John",
                "email": "not-an-email",
            },
        }
        report = self.validator.validate_entity(entity)
        assert any(
            i.rule == "type" and "email" in i.path for i in report.issues
        )

    def test_integer_type_wrong_type(self):
        """Test non-integer value for integer property."""
        entity = {
            "type": "Person",
            "name": "John",
            "properties": {
                "name": "John",
                "age": "thirty",
            },
        }
        report = self.validator.validate_entity(entity)
        assert any(
            i.rule == "type" and "age" in i.path for i in report.issues
        )


class TestValidateRelationship:
    """Tests for OntologyValidator.validate_relationship."""

    def setup_method(self):
        """Set up validator for each test."""
        self.ontology = make_test_ontology()
        self.validator = OntologyValidator(self.ontology)

    def test_valid_relationship(self):
        """Test validating a fully valid relationship."""
        rel = {
            "subject": "John Doe",
            "predicate": "WORKS_AT",
            "object": "Acme Corp",
        }
        report = self.validator.validate_relationship(
            rel, subject_type="Person", object_type="Organization"
        )
        assert report.is_valid is True
        assert report.error_count == 0

    def test_missing_predicate_is_error(self):
        """Test that relationship without predicate produces an error."""
        rel = {
            "subject": "John Doe",
            "object": "Acme Corp",
        }
        report = self.validator.validate_relationship(rel)
        assert report.is_valid is False
        assert any("predicate is required" in i.message for i in report.issues)

    def test_missing_subject_is_error(self):
        """Test that relationship without subject produces an error."""
        rel = {
            "predicate": "WORKS_AT",
            "object": "Acme Corp",
        }
        report = self.validator.validate_relationship(rel)
        assert any(
            i.severity == ValidationSeverity.ERROR and "subject is required" in i.message
            for i in report.issues
        )

    def test_missing_object_is_error(self):
        """Test that relationship without object produces an error."""
        rel = {
            "subject": "John Doe",
            "predicate": "WORKS_AT",
        }
        report = self.validator.validate_relationship(rel)
        assert any(
            i.severity == ValidationSeverity.ERROR and "object is required" in i.message
            for i in report.issues
        )

    def test_wrong_domain_type_is_warning(self):
        """Test that wrong subject type (domain) produces a warning."""
        rel = {
            "subject": "Acme Corp",
            "predicate": "WORKS_AT",
            "object": "Other Corp",
            "subject_type": "Organization",
        }
        report = self.validator.validate_relationship(rel)
        assert any(
            i.severity == ValidationSeverity.WARNING
            and "not in allowed domain" in i.message
            for i in report.issues
        )

    def test_wrong_range_type_is_warning(self):
        """Test that wrong object type (range) produces a warning."""
        rel = {
            "subject": "John",
            "predicate": "WORKS_AT",
            "object": "Jane",
        }
        report = self.validator.validate_relationship(
            rel, subject_type="Person", object_type="Person"
        )
        assert any(
            i.severity == ValidationSeverity.WARNING
            and "not in allowed range" in i.message
            for i in report.issues
        )

    def test_unknown_predicate_strict_is_error(self):
        """Test unknown predicate in strict mode is an error."""
        validator = OntologyValidator(self.ontology, strict=True)
        rel = {
            "subject": "John",
            "predicate": "UNKNOWN_REL",
            "object": "Jane",
        }
        report = validator.validate_relationship(rel)
        assert report.is_valid is False
        assert any(
            i.severity == ValidationSeverity.ERROR
            and "Unknown relationship type" in i.message
            for i in report.issues
        )

    def test_unknown_predicate_non_strict_is_warning(self):
        """Test unknown predicate in non-strict mode is a warning."""
        rel = {
            "subject": "John",
            "predicate": "UNKNOWN_REL",
            "object": "Jane",
        }
        report = self.validator.validate_relationship(rel)
        assert report.is_valid is True
        assert any(
            i.severity == ValidationSeverity.WARNING
            and "Unknown relationship type" in i.message
            for i in report.issues
        )

    def test_subject_type_from_relationship_dict(self):
        """Test subject_type extracted from relationship dict itself."""
        rel = {
            "subject": "Acme Corp",
            "predicate": "WORKS_AT",
            "object": "Other Corp",
            "subject_type": "Organization",
            "object_type": "Organization",
        }
        report = self.validator.validate_relationship(rel)
        assert any(
            "not in allowed domain" in i.message for i in report.issues
        )


class TestValidateExtractionResult:
    """Tests for OntologyValidator.validate_extraction_result."""

    def test_valid_extraction(self):
        """Test validating a valid extraction result."""
        ontology = make_test_ontology()
        validator = OntologyValidator(ontology)

        entities = [
            {"type": "Person", "name": "John Doe"},
            {"type": "Organization", "name": "Acme Corp"},
        ]
        relationships = [
            {
                "subject": "John Doe",
                "predicate": "WORKS_AT",
                "object": "Acme Corp",
            },
        ]

        report = validator.validate_extraction_result(entities, relationships)
        assert report.entity_count == 2
        assert report.relationship_count == 1

    def test_combined_report_counts_issues(self):
        """Test that combined report aggregates issues from entities and relationships."""
        ontology = make_test_ontology()
        validator = OntologyValidator(ontology)

        entities = [
            {"name": "John Doe"},  # missing type
        ]
        relationships = [
            {"subject": "John", "object": "Acme"},  # missing predicate
        ]

        report = validator.validate_extraction_result(entities, relationships)
        assert report.is_valid is False
        assert report.error_count >= 2

    def test_entity_type_lookup_for_relationships(self):
        """Test that entity types from entities are used for relationship domain/range validation."""
        ontology = make_test_ontology()
        validator = OntologyValidator(ontology)

        entities = [
            {"type": "Organization", "name": "Acme Corp"},
            {"type": "Organization", "name": "Other Corp"},
        ]
        relationships = [
            {
                "subject": "Acme Corp",
                "predicate": "WORKS_AT",
                "object": "Other Corp",
            },
        ]

        report = validator.validate_extraction_result(entities, relationships)
        assert any(
            "not in allowed domain" in i.message for i in report.issues
        )

    def test_issue_paths_prefixed(self):
        """Test that issue paths are prefixed with entities[i] or relationships[i]."""
        ontology = make_test_ontology()
        validator = OntologyValidator(ontology)

        entities = [{"name": "Test"}]
        relationships = [{"subject": "A", "object": "B"}]

        report = validator.validate_extraction_result(entities, relationships)
        entity_issues = [i for i in report.issues if i.path.startswith("entities[")]
        rel_issues = [i for i in report.issues if i.path.startswith("relationships[")]
        assert len(entity_issues) > 0
        assert len(rel_issues) > 0

    def test_empty_extraction(self):
        """Test validating empty extraction result."""
        ontology = make_test_ontology()
        validator = OntologyValidator(ontology)

        report = validator.validate_extraction_result([], [])
        assert report.is_valid is True
        assert report.entity_count == 0
        assert report.relationship_count == 0
        assert len(report.issues) == 0
