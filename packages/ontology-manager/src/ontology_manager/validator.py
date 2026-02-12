"""
SHACL-like validation for extracted knowledge.

Validates entities and relationships against ontology definitions,
checking property constraints, types, and cardinality rules.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger

from ontology_manager.schema import (
    DataType,
    EntityTypeDefinition,
    Ontology,
    PropertyDefinition,
    RelationshipTypeDefinition,
    ValidationRules,
)


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""

    ERROR = "error"  # Blocks processing
    WARNING = "warning"  # Should be fixed
    INFO = "info"  # Informational only


@dataclass
class ValidationIssue:
    """A single validation issue."""

    severity: ValidationSeverity
    path: str  # Property path (e.g., "entity.name", "relationship.domain")
    message: str
    value: Optional[Any] = None
    expected: Optional[Any] = None
    rule: Optional[str] = None  # Name of the validation rule

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.path}: {self.message}"


@dataclass
class ValidationReport:
    """Complete validation report for an extraction."""

    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    entity_count: int = 0
    relationship_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue to the report."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.error_count += 1
            self.is_valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warning_count += 1
        else:
            self.info_count += 1

    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return self.error_count > 0

    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return self.warning_count > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "is_valid": self.is_valid,
            "entity_count": self.entity_count,
            "relationship_count": self.relationship_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "issues": [
                {
                    "severity": i.severity.value,
                    "path": i.path,
                    "message": i.message,
                    "value": str(i.value) if i.value is not None else None,
                    "expected": str(i.expected) if i.expected is not None else None,
                    "rule": i.rule,
                }
                for i in self.issues
            ],
        }


class OntologyValidator:
    """
    Validates extracted knowledge against ontology definitions.

    Performs SHACL-like validation including:
    - Entity type validation
    - Property type and constraint checking
    - Relationship domain/range validation
    - Cardinality constraints
    """

    def __init__(self, ontology: Ontology, strict: bool = False):
        """
        Initialize validator.

        Args:
            ontology: The ontology to validate against
            strict: If True, unknown types are errors; if False, warnings
        """
        self.ontology = ontology
        self.strict = strict

    def validate_entity(
        self,
        entity: Dict[str, Any],
        entity_type_name: Optional[str] = None,
    ) -> ValidationReport:
        """
        Validate a single entity against the ontology.

        Args:
            entity: Dictionary with entity data (name, type, properties)
            entity_type_name: Override entity type (uses entity["type"] if None)

        Returns:
            ValidationReport with any issues found
        """
        report = ValidationReport(is_valid=True, entity_count=1)

        # Get entity type name
        type_name = entity_type_name or entity.get("type") or entity.get("entity_type")
        if not type_name:
            report.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    path="entity.type",
                    message="Entity type is required",
                )
            )
            return report

        # Look up entity type definition
        entity_def = self.ontology.get_entity_type(type_name)
        if not entity_def:
            severity = ValidationSeverity.ERROR if self.strict else ValidationSeverity.WARNING
            report.add_issue(
                ValidationIssue(
                    severity=severity,
                    path="entity.type",
                    message=f"Unknown entity type '{type_name}'",
                    value=type_name,
                    expected=list(self.ontology.entity_types.keys()),
                )
            )
            if self.strict:
                return report

        # Validate entity name
        name = entity.get("name")
        if not name:
            report.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    path="entity.name",
                    message="Entity name is required",
                )
            )
        elif entity_def:
            name_prop = next(
                (p for p in entity_def.properties if p.name == "name"), None
            )
            if name_prop:
                self._validate_property_value(
                    report, "entity.name", name, name_prop
                )

        # Validate properties if entity definition exists
        if entity_def:
            properties = entity.get("properties", {})
            self._validate_entity_properties(
                report, entity_def, properties, prefix="entity.properties"
            )

        return report

    def validate_relationship(
        self,
        relationship: Dict[str, Any],
        subject_type: Optional[str] = None,
        object_type: Optional[str] = None,
    ) -> ValidationReport:
        """
        Validate a single relationship against the ontology.

        Args:
            relationship: Dictionary with relationship data (subject, predicate, object)
            subject_type: Type of the subject entity (optional)
            object_type: Type of the object entity (optional)

        Returns:
            ValidationReport with any issues found
        """
        report = ValidationReport(is_valid=True, relationship_count=1)

        # Get predicate/relationship type
        predicate = relationship.get("predicate")
        if not predicate:
            report.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    path="relationship.predicate",
                    message="Relationship predicate is required",
                )
            )
            return report

        # Look up relationship definition
        rel_def = self.ontology.get_relationship_type(predicate)
        if not rel_def:
            severity = ValidationSeverity.ERROR if self.strict else ValidationSeverity.WARNING
            report.add_issue(
                ValidationIssue(
                    severity=severity,
                    path="relationship.predicate",
                    message=f"Unknown relationship type '{predicate}'",
                    value=predicate,
                    expected=list(self.ontology.relationship_types.keys()),
                )
            )
            if self.strict:
                return report

        # Validate subject
        subject = relationship.get("subject")
        if not subject:
            report.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    path="relationship.subject",
                    message="Relationship subject is required",
                )
            )

        # Validate object
        obj = relationship.get("object")
        if not obj:
            report.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    path="relationship.object",
                    message="Relationship object is required",
                )
            )

        # Validate domain/range if relationship definition exists
        if rel_def:
            # Validate subject type against domain
            subj_type = subject_type or relationship.get("subject_type")
            if subj_type:
                domain_lower = [d.lower() for d in rel_def.domain]
                if subj_type.lower() not in domain_lower and "entity" not in domain_lower:
                    report.add_issue(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            path="relationship.subject_type",
                            message=f"Subject type '{subj_type}' not in allowed domain",
                            value=subj_type,
                            expected=rel_def.domain,
                            rule="domain_constraint",
                        )
                    )

            # Validate object type against range
            obj_type = object_type or relationship.get("object_type")
            if obj_type:
                range_lower = [r.lower() for r in rel_def.range]
                if obj_type.lower() not in range_lower and "entity" not in range_lower:
                    report.add_issue(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            path="relationship.object_type",
                            message=f"Object type '{obj_type}' not in allowed range",
                            value=obj_type,
                            expected=rel_def.range,
                            rule="range_constraint",
                        )
                    )

            # Validate relationship properties if present
            if rel_def.properties:
                rel_props = relationship.get("properties", {})
                for prop_def in rel_def.properties:
                    if prop_def.name in rel_props:
                        self._validate_property_value(
                            report,
                            f"relationship.properties.{prop_def.name}",
                            rel_props[prop_def.name],
                            prop_def,
                        )

        return report

    def validate_extraction_result(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> ValidationReport:
        """
        Validate a complete extraction result.

        Args:
            entities: List of extracted entities
            relationships: List of extracted relationships

        Returns:
            Combined ValidationReport for all entities and relationships
        """
        combined_report = ValidationReport(
            is_valid=True,
            entity_count=len(entities),
            relationship_count=len(relationships),
        )

        # Validate each entity
        for i, entity in enumerate(entities):
            entity_report = self.validate_entity(entity)
            for issue in entity_report.issues:
                issue.path = f"entities[{i}].{issue.path}"
                combined_report.add_issue(issue)

        # Build entity lookup for relationship validation
        entity_types = {}
        for entity in entities:
            name = entity.get("name", "").lower()
            etype = entity.get("type") or entity.get("entity_type")
            if name and etype:
                entity_types[name] = etype

        # Validate each relationship
        for i, rel in enumerate(relationships):
            # Try to get subject/object types from entities
            subj_type = entity_types.get(rel.get("subject", "").lower())
            obj_type = entity_types.get(rel.get("object", "").lower())

            rel_report = self.validate_relationship(rel, subj_type, obj_type)
            for issue in rel_report.issues:
                issue.path = f"relationships[{i}].{issue.path}"
                combined_report.add_issue(issue)

        return combined_report

    def _validate_entity_properties(
        self,
        report: ValidationReport,
        entity_def: EntityTypeDefinition,
        properties: Dict[str, Any],
        prefix: str = "properties",
    ) -> None:
        """Validate entity properties against definition."""
        # Check required properties
        for prop_def in entity_def.properties:
            if prop_def.validation and prop_def.validation.required:
                if prop_def.name not in properties:
                    report.add_issue(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            path=f"{prefix}.{prop_def.name}",
                            message=f"Required property '{prop_def.name}' is missing",
                            rule="required",
                        )
                    )

        # Validate provided properties
        for prop_name, prop_value in properties.items():
            prop_def = next(
                (p for p in entity_def.properties if p.name == prop_name), None
            )
            if prop_def:
                self._validate_property_value(
                    report, f"{prefix}.{prop_name}", prop_value, prop_def
                )
            else:
                report.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        path=f"{prefix}.{prop_name}",
                        message=f"Unknown property '{prop_name}'",
                        value=prop_value,
                    )
                )

    def _validate_property_value(
        self,
        report: ValidationReport,
        path: str,
        value: Any,
        prop_def: PropertyDefinition,
    ) -> None:
        """Validate a single property value."""
        if value is None:
            return

        validation = prop_def.validation or ValidationRules()

        # Type validation
        if not self._check_data_type(value, prop_def.data_type):
            report.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    path=path,
                    message=f"Invalid type for property",
                    value=type(value).__name__,
                    expected=prop_def.data_type.value,
                    rule="type",
                )
            )
            return

        # String constraints
        if isinstance(value, str):
            if validation.min_length and len(value) < validation.min_length:
                report.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        path=path,
                        message=f"String too short (min: {validation.min_length})",
                        value=len(value),
                        expected=validation.min_length,
                        rule="min_length",
                    )
                )

            if validation.max_length and len(value) > validation.max_length:
                report.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        path=path,
                        message=f"String too long (max: {validation.max_length})",
                        value=len(value),
                        expected=validation.max_length,
                        rule="max_length",
                    )
                )

            if validation.pattern:
                if not re.match(validation.pattern, value):
                    report.add_issue(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            path=path,
                            message=f"String doesn't match pattern",
                            value=value,
                            expected=validation.pattern,
                            rule="pattern",
                        )
                    )

        # Numeric constraints
        if isinstance(value, (int, float)):
            if validation.min_value is not None and value < validation.min_value:
                report.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        path=path,
                        message=f"Value below minimum ({validation.min_value})",
                        value=value,
                        expected=validation.min_value,
                        rule="min_value",
                    )
                )

            if validation.max_value is not None and value > validation.max_value:
                report.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        path=path,
                        message=f"Value above maximum ({validation.max_value})",
                        value=value,
                        expected=validation.max_value,
                        rule="max_value",
                    )
                )

        # Enum constraints
        if validation.allowed_values:
            if value not in validation.allowed_values:
                report.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        path=path,
                        message=f"Value not in allowed values",
                        value=value,
                        expected=validation.allowed_values,
                        rule="allowed_values",
                    )
                )

    def _check_data_type(self, value: Any, expected_type: DataType) -> bool:
        """Check if value matches expected data type."""
        if value is None:
            return True

        type_checks = {
            DataType.STRING: lambda v: isinstance(v, str),
            DataType.TEXT: lambda v: isinstance(v, str),
            DataType.INTEGER: lambda v: isinstance(v, int) and not isinstance(v, bool),
            DataType.FLOAT: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            DataType.BOOLEAN: lambda v: isinstance(v, bool),
            DataType.DATE: lambda v: isinstance(v, str),  # ISO date string
            DataType.DATETIME: lambda v: isinstance(v, str),  # ISO datetime string
            DataType.URL: lambda v: isinstance(v, str) and (v.startswith("http://") or v.startswith("https://")),
            DataType.EMAIL: lambda v: isinstance(v, str) and "@" in v,
            DataType.ENUM: lambda v: isinstance(v, str),
            DataType.ARRAY: lambda v: isinstance(v, list),
            DataType.REFERENCE: lambda v: isinstance(v, str),
        }

        checker = type_checks.get(expected_type)
        return checker(value) if checker else True
