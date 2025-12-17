"""
Ontology Validation Gate for Knowledge Graph Extraction.

Validates extracted entities and relationships against ontology schema.
Provides validation errors that can trigger LLM fallback for correction.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from loguru import logger


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    ERROR = "error"  # Must be fixed
    WARNING = "warning"  # Should be reviewed
    INFO = "info"  # Informational only


@dataclass
class ValidationIssue:
    """A single validation issue."""
    severity: ValidationSeverity
    code: str
    message: str
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None
    property_name: Optional[str] = None
    value: Optional[Any] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validation with issues and statistics."""
    valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    entities_validated: int = 0
    entities_passed: int = 0
    entities_failed: int = 0

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.valid = False

    def get_errors(self) -> List[ValidationIssue]:
        """Get all error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    def get_warnings(self) -> List[ValidationIssue]:
        """Get all warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            "valid": self.valid,
            "entities_validated": self.entities_validated,
            "entities_passed": self.entities_passed,
            "entities_failed": self.entities_failed,
            "error_count": len(self.get_errors()),
            "warning_count": len(self.get_warnings()),
        }


class OntologyValidator:
    """
    Validates extracted knowledge against ontology schema.

    Performs:
    - Entity type validation against defined types
    - Required property validation
    - Property value type validation
    - Cardinality checks for relationships
    """

    def __init__(self, ontology_name: str = "general"):
        """
        Initialize validator with ontology.

        Args:
            ontology_name: Name of ontology to validate against
        """
        self.ontology_name = ontology_name
        self._ontology = None
        self._entity_types: Dict[str, Any] = {}
        self._relationship_types: Dict[str, Any] = {}

    async def initialize(self) -> bool:
        """Load ontology for validation."""
        try:
            from open_notebook.ontology.registry import OntologyRegistry

            registry = OntologyRegistry()
            self._ontology = await registry.get(self.ontology_name)

            if not self._ontology:
                logger.warning(f"Ontology '{self.ontology_name}' not found")
                return False

            # Index entity types by name
            for entity_type in self._ontology.entity_types:
                self._entity_types[entity_type.name.lower()] = entity_type
                # Also index by aliases
                if entity_type.aliases:
                    for alias in entity_type.aliases:
                        self._entity_types[alias.lower()] = entity_type

            # Index relationship types by name
            for rel_type in self._ontology.relationship_types:
                self._relationship_types[rel_type.name.lower()] = rel_type

            logger.info(
                f"Validator initialized with {len(self._entity_types)} entity types, "
                f"{len(self._relationship_types)} relationship types"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize validator: {e}")
            return False

    async def validate_entity(
        self,
        name: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate a single entity against ontology.

        Args:
            name: Entity name
            entity_type: Declared entity type
            properties: Entity properties to validate

        Returns:
            ValidationResult with any issues found
        """
        result = ValidationResult()
        result.entities_validated = 1

        # Ensure ontology is loaded
        if not self._ontology:
            await self.initialize()

        # Validate entity type
        type_def = self._entity_types.get(entity_type.lower())
        if not type_def:
            # Check if it's a known type that's just not in this ontology
            known_types = {"person", "organization", "location", "event", "concept", "work"}
            if entity_type.lower() in known_types:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="UNKNOWN_TYPE",
                    message=f"Entity type '{entity_type}' not in ontology '{self.ontology_name}'",
                    entity_name=name,
                    entity_type=entity_type,
                    suggestion=f"Consider adding '{entity_type}' to the ontology or mapping to existing type",
                ))
            else:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_TYPE",
                    message=f"Unknown entity type '{entity_type}'",
                    entity_name=name,
                    entity_type=entity_type,
                    suggestion="Map to a known entity type or add to ontology",
                ))
                result.entities_failed = 1
                return result

        # Validate required properties
        if type_def and properties is not None:
            required_props = type_def.get_required_properties() if hasattr(type_def, 'get_required_properties') else []
            for prop in required_props:
                if prop.name not in properties or properties[prop.name] is None:
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="MISSING_REQUIRED_PROPERTY",
                        message=f"Required property '{prop.name}' is missing",
                        entity_name=name,
                        entity_type=entity_type,
                        property_name=prop.name,
                    ))

            # Validate property values
            for prop_name, value in (properties or {}).items():
                prop_def = next(
                    (p for p in type_def.properties if p.name == prop_name), None
                ) if hasattr(type_def, 'properties') else None

                if prop_def:
                    prop_issues = self._validate_property_value(
                        name, entity_type, prop_def, value
                    )
                    for issue in prop_issues:
                        result.add_issue(issue)

        if result.valid:
            result.entities_passed = 1
        else:
            result.entities_failed = 1

        return result

    def _validate_property_value(
        self,
        entity_name: str,
        entity_type: str,
        prop_def: Any,
        value: Any,
    ) -> List[ValidationIssue]:
        """Validate a property value against its definition."""
        issues = []

        if value is None:
            return issues

        # Type validation
        data_type = prop_def.data_type if hasattr(prop_def, 'data_type') else None
        if data_type:
            type_valid, type_msg = self._check_data_type(value, str(data_type))
            if not type_valid:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="INVALID_PROPERTY_TYPE",
                    message=type_msg,
                    entity_name=entity_name,
                    entity_type=entity_type,
                    property_name=prop_def.name,
                    value=value,
                ))

        # Validation rules
        validation = prop_def.validation if hasattr(prop_def, 'validation') else None
        if validation:
            # Length constraints
            if isinstance(value, str):
                if validation.min_length and len(value) < validation.min_length:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="VALUE_TOO_SHORT",
                        message=f"Value length {len(value)} below minimum {validation.min_length}",
                        entity_name=entity_name,
                        entity_type=entity_type,
                        property_name=prop_def.name,
                        value=value,
                    ))
                if validation.max_length and len(value) > validation.max_length:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="VALUE_TOO_LONG",
                        message=f"Value length {len(value)} exceeds maximum {validation.max_length}",
                        entity_name=entity_name,
                        entity_type=entity_type,
                        property_name=prop_def.name,
                        value=value,
                    ))

            # Numeric constraints
            if isinstance(value, (int, float)):
                if validation.min_value is not None and value < validation.min_value:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="VALUE_TOO_SMALL",
                        message=f"Value {value} below minimum {validation.min_value}",
                        entity_name=entity_name,
                        entity_type=entity_type,
                        property_name=prop_def.name,
                        value=value,
                    ))
                if validation.max_value is not None and value > validation.max_value:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="VALUE_TOO_LARGE",
                        message=f"Value {value} exceeds maximum {validation.max_value}",
                        entity_name=entity_name,
                        entity_type=entity_type,
                        property_name=prop_def.name,
                        value=value,
                    ))

            # Pattern validation
            if validation.pattern and isinstance(value, str):
                if not re.match(validation.pattern, value):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="PATTERN_MISMATCH",
                        message=f"Value does not match pattern '{validation.pattern}'",
                        entity_name=entity_name,
                        entity_type=entity_type,
                        property_name=prop_def.name,
                        value=value,
                    ))

            # Enum validation
            if validation.allowed_values and value not in validation.allowed_values:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="INVALID_ENUM_VALUE",
                    message=f"Value '{value}' not in allowed values: {validation.allowed_values}",
                    entity_name=entity_name,
                    entity_type=entity_type,
                    property_name=prop_def.name,
                    value=value,
                    suggestion=f"Use one of: {', '.join(validation.allowed_values)}",
                ))

        return issues

    def _check_data_type(self, value: Any, expected_type: str) -> tuple[bool, str]:
        """Check if value matches expected data type."""
        type_checks = {
            "string": lambda v: isinstance(v, str),
            "text": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
            "json": lambda v: isinstance(v, (dict, list)),
        }

        # Normalize type name
        expected_lower = expected_type.lower().replace("datatype.", "")

        check_fn = type_checks.get(expected_lower)
        if check_fn is None:
            # Unknown type, accept anything
            return True, ""

        if check_fn(value):
            return True, ""

        return False, f"Expected {expected_type}, got {type(value).__name__}"

    async def validate_relationship(
        self,
        source_type: str,
        relationship_type: str,
        target_type: str,
    ) -> ValidationResult:
        """
        Validate a relationship against ontology.

        Args:
            source_type: Source entity type
            relationship_type: Relationship type name
            target_type: Target entity type

        Returns:
            ValidationResult with any issues found
        """
        result = ValidationResult()

        # Ensure ontology is loaded
        if not self._ontology:
            await self.initialize()

        rel_def = self._relationship_types.get(relationship_type.lower())
        if not rel_def:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="UNKNOWN_RELATIONSHIP",
                message=f"Relationship type '{relationship_type}' not in ontology",
                suggestion="Consider adding to ontology or mapping to existing type",
            ))
            return result

        # Check domain (source type)
        if rel_def.domain and source_type.lower() not in [d.lower() for d in rel_def.domain]:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="INVALID_DOMAIN",
                message=f"Source type '{source_type}' not valid for relationship '{relationship_type}'",
                suggestion=f"Valid source types: {', '.join(rel_def.domain)}",
            ))

        # Check range (target type)
        if rel_def.range and target_type.lower() not in [r.lower() for r in rel_def.range]:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="INVALID_RANGE",
                message=f"Target type '{target_type}' not valid for relationship '{relationship_type}'",
                suggestion=f"Valid target types: {', '.join(rel_def.range)}",
            ))

        return result

    async def validate_entities_batch(
        self,
        entities: List[Dict[str, Any]],
    ) -> ValidationResult:
        """
        Validate a batch of entities.

        Args:
            entities: List of entity dicts with 'name', 'entity_type', and optional 'properties'

        Returns:
            Combined ValidationResult
        """
        combined = ValidationResult()

        for entity in entities:
            result = await self.validate_entity(
                name=entity.get("name", ""),
                entity_type=entity.get("entity_type", entity.get("type", "")),
                properties=entity.get("properties", {}),
            )
            combined.entities_validated += result.entities_validated
            combined.entities_passed += result.entities_passed
            combined.entities_failed += result.entities_failed
            combined.issues.extend(result.issues)

        combined.valid = combined.entities_failed == 0
        return combined

    def get_valid_entity_types(self) -> Set[str]:
        """Get all valid entity type names."""
        return set(self._entity_types.keys())

    def get_valid_relationship_types(self) -> Set[str]:
        """Get all valid relationship type names."""
        return set(self._relationship_types.keys())

    def suggest_entity_type(self, extracted_type: str) -> Optional[str]:
        """
        Suggest a valid entity type for an extracted type.

        Uses fuzzy matching to find the closest valid type.
        """
        extracted_lower = extracted_type.lower()

        # Exact match
        if extracted_lower in self._entity_types:
            return self._entity_types[extracted_lower].name

        # Common mappings
        type_mappings = {
            "per": "person",
            "org": "organization",
            "loc": "location",
            "gpe": "location",  # Geo-Political Entity
            "fac": "location",  # Facility
            "date": "event",
            "time": "event",
            "norp": "organization",  # Nationality/Religious/Political groups
            "product": "work",
            "work_of_art": "work",
            "event": "event",
            "law": "concept",
            "language": "concept",
        }

        mapped = type_mappings.get(extracted_lower)
        if mapped and mapped in self._entity_types:
            return self._entity_types[mapped].name

        return None


# Convenience functions

async def validate_entity(
    name: str,
    entity_type: str,
    properties: Optional[Dict[str, Any]] = None,
    ontology_name: str = "general",
) -> ValidationResult:
    """Validate a single entity."""
    validator = OntologyValidator(ontology_name)
    return await validator.validate_entity(name, entity_type, properties)


async def validate_entities_batch(
    entities: List[Dict[str, Any]],
    ontology_name: str = "general",
) -> ValidationResult:
    """Validate a batch of entities."""
    validator = OntologyValidator(ontology_name)
    return await validator.validate_entities_batch(entities)


async def get_validation_report(
    entities: List[Dict[str, Any]],
    ontology_name: str = "general",
) -> Dict[str, Any]:
    """
    Get a detailed validation report for entities.

    Returns a report suitable for logging or LLM correction prompts.
    """
    validator = OntologyValidator(ontology_name)
    result = await validator.validate_entities_batch(entities)

    report = {
        "summary": result.summary(),
        "errors": [
            {
                "entity": i.entity_name,
                "type": i.entity_type,
                "code": i.code,
                "message": i.message,
                "suggestion": i.suggestion,
            }
            for i in result.get_errors()
        ],
        "warnings": [
            {
                "entity": i.entity_name,
                "type": i.entity_type,
                "property": i.property_name,
                "code": i.code,
                "message": i.message,
            }
            for i in result.get_warnings()
        ],
    }

    return report
