"""
Ontology constraint filter.

Validates entities and relations against an ontology definition,
removing those that fail validation. Uses OntologyValidator from
the ontology-manager package.

When ontology-manager is unavailable, the filter is a no-op.
"""

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# Optional dependency guard
try:
    from ontology_manager.schema import Ontology
    from ontology_manager.validator import OntologyValidator, ValidationReport

    _ONTOLOGY_AVAILABLE = True
except ImportError:
    _ONTOLOGY_AVAILABLE = False


class OntologyConstraintFilter:
    """Filter entities and relations that violate ontology constraints.

    Args:
        ontology: Ontology definition to validate against. When None,
            the filter is a no-op.
        strict: When True, unknown types are treated as errors.
        filter_invalid_entities: Whether to remove invalid entities.
        filter_invalid_relations: Whether to remove invalid relations.
    """

    def __init__(
        self,
        ontology: Optional[Any] = None,  # Actually Ontology, but use Any for optional dep
        strict: bool = False,
        filter_invalid_entities: bool = True,
        filter_invalid_relations: bool = True,
    ) -> None:
        self._ontology = ontology
        self._strict = strict
        self._filter_entities = filter_invalid_entities
        self._filter_relations = filter_invalid_relations
        self._validator = None

        if _ONTOLOGY_AVAILABLE and ontology is not None:
            self._validator = OntologyValidator(ontology, strict=strict)

    def filter(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """Filter entities and relations against ontology constraints.

        The OntologyValidator uses:
        - entity["type"] for entity type (but our pipeline entities use entity["label"])
        - entity["name"] for entity name (but our entities use entity["text"])
        - relationship["predicate"] for relation type (but we use relation["relation_type"])
        - relationship["subject"] / ["object"] for endpoints (we use source_entity/target_entity)

        So we need to TRANSLATE our dict format to the validator's expected format.

        Args:
            entities: Entity dicts from the pipeline (with "text", "label" keys).
            relations: Relation dicts from the pipeline (with "source_entity",
                "target_entity", "relation_type" keys).

        Returns:
            Tuple of (valid_entities, valid_relations, validation_report_dict).
            The report dict contains:
            - total_entities: int
            - total_relations: int
            - valid_entities: int
            - invalid_entities: int
            - valid_relations: int
            - invalid_relations: int
            - entity_issues: list of issue dicts
            - relation_issues: list of issue dicts
        """
        report: Dict[str, Any] = {
            "total_entities": len(entities),
            "total_relations": len(relations),
            "valid_entities": len(entities),
            "invalid_entities": 0,
            "valid_relations": len(relations),
            "invalid_relations": 0,
            "entity_issues": [],
            "relation_issues": [],
        }

        if self._validator is None:
            if not _ONTOLOGY_AVAILABLE:
                logger.debug("ontology-manager not available, skipping validation")
            elif self._ontology is None:
                logger.debug("No ontology provided, skipping validation")
            return entities, relations, report

        # Build entity type lookup for relation domain/range validation
        entity_type_map: Dict[str, str] = {}
        for ent in entities:
            text = ent.get("text", "")
            label = ent.get("label", "")
            if text and label:
                entity_type_map[text] = label

        # Validate entities
        valid_entities = []
        for entity in entities:
            # Translate pipeline format to validator format
            validator_entity = {
                "type": entity.get("label", ""),
                "name": entity.get("text", ""),
                "properties": entity.get("properties", {}),
            }

            vr = self._validator.validate_entity(validator_entity)

            if vr.has_errors() and self._filter_entities:
                report["invalid_entities"] += 1
                report["entity_issues"].append(
                    {
                        "entity_text": entity.get("text", ""),
                        "entity_label": entity.get("label", ""),
                        "issues": [
                            {
                                "severity": i.severity.value,
                                "message": i.message,
                                "path": i.path,
                            }
                            for i in vr.issues
                        ],
                    }
                )
            else:
                valid_entities.append(entity)

        report["valid_entities"] = len(valid_entities)

        # Build updated entity text set for relation filtering
        valid_entity_texts = {e.get("text", "") for e in valid_entities}

        # Validate relations
        valid_relations = []
        for rel in relations:
            # Skip relations whose endpoints were removed
            source = rel.get("source_entity", "")
            target = rel.get("target_entity", "")
            if source not in valid_entity_texts or target not in valid_entity_texts:
                report["invalid_relations"] += 1
                continue

            # Translate pipeline format to validator format
            validator_rel = {
                "subject": source,
                "predicate": rel.get("relation_type", ""),
                "object": target,
                "properties": rel.get("properties", {}),
            }

            subject_type = entity_type_map.get(source)
            object_type = entity_type_map.get(target)

            vr = self._validator.validate_relationship(
                validator_rel, subject_type=subject_type, object_type=object_type
            )

            if vr.has_errors() and self._filter_relations:
                report["invalid_relations"] += 1
                report["relation_issues"].append(
                    {
                        "source": source,
                        "target": target,
                        "relation_type": rel.get("relation_type", ""),
                        "issues": [
                            {
                                "severity": i.severity.value,
                                "message": i.message,
                                "path": i.path,
                            }
                            for i in vr.issues
                        ],
                    }
                )
            else:
                valid_relations.append(rel)

        report["valid_relations"] = len(valid_relations)

        logger.debug(
            "OntologyConstraintFilter: {}/{} entities valid, {}/{} relations valid",
            report["valid_entities"],
            report["total_entities"],
            report["valid_relations"],
            report["total_relations"],
        )

        return valid_entities, valid_relations, report
