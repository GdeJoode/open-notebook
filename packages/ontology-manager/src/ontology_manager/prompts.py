"""
Ontology-driven prompt generation for knowledge extraction.

Generates LLM prompts based on entity and relationship definitions
from the ontology schema.
"""

from typing import List, Optional

from ontology_manager.schema import (
    ClaimTypeDefinition,
    ConceptDefinition,
    EntityTypeDefinition,
    ExtractionStrategy,
    Ontology,
    RelationshipTypeDefinition,
)


class OntologyPromptGenerator:
    """
    Generates extraction prompts based on ontology definitions.

    Produces structured prompts that guide LLM extraction according
    to the entity types, properties, and relationships defined in
    the ontology.
    """

    def __init__(self, ontology: Optional[Ontology]):
        self.ontology = ontology

    def generate_entity_extraction_prompt(
        self,
        entity_types: Optional[List[str]] = None,
        include_properties: bool = True,
    ) -> str:
        """
        Generate a prompt for entity extraction.

        Args:
            entity_types: Specific types to extract (None = all)
            include_properties: Whether to include property extraction hints

        Returns:
            Formatted prompt string
        """
        if entity_types:
            types_to_extract = [
                self.ontology.get_entity_type(t)
                for t in entity_types
                if self.ontology.get_entity_type(t)
            ]
        else:
            types_to_extract = list(self.ontology.entity_types.values())

        prompt_parts = [
            "Extract entities from the following text according to these definitions:",
            "",
            "## Entity Types to Extract",
            "",
        ]

        for entity_type in types_to_extract:
            prompt_parts.extend(self._format_entity_type(entity_type, include_properties))

        prompt_parts.extend([
            "",
            "## Output Format",
            "",
            "Return ONLY a valid JSON array (no other text):",
            "```json",
            "[",
            '  {"name": "Entity Name", "entity_type": "Person", "description": "optional", "confidence": 0.9}',
            "]",
            "```",
            "",
            "## Guidelines",
            "- Extract ALL entities matching the defined types",
            "- Use `entity_type` field with exact type names from definitions (Person, Organization, etc.)",
            "- Include confidence score (0.0-1.0) based on extraction certainty",
            "- For properties, only include those you can confidently extract",
            "- Normalize entity names (capitalize properly, remove extra whitespace)",
            "- Return ONLY valid JSON, no explanatory text",
            "",
        ])

        return "\n".join(prompt_parts)

    def _format_entity_type(
        self, entity_type: EntityTypeDefinition, include_properties: bool
    ) -> List[str]:
        """Format a single entity type for the prompt."""
        lines = [
            f"### {entity_type.name}",
            f"**Description**: {entity_type.description}",
        ]

        if entity_type.extraction_hints:
            lines.append("**Extraction hints**:")
            for hint in entity_type.extraction_hints:
                lines.append(f"  - {hint}")

        if entity_type.examples:
            lines.append(f"**Examples**: {', '.join(entity_type.examples)}")

        if include_properties and entity_type.properties:
            extractable = [
                p for p in entity_type.properties
                if p.extraction_strategy != ExtractionStrategy.NONE
            ]
            if extractable:
                lines.append("**Properties to extract**:")
                for prop in extractable:
                    required = "(required)" if prop.validation and prop.validation.required else ""
                    lines.append(f"  - `{prop.name}`: {prop.description} {required}")
                    if prop.examples:
                        lines.append(f"    Examples: {', '.join(prop.examples)}")

        lines.append("")
        return lines

    def generate_relationship_extraction_prompt(
        self,
        relationship_types: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a prompt for relationship extraction.

        Args:
            relationship_types: Specific relationships to extract (None = all)

        Returns:
            Formatted prompt string
        """
        if relationship_types:
            rels_to_extract = [
                self.ontology.get_relationship_type(r)
                for r in relationship_types
                if self.ontology.get_relationship_type(r)
            ]
        else:
            rels_to_extract = list(self.ontology.relationship_types.values())

        prompt_parts = [
            "Extract relationships between entities according to these definitions:",
            "",
            "## Relationship Types to Extract",
            "",
        ]

        for rel_type in rels_to_extract:
            prompt_parts.extend(self._format_relationship_type(rel_type))

        prompt_parts.extend([
            "",
            "## Output Format",
            "",
            "Return relationships as a JSON array of triples:",
            "```json",
            "[",
            '  {"subject": "entity name", "subject_type": "type", "predicate": "RELATIONSHIP_NAME", "object": "entity name", "object_type": "type", "confidence": 0.0-1.0, "context": "snippet"}',
            "]",
            "```",
            "",
            "## Guidelines",
            "- Only extract relationships matching the defined types",
            "- Ensure subject and object types match the domain/range constraints",
            "- Include a brief context snippet showing where the relationship was found",
            "- Include confidence score (0.0-1.0) based on extraction certainty",
            "",
        ])

        return "\n".join(prompt_parts)

    def _format_relationship_type(
        self, rel_type: RelationshipTypeDefinition
    ) -> List[str]:
        """Format a single relationship type for the prompt."""
        lines = [
            f"### {rel_type.name}",
            f"**Description**: {rel_type.description}",
            f"**Domain** (subject types): {', '.join(rel_type.domain)}",
            f"**Range** (object types): {', '.join(rel_type.range)}",
        ]

        if rel_type.symmetric:
            lines.append("**Note**: This relationship is symmetric (A→B implies B→A)")

        if rel_type.extraction_hints:
            lines.append("**Extraction hints**:")
            for hint in rel_type.extraction_hints:
                lines.append(f"  - {hint}")

        if rel_type.properties:
            lines.append("**Additional properties**:")
            for prop in rel_type.properties:
                lines.append(f"  - `{prop.name}`: {prop.description}")

        lines.append("")
        return lines

    def generate_combined_extraction_prompt(
        self,
        entity_types: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        include_concepts: bool = True,
        include_claims: bool = True,
    ) -> str:
        """
        Generate a combined prompt for entity, relationship, concept, and claim extraction.

        This is more efficient than separate calls as it allows the LLM
        to extract all knowledge types in a single pass.

        Args:
            entity_types: Specific entity types to extract (None = all)
            relationship_types: Specific relationships to extract (None = all)
            include_concepts: Whether to include concept extraction
            include_claims: Whether to include claim extraction

        Returns:
            Formatted prompt string
        """
        # Get types - prioritize domain-specific types over generic inherited types
        if entity_types:
            types_to_extract = [
                self.ontology.get_entity_type(t)
                for t in entity_types
                if self.ontology.get_entity_type(t)
            ]
        else:
            # Sort by inheritance depth - domain-specific types first
            types_to_extract = self._sort_types_by_specificity(
                list(self.ontology.entity_types.values())
            )

        if relationship_types:
            rels_to_extract = [
                self.ontology.get_relationship_type(r)
                for r in relationship_types
                if self.ontology.get_relationship_type(r)
            ]
        else:
            rels_to_extract = list(self.ontology.relationship_types.values())

        # Get concepts and claim types
        concepts_to_extract = list(self.ontology.concepts.values()) if include_concepts else []
        claim_types_to_extract = list(self.ontology.claim_types.values()) if include_claims else []

        prompt_parts = [
            f"# Knowledge Extraction using {self.ontology.metadata.name} Ontology v{self.ontology.metadata.version}",
            "",
        ]

        if self.ontology.metadata.description:
            prompt_parts.append(f"**Context**: {self.ontology.metadata.description}")
            prompt_parts.append("")

        prompt_parts.extend([
            "Extract entities, relationships, thematic concepts, and claims from the text according to these schema definitions.",
            "",
            "---",
            "",
            "## Entity Types (DOMAIN-SPECIFIC FIRST)",
            "",
        ])

        for entity_type in types_to_extract:
            prompt_parts.extend(self._format_entity_type_compact(entity_type))

        prompt_parts.extend([
            "---",
            "",
            "## Relationship Types",
            "",
        ])

        for rel_type in rels_to_extract:
            prompt_parts.extend(self._format_relationship_type_compact(rel_type))

        # Add concepts section
        if concepts_to_extract:
            prompt_parts.extend([
                "---",
                "",
                "## Thematic Concepts",
                "",
                "Extract mentions of these thematic concepts and link them to entities:",
                "",
            ])
            for concept in concepts_to_extract:
                prompt_parts.extend(self._format_concept_compact(concept))

        # Add claim types section
        if claim_types_to_extract:
            prompt_parts.extend([
                "---",
                "",
                "## Claim Types",
                "",
                "Extract factual claims and statements of these types:",
                "",
            ])
            for claim_type in claim_types_to_extract:
                prompt_parts.extend(self._format_claim_type_compact(claim_type))

        prompt_parts.extend([
            "---",
            "",
            "## Output Format",
            "",
            "Return ONLY a valid JSON object (no other text before or after):",
            "```json",
            "{",
            '  "entities": [',
            '    {',
            '      "name": "Entity Name",',
            '      "entity_type": "RegioDeal",',
            '      "description": "description of the entity",',
            '      "confidence": 0.9,',
            '      "properties": {"key": "value"}',
            '    }',
            "  ],",
            '  "relationships": [',
            '    {"subject": "Entity A", "subject_type": "Ministerie", "predicate": "FINANCIERT", "object": "Entity B", "object_type": "RegioDeal", "confidence": 0.8, "context": "evidence snippet"}',
            "  ],",
            '  "concepts": [',
            '    {"name": "BredeWelvaart", "mentions": ["context snippet 1", "context snippet 2"], "related_entities": ["Entity A", "Entity B"]}',
            "  ],",
            '  "claims": [',
            '    {"statement": "Vergrijzing leidt tot minder voorzieningen", "claim_type": "Causaal", "entities_involved": ["Vergrijzing"], "confidence": 0.8, "evidence": "evidence snippet"}',
            "  ]",
            "}",
            "```",
            "",
            "## Extraction Guidelines",
            "",
            "### CRITICAL — Exhaustive extraction (complete recall, not a shortlist)",
            "Extract EVERY entity the text mentions — do NOT select only the most important "
            "ones and do NOT summarize. Go through the text systematically and emit every "
            "distinct entity; a document of this kind yields many dozens. Returning only a "
            "handful means you MISSED most of them. Do NOT deduplicate, merge, or filter — "
            "downstream handles that; your job is COMPLETE RECALL. Never silently drop an "
            "entity because it seems minor or is hard to classify.",
            "",
            "1. **Entities**: Extract every entity exhaustively, then type each one",
            "   - **Map aggressively to the schema**: use the MOST SPECIFIC defined type (e.g., Gemeente, Ministerie, RegioDeal, BeleidsThema) over generic ones (Organization, Place)",
            "   - **Fallback**: if an entity genuinely fits NO defined type, set `entity_type` to \"other\" — never drop it, never force a wrong type",
            "   - Use `entity_type` field with values EXACTLY matching the defined types (or \"other\")",
            "   - Normalize names (proper capitalization, no extra whitespace)",
            "   - Extract properties listed for each entity type",
            "   - Assign confidence scores (0.0-1.0)",
            "",
            "2. **Relationships**: Extract connections between entities",
            "   - Use relationship names EXACTLY as defined (e.g., FINANCIERT, LEIDT_TOT, RICHT_ZICH_OP)",
            "   - Subject type must be in the relationship's domain list",
            "   - Object type must be in the relationship's range list",
            "   - Include context snippet showing the evidence",
            "   - Look for causal relationships (leidt tot, veroorzaakt, resulteert in)",
            "",
        ])

        # Add concept-specific guidelines if concepts present
        if concepts_to_extract:
            prompt_parts.extend([
                "3. **Concepts**: Extract thematic concept mentions",
                "   - Identify text passages that discuss each concept",
                "   - Link concepts to relevant entities mentioned in same context",
                "   - Use concept names EXACTLY as defined",
                "",
            ])

        # Add claim-specific guidelines if claims present
        if claim_types_to_extract:
            prompt_parts.extend([
                "4. **Claims**: Extract factual statements and assertions",
                "   - Classify each claim by type (Feitelijk, Causaal, Normatief, Predictief)",
                "   - Link claims to entities they reference",
                "   - Include the source text as evidence",
                "   - Pay special attention to causal claims (X leidt tot Y)",
                "",
            ])

        prompt_parts.extend([
            "5. **Output**: Return ONLY valid JSON, no explanatory text",
            "   - Prefer precision over recall",
            "   - Use lower confidence for inferred information",
            "",
        ])

        return "\n".join(prompt_parts)

    def _sort_types_by_specificity(
        self, entity_types: List[EntityTypeDefinition]
    ) -> List[EntityTypeDefinition]:
        """
        Sort entity types to prioritize domain-specific types over generic inherited types.

        Types with parent_type are more specific and should come first.
        Types without parent_type are base types and come last.
        """
        def specificity_key(et: EntityTypeDefinition) -> tuple:
            # Generic schema.org types come last
            generic_types = {'Thing', 'CreativeWork', 'Article', 'Report', 'Book',
                           'WebPage', 'Dataset', 'Person', 'Organization', 'Place',
                           'Event', 'Action', 'Product', 'Service'}
            is_generic = et.name in generic_types
            has_parent = et.parent_type is not None
            # Lower tuple = higher priority: (0, False, name) beats (1, True, name)
            return (1 if is_generic else 0, not has_parent, et.name)

        return sorted(entity_types, key=specificity_key)

    def _format_concept_compact(
        self, concept: ConceptDefinition
    ) -> List[str]:
        """Format a concept definition for the prompt."""
        lines = [f"**{concept.name}**: {concept.description}"]

        if concept.indicators:
            lines.append(f"  - Indicators: {', '.join(concept.indicators[:5])}")

        if concept.related_concepts:
            lines.append(f"  - Related: {', '.join(concept.related_concepts[:5])}")

        if concept.causal_patterns:
            lines.append(f"  - Causal patterns: {'; '.join(concept.causal_patterns[:3])}")

        lines.append("")
        return lines

    def _format_claim_type_compact(
        self, claim_type: ClaimTypeDefinition
    ) -> List[str]:
        """Format a claim type definition for the prompt."""
        lines = [f"**{claim_type.name}**: {claim_type.description}"]

        if claim_type.indicators:
            lines.append(f"  - Look for: {', '.join(claim_type.indicators[:5])}")

        lines.append("")
        return lines

    def _format_entity_type_compact(
        self, entity_type: EntityTypeDefinition
    ) -> List[str]:
        """Format entity type in compact form."""
        lines = [f"**{entity_type.name}**: {entity_type.description}"]

        if entity_type.examples:
            lines.append(f"  - Examples: {', '.join(entity_type.examples[:3])}")

        # List extraction hints if available
        if entity_type.extraction_hints:
            lines.append(f"  - Look for: {'; '.join(entity_type.extraction_hints[:3])}")

        # List ALL extractable properties (not just required)
        if entity_type.properties:
            extractable_props = [
                p for p in entity_type.properties
                if p.extraction_strategy != ExtractionStrategy.NONE
            ]
            if extractable_props:
                prop_details = []
                for p in extractable_props[:8]:  # Limit to 8 most important
                    req_marker = "*" if (p.validation and p.validation.required) else ""
                    prop_details.append(f"`{p.name}`{req_marker}")
                lines.append(f"  - Properties to extract: {', '.join(prop_details)}")

        lines.append("")
        return lines

    def _format_relationship_type_compact(
        self, rel_type: RelationshipTypeDefinition
    ) -> List[str]:
        """Format relationship type in compact form."""
        domain_str = "/".join(rel_type.domain)
        range_str = "/".join(rel_type.range)
        lines = [
            f"**{rel_type.name}**: {rel_type.description}",
            f"  - Pattern: [{domain_str}] --{rel_type.name}--> [{range_str}]",
        ]

        if rel_type.extraction_hints:
            lines.append(f"  - Look for: {'; '.join(rel_type.extraction_hints[:2])}")

        lines.append("")
        return lines

    def generate_document_type_detection_prompt(self) -> str:
        """
        Generate a prompt to detect document type.

        Returns:
            Formatted prompt for document type classification
        """
        return """Analyze the following text and classify the document type.

## Document Types
- **academic_paper**: Scholarly articles, research papers, journal publications
- **policy_document**: Government policies, regulations, official guidelines
- **policy_advice**: Expert recommendations, advisory reports, consultations
- **news_article**: Journalism, news reports, press releases
- **report**: Business reports, technical reports, white papers
- **legal_document**: Laws, contracts, legal opinions
- **presentation**: Slides, talk transcripts, lecture notes
- **social_media**: Posts, comments, discussions
- **other**: Any other document type

## Output Format
Return a JSON object:
```json
{
  "document_type": "type_name",
  "confidence": 0.0-1.0,
  "indicators": ["reason 1", "reason 2"]
}
```

## Guidelines
- Base classification on structure, language, and content
- Include specific indicators that led to the classification
- Use confidence < 0.7 if classification is uncertain
"""
