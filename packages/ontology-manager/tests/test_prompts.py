"""Tests for ontology-driven prompt generation."""

import pytest

from ontology_manager.prompts import OntologyPromptGenerator
from ontology_manager.schema import (
    ClaimTypeDefinition,
    ConceptDefinition,
    DataType,
    EntityTypeDefinition,
    ExtractionStrategy,
    Ontology,
    OntologyMetadata,
    PropertyDefinition,
    RelationshipTypeDefinition,
    ValidationRules,
)


def make_test_ontology():
    """Create a test ontology for prompt generation tests."""
    return Ontology(
        metadata=OntologyMetadata(
            name="test", version="1.0", description="Test ontology for prompts"
        ),
        entity_types={
            "Person": EntityTypeDefinition(
                name="Person",
                description="A person",
                aliases=["Human", "Individual"],
                examples=["John Doe", "Jane Smith"],
                extraction_hints=["Look for proper nouns"],
                properties=[
                    PropertyDefinition(
                        name="name",
                        description="Full name",
                        data_type=DataType.STRING,
                        validation=ValidationRules(required=True, min_length=1),
                    ),
                    PropertyDefinition(
                        name="age",
                        description="Age in years",
                        data_type=DataType.INTEGER,
                    ),
                    PropertyDefinition(
                        name="internal_id",
                        description="Internal ID",
                        data_type=DataType.STRING,
                        extraction_strategy=ExtractionStrategy.NONE,
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
                extraction_hints=["works at", "employed by"],
                properties=[
                    PropertyDefinition(
                        name="role",
                        description="Job role",
                        data_type=DataType.STRING,
                    ),
                ],
            ),
            "KNOWS": RelationshipTypeDefinition(
                name="KNOWS",
                description="Knows another person",
                domain=["Person"],
                range=["Person"],
                symmetric=True,
            ),
        },
        concepts={
            "Innovation": ConceptDefinition(
                name="Innovation",
                description="Innovation concept",
                indicators=["disruptive", "novel"],
                related_concepts=["Technology"],
                causal_patterns=["X leads to innovation"],
            ),
        },
        claim_types={
            "Factual": ClaimTypeDefinition(
                name="Factual",
                description="A factual claim",
                indicators=["research shows", "data indicates"],
            ),
        },
    )


class TestGenerateEntityExtractionPrompt:
    """Tests for generate_entity_extraction_prompt."""

    def test_contains_entity_types_section(self):
        """Test prompt includes Entity Types section header."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_entity_extraction_prompt()

        assert "## Entity Types to Extract" in prompt

    def test_contains_output_format(self):
        """Test prompt includes output format section."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_entity_extraction_prompt()

        assert "## Output Format" in prompt

    def test_contains_guidelines(self):
        """Test prompt includes guidelines section."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_entity_extraction_prompt()

        assert "## Guidelines" in prompt

    def test_contains_entity_type_names(self):
        """Test prompt mentions entity type names."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_entity_extraction_prompt()

        assert "### Person" in prompt
        assert "### Organization" in prompt

    def test_contains_extraction_hints(self):
        """Test prompt includes extraction hints."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_entity_extraction_prompt()

        assert "Look for proper nouns" in prompt

    def test_contains_examples(self):
        """Test prompt includes examples."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_entity_extraction_prompt()

        assert "John Doe" in prompt
        assert "Jane Smith" in prompt

    def test_includes_extractable_properties(self):
        """Test prompt includes extractable properties but not NONE strategy."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_entity_extraction_prompt(include_properties=True)

        assert "`name`" in prompt
        assert "`age`" in prompt
        assert "internal_id" not in prompt

    def test_excludes_properties_when_flag_false(self):
        """Test prompt excludes properties when include_properties is False."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_entity_extraction_prompt(include_properties=False)

        assert "Properties to extract" not in prompt

    def test_filter_specific_entity_types(self):
        """Test extracting only specific entity types."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_entity_extraction_prompt(entity_types=["Person"])

        assert "### Person" in prompt
        assert "### Organization" not in prompt

    def test_marks_required_properties(self):
        """Test that required properties are indicated."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_entity_extraction_prompt()

        assert "(required)" in prompt


class TestGenerateRelationshipExtractionPrompt:
    """Tests for generate_relationship_extraction_prompt."""

    def test_contains_relationship_types_section(self):
        """Test prompt includes Relationship Types section header."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_relationship_extraction_prompt()

        assert "## Relationship Types to Extract" in prompt

    def test_contains_output_format(self):
        """Test prompt includes output format with JSON template."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_relationship_extraction_prompt()

        assert "## Output Format" in prompt
        assert "predicate" in prompt

    def test_contains_relationship_names(self):
        """Test prompt mentions relationship type names."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_relationship_extraction_prompt()

        assert "### WORKS_AT" in prompt
        assert "### KNOWS" in prompt

    def test_contains_domain_range(self):
        """Test prompt includes domain and range information."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_relationship_extraction_prompt()

        assert "**Domain**" in prompt
        assert "**Range**" in prompt
        assert "Person" in prompt
        assert "Organization" in prompt

    def test_contains_extraction_hints(self):
        """Test prompt includes relationship extraction hints."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_relationship_extraction_prompt()

        assert "works at" in prompt
        assert "employed by" in prompt

    def test_notes_symmetric_relationship(self):
        """Test prompt notes symmetric relationships."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_relationship_extraction_prompt()

        assert "symmetric" in prompt.lower()

    def test_includes_relationship_properties(self):
        """Test prompt includes additional properties for relationships."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_relationship_extraction_prompt()

        assert "`role`" in prompt

    def test_filter_specific_relationship_types(self):
        """Test extracting only specific relationship types."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_relationship_extraction_prompt(
            relationship_types=["WORKS_AT"]
        )

        assert "### WORKS_AT" in prompt
        assert "### KNOWS" not in prompt


class TestGenerateCombinedExtractionPrompt:
    """Tests for generate_combined_extraction_prompt."""

    def test_contains_all_sections(self):
        """Test combined prompt includes entity, relationship, concept, and claim sections."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_combined_extraction_prompt()

        assert "## Entity Types" in prompt
        assert "## Relationship Types" in prompt
        assert "## Thematic Concepts" in prompt
        assert "## Claim Types" in prompt
        assert "## Output Format" in prompt
        assert "## Extraction Guidelines" in prompt

    def test_includes_ontology_metadata(self):
        """Test combined prompt includes ontology name and version."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_combined_extraction_prompt()

        assert "test" in prompt
        assert "1.0" in prompt

    def test_includes_description(self):
        """Test combined prompt includes ontology description."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_combined_extraction_prompt()

        assert "Test ontology for prompts" in prompt

    def test_includes_concept_details(self):
        """Test combined prompt includes concept indicators and related concepts."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_combined_extraction_prompt()

        assert "Innovation" in prompt
        assert "disruptive" in prompt
        assert "Technology" in prompt

    def test_includes_claim_type_details(self):
        """Test combined prompt includes claim type indicators."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_combined_extraction_prompt()

        assert "Factual" in prompt
        assert "research shows" in prompt

    def test_exclude_concepts(self):
        """Test combined prompt excludes concepts when flag is False."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_combined_extraction_prompt(include_concepts=False)

        assert "## Thematic Concepts" not in prompt

    def test_exclude_claims(self):
        """Test combined prompt excludes claims when flag is False."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_combined_extraction_prompt(include_claims=False)

        assert "## Claim Types" not in prompt

    def test_exclude_both_concepts_and_claims(self):
        """Test combined prompt with both concepts and claims excluded."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_combined_extraction_prompt(
            include_concepts=False, include_claims=False
        )

        assert "## Thematic Concepts" not in prompt
        assert "## Claim Types" not in prompt
        assert "## Entity Types" in prompt
        assert "## Relationship Types" in prompt

    def test_json_output_template(self):
        """Test combined prompt contains proper JSON output template."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_combined_extraction_prompt()

        assert '"entities"' in prompt
        assert '"relationships"' in prompt
        assert '"concepts"' in prompt
        assert '"claims"' in prompt


class TestSortTypesBySpecificity:
    """Tests for _sort_types_by_specificity."""

    def test_domain_specific_before_generic(self):
        """Test that domain-specific types come before generic types."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)

        types = [
            EntityTypeDefinition(name="Person", description="Generic person"),
            EntityTypeDefinition(
                name="Researcher",
                description="A researcher",
                parent_type="Person",
            ),
            EntityTypeDefinition(name="Organization", description="Generic org"),
        ]

        sorted_types = gen._sort_types_by_specificity(types)

        # Researcher (has parent_type, not in generic set) should be first
        assert sorted_types[0].name == "Researcher"

    def test_generic_schema_org_types_last(self):
        """Test that common schema.org types are sorted to the end."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)

        types = [
            EntityTypeDefinition(name="Person", description="Generic person"),
            EntityTypeDefinition(name="Gemeente", description="A municipality"),
            EntityTypeDefinition(name="Organization", description="Generic org"),
        ]

        sorted_types = gen._sort_types_by_specificity(types)

        # Person and Organization are in the generic set, Gemeente is not
        names = [t.name for t in sorted_types]
        gemeente_idx = names.index("Gemeente")
        person_idx = names.index("Person")
        org_idx = names.index("Organization")
        assert gemeente_idx < person_idx
        assert gemeente_idx < org_idx

    def test_type_with_parent_has_higher_priority(self):
        """Test that types with parent_type have higher priority."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)

        types = [
            EntityTypeDefinition(name="CustomType", description="No parent"),
            EntityTypeDefinition(
                name="ChildType",
                description="Has parent",
                parent_type="CustomType",
            ),
        ]

        sorted_types = gen._sort_types_by_specificity(types)
        assert sorted_types[0].name == "ChildType"


class TestGenerateDocumentTypeDetectionPrompt:
    """Tests for generate_document_type_detection_prompt."""

    def test_contains_document_types(self):
        """Test prompt lists document types."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_document_type_detection_prompt()

        assert "academic_paper" in prompt
        assert "policy_document" in prompt
        assert "news_article" in prompt
        assert "report" in prompt
        assert "legal_document" in prompt
        assert "other" in prompt

    def test_contains_output_format(self):
        """Test prompt includes JSON output format."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_document_type_detection_prompt()

        assert "document_type" in prompt
        assert "confidence" in prompt
        assert "indicators" in prompt

    def test_contains_guidelines(self):
        """Test prompt includes classification guidelines."""
        ontology = make_test_ontology()
        gen = OntologyPromptGenerator(ontology)
        prompt = gen.generate_document_type_detection_prompt()

        assert "classification" in prompt.lower()
