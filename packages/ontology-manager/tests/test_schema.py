"""Tests for ontology schema models."""

import pytest

from ontology_manager.schema import (
    Cardinality,
    ClaimTypeDefinition,
    ConceptDefinition,
    DataType,
    EntityTypeDefinition,
    ExtractionPatternDefinition,
    ExtractionStrategy,
    Ontology,
    OntologyMetadata,
    PropertyDefinition,
    RelationshipTypeDefinition,
    ValidationRules,
)


def make_test_ontology():
    """Create a test ontology for reuse across tests."""
    return Ontology(
        metadata=OntologyMetadata(name="test", version="1.0", description="Test ontology"),
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
                properties=[PropertyDefinition(name="role", data_type=DataType.STRING)],
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


class TestDataTypeEnum:
    """Tests for the DataType enum."""

    def test_all_data_types_exist(self):
        """Test that all expected data types are defined."""
        expected = [
            "string", "text", "integer", "float", "boolean",
            "date", "datetime", "url", "email", "enum",
            "array", "reference", "object", "json",
        ]
        for value in expected:
            assert DataType(value) is not None

    def test_data_type_is_str_enum(self):
        """Test DataType values are strings."""
        assert DataType.STRING == "string"
        assert DataType.INTEGER == "integer"
        assert isinstance(DataType.STRING, str)


class TestExtractionStrategyEnum:
    """Tests for the ExtractionStrategy enum."""

    def test_all_strategies_exist(self):
        """Test that all extraction strategies are defined."""
        expected = ["direct", "inferred", "lookup", "computed", "none", "llm"]
        for value in expected:
            assert ExtractionStrategy(value) is not None

    def test_default_strategy(self):
        """Test default extraction strategy is DIRECT."""
        prop = PropertyDefinition(name="test", data_type=DataType.STRING)
        assert prop.extraction_strategy == ExtractionStrategy.DIRECT


class TestCardinalityEnum:
    """Tests for the Cardinality enum."""

    def test_hyphen_variants(self):
        """Test cardinality with hyphen separators."""
        assert Cardinality.ONE_TO_ONE == "one-to-one"
        assert Cardinality.ONE_TO_MANY == "one-to-many"
        assert Cardinality.MANY_TO_ONE == "many-to-one"
        assert Cardinality.MANY_TO_MANY == "many-to-many"

    def test_underscore_variants(self):
        """Test cardinality with underscore separators for YAML compatibility."""
        assert Cardinality.ONE_TO_ONE_ALT == "one_to_one"
        assert Cardinality.ONE_TO_MANY_ALT == "one_to_many"
        assert Cardinality.MANY_TO_ONE_ALT == "many_to_one"
        assert Cardinality.MANY_TO_MANY_ALT == "many_to_many"


class TestValidationRules:
    """Tests for the ValidationRules model."""

    def test_defaults(self):
        """Test default validation rules."""
        rules = ValidationRules()
        assert rules.required is False
        assert rules.min_length is None
        assert rules.max_length is None
        assert rules.min_value is None
        assert rules.max_value is None
        assert rules.pattern is None
        assert rules.allowed_values is None
        assert rules.unique is False

    def test_full_rules(self):
        """Test ValidationRules with all fields set."""
        rules = ValidationRules(
            required=True,
            min_length=1,
            max_length=100,
            min_value=0,
            max_value=999,
            pattern=r"^[A-Z]",
            allowed_values=["A", "B", "C"],
            unique=True,
        )
        assert rules.required is True
        assert rules.min_length == 1
        assert rules.max_length == 100
        assert rules.min_value == 0
        assert rules.max_value == 999
        assert rules.pattern == r"^[A-Z]"
        assert rules.allowed_values == ["A", "B", "C"]
        assert rules.unique is True

    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored."""
        rules = ValidationRules(required=True, unknown_field="ignored")
        assert rules.required is True


class TestPropertyDefinition:
    """Tests for the PropertyDefinition model."""

    def test_minimal_property(self):
        """Test property with only required fields."""
        prop = PropertyDefinition(name="test_prop", data_type=DataType.STRING)
        assert prop.name == "test_prop"
        assert prop.data_type == DataType.STRING
        assert prop.extraction_strategy == ExtractionStrategy.DIRECT
        assert prop.validation is None

    def test_property_with_validation_dict(self):
        """Test that validation can be passed as dict and gets parsed."""
        prop = PropertyDefinition(
            name="name",
            data_type=DataType.STRING,
            validation={"required": True, "min_length": 1},
        )
        assert isinstance(prop.validation, ValidationRules)
        assert prop.validation.required is True
        assert prop.validation.min_length == 1

    def test_property_with_all_fields(self):
        """Test property with all optional fields set."""
        prop = PropertyDefinition(
            name="url_field",
            description="A URL property",
            data_type=DataType.URL,
            extraction_strategy=ExtractionStrategy.INFERRED,
            schema_org_mapping="schema:url",
            dublin_core_mapping="dc:identifier",
            examples=["https://example.com"],
            default_value="https://default.com",
            reference_type="WebPage",
        )
        assert prop.description == "A URL property"
        assert prop.extraction_strategy == ExtractionStrategy.INFERRED
        assert prop.schema_org_mapping == "schema:url"
        assert prop.dublin_core_mapping == "dc:identifier"
        assert prop.examples == ["https://example.com"]
        assert prop.default_value == "https://default.com"
        assert prop.reference_type == "WebPage"


class TestEntityTypeDefinition:
    """Tests for the EntityTypeDefinition model."""

    def test_minimal_entity_type(self):
        """Test entity type with only required fields."""
        et = EntityTypeDefinition(name="Thing")
        assert et.name == "Thing"
        assert et.description is None
        assert et.properties == []
        assert et.aliases is None
        assert et.parent_type is None

    def test_entity_type_with_properties_as_dicts(self):
        """Test that properties list of dicts gets parsed."""
        et = EntityTypeDefinition(
            name="Person",
            properties=[
                {"name": "age", "data_type": "integer"},
                {"name": "email", "data_type": "email"},
            ],
        )
        assert len(et.properties) == 2
        assert isinstance(et.properties[0], PropertyDefinition)
        assert et.properties[0].name == "age"
        assert et.properties[0].data_type == DataType.INTEGER
        assert et.properties[1].data_type == DataType.EMAIL

    def test_entity_type_with_aliases(self):
        """Test entity type with aliases."""
        et = EntityTypeDefinition(
            name="Person",
            aliases=["Human", "Individual"],
        )
        assert et.aliases == ["Human", "Individual"]

    def test_get_required_properties(self):
        """Test get_required_properties returns only required ones."""
        et = EntityTypeDefinition(
            name="Person",
            properties=[
                PropertyDefinition(
                    name="name",
                    data_type=DataType.STRING,
                    validation=ValidationRules(required=True),
                ),
                PropertyDefinition(name="age", data_type=DataType.INTEGER),
                PropertyDefinition(
                    name="email",
                    data_type=DataType.EMAIL,
                    validation=ValidationRules(required=True),
                ),
            ],
        )
        required = et.get_required_properties()
        assert len(required) == 2
        assert required[0].name == "name"
        assert required[1].name == "email"

    def test_get_extraction_properties(self):
        """Test get_extraction_properties excludes NONE strategy."""
        et = EntityTypeDefinition(
            name="Person",
            properties=[
                PropertyDefinition(name="name", data_type=DataType.STRING),
                PropertyDefinition(
                    name="internal_id",
                    data_type=DataType.STRING,
                    extraction_strategy=ExtractionStrategy.NONE,
                ),
                PropertyDefinition(
                    name="bio",
                    data_type=DataType.TEXT,
                    extraction_strategy=ExtractionStrategy.LLM,
                ),
            ],
        )
        extractable = et.get_extraction_properties()
        assert len(extractable) == 2
        assert extractable[0].name == "name"
        assert extractable[1].name == "bio"

    def test_entity_type_with_parent(self):
        """Test entity type with parent_type for inheritance."""
        et = EntityTypeDefinition(name="Student", parent_type="Person")
        assert et.parent_type == "Person"


class TestRelationshipTypeDefinition:
    """Tests for the RelationshipTypeDefinition model."""

    def test_minimal_relationship(self):
        """Test relationship with only required fields."""
        rel = RelationshipTypeDefinition(name="RELATED_TO")
        assert rel.name == "RELATED_TO"
        assert rel.domain == []
        assert rel.range == []
        assert rel.cardinality == Cardinality.MANY_TO_MANY
        assert rel.symmetric is False
        assert rel.transitive is False

    def test_relationship_with_domain_range(self):
        """Test relationship with domain and range."""
        rel = RelationshipTypeDefinition(
            name="WORKS_AT",
            domain=["Person"],
            range=["Organization"],
        )
        assert rel.domain == ["Person"]
        assert rel.range == ["Organization"]

    def test_source_types_target_types_alias(self):
        """Test that source_types/target_types populate domain/range."""
        rel = RelationshipTypeDefinition(
            name="BELONGS_TO",
            source_types=["Student"],
            target_types=["University"],
        )
        assert rel.domain == ["Student"]
        assert rel.range == ["University"]

    def test_source_types_does_not_override_domain(self):
        """Test that source_types does not override explicit domain."""
        rel = RelationshipTypeDefinition(
            name="WORKS_AT",
            domain=["Person"],
            source_types=["Employee"],
        )
        assert rel.domain == ["Person"]

    def test_relationship_with_properties_as_dicts(self):
        """Test that relationship properties dicts get parsed."""
        rel = RelationshipTypeDefinition(
            name="WORKS_AT",
            properties=[{"name": "start_date", "data_type": "date"}],
        )
        assert len(rel.properties) == 1
        assert isinstance(rel.properties[0], PropertyDefinition)
        assert rel.properties[0].name == "start_date"

    def test_symmetric_relationship(self):
        """Test symmetric relationship."""
        rel = RelationshipTypeDefinition(
            name="KNOWS",
            symmetric=True,
        )
        assert rel.symmetric is True

    def test_transitive_relationship(self):
        """Test transitive relationship."""
        rel = RelationshipTypeDefinition(
            name="PART_OF",
            transitive=True,
        )
        assert rel.transitive is True

    def test_inverse_relationship(self):
        """Test relationship with inverse."""
        rel = RelationshipTypeDefinition(
            name="WORKS_AT",
            inverse="EMPLOYS",
        )
        assert rel.inverse == "EMPLOYS"


class TestConceptDefinition:
    """Tests for the ConceptDefinition model."""

    def test_minimal_concept(self):
        """Test concept with only required fields."""
        concept = ConceptDefinition(name="Innovation")
        assert concept.name == "Innovation"
        assert concept.description is None
        assert concept.related_concepts is None
        assert concept.indicators is None
        assert concept.causal_patterns is None

    def test_full_concept(self):
        """Test concept with all fields."""
        concept = ConceptDefinition(
            name="Innovation",
            description="Innovation concept",
            indicators=["disruptive", "novel"],
            related_concepts=["Technology", "Research"],
            causal_patterns=["X leads to innovation"],
            examples=["AI breakthrough"],
        )
        assert concept.description == "Innovation concept"
        assert concept.indicators == ["disruptive", "novel"]
        assert concept.related_concepts == ["Technology", "Research"]
        assert concept.causal_patterns == ["X leads to innovation"]
        assert concept.examples == ["AI breakthrough"]


class TestClaimTypeDefinition:
    """Tests for the ClaimTypeDefinition model."""

    def test_minimal_claim_type(self):
        """Test claim type with only required fields."""
        ct = ClaimTypeDefinition(name="Factual")
        assert ct.name == "Factual"
        assert ct.description is None
        assert ct.indicators is None

    def test_full_claim_type(self):
        """Test claim type with all fields."""
        ct = ClaimTypeDefinition(
            name="Factual",
            description="A factual claim",
            indicators=["research shows", "data indicates"],
        )
        assert ct.description == "A factual claim"
        assert ct.indicators == ["research shows", "data indicates"]


class TestOntologyMetadata:
    """Tests for the OntologyMetadata model."""

    def test_minimal_metadata(self):
        """Test metadata with only required fields."""
        meta = OntologyMetadata(name="test", version="1.0")
        assert meta.name == "test"
        assert meta.version == "1.0"
        assert meta.description is None
        assert meta.extends is None

    def test_full_metadata(self):
        """Test metadata with all fields."""
        meta = OntologyMetadata(
            name="scholarly",
            version="2.0",
            description="Scholarly ontology",
            authors=["Author A"],
            license="MIT",
            extends="general",
            created="2024-01-01",
            updated="2024-06-01",
        )
        assert meta.extends == "general"
        assert meta.authors == ["Author A"]
        assert meta.license == "MIT"


class TestOntology:
    """Tests for the Ontology model."""

    def test_create_from_objects(self):
        """Test creating ontology from Pydantic model objects."""
        ontology = make_test_ontology()
        assert ontology.metadata.name == "test"
        assert len(ontology.entity_types) == 2
        assert len(ontology.relationship_types) == 1
        assert len(ontology.concepts) == 1
        assert len(ontology.claim_types) == 1

    def test_entity_types_from_list(self):
        """Test that entity_types can be provided as a list (converted to dict)."""
        ontology = Ontology(
            metadata=OntologyMetadata(name="test", version="1.0"),
            entity_types=[
                {"name": "Person", "description": "A person"},
                {"name": "Place", "description": "A place"},
            ],
        )
        assert "Person" in ontology.entity_types
        assert "Place" in ontology.entity_types
        assert isinstance(ontology.entity_types["Person"], EntityTypeDefinition)

    def test_entity_types_from_dict(self):
        """Test that entity_types dict values get parsed."""
        ontology = Ontology(
            metadata=OntologyMetadata(name="test", version="1.0"),
            entity_types={
                "Person": {"name": "Person", "description": "A person"},
            },
        )
        assert isinstance(ontology.entity_types["Person"], EntityTypeDefinition)
        assert ontology.entity_types["Person"].description == "A person"

    def test_relationship_types_from_list(self):
        """Test that relationship_types can be provided as a list."""
        ontology = Ontology(
            metadata=OntologyMetadata(name="test", version="1.0"),
            relationship_types=[
                {"name": "KNOWS", "description": "Knows relationship"},
            ],
        )
        assert "KNOWS" in ontology.relationship_types
        assert isinstance(ontology.relationship_types["KNOWS"], RelationshipTypeDefinition)

    def test_concepts_from_list(self):
        """Test that concepts can be provided as a list."""
        ontology = Ontology(
            metadata=OntologyMetadata(name="test", version="1.0"),
            concepts=[
                {"name": "Innovation", "description": "Innovation concept"},
            ],
        )
        assert "Innovation" in ontology.concepts

    def test_claim_types_from_list(self):
        """Test that claim_types can be provided as a list."""
        ontology = Ontology(
            metadata=OntologyMetadata(name="test", version="1.0"),
            claim_types=[
                {"name": "Factual", "description": "A factual claim"},
            ],
        )
        assert "Factual" in ontology.claim_types

    def test_extraction_patterns_from_list(self):
        """Test that extraction_patterns can be provided as a list."""
        ontology = Ontology(
            metadata=OntologyMetadata(name="test", version="1.0"),
            extraction_patterns=[
                {"name": "DatePattern", "pattern": r"\d{4}-\d{2}-\d{2}"},
            ],
        )
        assert "DatePattern" in ontology.extraction_patterns
        assert isinstance(
            ontology.extraction_patterns["DatePattern"],
            ExtractionPatternDefinition,
        )

    def test_get_entity_type_exact_match(self):
        """Test get_entity_type with exact name."""
        ontology = make_test_ontology()
        result = ontology.get_entity_type("Person")
        assert result is not None
        assert result.name == "Person"

    def test_get_entity_type_case_insensitive(self):
        """Test get_entity_type with different casing."""
        ontology = make_test_ontology()
        result = ontology.get_entity_type("person")
        assert result is not None
        assert result.name == "Person"

        result_upper = ontology.get_entity_type("PERSON")
        assert result_upper is not None
        assert result_upper.name == "Person"

    def test_get_entity_type_by_alias(self):
        """Test get_entity_type looks up aliases."""
        ontology = make_test_ontology()
        result = ontology.get_entity_type("Human")
        assert result is not None
        assert result.name == "Person"

    def test_get_entity_type_by_alias_case_insensitive(self):
        """Test get_entity_type alias lookup is case-insensitive."""
        ontology = make_test_ontology()
        result = ontology.get_entity_type("individual")
        assert result is not None
        assert result.name == "Person"

    def test_get_entity_type_not_found(self):
        """Test get_entity_type returns None for unknown type."""
        ontology = make_test_ontology()
        result = ontology.get_entity_type("Unknown")
        assert result is None

    def test_get_relationship_type_exact_match(self):
        """Test get_relationship_type with exact name."""
        ontology = make_test_ontology()
        result = ontology.get_relationship_type("WORKS_AT")
        assert result is not None
        assert result.name == "WORKS_AT"

    def test_get_relationship_type_case_insensitive(self):
        """Test get_relationship_type with different casing."""
        ontology = make_test_ontology()
        result = ontology.get_relationship_type("works_at")
        assert result is not None
        assert result.name == "WORKS_AT"

    def test_get_relationship_type_not_found(self):
        """Test get_relationship_type returns None for unknown type."""
        ontology = make_test_ontology()
        result = ontology.get_relationship_type("UNKNOWN_REL")
        assert result is None

    def test_get_valid_relationships_for_entity(self):
        """Test getting valid relationships for a given entity type."""
        ontology = make_test_ontology()
        rels = ontology.get_valid_relationships_for_entity("Person")
        assert len(rels) == 1
        assert rels[0].name == "WORKS_AT"

    def test_get_valid_relationships_for_entity_no_match(self):
        """Test getting valid relationships returns empty for non-domain entity."""
        ontology = make_test_ontology()
        rels = ontology.get_valid_relationships_for_entity("Organization")
        assert len(rels) == 0

    def test_get_concept(self):
        """Test get_concept case-insensitive lookup."""
        ontology = make_test_ontology()
        result = ontology.get_concept("innovation")
        assert result is not None
        assert result.name == "Innovation"

    def test_get_concept_not_found(self):
        """Test get_concept returns None for unknown concept."""
        ontology = make_test_ontology()
        result = ontology.get_concept("Nonexistent")
        assert result is None

    def test_get_all_entity_type_names(self):
        """Test get_all_entity_type_names returns all names."""
        ontology = make_test_ontology()
        names = ontology.get_all_entity_type_names()
        assert set(names) == {"Person", "Organization"}

    def test_get_all_relationship_type_names(self):
        """Test get_all_relationship_type_names returns all names."""
        ontology = make_test_ontology()
        names = ontology.get_all_relationship_type_names()
        assert names == ["WORKS_AT"]

    def test_get_all_concept_names(self):
        """Test get_all_concept_names returns all names."""
        ontology = make_test_ontology()
        names = ontology.get_all_concept_names()
        assert names == ["Innovation"]

    def test_get_all_claim_type_names(self):
        """Test get_all_claim_type_names returns all names."""
        ontology = make_test_ontology()
        names = ontology.get_all_claim_type_names()
        assert names == ["Factual"]

    def test_merge_with_parent(self):
        """Test merge_with combines child and parent ontologies."""
        parent = Ontology(
            metadata=OntologyMetadata(name="parent", version="1.0"),
            entity_types={
                "Thing": EntityTypeDefinition(name="Thing", description="Base thing"),
            },
            relationship_types={
                "RELATED_TO": RelationshipTypeDefinition(
                    name="RELATED_TO", description="Generic relation"
                ),
            },
            concepts={
                "General": ConceptDefinition(name="General", description="General concept"),
            },
        )

        child = Ontology(
            metadata=OntologyMetadata(name="child", version="1.0", extends="parent"),
            entity_types={
                "Person": EntityTypeDefinition(name="Person", description="A person"),
            },
            relationship_types={
                "KNOWS": RelationshipTypeDefinition(
                    name="KNOWS", description="Knows relation"
                ),
            },
        )

        merged = child.merge_with(parent)

        assert merged.metadata.name == "child"
        assert "Thing" in merged.entity_types
        assert "Person" in merged.entity_types
        assert "RELATED_TO" in merged.relationship_types
        assert "KNOWS" in merged.relationship_types
        assert "General" in merged.concepts

    def test_merge_with_child_overrides_parent(self):
        """Test that child entity types override parent types with same key."""
        parent = Ontology(
            metadata=OntologyMetadata(name="parent", version="1.0"),
            entity_types={
                "Person": EntityTypeDefinition(name="Person", description="Base person"),
            },
        )

        child = Ontology(
            metadata=OntologyMetadata(name="child", version="1.0"),
            entity_types={
                "Person": EntityTypeDefinition(
                    name="Person", description="Extended person"
                ),
            },
        )

        merged = child.merge_with(parent)
        assert merged.entity_types["Person"].description == "Extended person"

    def test_from_yaml_string(self):
        """Test loading an ontology from a YAML string."""
        yaml_content = """
metadata:
  name: test_yaml
  version: "1.0"
  description: "YAML test ontology"
entity_types:
  Person:
    name: Person
    description: A person entity
    aliases:
      - Human
    properties:
      - name: name
        data_type: string
        validation:
          required: true
relationship_types:
  KNOWS:
    name: KNOWS
    description: Knows someone
    domain:
      - Person
    range:
      - Person
"""
        ontology = Ontology.from_yaml_string(yaml_content)
        assert ontology.metadata.name == "test_yaml"
        assert ontology.metadata.version == "1.0"
        assert "Person" in ontology.entity_types
        assert ontology.entity_types["Person"].aliases == ["Human"]
        assert len(ontology.entity_types["Person"].properties) == 1
        assert ontology.entity_types["Person"].properties[0].validation.required is True
        assert "KNOWS" in ontology.relationship_types
        assert ontology.relationship_types["KNOWS"].domain == ["Person"]

    def test_from_yaml_string_list_format(self):
        """Test loading ontology from YAML with list-format entity_types."""
        yaml_content = """
metadata:
  name: list_format
  version: "1.0"
entity_types:
  - name: Person
    description: A person
  - name: Place
    description: A place
"""
        ontology = Ontology.from_yaml_string(yaml_content)
        assert "Person" in ontology.entity_types
        assert "Place" in ontology.entity_types

    def test_to_yaml_and_load(self, tmp_path):
        """Test round-tripping ontology through YAML file.

        Uses a simple ontology without Cardinality enum defaults because
        model_dump() serializes str-enums as Python tagged objects that
        yaml.safe_load cannot deserialize.
        """
        import yaml

        ontology = Ontology(
            metadata=OntologyMetadata(name="roundtrip", version="1.0", description="RT"),
            entity_types={
                "Person": EntityTypeDefinition(name="Person", description="A person"),
            },
        )

        yaml_path = tmp_path / "test_ontology.yaml"
        # Use mode="json" to ensure enums serialize as plain strings
        data = ontology.model_dump(mode="json")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        loaded = Ontology.from_yaml(yaml_path)
        assert loaded.metadata.name == "roundtrip"
        assert "Person" in loaded.entity_types

    def test_empty_ontology(self):
        """Test creating ontology with no types."""
        ontology = Ontology(
            metadata=OntologyMetadata(name="empty", version="0.1"),
        )
        assert len(ontology.entity_types) == 0
        assert len(ontology.relationship_types) == 0
        assert len(ontology.concepts) == 0
        assert len(ontology.claim_types) == 0
        assert len(ontology.extraction_patterns) == 0
        assert ontology.prefixes == {}
