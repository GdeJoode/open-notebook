"""Tests for the OntologyManager singleton facade."""

import pytest

from ontology_manager.config import OntologyManagerConfig
from ontology_manager.manager import OntologyManager, get_ontology_manager
from ontology_manager.prompts import OntologyPromptGenerator
from ontology_manager.schema import (
    DataType,
    EntityTypeDefinition,
    Ontology,
    OntologyMetadata,
    PropertyDefinition,
    RelationshipTypeDefinition,
    ValidationRules,
)
from ontology_manager.validator import OntologyValidator


def make_test_ontology():
    """Create a minimal test ontology."""
    return Ontology(
        metadata=OntologyMetadata(name="test", version="1.0", description="Test"),
        entity_types={
            "Person": EntityTypeDefinition(
                name="Person",
                description="A person",
                properties=[
                    PropertyDefinition(
                        name="name",
                        data_type=DataType.STRING,
                        validation=ValidationRules(required=True),
                    ),
                ],
            ),
        },
        relationship_types={
            "KNOWS": RelationshipTypeDefinition(
                name="KNOWS",
                description="Knows someone",
                domain=["Person"],
                range=["Person"],
            ),
        },
    )


class TestOntologyManagerSingleton:
    """Tests for OntologyManager singleton behavior."""

    def setup_method(self):
        """Reset singleton before each test."""
        OntologyManager.reset()
        import ontology_manager.manager as manager_module
        manager_module._manager = None

    def teardown_method(self):
        """Reset singleton after each test."""
        OntologyManager.reset()
        import ontology_manager.manager as manager_module
        manager_module._manager = None

    def test_singleton_returns_same_instance(self):
        """Test that OntologyManager returns same instance."""
        config = OntologyManagerConfig(_env_file=None)
        mgr_a = OntologyManager(config)
        mgr_b = OntologyManager(config)
        assert mgr_a is mgr_b

    def test_singleton_reset(self):
        """Test that reset creates a fresh instance."""
        config = OntologyManagerConfig(_env_file=None)
        mgr_a = OntologyManager(config)
        OntologyManager.reset()
        mgr_b = OntologyManager(config)
        assert mgr_a is not mgr_b

    def test_config_property(self):
        """Test that config property returns the configuration."""
        config = OntologyManagerConfig(
            default_ontology="scholarly",
            _env_file=None,
        )
        mgr = OntologyManager(config)
        assert mgr.config.default_ontology == "scholarly"

    def test_registry_property(self):
        """Test that registry property returns an OntologyRegistry."""
        config = OntologyManagerConfig(_env_file=None)
        mgr = OntologyManager(config)
        assert mgr.registry is not None

    def test_evolution_property(self):
        """Test that evolution property returns an OntologyEvolutionAgent."""
        config = OntologyManagerConfig(_env_file=None)
        mgr = OntologyManager(config)
        assert mgr.evolution is not None


class TestGetOntologyManager:
    """Tests for the module-level get_ontology_manager function."""

    def setup_method(self):
        """Reset singleton before each test."""
        OntologyManager.reset()
        import ontology_manager.manager as manager_module
        manager_module._manager = None

    def teardown_method(self):
        """Reset singleton after each test."""
        OntologyManager.reset()
        import ontology_manager.manager as manager_module
        manager_module._manager = None

    def test_returns_manager_instance(self):
        """Test get_ontology_manager returns an OntologyManager."""
        mgr = get_ontology_manager(
            config=OntologyManagerConfig(_env_file=None)
        )
        assert isinstance(mgr, OntologyManager)

    def test_returns_same_instance(self):
        """Test get_ontology_manager returns cached instance."""
        config = OntologyManagerConfig(_env_file=None)
        mgr_a = get_ontology_manager(config)
        mgr_b = get_ontology_manager()
        assert mgr_a is mgr_b


class TestCreateValidator:
    """Tests for OntologyManager.create_validator."""

    def setup_method(self):
        """Reset singleton before each test."""
        OntologyManager.reset()
        import ontology_manager.manager as manager_module
        manager_module._manager = None

    def teardown_method(self):
        """Reset singleton after each test."""
        OntologyManager.reset()
        import ontology_manager.manager as manager_module
        manager_module._manager = None

    def test_creates_validator(self):
        """Test create_validator returns an OntologyValidator."""
        config = OntologyManagerConfig(_env_file=None)
        mgr = OntologyManager(config)
        ontology = make_test_ontology()
        validator = mgr.create_validator(ontology)
        assert isinstance(validator, OntologyValidator)

    def test_creates_strict_validator(self):
        """Test create_validator with strict mode."""
        config = OntologyManagerConfig(_env_file=None)
        mgr = OntologyManager(config)
        ontology = make_test_ontology()
        validator = mgr.create_validator(ontology, strict=True)
        assert isinstance(validator, OntologyValidator)
        assert validator.strict is True

    def test_creates_non_strict_validator(self):
        """Test create_validator defaults to non-strict mode."""
        config = OntologyManagerConfig(_env_file=None)
        mgr = OntologyManager(config)
        ontology = make_test_ontology()
        validator = mgr.create_validator(ontology)
        assert validator.strict is False

    def test_validator_works_with_entity(self):
        """Test that created validator can validate an entity."""
        config = OntologyManagerConfig(_env_file=None)
        mgr = OntologyManager(config)
        ontology = make_test_ontology()
        validator = mgr.create_validator(ontology)

        entity = {"type": "Person", "name": "John"}
        report = validator.validate_entity(entity)
        assert report.is_valid is True


class TestCreatePromptGenerator:
    """Tests for OntologyManager.create_prompt_generator."""

    def setup_method(self):
        """Reset singleton before each test."""
        OntologyManager.reset()
        import ontology_manager.manager as manager_module
        manager_module._manager = None

    def teardown_method(self):
        """Reset singleton after each test."""
        OntologyManager.reset()
        import ontology_manager.manager as manager_module
        manager_module._manager = None

    def test_creates_prompt_generator(self):
        """Test create_prompt_generator returns an OntologyPromptGenerator."""
        config = OntologyManagerConfig(_env_file=None)
        mgr = OntologyManager(config)
        ontology = make_test_ontology()
        generator = mgr.create_prompt_generator(ontology)
        assert isinstance(generator, OntologyPromptGenerator)

    def test_generator_uses_correct_ontology(self):
        """Test that created generator uses the provided ontology."""
        config = OntologyManagerConfig(_env_file=None)
        mgr = OntologyManager(config)
        ontology = make_test_ontology()
        generator = mgr.create_prompt_generator(ontology)
        assert generator.ontology is ontology

    def test_generator_produces_prompt(self):
        """Test that created generator can produce an extraction prompt."""
        config = OntologyManagerConfig(_env_file=None)
        mgr = OntologyManager(config)
        ontology = make_test_ontology()
        generator = mgr.create_prompt_generator(ontology)

        prompt = generator.generate_entity_extraction_prompt()
        assert isinstance(prompt, str)
        assert "Person" in prompt
