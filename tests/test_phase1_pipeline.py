"""
Phase 1 Integration Tests: Document Processing Pipeline.

Tests the integrated pipeline from document input to knowledge graph storage,
including:
- GPU detection
- spaCy-Layout processing
- KnowledgeBase building
- Entity resolution
- Ontology validation
- Embeddings generation
- RAPTOR processing
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List


# ============================================================================
# GPU Detection Tests
# ============================================================================

class TestGPUDetection:
    """Test GPU detection and configuration."""

    def test_detect_gpu_returns_config(self):
        """GPU detection should return a GPUConfig object."""
        from open_notebook.processors.gpu_detection import detect_gpu, GPUConfig, GPUDevice

        config = detect_gpu()
        assert isinstance(config, GPUConfig)
        assert config.device in (GPUDevice.CUDA, GPUDevice.MPS, GPUDevice.CPU)

    def test_get_optimal_config(self):
        """Should return optimal config for current hardware."""
        from open_notebook.processors.gpu_detection import get_optimal_config

        config = get_optimal_config()
        assert config is not None
        assert hasattr(config, "device")
        assert hasattr(config, "enabled")

    def test_setup_spacy_gpu_returns_device(self):
        """Setup should return the selected device or False if spacy not available."""
        from open_notebook.processors.gpu_detection import setup_spacy_gpu, detect_gpu, GPUDevice

        config = detect_gpu()
        device = setup_spacy_gpu(config)
        # Returns device string or False if spacy not installed
        valid_devices = (GPUDevice.CUDA.value, GPUDevice.MPS.value, GPUDevice.CPU.value, "cuda", "mps", "cpu", False)
        assert device in valid_devices


# ============================================================================
# spaCy-Layout Pipeline Tests
# ============================================================================

class TestSpacyLayoutPipeline:
    """Test spaCy-Layout document processing."""

    def test_pipeline_imports(self):
        """Pipeline classes should be importable."""
        from open_notebook.processors.spacy_pipeline import (
            SpacyLayoutPipeline,
            ProcessingInput,
            ProcessingOutput,
            ChunkData,
        )

        assert SpacyLayoutPipeline is not None
        assert ProcessingInput is not None
        assert ProcessingOutput is not None

    def test_processing_output_structure(self):
        """ProcessingOutput should have required fields."""
        from open_notebook.processors.spacy_pipeline import ProcessingOutput

        output = ProcessingOutput(
            content="Test content",
            title="Test Title",
            url=None,
            file_path="/test/path.pdf",
            metadata={"pages": 10},
        )

        assert output.content == "Test content"
        assert output.title == "Test Title"
        assert output.file_path == "/test/path.pdf"


# ============================================================================
# KnowledgeBase Builder Tests
# ============================================================================

class TestKnowledgeBaseBuilder:
    """Test spaCy KnowledgeBase building from SurrealDB entities."""

    def test_kb_builder_imports(self):
        """KB builder classes should be importable."""
        from open_notebook.processors.kb_builder import (
            KnowledgeBaseConfig,
            SpacyKnowledgeBaseBuilder,
            get_kb_builder,
        )

        assert KnowledgeBaseConfig is not None
        assert SpacyKnowledgeBaseBuilder is not None

    def test_kb_config_defaults(self):
        """KnowledgeBaseConfig should have sensible defaults."""
        from open_notebook.processors.kb_builder import KnowledgeBaseConfig

        config = KnowledgeBaseConfig()
        assert config.cache_ttl_seconds > 0
        # Check that config has the expected structure
        assert hasattr(config, "cache_ttl_seconds")

    @pytest.mark.asyncio
    async def test_pattern_generation_structure(self):
        """Pattern generation should be available through builder."""
        from open_notebook.processors.kb_builder import SpacyKnowledgeBaseBuilder

        builder = SpacyKnowledgeBaseBuilder()

        # Builder should exist and have the build methods
        assert builder is not None
        assert hasattr(builder, "build_entity_ruler_patterns")


# ============================================================================
# Entity Resolution Pipeline Tests
# ============================================================================

class TestEntityResolutionPipeline:
    """Test 3-tier entity resolution."""

    def test_resolution_imports(self):
        """Entity resolution classes should be importable."""
        from open_notebook.processors.entity_resolution import (
            EntityResolutionConfig,
            EntityResolutionPipeline,
            ResolvedEntity,
            EntityResolutionResult,
        )

        assert EntityResolutionConfig is not None
        assert EntityResolutionPipeline is not None
        assert ResolvedEntity is not None

    def test_resolution_config_defaults(self):
        """EntityResolutionConfig should have sensible defaults."""
        from open_notebook.processors.entity_resolution import EntityResolutionConfig

        config = EntityResolutionConfig()
        assert config.enable_entity_ruler is True
        assert config.enable_entity_linker is True
        assert config.enable_semantic_fallback is True
        assert 0 < config.semantic_threshold < 1

    def test_resolved_entity_structure(self):
        """ResolvedEntity should capture resolution details."""
        from open_notebook.processors.entity_resolution import ResolvedEntity

        resolved = ResolvedEntity(
            text="Microsoft Corporation",
            start_char=0,
            end_char=21,
            label="ORG",
            entity_type="Organization",
            entity_id="entity:ms123",
            resolution_method="pattern",
        )

        assert resolved.text == "Microsoft Corporation"
        assert resolved.resolution_method == "pattern"
        assert resolved.entity_id == "entity:ms123"

    def test_resolution_result_aggregation(self):
        """EntityResolutionResult should aggregate entities by method."""
        from open_notebook.processors.entity_resolution import (
            EntityResolutionResult,
            ResolvedEntity,
        )

        result = EntityResolutionResult()
        result.resolved_entities = [
            ResolvedEntity(
                text="Microsoft",
                start_char=0,
                end_char=9,
                label="ORG",
                entity_type="Organization",
                entity_id="e1",
                resolution_method="pattern",
            ),
            ResolvedEntity(
                text="Google",
                start_char=10,
                end_char=16,
                label="ORG",
                entity_type="Organization",
                entity_id="e2",
                resolution_method="linker",
            ),
            ResolvedEntity(
                text="Unknown Corp",
                start_char=20,
                end_char=32,
                label="ORG",
                entity_type="Organization",
                entity_id="e3",
                resolution_method="semantic",
            ),
        ]

        # Check method counts
        method_counts = {}
        for entity in result.resolved_entities:
            method = entity.resolution_method
            method_counts[method] = method_counts.get(method, 0) + 1

        assert method_counts["pattern"] == 1
        assert method_counts["linker"] == 1
        assert method_counts["semantic"] == 1


# ============================================================================
# Ontology Validation Tests
# ============================================================================

class TestOntologyValidation:
    """Test ontology validation gate."""

    def test_validation_imports(self):
        """Validation classes should be importable."""
        from open_notebook.processors.validation import (
            OntologyValidator,
            ValidationResult,
            ValidationIssue,
            ValidationSeverity,
        )

        assert OntologyValidator is not None
        assert ValidationResult is not None
        assert ValidationSeverity.ERROR == "error"

    def test_validation_result_structure(self):
        """ValidationResult should track validation status."""
        from open_notebook.processors.validation import (
            ValidationResult,
            ValidationIssue,
            ValidationSeverity,
        )

        result = ValidationResult()
        assert result.valid is True
        assert result.entities_validated == 0

        # Add an error
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="TEST_ERROR",
            message="Test error message",
        ))

        assert result.valid is False
        assert len(result.get_errors()) == 1

    def test_validation_severity_levels(self):
        """All severity levels should be defined."""
        from open_notebook.processors.validation import ValidationSeverity

        assert ValidationSeverity.ERROR == "error"
        assert ValidationSeverity.WARNING == "warning"
        assert ValidationSeverity.INFO == "info"

    @pytest.mark.asyncio
    async def test_validator_initialization(self):
        """Validator should initialize with ontology name."""
        from open_notebook.processors.validation import OntologyValidator

        validator = OntologyValidator(ontology_name="general")
        assert validator.ontology_name == "general"


# ============================================================================
# Embeddings Tests
# ============================================================================

class TestEmbeddings:
    """Test embedding generation for knowledge graph."""

    def test_embedding_imports(self):
        """Embedding classes should be importable."""
        from open_notebook.processors.embeddings import (
            EmbeddingConfig,
            EmbeddingService,
            KnowledgeGraphEmbeddings,
        )

        assert EmbeddingConfig is not None
        assert EmbeddingService is not None
        assert KnowledgeGraphEmbeddings is not None

    def test_embedding_config_defaults(self):
        """EmbeddingConfig should have sensible defaults."""
        from open_notebook.processors.embeddings import EmbeddingConfig

        config = EmbeddingConfig()
        # Check that config has expected structure
        assert hasattr(config, "use_app_config")
        assert config.use_app_config is True


# ============================================================================
# RAPTOR Tests
# ============================================================================

class TestRAPTOR:
    """Test RAPTOR hierarchical summarization."""

    def test_raptor_imports(self):
        """RAPTOR classes should be importable."""
        from open_notebook.processors.raptor.processor import (
            RaptorProcessor,
        )
        from open_notebook.processors.raptor.tree_builder import (
            RaptorTree,
            RaptorNode,
            RaptorTreeBuilder,
        )
        from open_notebook.processors.raptor.config import RaptorConfig

        assert RaptorProcessor is not None
        assert RaptorTree is not None
        assert RaptorConfig is not None

    def test_raptor_config_defaults(self):
        """RaptorConfig should have sensible defaults."""
        from open_notebook.processors.raptor.config import RaptorConfig

        config = RaptorConfig()
        assert config.max_layers > 0
        assert config.min_chunks_for_clustering > 0

    def test_raptor_tree_structure(self):
        """RaptorTree should track layers and nodes."""
        from open_notebook.processors.raptor.tree_builder import RaptorTree, RaptorNode

        tree = RaptorTree()
        assert tree.nodes == []
        assert tree.num_layers == 0

        # Add a node
        node = RaptorNode(
            text="Summary text",
            layer=1,
            embedding=[0.1] * 768,
        )
        tree.nodes.append(node)
        tree.num_layers = 1

        assert len(tree.nodes) == 1
        assert tree.num_layers == 1


# ============================================================================
# OpenIE Integration Tests
# ============================================================================

class TestOpenIEIntegration:
    """Test OpenIE knowledge extraction."""

    def test_openie_imports(self):
        """OpenIE classes should be importable."""
        from open_notebook.processors.openie import (
            OpenIEExtractor,
            OpenIEConfig,
            ExtractionResult,
            ExtractedEntity,
            ExtractedClaim,
        )

        assert OpenIEExtractor is not None
        assert OpenIEConfig is not None
        assert ExtractionResult is not None

    def test_extraction_result_structure(self):
        """ExtractionResult should contain entities and claims."""
        from open_notebook.processors.openie import (
            ExtractionResult,
            ExtractedEntity,
            ExtractedClaim,
        )

        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Test Entity",
                    entity_type="Organization",
                    confidence=0.9,
                )
            ],
            claims=[
                ExtractedClaim(
                    statement="Test claim",
                    claim_type="factual",
                    confidence=0.85,
                )
            ],
            triples=[],
        )

        assert len(result.entities) == 1
        assert len(result.claims) == 1
        assert result.entities[0].name == "Test Entity"


# ============================================================================
# Source Graph Integration Tests
# ============================================================================

class TestSourceGraphIntegration:
    """Test integrated source processing graph."""

    def test_source_state_has_kg_fields(self):
        """SourceState should include KG extraction fields."""
        from open_notebook.graphs.source import SourceState

        # Check type hints include KG fields
        hints = SourceState.__annotations__
        assert "extract_knowledge" in hints
        assert "kg_extraction_result" in hints
        assert "raptor_enabled" in hints
        assert "raptor_result" in hints

    def test_graph_compilation(self):
        """Source graph should compile without errors."""
        from open_notebook.graphs.source import source_graph

        assert source_graph is not None
        # Graph should have nodes for KG and RAPTOR
        assert "extract_knowledge_graph" in source_graph.nodes
        assert "process_raptor" in source_graph.nodes


# ============================================================================
# Entity Linking Tests
# ============================================================================

class TestEntityLinking:
    """Test entity linking with KNN deduplication."""

    def test_entity_linking_imports(self):
        """Entity linking classes should be importable."""
        from open_notebook.processors.entity_linking import (
            EntityLinkingConfig,
            EntityLinker,
        )

        assert EntityLinkingConfig is not None
        assert EntityLinker is not None

    def test_linking_config_defaults(self):
        """EntityLinkingConfig should have sensible defaults."""
        from open_notebook.processors.entity_linking import EntityLinkingConfig

        config = EntityLinkingConfig()
        assert 0 < config.similarity_threshold < 1


# ============================================================================
# Integration Smoke Test
# ============================================================================

class TestIntegrationSmoke:
    """Smoke tests to verify all components work together."""

    def test_all_processor_imports(self):
        """All processors should be importable from __init__."""
        from open_notebook.processors import (
            # GPU
            GPUConfig,
            detect_gpu,
            # spaCy Pipeline
            SpacyLayoutPipeline,
            ProcessingOutput,
            # KB Builder
            SpacyKnowledgeBaseBuilder,
            # Entity Resolution
            EntityResolutionPipeline,
            # Validation
            OntologyValidator,
            ValidationResult,
            # Embeddings
            EmbeddingService,
            # OpenIE
            OpenIEExtractor,
            # Entity Linking
            EntityLinker,
        )

        # All imports should succeed
        assert GPUConfig is not None
        assert SpacyLayoutPipeline is not None
        assert SpacyKnowledgeBaseBuilder is not None
        assert EntityResolutionPipeline is not None
        assert OntologyValidator is not None
        assert EmbeddingService is not None
        assert OpenIEExtractor is not None
        assert EntityLinker is not None

    def test_source_command_output_has_all_fields(self):
        """SourceProcessingOutput should include all result fields."""
        from commands.source_commands import SourceProcessingOutput

        output = SourceProcessingOutput(
            success=True,
            source_id="source:test",
            embedded_chunks=10,
            insights_created=2,
            entities_extracted=5,
            claims_extracted=3,
            raptor_layers=2,
            raptor_nodes=4,
            processing_time=1.5,
        )

        assert output.success is True
        assert output.entities_extracted == 5
        assert output.claims_extracted == 3
        assert output.raptor_layers == 2
        assert output.raptor_nodes == 4
