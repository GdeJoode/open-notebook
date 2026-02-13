"""Tests for shared types and enumerations."""

import pytest

from shared.types import (
    ChunkingMethod,
    ContentProcessingEngine,
    DoclingPipeline,
    ElementType,
    EmbeddingOption,
    FileOperation,
    GpuDevice,
    InsightType,
    NamingScheme,
    NoteType,
    OcrEngine,
    PipelineConfig,
    PipelineTask,
    TableMode,
    TaskPriority,
    TaskResult,
    TaskStatus,
    UrlProcessingEngine,
    VlmFramework,
    VlmModel,
)


class TestEnums:
    """Tests for enumeration types."""

    def test_note_type_values(self):
        """Test NoteType enum values."""
        assert NoteType.HUMAN.value == "human"
        assert NoteType.AI.value == "ai"

    def test_content_processing_engine_values(self):
        """Test ContentProcessingEngine enum values."""
        assert ContentProcessingEngine.AUTO.value == "auto"
        assert ContentProcessingEngine.DOCLING.value == "docling"
        assert ContentProcessingEngine.SIMPLE.value == "simple"

    def test_url_processing_engine_values(self):
        """Test UrlProcessingEngine enum values."""
        assert UrlProcessingEngine.FIRECRAWL.value == "firecrawl"
        assert UrlProcessingEngine.JINA.value == "jina"

    def test_gpu_device_values(self):
        """Test GpuDevice enum values."""
        assert GpuDevice.AUTO.value == "auto"
        assert GpuDevice.CUDA.value == "cuda"
        assert GpuDevice.CPU.value == "cpu"

    def test_docling_pipeline_values(self):
        """Test DoclingPipeline enum values."""
        assert DoclingPipeline.STANDARD.value == "standard"
        assert DoclingPipeline.VLM.value == "vlm"

    def test_ocr_engine_values(self):
        """Test OcrEngine enum values."""
        assert OcrEngine.EASYOCR.value == "easyocr"
        assert OcrEngine.TESSERACT.value == "tesseract"

    def test_element_type_values(self):
        """Test ElementType enum values."""
        assert ElementType.PARAGRAPH.value == "paragraph"
        assert ElementType.TABLE.value == "table"
        assert ElementType.TITLE.value == "title"

    def test_insight_type_values(self):
        """Test InsightType enum values."""
        assert InsightType.SUMMARY.value == "summary"
        assert InsightType.KEY_POINTS.value == "key_points"
        assert InsightType.ENTITIES.value == "entities"


class TestPipelineTask:
    """Tests for PipelineTask enum."""

    def test_ingestion_tasks(self):
        """Test ingestion pipeline tasks exist."""
        assert PipelineTask.INGEST_PARSE.value == "ingest_parse"
        assert PipelineTask.INGEST_CHUNK.value == "ingest_chunk"
        assert PipelineTask.INGEST_CREATE_SOURCE.value == "ingest_create_source"

    def test_extraction_tasks(self):
        """Test extraction pipeline tasks exist."""
        assert PipelineTask.EXTRACT_ENTITIES.value == "extract_entities"
        assert PipelineTask.EXTRACT_RELATIONS.value == "extract_relations"

    def test_summarization_tasks(self):
        """Test summarization pipeline tasks exist."""
        assert PipelineTask.SUMMARIZE_DOCUMENT.value == "summarize_document"
        assert PipelineTask.SUMMARIZE_RAPTOR.value == "summarize_raptor"

    def test_embedding_tasks(self):
        """Test embedding pipeline tasks exist."""
        assert PipelineTask.EMBED_CHUNKS.value == "embed_chunks"
        assert PipelineTask.EMBED_ENTITIES.value == "embed_entities"

    def test_retrieval_tasks(self):
        """Test retrieval pipeline tasks exist."""
        assert PipelineTask.RETRIEVE_VECTOR_SEARCH.value == "retrieve_vector_search"
        assert PipelineTask.RETRIEVE_HYBRID_SEARCH.value == "retrieve_hybrid_search"


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_all_statuses(self):
        """Test all task statuses exist."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"
        assert TaskStatus.SKIPPED.value == "skipped"


class TestTaskResult:
    """Tests for TaskResult model."""

    def test_task_result_creation(self):
        """Test TaskResult creation."""
        result = TaskResult(
            task=PipelineTask.INGEST_PARSE,
            status=TaskStatus.COMPLETED,
            source_id="source:123",
        )
        assert result.task == PipelineTask.INGEST_PARSE
        assert result.status == TaskStatus.COMPLETED
        assert result.source_id == "source:123"

    def test_task_result_with_error(self):
        """Test TaskResult with error."""
        result = TaskResult(
            task=PipelineTask.INGEST_PARSE,
            status=TaskStatus.FAILED,
            error="Something went wrong",
        )
        assert result.error == "Something went wrong"


class TestPipelineConfig:
    """Tests for PipelineConfig model."""

    def test_pipeline_config_creation(self):
        """Test PipelineConfig creation."""
        config = PipelineConfig(
            tasks=[PipelineTask.INGEST_PARSE, PipelineTask.INGEST_CHUNK],
            source_id="source:123",
        )
        assert len(config.tasks) == 2
        assert config.source_id == "source:123"
        assert config.priority == TaskPriority.NORMAL

    def test_pipeline_config_options(self):
        """Test PipelineConfig with options."""
        config = PipelineConfig(
            tasks=[PipelineTask.INGEST_PARSE],
            priority=TaskPriority.HIGH,
            options={"custom_option": "value"},
            retry_count=3,
        )
        assert config.priority == TaskPriority.HIGH
        assert config.options["custom_option"] == "value"
        assert config.retry_count == 3
