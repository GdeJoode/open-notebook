"""Tests for ingestion configuration."""

import os
from pathlib import Path

import pytest

from ingestion.config import (
    AcceleratorDevice,
    DestinationType,
    DoclingConfig,
    ImageRefMode,
    IngestionConfig,
    OcrEngine,
    OutputConfig,
    TableFormat,
    TableMode,
    WhisperModel,
    WhisperXConfig,
)


class TestDoclingConfig:
    """Tests for DoclingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DoclingConfig()

        assert config.device == AcceleratorDevice.AUTO
        assert config.num_threads == 4
        assert config.do_ocr is True
        assert config.ocr_engine == OcrEngine.EASYOCR
        assert config.do_table_structure is True
        assert config.table_mode == TableMode.ACCURATE
        assert config.generate_picture_images is True
        assert config.do_picture_description is True
        assert config.organize_by_page is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DoclingConfig(
            device=AcceleratorDevice.CUDA,
            do_ocr=False,
            table_mode=TableMode.FAST,
        )

        assert config.device == AcceleratorDevice.CUDA
        assert config.do_ocr is False
        assert config.table_mode == TableMode.FAST

    def test_ocr_languages(self):
        """Test OCR language configuration."""
        config = DoclingConfig(ocr_lang=["en", "de", "fr"])
        assert config.ocr_lang == ["en", "de", "fr"]


class TestWhisperXConfig:
    """Tests for WhisperXConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = WhisperXConfig()

        assert config.model_size == WhisperModel.LARGE
        assert config.language is None
        assert config.device == AcceleratorDevice.AUTO
        assert config.enable_diarization is True
        assert config.include_word_timestamps is True

    def test_get_device_cpu(self):
        """Test device selection when CUDA not available."""
        config = WhisperXConfig(device=AcceleratorDevice.CPU)
        assert config.get_device() == "cpu"

    def test_custom_model(self):
        """Test custom model selection."""
        config = WhisperXConfig(model_size=WhisperModel.SMALL)
        assert config.model_size == WhisperModel.SMALL


class TestOutputConfig:
    """Tests for OutputConfig."""

    def test_default_structure(self):
        """Test default directory structure."""
        config = OutputConfig()

        assert config.original_subdir == "original_file"
        assert config.output_subdir == "output"
        assert config.extracted_subdir == "extracted_info"
        assert config.images_subdir == "images"
        assert config.tables_subdir == "tables"

    def test_get_source_dir(self):
        """Test source directory path generation."""
        config = OutputConfig(base_dir=Path("/tmp/test"))
        source_dir = config.get_source_dir("my_document")

        assert source_dir == Path("/tmp/test/my_document")

    def test_get_output_dir(self):
        """Test output directory path generation."""
        config = OutputConfig(base_dir=Path("/tmp/test"))
        output_dir = config.get_output_dir("my_document")

        assert output_dir == Path("/tmp/test/my_document/output/extracted_info")

    def test_get_images_dir(self):
        """Test images directory path generation."""
        config = OutputConfig(base_dir=Path("/tmp/test"))
        images_dir = config.get_images_dir("my_document")

        assert images_dir == Path("/tmp/test/my_document/output/extracted_info/images")

    def test_sanitize_name(self):
        """Test name sanitization for directory names."""
        config = OutputConfig()

        # Test special characters removal
        assert config._sanitize_name("my:file?name") == "my_file_name"

        # Test length limit
        long_name = "a" * 100
        assert len(config._sanitize_name(long_name)) == 80


class TestIngestionConfig:
    """Tests for IngestionConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = IngestionConfig()

        assert config.destination == DestinationType.ASK
        assert config.store_original_for_av is False
        assert config.concurrent_processing is True
        assert isinstance(config.docling, DoclingConfig)
        assert isinstance(config.whisperx, WhisperXConfig)
        assert isinstance(config.output, OutputConfig)

    def test_is_document(self):
        """Test document type detection."""
        config = IngestionConfig()

        assert config.is_document(Path("test.pdf")) is True
        assert config.is_document(Path("test.docx")) is True
        assert config.is_document(Path("test.mp3")) is False

    def test_is_audio(self):
        """Test audio type detection."""
        config = IngestionConfig()

        assert config.is_audio(Path("test.mp3")) is True
        assert config.is_audio(Path("test.wav")) is True
        assert config.is_audio(Path("test.pdf")) is False

    def test_is_video(self):
        """Test video type detection."""
        config = IngestionConfig()

        assert config.is_video(Path("test.mp4")) is True
        assert config.is_video(Path("test.mkv")) is True
        assert config.is_video(Path("test.mp3")) is False

    def test_is_av(self):
        """Test audio/video detection."""
        config = IngestionConfig()

        assert config.is_av(Path("test.mp3")) is True
        assert config.is_av(Path("test.mp4")) is True
        assert config.is_av(Path("test.pdf")) is False

    def test_is_supported(self):
        """Test supported file type detection."""
        config = IngestionConfig()

        assert config.is_supported(Path("test.pdf")) is True
        assert config.is_supported(Path("test.mp3")) is True
        assert config.is_supported(Path("test.xyz")) is False

    def test_for_cli(self):
        """Test CLI configuration factory."""
        config = IngestionConfig.for_cli(output_dir=Path("/custom/output"))

        assert config.destination == DestinationType.ASK
        assert config.output.base_dir == Path("/custom/output")

    def test_for_api(self):
        """Test API configuration factory."""
        config = IngestionConfig.for_api(
            destination=DestinationType.PROJECT,
            project_id="project-123",
        )

        assert config.destination == DestinationType.PROJECT
        assert config.project_id == "project-123"

    def test_from_env(self, monkeypatch):
        """Test environment variable loading."""
        monkeypatch.setenv("INGESTION_OUTPUT_DIR", "/env/output")
        monkeypatch.setenv("INGESTION_DEVICE", "cuda")
        monkeypatch.setenv("INGESTION_OCR_LANG", "en,de")

        config = IngestionConfig.from_env()

        assert config.output.base_dir == Path("/env/output")
        assert config.docling.device == AcceleratorDevice.CUDA
        assert config.docling.ocr_lang == ["en", "de"]
