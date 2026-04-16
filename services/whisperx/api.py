"""
WhisperX transcription service.

Standalone HTTP API for audio/video transcription via WhisperX.
Accepts a file path (inside mounted volume) and writes all output formats
(markdown, plain text, timestamped text, JSON, SRT, WebVTT, metadata)
to the same directory as the input file.

All pipeline options configured via environment variables, with
per-request overrides.
"""

import os
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from ingestion.audio.transcriber import WhisperXTranscriber
from ingestion.config import AcceleratorDevice, WhisperModel, WhisperXConfig
from ingestion.exporters import TranscriptionExporter
from ingestion.config import OutputConfig

app = FastAPI(
    title="WhisperX Transcription Service",
    description="Audio/video transcription with speaker diarization, word alignment, and subtitle export",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Environment-based defaults
# ---------------------------------------------------------------------------

SUPPORTED_AUDIO = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma"}
SUPPORTED_VIDEO = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv"}
SUPPORTED_ALL = SUPPORTED_AUDIO | SUPPORTED_VIDEO


def _bool_env(key: str, default: bool) -> bool:
    val = os.getenv(key, "").lower()
    if not val:
        return default
    return val in ("1", "true", "yes")


def _int_env(key: str, default: int) -> int:
    val = os.getenv(key)
    return int(val) if val else default


def _opt_int_env(key: str) -> Optional[int]:
    val = os.getenv(key)
    return int(val) if val else None


def build_whisperx_config(override: Optional["TranscribeOverride"] = None) -> WhisperXConfig:
    """Build WhisperXConfig from env defaults + optional per-request override."""
    cfg = WhisperXConfig()

    # Model
    if model := os.getenv("WHISPERX_MODEL"):
        cfg.model_size = WhisperModel(model)
    if override and override.model:
        cfg.model_size = WhisperModel(override.model)

    # Language
    cfg.language = os.getenv("WHISPERX_LANGUAGE") or None
    if override and override.language is not None:
        cfg.language = override.language or None  # empty string = auto-detect

    # Device
    if device := os.getenv("WHISPERX_DEVICE"):
        cfg.device = AcceleratorDevice(device)
    if override and override.device:
        cfg.device = AcceleratorDevice(override.device)

    # Compute type
    cfg.compute_type = os.getenv("WHISPERX_COMPUTE_TYPE", cfg.compute_type)
    if override and override.compute_type:
        cfg.compute_type = override.compute_type

    # Batch size
    cfg.batch_size = _int_env("WHISPERX_BATCH_SIZE", cfg.batch_size)
    if override and override.batch_size is not None:
        cfg.batch_size = override.batch_size

    # Diarization
    cfg.enable_diarization = _bool_env("WHISPERX_DIARIZATION", cfg.enable_diarization)
    if override and override.enable_diarization is not None:
        cfg.enable_diarization = override.enable_diarization

    cfg.min_speakers = _opt_int_env("WHISPERX_MIN_SPEAKERS")
    if override and override.min_speakers is not None:
        cfg.min_speakers = override.min_speakers

    cfg.max_speakers = _opt_int_env("WHISPERX_MAX_SPEAKERS")
    if override and override.max_speakers is not None:
        cfg.max_speakers = override.max_speakers

    # Word timestamps
    cfg.include_word_timestamps = _bool_env("WHISPERX_WORD_TIMESTAMPS", cfg.include_word_timestamps)
    if override and override.include_word_timestamps is not None:
        cfg.include_word_timestamps = override.include_word_timestamps

    # HuggingFace token (for diarization)
    cfg.hf_token = os.getenv("HF_TOKEN") or None
    if override and override.hf_token:
        cfg.hf_token = override.hf_token

    return cfg


# Lazy-initialized transcriber (heavy model loading, do once)
_transcriber: Optional[WhisperXTranscriber] = None
_transcriber_config_hash: Optional[str] = None


def get_transcriber(config: WhisperXConfig) -> WhisperXTranscriber:
    """Get or create a WhisperXTranscriber, reinitializing if config changed."""
    global _transcriber, _transcriber_config_hash
    config_hash = f"{config.model_size.value}_{config.device.value}_{config.compute_type}"
    if _transcriber is None or _transcriber_config_hash != config_hash:
        _transcriber = WhisperXTranscriber(config)
        _transcriber_config_hash = config_hash
    else:
        # Update runtime settings that don't require model reload
        _transcriber.config = config
    return _transcriber


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class TranscribeOverride(BaseModel):
    """Optional per-request config overrides."""
    model: Optional[str] = Field(default=None, description="Model size: tiny, base, small, medium, large-v3")
    language: Optional[str] = Field(default=None, description="Language code (e.g. 'en', 'nl'). Empty = auto-detect")
    device: Optional[str] = Field(default=None, description="Device: auto, cuda, mps, cpu")
    compute_type: Optional[str] = Field(default=None, description="Compute type: float16, int8, float32")
    batch_size: Optional[int] = Field(default=None, description="Batch size for transcription")
    enable_diarization: Optional[bool] = Field(default=None, description="Enable speaker diarization")
    min_speakers: Optional[int] = Field(default=None, description="Minimum speakers for diarization")
    max_speakers: Optional[int] = Field(default=None, description="Maximum speakers for diarization")
    include_word_timestamps: Optional[bool] = Field(default=None, description="Include word-level timestamps")
    hf_token: Optional[str] = Field(default=None, description="HuggingFace token for diarization")


class TranscribeRequest(BaseModel):
    file_path: str = Field(
        ...,
        description="Path to audio/video file inside the container",
    )
    export_subtitles: bool = Field(default=True, description="Export SRT and WebVTT subtitle files")
    override: Optional[TranscribeOverride] = Field(default=None, description="Override default settings")


class TranscribeResponse(BaseModel):
    input_file: str
    output_dir: str
    success: bool
    source_type: str  # "audio" or "video"
    language: Optional[str] = None
    language_probability: Optional[float] = None
    duration_seconds: float = 0.0
    segment_count: int = 0
    speaker_count: int = 0
    word_count: int = 0
    output_files: dict = {}
    processing_seconds: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok", "service": "whisperx"}


@app.get("/formats")
def formats():
    """List supported input formats."""
    return {
        "audio": sorted(SUPPORTED_AUDIO),
        "video": sorted(SUPPORTED_VIDEO),
    }


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest):
    """Transcribe an audio or video file.

    All output files are written to the same directory as the input file.
    """
    file_path = Path(req.file_path)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {req.file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_ALL:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {suffix}. Supported: {sorted(SUPPORTED_ALL)}",
        )

    source_type = "video" if suffix in SUPPORTED_VIDEO else "audio"

    # Build config
    config = build_whisperx_config(req.override)
    transcriber = get_transcriber(config)

    logger.info(f"Transcribing {source_type}: {file_path.name}")
    logger.info(f"  Model: {config.model_size.value}, Device: {config.get_device()}")
    logger.info(f"  Diarization: {config.enable_diarization}, Word timestamps: {config.include_word_timestamps}")

    t0 = time.time()

    try:
        result = transcriber.transcribe(file_path)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return TranscribeResponse(
            input_file=str(file_path),
            output_dir=str(file_path.parent),
            success=False,
            source_type=source_type,
            processing_seconds=round(time.time() - t0, 2),
            error=str(e),
        )

    # Export all formats to the same directory as the input file
    output_dir = file_path.parent
    stem = file_path.stem
    output_files = {}

    try:
        # Markdown
        md_path = output_dir / f"{stem}_transcription.md"
        md_path.write_text(result.to_markdown(), encoding="utf-8")
        output_files["markdown"] = str(md_path)

        # Plain text
        txt_path = output_dir / f"{stem}_transcription.txt"
        txt_path.write_text(result.to_plain_text(), encoding="utf-8")
        output_files["plain_text"] = str(txt_path)

        # Timestamped text
        ts_path = output_dir / f"{stem}_transcription_timestamps.txt"
        ts_path.write_text(result.to_text_with_timestamps(), encoding="utf-8")
        output_files["timestamped_text"] = str(ts_path)

        # JSON
        json_path = output_dir / f"{stem}_transcription.json"
        json_path.write_text(result.to_json(), encoding="utf-8")
        output_files["json"] = str(json_path)

        # Subtitles
        if req.export_subtitles:
            srt_path = output_dir / f"{stem}.srt"
            srt_path.write_text(result.to_srt(), encoding="utf-8")
            output_files["srt"] = str(srt_path)

            vtt_path = output_dir / f"{stem}.vtt"
            vtt_path.write_text(result.to_vtt(), encoding="utf-8")
            output_files["vtt"] = str(vtt_path)

        # Metadata
        import json
        meta_path = output_dir / f"{stem}_metadata.json"
        metadata = {
            "source_file": str(file_path),
            "source_type": source_type,
            "language": result.language,
            "language_probability": result.language_probability,
            "duration_seconds": result.duration_seconds,
            "segment_count": result.segment_count,
            "speaker_count": result.speaker_count,
            "word_count": result.word_count,
            "model": result.model_name,
            "compute_type": result.compute_type,
            "device": result.device,
            "transcription_time_seconds": result.transcription_time_seconds,
            "speakers": [s.to_dict() for s in result.speakers],
        }
        meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        output_files["metadata"] = str(meta_path)

    except Exception as e:
        logger.error(f"Export failed: {e}")
        return TranscribeResponse(
            input_file=str(file_path),
            output_dir=str(output_dir),
            success=False,
            source_type=source_type,
            processing_seconds=round(time.time() - t0, 2),
            error=f"Transcription succeeded but export failed: {e}",
        )

    elapsed = time.time() - t0

    logger.info(
        f"Transcription complete: {result.segment_count} segments, "
        f"{result.speaker_count} speakers, {result.word_count} words, "
        f"{elapsed:.1f}s"
    )

    return TranscribeResponse(
        input_file=str(file_path),
        output_dir=str(output_dir),
        success=True,
        source_type=source_type,
        language=result.language,
        language_probability=result.language_probability,
        duration_seconds=result.duration_seconds,
        segment_count=result.segment_count,
        speaker_count=result.speaker_count,
        word_count=result.word_count,
        output_files=output_files,
        processing_seconds=round(elapsed, 2),
    )
