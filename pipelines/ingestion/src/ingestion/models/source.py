"""
Source metadata and ingestion result models.

Provides:
- SourceMetadata: Information about original files
- IngestionResult: Complete ingestion output combining documents and transcriptions
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from .document import ExtractedDocument
from .transcription import TranscriptionResult


class SourceType(str, Enum):
    """Type of source being ingested."""
    DOCUMENT = "document"  # PDF, DOCX, etc.
    AUDIO = "audio"  # MP3, WAV, etc.
    VIDEO = "video"  # MP4, MKV, etc.
    URL = "url"  # Web page
    TEXT = "text"  # Plain text


@dataclass
class SourceMetadata:
    """
    Metadata about the original source file.

    For audio/video, only metadata is stored (not the original file).
    For documents, the original file can optionally be stored.
    """
    # Basic info
    filename: str
    source_type: SourceType
    file_extension: str

    # File metadata
    file_size_bytes: int = 0
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None

    # Document-specific
    page_count: int = 0

    # Audio/video-specific
    duration_seconds: float = 0.0
    bitrate: Optional[int] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    codec: Optional[str] = None

    # Video-specific
    resolution: Optional[tuple[int, int]] = None
    fps: Optional[float] = None

    # URL-specific
    url: Optional[str] = None
    title: Optional[str] = None

    # Processing info
    original_path: Optional[Path] = None
    stored_original: bool = False  # Whether original file was stored

    @classmethod
    def from_path(cls, path: Path, source_type: SourceType) -> "SourceMetadata":
        """
        Create metadata from a file path.

        Extracts basic file info. Media-specific info should be
        populated separately using appropriate libraries.
        """
        stat = path.stat() if path.exists() else None

        return cls(
            filename=path.name,
            source_type=source_type,
            file_extension=path.suffix.lower(),
            file_size_bytes=stat.st_size if stat else 0,
            created_at=datetime.fromtimestamp(stat.st_ctime) if stat else None,
            modified_at=datetime.fromtimestamp(stat.st_mtime) if stat else None,
            original_path=path,
        )

    @classmethod
    def from_media_file(cls, path: Path, source_type: SourceType) -> "SourceMetadata":
        """
        Create metadata from audio/video file using ffprobe or similar.

        Falls back to basic metadata if media libraries unavailable.
        """
        metadata = cls.from_path(path, source_type)

        try:
            import subprocess
            import json

            # Use ffprobe to get media info
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    str(path),
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                info = json.loads(result.stdout)
                fmt = info.get("format", {})
                streams = info.get("streams", [])

                # Duration
                metadata.duration_seconds = float(fmt.get("duration", 0))

                # Bitrate
                if "bit_rate" in fmt:
                    metadata.bitrate = int(fmt["bit_rate"])

                # Find audio and video streams
                for stream in streams:
                    if stream.get("codec_type") == "audio":
                        metadata.sample_rate = int(stream.get("sample_rate", 0))
                        metadata.channels = stream.get("channels")
                        if not metadata.codec:
                            metadata.codec = stream.get("codec_name")

                    elif stream.get("codec_type") == "video":
                        metadata.resolution = (
                            stream.get("width", 0),
                            stream.get("height", 0),
                        )
                        fps_str = stream.get("r_frame_rate", "0/1")
                        if "/" in fps_str:
                            num, den = fps_str.split("/")
                            metadata.fps = float(num) / float(den) if float(den) > 0 else 0
                        metadata.codec = stream.get("codec_name")

        except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError):
            # ffprobe not available or failed
            pass

        return metadata

    def format_duration(self) -> str:
        """Format duration as human readable string."""
        if self.duration_seconds <= 0:
            return "Unknown"

        seconds = self.duration_seconds
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def format_size(self) -> str:
        """Format file size as human readable string."""
        size = self.file_size_bytes
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "filename": self.filename,
            "source_type": self.source_type.value,
            "file_extension": self.file_extension,
            "file_size_bytes": self.file_size_bytes,
            "file_size_formatted": self.format_size(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "page_count": self.page_count,
            "duration_seconds": self.duration_seconds,
            "duration_formatted": self.format_duration() if self.duration_seconds > 0 else None,
            "bitrate": self.bitrate,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "codec": self.codec,
            "resolution": list(self.resolution) if self.resolution else None,
            "fps": self.fps,
            "url": self.url,
            "title": self.title,
            "original_path": str(self.original_path) if self.original_path else None,
            "stored_original": self.stored_original,
        }


@dataclass
class IngestionResult:
    """
    Complete result of the ingestion pipeline.

    Combines:
    - Source metadata
    - Document extraction (for documents)
    - Transcription (for audio/video)
    - Output file paths
    """
    source_metadata: SourceMetadata
    success: bool = True
    error_message: Optional[str] = None

    # Extracted content (mutually exclusive based on source type)
    document: Optional[ExtractedDocument] = None
    transcription: Optional[TranscriptionResult] = None

    # Output paths
    output_directory: Optional[Path] = None
    markdown_path: Optional[Path] = None
    json_path: Optional[Path] = None
    original_path: Optional[Path] = None  # Where original was stored (if applicable)

    # Table/image paths (for documents)
    table_paths: list[Path] = field(default_factory=list)
    image_paths: list[Path] = field(default_factory=list)

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    processing_time_seconds: float = 0.0

    @property
    def source_type(self) -> SourceType:
        """Get the source type."""
        return self.source_metadata.source_type

    @property
    def is_document(self) -> bool:
        """Check if this is a document ingestion."""
        return self.source_type == SourceType.DOCUMENT

    @property
    def is_audio_video(self) -> bool:
        """Check if this is an audio/video ingestion."""
        return self.source_type in (SourceType.AUDIO, SourceType.VIDEO)

    def complete(self, success: bool = True, error: Optional[str] = None) -> "IngestionResult":
        """Mark ingestion as complete."""
        self.completed_at = datetime.now()
        self.processing_time_seconds = (self.completed_at - self.started_at).total_seconds()
        self.success = success
        self.error_message = error
        return self

    def get_content(self) -> Union[ExtractedDocument, TranscriptionResult, None]:
        """Get the extracted content (document or transcription)."""
        return self.document or self.transcription

    def get_markdown(self) -> str:
        """Get markdown representation of the content."""
        if self.document:
            return self.document.to_markdown()
        elif self.transcription:
            return self.transcription.to_markdown()
        return ""

    def get_plain_text(self) -> str:
        """Get plain text representation of the content."""
        if self.document:
            return self.document.full_text
        elif self.transcription:
            return self.transcription.to_plain_text()
        return ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "success": self.success,
            "error_message": self.error_message,
            "source_metadata": self.source_metadata.to_dict(),
            "source_type": self.source_type.value,
            "output_directory": str(self.output_directory) if self.output_directory else None,
            "markdown_path": str(self.markdown_path) if self.markdown_path else None,
            "json_path": str(self.json_path) if self.json_path else None,
            "original_path": str(self.original_path) if self.original_path else None,
            "table_paths": [str(p) for p in self.table_paths],
            "image_paths": [str(p) for p in self.image_paths],
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_time_seconds": self.processing_time_seconds,
        }

        if self.document:
            result["document"] = self.document.to_dict()
        if self.transcription:
            result["transcription"] = self.transcription.to_dict()

        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def create_error(cls, source_path: Path, error: str) -> "IngestionResult":
        """Create a failed ingestion result."""
        # Determine source type from extension
        ext = source_path.suffix.lower()
        if ext in [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma"]:
            source_type = SourceType.AUDIO
        elif ext in [".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv"]:
            source_type = SourceType.VIDEO
        else:
            source_type = SourceType.DOCUMENT

        metadata = SourceMetadata.from_path(source_path, source_type)

        return cls(
            source_metadata=metadata,
            success=False,
            error_message=error,
        ).complete(success=False, error=error)
