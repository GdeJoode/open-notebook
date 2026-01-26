"""
Audio/video transcription models.

Models for WhisperX transcription output including:
- Word-level timestamps
- Speaker diarization
- Segment organization
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional


@dataclass
class TranscriptionWord:
    """
    Single word with timing information.

    WhisperX provides word-level timestamps for precise synchronization.
    """
    word: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        """Duration of the word in seconds."""
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }


@dataclass
class SpeakerInfo:
    """
    Speaker identification from diarization.

    Tracks speaker statistics and timing.
    """
    speaker_id: str  # e.g., "SPEAKER_00", "SPEAKER_01"
    label: Optional[str] = None  # User-assigned label
    total_time: float = 0.0  # Total speaking time in seconds
    segment_count: int = 0  # Number of segments

    @property
    def display_name(self) -> str:
        """Get display name for speaker."""
        return self.label or self.speaker_id

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "speaker_id": self.speaker_id,
            "label": self.label,
            "display_name": self.display_name,
            "total_time": self.total_time,
            "segment_count": self.segment_count,
        }


@dataclass
class TranscriptionSegment:
    """
    Transcription segment with speaker and timing.

    A segment is a continuous stretch of speech, typically
    separated by pauses or speaker changes.
    """
    text: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    speaker: Optional[str] = None  # Speaker ID if diarization enabled
    words: list[TranscriptionWord] = field(default_factory=list)
    confidence: float = 1.0
    language: Optional[str] = None

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end - self.start

    @property
    def word_count(self) -> int:
        """Number of words in segment."""
        return len(self.words) if self.words else len(self.text.split())

    def format_timestamp(self, time_seconds: float) -> str:
        """Format time in HH:MM:SS.mmm format."""
        td = timedelta(seconds=time_seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        ms = int(td.microseconds / 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"

    @property
    def start_formatted(self) -> str:
        """Formatted start time."""
        return self.format_timestamp(self.start)

    @property
    def end_formatted(self) -> str:
        """Formatted end time."""
        return self.format_timestamp(self.end)

    def to_srt_format(self, index: int) -> str:
        """Format as SRT subtitle entry."""
        start_srt = self.format_timestamp(self.start).replace(".", ",")
        end_srt = self.format_timestamp(self.end).replace(".", ",")
        text = f"[{self.speaker}] {self.text}" if self.speaker else self.text
        return f"{index}\n{start_srt} --> {end_srt}\n{text}\n"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "start_formatted": self.start_formatted,
            "end_formatted": self.end_formatted,
            "duration": self.duration,
            "speaker": self.speaker,
            "words": [w.to_dict() for w in self.words],
            "word_count": self.word_count,
            "confidence": self.confidence,
            "language": self.language,
        }


@dataclass
class TranscriptionResult:
    """
    Complete transcription result with metadata.

    Contains:
    - Full transcript text
    - Segmented transcript with timing
    - Speaker information (if diarization enabled)
    - Source file metadata
    """
    source_path: Path
    segments: list[TranscriptionSegment] = field(default_factory=list)
    speakers: list[SpeakerInfo] = field(default_factory=list)

    # Full text versions
    full_text: str = ""
    full_text_with_speakers: str = ""
    full_text_with_timestamps: str = ""

    # Metadata
    language: str = ""
    language_probability: float = 0.0
    duration_seconds: float = 0.0
    transcribed_at: datetime = field(default_factory=datetime.now)
    transcription_time_seconds: float = 0.0

    # Source file metadata (for audio/video - not stored)
    source_filename: str = ""
    source_duration: float = 0.0
    source_format: str = ""
    source_size_bytes: int = 0
    source_created_at: Optional[datetime] = None

    # Model info
    model_name: str = ""
    compute_type: str = ""
    device: str = ""

    @property
    def segment_count(self) -> int:
        """Number of segments."""
        return len(self.segments)

    @property
    def speaker_count(self) -> int:
        """Number of unique speakers."""
        return len(self.speakers)

    @property
    def word_count(self) -> int:
        """Total word count."""
        return sum(s.word_count for s in self.segments)

    def get_speaker_segments(self, speaker_id: str) -> list[TranscriptionSegment]:
        """Get all segments for a specific speaker."""
        return [s for s in self.segments if s.speaker == speaker_id]

    def to_plain_text(self) -> str:
        """Export as plain text without any formatting."""
        if self.full_text:
            return self.full_text
        return " ".join(s.text for s in self.segments)

    def to_text_with_speakers(self) -> str:
        """Export with speaker labels."""
        if self.full_text_with_speakers:
            return self.full_text_with_speakers

        lines = []
        current_speaker = None
        current_text = []

        for segment in self.segments:
            if segment.speaker != current_speaker:
                if current_text:
                    speaker_label = current_speaker or "Unknown"
                    lines.append(f"[{speaker_label}]: {' '.join(current_text)}")
                current_speaker = segment.speaker
                current_text = []
            current_text.append(segment.text)

        if current_text:
            speaker_label = current_speaker or "Unknown"
            lines.append(f"[{speaker_label}]: {' '.join(current_text)}")

        return "\n\n".join(lines)

    def to_text_with_timestamps(self) -> str:
        """Export with timestamps."""
        if self.full_text_with_timestamps:
            return self.full_text_with_timestamps

        lines = []
        for segment in self.segments:
            prefix = f"[{segment.start_formatted}]"
            if segment.speaker:
                prefix = f"[{segment.start_formatted}] [{segment.speaker}]"
            lines.append(f"{prefix} {segment.text}")

        return "\n".join(lines)

    def to_srt(self) -> str:
        """Export as SRT subtitle format."""
        entries = []
        for i, segment in enumerate(self.segments, 1):
            entries.append(segment.to_srt_format(i))
        return "\n".join(entries)

    def to_vtt(self) -> str:
        """Export as WebVTT subtitle format."""
        lines = ["WEBVTT", ""]
        for segment in self.segments:
            text = f"<v {segment.speaker}>{segment.text}" if segment.speaker else segment.text
            lines.extend([
                f"{segment.start_formatted} --> {segment.end_formatted}",
                text,
                "",
            ])
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Export as Markdown with sections."""
        parts = [f"# Transcription: {self.source_filename or self.source_path.name}"]

        # Metadata section
        parts.append("\n## Metadata\n")
        parts.append(f"- **Duration**: {self._format_duration(self.duration_seconds)}")
        parts.append(f"- **Language**: {self.language}")
        parts.append(f"- **Speakers**: {self.speaker_count}")
        parts.append(f"- **Segments**: {self.segment_count}")
        parts.append(f"- **Words**: {self.word_count}")

        # Speakers section
        if self.speakers:
            parts.append("\n## Speakers\n")
            for speaker in self.speakers:
                time_str = self._format_duration(speaker.total_time)
                parts.append(f"- **{speaker.display_name}**: {time_str} ({speaker.segment_count} segments)")

        # Transcript section
        parts.append("\n## Transcript\n")
        parts.append(self.to_text_with_speakers())

        return "\n".join(parts)

    def _format_duration(self, seconds: float) -> str:
        """Format duration as human readable string."""
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_path": str(self.source_path),
            "source_filename": self.source_filename,
            "source_duration": self.source_duration,
            "source_format": self.source_format,
            "source_size_bytes": self.source_size_bytes,
            "source_created_at": self.source_created_at.isoformat() if self.source_created_at else None,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration_seconds": self.duration_seconds,
            "segment_count": self.segment_count,
            "speaker_count": self.speaker_count,
            "word_count": self.word_count,
            "speakers": [s.to_dict() for s in self.speakers],
            "segments": [s.to_dict() for s in self.segments],
            "model_name": self.model_name,
            "compute_type": self.compute_type,
            "device": self.device,
            "transcribed_at": self.transcribed_at.isoformat(),
            "transcription_time_seconds": self.transcription_time_seconds,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
