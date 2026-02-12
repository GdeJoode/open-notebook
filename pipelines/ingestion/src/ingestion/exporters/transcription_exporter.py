"""
Transcription content exporter.

Exports transcription results to a fixed directory structure:
    ./[source_name]/
        output/
            extracted_info/
                transcription.md    # Full markdown with speakers
                transcription.txt   # Plain text
                transcription.json  # Structured JSON
                transcription.srt   # SRT subtitles
                transcription.vtt   # WebVTT subtitles
                metadata.json       # Source and processing metadata
"""

import json
from pathlib import Path
from typing import Optional

from loguru import logger

from ingestion.config import OutputConfig
from ingestion.models import TranscriptionResult


class TranscriptionExporter:
    """
    Export transcription results to the standard directory structure.

    Features:
    - Multiple output formats (MD, TXT, JSON, SRT, VTT)
    - Speaker-aware formatting
    - Source metadata preservation
    - No original file storage (only metadata)
    """

    def __init__(self, config: Optional[OutputConfig] = None):
        """
        Initialize the transcription exporter.

        Args:
            config: OutputConfig instance. If None, uses defaults.
        """
        self.config = config or OutputConfig()

    def export(
        self,
        transcription: TranscriptionResult,
        source_name: Optional[str] = None,
        export_subtitles: bool = True,
        source_folder_path: Optional[Path] = None,
        source_id: Optional[str] = None,
    ) -> dict[str, Path]:
        """
        Export a transcription to the standard directory structure.

        Note: Original audio/video files are NOT stored, only metadata.

        Args:
            transcription: Transcription result to export
            source_name: Name for the output directory (defaults to source filename)
            export_subtitles: Whether to export SRT and VTT files
            source_folder_path: If provided, use this pre-existing source folder
                instead of creating a new directory.
            source_id: If provided (with source_folder_path), prefix all output
                files with this ID for global uniqueness.

        Returns:
            Dictionary of exported file paths
        """
        # Determine source name
        if source_name is None:
            source_name = transcription.source_path.stem

        logger.info(f"Exporting transcription: {source_name}")

        # File prefix for source-folder mode
        prefix = f"{source_id}_" if source_id else ""

        # Create directories
        if source_folder_path is not None:
            dirs = self.config.create_directories_in_source_folder(
                source_folder_path, source_type="audio"
            )
            # In source-folder mode, transcription outputs go to transcription/ dir
            output_dir = dirs.get("transcription", dirs["output"])
        else:
            dirs = self.config.create_directories(source_name, include_original=False)
            output_dir = dirs["output"]

        exported_paths: dict[str, Path] = {
            "root": dirs["root"],
            "output": output_dir,
        }

        # Export markdown
        markdown_path = output_dir / f"{prefix}transcription.md"
        markdown_path.write_text(transcription.to_markdown(), encoding="utf-8")
        exported_paths["markdown"] = markdown_path
        logger.debug(f"Exported markdown to: {markdown_path}")

        # Export plain text
        text_path = output_dir / f"{prefix}transcription.txt"
        text_path.write_text(transcription.to_plain_text(), encoding="utf-8")
        exported_paths["text"] = text_path

        # Export with timestamps
        timestamps_path = output_dir / f"{prefix}transcription_timestamps.txt"
        timestamps_path.write_text(transcription.to_text_with_timestamps(), encoding="utf-8")
        exported_paths["timestamps"] = timestamps_path

        # Export JSON
        json_path = output_dir / f"{prefix}transcription.json"
        json_path.write_text(transcription.to_json(), encoding="utf-8")
        exported_paths["json"] = json_path
        logger.debug(f"Exported JSON to: {json_path}")

        # Export subtitles
        if export_subtitles:
            # SRT format
            srt_path = output_dir / f"{prefix}transcription.srt"
            srt_path.write_text(transcription.to_srt(), encoding="utf-8")
            exported_paths["srt"] = srt_path

            # VTT format
            vtt_path = output_dir / f"{prefix}transcription.vtt"
            vtt_path.write_text(transcription.to_vtt(), encoding="utf-8")
            exported_paths["vtt"] = vtt_path

        # Export metadata (source file info without storing the file)
        metadata_path = output_dir / f"{prefix}metadata.json"
        metadata = self._generate_metadata(transcription)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        exported_paths["metadata"] = metadata_path

        logger.info(
            f"Transcription exported: {transcription.segment_count} segments, "
            f"{transcription.speaker_count} speakers"
        )

        return exported_paths

    def _generate_metadata(self, transcription: TranscriptionResult) -> dict:
        """Generate metadata for the transcription."""
        return {
            "source_file": {
                "filename": transcription.source_filename,
                "path": str(transcription.source_path),
                "format": transcription.source_format,
                "duration_seconds": transcription.source_duration,
                "size_bytes": transcription.source_size_bytes,
                "created_at": (
                    transcription.source_created_at.isoformat()
                    if transcription.source_created_at
                    else None
                ),
                "note": "Original file not stored - only metadata preserved",
            },
            "transcription": {
                "language": transcription.language,
                "language_probability": transcription.language_probability,
                "duration_seconds": transcription.duration_seconds,
                "segment_count": transcription.segment_count,
                "speaker_count": transcription.speaker_count,
                "word_count": transcription.word_count,
            },
            "speakers": [s.to_dict() for s in transcription.speakers],
            "processing": {
                "model_name": transcription.model_name,
                "compute_type": transcription.compute_type,
                "device": transcription.device,
                "transcribed_at": transcription.transcribed_at.isoformat(),
                "transcription_time_seconds": transcription.transcription_time_seconds,
            },
        }


def export_transcription(
    transcription: TranscriptionResult,
    output_dir: Optional[Path] = None,
    source_name: Optional[str] = None,
    export_subtitles: bool = True,
) -> dict[str, Path]:
    """
    Convenience function to export a transcription.

    Args:
        transcription: Transcription result to export
        output_dir: Base output directory (optional)
        source_name: Name for output directory (optional)
        export_subtitles: Whether to export SRT and VTT files

    Returns:
        Dictionary of exported file paths
    """
    config = OutputConfig()
    if output_dir:
        config.base_dir = output_dir

    exporter = TranscriptionExporter(config)
    return exporter.export(
        transcription=transcription,
        source_name=source_name,
        export_subtitles=export_subtitles,
    )
