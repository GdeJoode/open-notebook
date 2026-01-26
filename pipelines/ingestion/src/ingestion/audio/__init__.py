"""
Audio/video transcription for the ingestion pipeline.

Provides:
- WhisperXTranscriber: Transcription with speaker diarization
- transcribe: Convenience function for transcription
"""

from .transcriber import WhisperXTranscriber, transcribe

__all__ = [
    "WhisperXTranscriber",
    "transcribe",
]
