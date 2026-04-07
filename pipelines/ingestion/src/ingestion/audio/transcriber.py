"""
WhisperX-based audio/video transcription.

Features:
- Multiple model sizes (tiny to large-v3)
- Speaker diarization with pyannote
- Word-level timestamps
- GPU acceleration (CUDA, MPS, CPU)
- Audio extraction from video
"""

import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from ingestion.config import WhisperXConfig
from ingestion.models import (
    SourceMetadata,
    SourceType,
    SpeakerInfo,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)


class WhisperXTranscriber:
    """
    Audio/video transcriber using WhisperX.

    WhisperX provides:
    - Fast transcription with batch processing
    - Word-level timestamps via forced alignment
    - Speaker diarization via pyannote
    - GPU acceleration support
    """

    def __init__(self, config: Optional[WhisperXConfig] = None):
        """
        Initialize the WhisperX transcriber.

        Args:
            config: WhisperXConfig instance. If None, uses defaults.
        """
        self.config = config or WhisperXConfig()
        self._model = None
        self._diarize_model = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazily initialize WhisperX models."""
        if self._initialized:
            return

        try:
            import whisperx
        except ImportError:
            raise ImportError(
                "WhisperX not installed. Install with: "
                "pip install whisperx torch torchaudio"
            )

        device = self.config.get_device()
        compute_type = self.config.compute_type

        # Adjust compute type for CPU
        if device == "cpu" and compute_type == "float16":
            compute_type = "float32"
            logger.info("Adjusted compute_type to float32 for CPU")

        logger.info(
            f"Loading WhisperX model: {self.config.model_size.value} "
            f"on {device} with {compute_type}"
        )

        # Load main transcription model
        self._model = whisperx.load_model(
            self.config.model_size.value,
            device=device,
            compute_type=compute_type,
            language=self.config.language,
        )

        # Load diarization model if enabled
        if self.config.enable_diarization:
            hf_token = self.config.hf_token or os.getenv("HF_TOKEN")
            if hf_token:
                try:
                    from whisperx.diarize import DiarizationPipeline
                    self._diarize_model = DiarizationPipeline(
                        use_auth_token=hf_token,
                        device=device,
                    )
                    logger.info("Speaker diarization enabled")
                except Exception as e:
                    logger.warning(f"Failed to load diarization model: {e}")
                    self._diarize_model = None
            else:
                logger.warning(
                    "HF_TOKEN not set - speaker diarization disabled. "
                    "Set HF_TOKEN environment variable for diarization support."
                )

        self._initialized = True

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """
        Transcribe an audio or video file.

        For video files, audio is automatically extracted.

        Args:
            audio_path: Path to audio or video file

        Returns:
            TranscriptionResult with segments and speaker info
        """
        self._ensure_initialized()

        start_time = time.time()
        logger.info(f"Transcribing: {audio_path}")

        # Handle video files - extract audio first
        is_video = audio_path.suffix.lower() in [".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv"]
        temp_audio = None

        if is_video:
            temp_audio = self._extract_audio(audio_path)
            processing_path = temp_audio
        else:
            processing_path = audio_path

        try:
            # Load and transcribe audio
            import whisperx

            device = self.config.get_device()

            # Load audio
            audio = whisperx.load_audio(str(processing_path))

            # Transcribe
            result = self._model.transcribe(audio, batch_size=16)
            detected_language = result.get("language", self.config.language or "en")

            # Align for word-level timestamps
            if self.config.include_word_timestamps:
                try:
                    model_a, metadata = whisperx.load_align_model(
                        language_code=detected_language,
                        device=device,
                    )
                    result = whisperx.align(
                        result["segments"],
                        model_a,
                        metadata,
                        audio,
                        device,
                        return_char_alignments=False,
                    )
                except Exception as e:
                    logger.warning(f"Word alignment failed: {e}")

            # Speaker diarization
            if self._diarize_model is not None:
                try:
                    diarize_segments = self._diarize_model(
                        audio,
                        min_speakers=self.config.min_speakers,
                        max_speakers=self.config.max_speakers,
                    )
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                except Exception as e:
                    logger.warning(f"Diarization failed: {e}")

            # Convert to our models
            transcription_result = self._build_result(
                result=result,
                source_path=audio_path,
                detected_language=detected_language,
                start_time=start_time,
                is_video=is_video,
            )

            return transcription_result

        finally:
            # Clean up temporary audio file
            if temp_audio and temp_audio.exists():
                temp_audio.unlink()

    def _extract_audio(self, video_path: Path) -> Path:
        """Extract audio from video file using ffmpeg."""
        logger.info(f"Extracting audio from video: {video_path}")

        # Create temporary file for audio
        temp_dir = tempfile.mkdtemp()
        temp_audio = Path(temp_dir) / "audio.wav"

        try:
            import subprocess

            result = subprocess.run(
                [
                    "ffmpeg",
                    "-i", str(video_path),
                    "-vn",  # No video
                    "-acodec", "pcm_s16le",  # PCM 16-bit
                    "-ar", "16000",  # 16kHz sample rate
                    "-ac", "1",  # Mono
                    "-y",  # Overwrite
                    str(temp_audio),
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")

            return temp_audio

        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg not found. Install FFmpeg for video support."
            )

    def _build_result(
        self,
        result: dict,
        source_path: Path,
        detected_language: str,
        start_time: float,
        is_video: bool,
    ) -> TranscriptionResult:
        """Build TranscriptionResult from WhisperX output."""
        segments = []
        speakers_dict: dict[str, SpeakerInfo] = {}

        raw_segments = result.get("segments", [])

        for seg in raw_segments:
            # Extract words
            words = []
            for word in seg.get("words", []):
                words.append(
                    TranscriptionWord(
                        word=word.get("word", ""),
                        start=word.get("start", 0),
                        end=word.get("end", 0),
                        confidence=word.get("score", 1.0),
                    )
                )

            # Get speaker
            speaker = seg.get("speaker")
            if speaker:
                if speaker not in speakers_dict:
                    speakers_dict[speaker] = SpeakerInfo(
                        speaker_id=speaker,
                        total_time=0.0,
                        segment_count=0,
                    )
                speakers_dict[speaker].total_time += seg.get("end", 0) - seg.get("start", 0)
                speakers_dict[speaker].segment_count += 1

            segment = TranscriptionSegment(
                text=seg.get("text", "").strip(),
                start=seg.get("start", 0),
                end=seg.get("end", 0),
                speaker=speaker,
                words=words,
                language=detected_language,
            )
            segments.append(segment)

        # Calculate total duration
        total_duration = 0.0
        if segments:
            total_duration = segments[-1].end

        # Get source metadata
        source_type = SourceType.VIDEO if is_video else SourceType.AUDIO
        source_metadata = SourceMetadata.from_media_file(source_path, source_type)

        processing_time = time.time() - start_time

        transcription = TranscriptionResult(
            source_path=source_path,
            segments=segments,
            speakers=list(speakers_dict.values()),
            language=detected_language,
            duration_seconds=total_duration,
            transcription_time_seconds=processing_time,
            source_filename=source_path.name,
            source_duration=source_metadata.duration_seconds,
            source_format=source_path.suffix.lower(),
            source_size_bytes=source_metadata.file_size_bytes,
            source_created_at=source_metadata.created_at,
            model_name=self.config.model_size.value,
            compute_type=self.config.compute_type,
            device=self.config.get_device(),
        )

        # Generate text versions
        transcription.full_text = transcription.to_plain_text()
        transcription.full_text_with_speakers = transcription.to_text_with_speakers()
        transcription.full_text_with_timestamps = transcription.to_text_with_timestamps()

        logger.info(
            f"Transcription complete: {len(segments)} segments, "
            f"{len(speakers_dict)} speakers, {total_duration:.1f}s duration "
            f"in {processing_time:.2f}s"
        )

        return transcription


def transcribe(
    audio_path: Path,
    config: Optional[WhisperXConfig] = None,
) -> TranscriptionResult:
    """
    Convenience function to transcribe an audio/video file.

    Args:
        audio_path: Path to audio or video file
        config: Optional WhisperXConfig

    Returns:
        TranscriptionResult with full transcription
    """
    transcriber = WhisperXTranscriber(config)
    return transcriber.transcribe(audio_path)
