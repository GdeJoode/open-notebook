"""Podcasts router - podcast episode management."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel, Field

from app_main.dependencies import get_notebook_service, get_podcast_service
from app_main.services.notebook_service import NotebookService
from app_main.services.podcast_service import PodcastService

router = APIRouter(prefix="/podcasts", tags=["podcast"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class PodcastGenerationRequest(BaseModel):
    episode_profile: str = Field(..., description="Episode profile name")
    speaker_profile: str = Field(..., description="Speaker profile name")
    episode_name: str = Field(..., description="Episode name")
    content: Optional[str] = Field(
        None, description="Content for podcast generation"
    )
    notebook_id: Optional[str] = Field(
        None, description="Notebook ID to use as content source"
    )
    briefing_suffix: Optional[str] = Field(
        None, description="Additional briefing suffix"
    )


class PodcastGenerationResponse(BaseModel):
    job_id: str
    status: str
    message: str
    episode_profile: str
    episode_name: str


class PodcastEpisodeResponse(BaseModel):
    id: str
    name: str
    episode_profile: dict
    speaker_profile: dict
    briefing: str
    audio_file: Optional[str] = None
    audio_url: Optional[str] = None
    transcript: Optional[dict] = None
    outline: Optional[dict] = None
    created: Optional[str] = None
    job_status: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_audio_path(audio_file: str) -> Path:
    """Resolve audio file path from various formats."""
    if audio_file.startswith("file://"):
        parsed = urlparse(audio_file)
        return Path(unquote(parsed.path))
    return Path(audio_file)


def _episode_to_response(
    episode, job_status: Optional[str] = None,
) -> PodcastEpisodeResponse:
    """Convert a PodcastEpisode domain model to a response."""
    audio_url = None
    if episode.audio_file:
        audio_path = _resolve_audio_path(episode.audio_file)
        if audio_path.exists():
            audio_url = f"/api/podcasts/episodes/{episode.id}/audio"

    return PodcastEpisodeResponse(
        id=str(episode.id),
        name=episode.name,
        episode_profile=episode.episode_profile,
        speaker_profile=episode.speaker_profile,
        briefing=episode.briefing,
        audio_file=episode.audio_file,
        audio_url=audio_url,
        transcript=episode.transcript,
        outline=episode.outline,
        created=str(episode.created) if episode.created else None,
        job_status=job_status,
    )


async def _get_episode_job_status(episode) -> Optional[str]:
    """Get job status for an episode."""
    if hasattr(episode, "command") and episode.command:
        try:
            from app_main.services.command_service import CommandService

            status_data = await CommandService.get_command_status(
                str(episode.command)
            )
            return status_data.get("status", "unknown")
        except Exception:
            return "unknown"
    elif episode.audio_file:
        return "completed"
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/generate", response_model=PodcastGenerationResponse)
async def generate_podcast(
    request: PodcastGenerationRequest,
    podcast_svc: PodcastService = Depends(get_podcast_service),
    notebook_svc: NotebookService = Depends(get_notebook_service),
):
    """Generate a podcast episode using Episode Profiles.

    Returns immediately with job ID for status tracking.
    """
    try:
        # Validate episode profile exists
        profiles = await podcast_svc.get_all_episode_profiles()
        ep_profile = next(
            (p for p in profiles if p.name == request.episode_profile),
            None,
        )
        if not ep_profile:
            raise HTTPException(
                status_code=404,
                detail=f"Episode profile '{request.episode_profile}' not found",
            )

        # Validate speaker profile exists
        sp_profiles = await podcast_svc.get_all_speaker_profiles()
        sp_profile = next(
            (p for p in sp_profiles if p.name == request.speaker_profile),
            None,
        )
        if not sp_profile:
            raise HTTPException(
                status_code=404,
                detail=f"Speaker profile '{request.speaker_profile}' not found",
            )

        # Get content from notebook if not provided directly
        content = request.content
        if not content and request.notebook_id:
            try:
                notebook = await notebook_svc.get(request.notebook_id)
                if notebook:
                    content = f"Notebook ID: {request.notebook_id}"
                else:
                    raise HTTPException(
                        status_code=404, detail="Notebook not found"
                    )
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(
                    f"Failed to get notebook content: {e}"
                )
                content = f"Notebook ID: {request.notebook_id}"

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Content is required -- provide either content or notebook_id",
            )

        # Submit command for background processing
        from app_main.services.command_service import CommandService

        command_args = {
            "episode_profile": request.episode_profile,
            "speaker_profile": request.speaker_profile,
            "episode_name": request.episode_name,
            "content": str(content),
            "briefing_suffix": request.briefing_suffix,
        }

        job_id = await CommandService.submit_command_job(
            "open_notebook",
            "generate_podcast",
            command_args,
        )

        logger.info(
            f"Submitted podcast generation job: {job_id} "
            f"for episode '{request.episode_name}'"
        )

        return PodcastGenerationResponse(
            job_id=str(job_id),
            status="submitted",
            message=(
                f"Podcast generation started for "
                f"episode '{request.episode_name}'"
            ),
            episode_profile=request.episode_profile,
            episode_name=request.episode_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating podcast: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate podcast: {str(e)}",
        )


@router.get("/jobs/{job_id}")
async def get_podcast_job_status(job_id: str):
    """Get the status of a podcast generation job."""
    try:
        from app_main.services.command_service import CommandService

        status_data = await CommandService.get_command_status(job_id)
        return {
            "job_id": job_id,
            "status": status_data.get("status", "unknown"),
            "result": status_data.get("result"),
            "error_message": status_data.get("error_message"),
            "created": status_data.get("created"),
            "updated": status_data.get("updated"),
            "progress": status_data.get("progress"),
        }
    except Exception as e:
        logger.error(f"Error fetching podcast job status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch job status: {str(e)}",
        )


@router.get(
    "/episodes",
    response_model=List[PodcastEpisodeResponse],
)
async def list_podcast_episodes(
    podcast_svc: PodcastService = Depends(get_podcast_service),
):
    """List all podcast episodes."""
    try:
        episodes = await podcast_svc.get_all_episodes()

        response_episodes = []
        for episode in episodes:
            # Skip incomplete episodes without command or audio
            has_command = hasattr(episode, "command") and episode.command
            if not has_command and not episode.audio_file:
                continue

            job_status = await _get_episode_job_status(episode)
            response_episodes.append(
                _episode_to_response(episode, job_status)
            )

        return response_episodes

    except Exception as e:
        logger.error(f"Error listing podcast episodes: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list podcast episodes: {str(e)}",
        )


@router.get(
    "/episodes/{episode_id}",
    response_model=PodcastEpisodeResponse,
)
async def get_podcast_episode(
    episode_id: str,
    podcast_svc: PodcastService = Depends(get_podcast_service),
):
    """Get a specific podcast episode."""
    try:
        episode = await podcast_svc.get_episode(episode_id)
        if not episode:
            raise HTTPException(
                status_code=404, detail="Episode not found"
            )

        job_status = await _get_episode_job_status(episode)
        return _episode_to_response(episode, job_status)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching podcast episode: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Episode not found: {str(e)}",
        )


@router.get("/episodes/{episode_id}/audio")
async def stream_podcast_episode_audio(
    episode_id: str,
    podcast_svc: PodcastService = Depends(get_podcast_service),
):
    """Stream the audio file associated with a podcast episode."""
    try:
        episode = await podcast_svc.get_episode(episode_id)
        if not episode:
            raise HTTPException(
                status_code=404, detail="Episode not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error fetching podcast episode for audio: {e}"
        )
        raise HTTPException(
            status_code=404,
            detail=f"Episode not found: {str(e)}",
        )

    if not episode.audio_file:
        raise HTTPException(
            status_code=404, detail="Episode has no audio file"
        )

    audio_path = _resolve_audio_path(episode.audio_file)
    if not audio_path.exists():
        raise HTTPException(
            status_code=404, detail="Audio file not found on disk"
        )

    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        filename=audio_path.name,
    )


@router.delete("/episodes/{episode_id}")
async def delete_podcast_episode(
    episode_id: str,
    podcast_svc: PodcastService = Depends(get_podcast_service),
):
    """Delete a podcast episode and its associated audio file."""
    try:
        episode = await podcast_svc.get_episode(episode_id)
        if not episode:
            raise HTTPException(
                status_code=404, detail="Episode not found"
            )

        # Delete the physical audio file if it exists
        if episode.audio_file:
            audio_path = _resolve_audio_path(episode.audio_file)
            if audio_path.exists():
                try:
                    audio_path.unlink()
                    logger.info(f"Deleted audio file: {audio_path}")
                except Exception as e:
                    logger.warning(
                        f"Failed to delete audio file {audio_path}: {e}"
                    )

        # Delete the episode from the database
        await podcast_svc.delete_episode(episode_id)

        logger.info(f"Deleted podcast episode: {episode_id}")
        return {
            "message": "Episode deleted successfully",
            "episode_id": episode_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting podcast episode: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete episode: {str(e)}",
        )
