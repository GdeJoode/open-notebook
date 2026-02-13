"""Episode profiles router - CRUD for podcast episode profiles."""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from app_main.dependencies import get_podcast_service
from app_main.services.podcast_service import PodcastService

router = APIRouter(prefix="/episode-profiles", tags=["podcast"])


def _profile_to_dict(profile) -> Dict[str, Any]:
    """Convert an EpisodeProfile domain model to a response dict."""
    return {
        "id": profile.id,
        "name": profile.name,
        "description": profile.description or "",
        "speaker_config": profile.speaker_config,
        "outline_provider": profile.outline_provider,
        "outline_model": profile.outline_model,
        "transcript_provider": profile.transcript_provider,
        "transcript_model": profile.transcript_model,
        "default_briefing": profile.default_briefing,
        "num_segments": profile.num_segments,
        "created": str(profile.created),
        "updated": str(profile.updated),
    }


@router.get("")
async def list_episode_profiles(
    podcast_service: PodcastService = Depends(get_podcast_service),
):
    """List all episode profiles."""
    profiles = await podcast_service.get_all_episode_profiles()
    return [_profile_to_dict(p) for p in profiles]


@router.post("", status_code=201)
async def create_episode_profile(
    body: Dict[str, Any],
    podcast_service: PodcastService = Depends(get_podcast_service),
):
    """Create a new episode profile."""
    profile = await podcast_service.create_episode_profile(body)
    logger.info("Created episode profile {}", profile.id)
    return _profile_to_dict(profile)


@router.get("/by-name/{profile_name}")
async def get_episode_profile_by_name(
    profile_name: str,
    podcast_service: PodcastService = Depends(get_podcast_service),
):
    """Get an episode profile by name."""
    profiles = await podcast_service.get_all_episode_profiles()
    profile = next((p for p in profiles if p.name == profile_name), None)
    if not profile:
        raise HTTPException(
            status_code=404,
            detail=f"Episode profile '{profile_name}' not found",
        )
    return _profile_to_dict(profile)


@router.get("/{profile_id}")
async def get_episode_profile(
    profile_id: str,
    podcast_service: PodcastService = Depends(get_podcast_service),
):
    """Get an episode profile by ID."""
    profile = await podcast_service.get_episode_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Episode profile not found")
    return _profile_to_dict(profile)


@router.put("/{profile_id}")
async def update_episode_profile(
    profile_id: str,
    body: Dict[str, Any],
    podcast_service: PodcastService = Depends(get_podcast_service),
):
    """Update an episode profile."""
    existing = await podcast_service.get_episode_profile(profile_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Episode profile not found")

    profile = await podcast_service.update_episode_profile(profile_id, body)
    logger.info("Updated episode profile {}", profile_id)
    return _profile_to_dict(profile)


@router.delete("/{profile_id}", status_code=204)
async def delete_episode_profile(
    profile_id: str,
    podcast_service: PodcastService = Depends(get_podcast_service),
):
    """Delete an episode profile."""
    existing = await podcast_service.get_episode_profile(profile_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Episode profile not found")
    await podcast_service.delete_episode_profile(profile_id)
    logger.info("Deleted episode profile {}", profile_id)


@router.post("/{profile_id}/duplicate", status_code=201)
async def duplicate_episode_profile(
    profile_id: str,
    podcast_service: PodcastService = Depends(get_podcast_service),
):
    """Duplicate an episode profile."""
    original = await podcast_service.get_episode_profile(profile_id)
    if not original:
        raise HTTPException(
            status_code=404,
            detail=f"Episode profile '{profile_id}' not found",
        )

    data = {
        "name": f"{original.name} - Copy",
        "description": original.description,
        "speaker_config": original.speaker_config,
        "outline_provider": original.outline_provider,
        "outline_model": original.outline_model,
        "transcript_provider": original.transcript_provider,
        "transcript_model": original.transcript_model,
        "default_briefing": original.default_briefing,
        "num_segments": original.num_segments,
    }
    duplicate = await podcast_service.create_episode_profile(data)
    logger.info("Duplicated episode profile {} -> {}", profile_id, duplicate.id)
    return _profile_to_dict(duplicate)
