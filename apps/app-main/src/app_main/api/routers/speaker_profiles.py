"""Speaker profiles router - CRUD for podcast speaker profiles."""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from app_main.dependencies import get_podcast_service
from app_main.services.podcast_service import PodcastService

router = APIRouter(prefix="/speaker-profiles", tags=["podcast"])


def _profile_to_dict(profile) -> Dict[str, Any]:
    """Convert a SpeakerProfile domain model to a response dict."""
    return {
        "id": profile.id,
        "name": profile.name,
        "description": profile.description or "",
        "tts_provider": profile.tts_provider,
        "tts_model": profile.tts_model,
        "speakers": profile.speakers,
        "created": str(profile.created),
        "updated": str(profile.updated),
    }


@router.get("")
async def list_speaker_profiles(
    podcast_service: PodcastService = Depends(get_podcast_service),
):
    """List all speaker profiles."""
    profiles = await podcast_service.get_all_speaker_profiles()
    return [_profile_to_dict(p) for p in profiles]


@router.post("", status_code=201)
async def create_speaker_profile(
    body: Dict[str, Any],
    podcast_service: PodcastService = Depends(get_podcast_service),
):
    """Create a new speaker profile."""
    profile = await podcast_service.create_speaker_profile(body)
    logger.info("Created speaker profile {}", profile.id)
    return _profile_to_dict(profile)


@router.get("/by-name/{profile_name}")
async def get_speaker_profile_by_name(
    profile_name: str,
    podcast_service: PodcastService = Depends(get_podcast_service),
):
    """Get a speaker profile by name."""
    profiles = await podcast_service.get_all_speaker_profiles()
    profile = next((p for p in profiles if p.name == profile_name), None)
    if not profile:
        raise HTTPException(
            status_code=404,
            detail=f"Speaker profile '{profile_name}' not found",
        )
    return _profile_to_dict(profile)


@router.get("/{profile_id}")
async def get_speaker_profile(
    profile_id: str,
    podcast_service: PodcastService = Depends(get_podcast_service),
):
    """Get a speaker profile by ID."""
    profile = await podcast_service.get_speaker_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Speaker profile not found")
    return _profile_to_dict(profile)


@router.put("/{profile_id}")
async def update_speaker_profile(
    profile_id: str,
    body: Dict[str, Any],
    podcast_service: PodcastService = Depends(get_podcast_service),
):
    """Update a speaker profile."""
    existing = await podcast_service.get_speaker_profile(profile_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Speaker profile not found")

    profile = await podcast_service.update_speaker_profile(profile_id, body)
    logger.info("Updated speaker profile {}", profile_id)
    return _profile_to_dict(profile)


@router.delete("/{profile_id}", status_code=204)
async def delete_speaker_profile(
    profile_id: str,
    podcast_service: PodcastService = Depends(get_podcast_service),
):
    """Delete a speaker profile."""
    existing = await podcast_service.get_speaker_profile(profile_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Speaker profile not found")
    await podcast_service.delete_speaker_profile(profile_id)
    logger.info("Deleted speaker profile {}", profile_id)


@router.post("/{profile_id}/duplicate", status_code=201)
async def duplicate_speaker_profile(
    profile_id: str,
    podcast_service: PodcastService = Depends(get_podcast_service),
):
    """Duplicate a speaker profile."""
    original = await podcast_service.get_speaker_profile(profile_id)
    if not original:
        raise HTTPException(
            status_code=404,
            detail=f"Speaker profile '{profile_id}' not found",
        )

    data = {
        "name": f"{original.name} - Copy",
        "description": original.description,
        "tts_provider": original.tts_provider,
        "tts_model": original.tts_model,
        "speakers": original.speakers,
    }
    duplicate = await podcast_service.create_speaker_profile(data)
    logger.info("Duplicated speaker profile {} -> {}", profile_id, duplicate.id)
    return _profile_to_dict(duplicate)
