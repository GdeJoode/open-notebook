"""
Podcast service - business logic for podcast operations.
"""

from typing import Any, Dict, List, Optional

from shared.models.podcast import EpisodeProfile, PodcastEpisode, SpeakerProfile
from surrealdb_service.repositories import (
    EpisodeProfileRepository,
    PodcastEpisodeRepository,
    SpeakerProfileRepository,
)


class PodcastService:
    """Service for podcast business logic."""

    def __init__(
        self,
        episode_repo: PodcastEpisodeRepository,
        episode_profile_repo: EpisodeProfileRepository,
        speaker_profile_repo: SpeakerProfileRepository,
    ):
        self.episode_repo = episode_repo
        self.episode_profile_repo = episode_profile_repo
        self.speaker_profile_repo = speaker_profile_repo

    # --- Episode operations ---

    async def get_episode(self, episode_id: str) -> Optional[PodcastEpisode]:
        """Get an episode by ID."""
        return await self.episode_repo.get(episode_id)

    async def get_all_episodes(self) -> List[PodcastEpisode]:
        """Get all episodes."""
        return await self.episode_repo.get_all(order_by="created DESC")

    async def create_episode(self, data: Dict[str, Any]) -> PodcastEpisode:
        """Create a new episode."""
        return await self.episode_repo.create(data)

    async def delete_episode(self, episode_id: str) -> bool:
        """Delete an episode."""
        return await self.episode_repo.delete(episode_id)

    # --- Episode Profile operations ---

    async def get_episode_profile(
        self, profile_id: str,
    ) -> Optional[EpisodeProfile]:
        """Get an episode profile by ID."""
        return await self.episode_profile_repo.get(profile_id)

    async def get_all_episode_profiles(self) -> List[EpisodeProfile]:
        """Get all episode profiles."""
        return await self.episode_profile_repo.get_all(order_by="name ASC")

    async def create_episode_profile(
        self, data: Dict[str, Any],
    ) -> EpisodeProfile:
        """Create a new episode profile."""
        return await self.episode_profile_repo.create(data)

    async def update_episode_profile(
        self, profile_id: str, data: Dict[str, Any],
    ) -> EpisodeProfile:
        """Update an episode profile."""
        return await self.episode_profile_repo.update(profile_id, data)

    async def delete_episode_profile(self, profile_id: str) -> bool:
        """Delete an episode profile."""
        return await self.episode_profile_repo.delete(profile_id)

    # --- Speaker Profile operations ---

    async def get_speaker_profile(
        self, profile_id: str,
    ) -> Optional[SpeakerProfile]:
        """Get a speaker profile by ID."""
        return await self.speaker_profile_repo.get(profile_id)

    async def get_all_speaker_profiles(self) -> List[SpeakerProfile]:
        """Get all speaker profiles."""
        return await self.speaker_profile_repo.get_all(order_by="name ASC")

    async def create_speaker_profile(
        self, data: Dict[str, Any],
    ) -> SpeakerProfile:
        """Create a new speaker profile."""
        return await self.speaker_profile_repo.create(data)

    async def update_speaker_profile(
        self, profile_id: str, data: Dict[str, Any],
    ) -> SpeakerProfile:
        """Update a speaker profile."""
        return await self.speaker_profile_repo.update(profile_id, data)

    async def delete_speaker_profile(self, profile_id: str) -> bool:
        """Delete a speaker profile."""
        return await self.speaker_profile_repo.delete(profile_id)
