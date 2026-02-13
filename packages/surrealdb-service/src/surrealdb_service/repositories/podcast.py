"""
Podcast repositories for speaker profiles, episode profiles, and episodes.
"""

from typing import Optional

from shared.models.podcast import EpisodeProfile, PodcastEpisode, SpeakerProfile
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.repositories.base import BaseRepository


class SpeakerProfileRepository(BaseRepository[SpeakerProfile]):
    """Repository for SpeakerProfile operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(SpeakerProfile, config)

    async def get_by_name(self, name: str) -> Optional[SpeakerProfile]:
        """Get a speaker profile by name."""
        results = await self.query("name = $name", {"name": name}, limit=1)
        return results[0] if results else None


class EpisodeProfileRepository(BaseRepository[EpisodeProfile]):
    """Repository for EpisodeProfile operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(EpisodeProfile, config)

    async def get_by_name(self, name: str) -> Optional[EpisodeProfile]:
        """Get an episode profile by name."""
        results = await self.query("name = $name", {"name": name}, limit=1)
        return results[0] if results else None


class PodcastEpisodeRepository(BaseRepository[PodcastEpisode]):
    """Repository for PodcastEpisode operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(PodcastEpisode, config)
