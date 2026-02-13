"""Tests for PodcastService."""

import pytest

from app_main.services.podcast_service import PodcastService
from tests.conftest import make_episode, make_episode_profile, make_speaker_profile


class TestPodcastServiceEpisodes:

    @pytest.mark.asyncio
    async def test_get_episode(self, episode_repo, episode_profile_repo, speaker_profile_repo):
        ep = make_episode()
        episode_repo.get.return_value = ep
        service = PodcastService(episode_repo, episode_profile_repo, speaker_profile_repo)

        result = await service.get_episode("podcast_episode:test1")

        episode_repo.get.assert_called_once_with("podcast_episode:test1")
        assert result == ep

    @pytest.mark.asyncio
    async def test_get_all_episodes(self, episode_repo, episode_profile_repo, speaker_profile_repo):
        episode_repo.get_all.return_value = [make_episode()]
        service = PodcastService(episode_repo, episode_profile_repo, speaker_profile_repo)

        result = await service.get_all_episodes()

        episode_repo.get_all.assert_called_once_with(order_by="created DESC")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_create_episode(self, episode_repo, episode_profile_repo, speaker_profile_repo):
        ep = make_episode()
        episode_repo.create.return_value = ep
        service = PodcastService(episode_repo, episode_profile_repo, speaker_profile_repo)

        result = await service.create_episode({"name": "Ep1"})

        episode_repo.create.assert_called_once_with({"name": "Ep1"})
        assert result == ep

    @pytest.mark.asyncio
    async def test_delete_episode(self, episode_repo, episode_profile_repo, speaker_profile_repo):
        episode_repo.delete.return_value = True
        service = PodcastService(episode_repo, episode_profile_repo, speaker_profile_repo)

        assert await service.delete_episode("ep:1") is True


class TestPodcastServiceEpisodeProfiles:

    @pytest.mark.asyncio
    async def test_get_episode_profile(self, episode_repo, episode_profile_repo, speaker_profile_repo):
        prof = make_episode_profile()
        episode_profile_repo.get.return_value = prof
        service = PodcastService(episode_repo, episode_profile_repo, speaker_profile_repo)

        result = await service.get_episode_profile("episode_profile:test1")

        assert result == prof

    @pytest.mark.asyncio
    async def test_get_all_episode_profiles(self, episode_repo, episode_profile_repo, speaker_profile_repo):
        episode_profile_repo.get_all.return_value = [make_episode_profile()]
        service = PodcastService(episode_repo, episode_profile_repo, speaker_profile_repo)

        result = await service.get_all_episode_profiles()

        episode_profile_repo.get_all.assert_called_once_with(order_by="name ASC")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_delete_episode_profile(self, episode_repo, episode_profile_repo, speaker_profile_repo):
        episode_profile_repo.delete.return_value = True
        service = PodcastService(episode_repo, episode_profile_repo, speaker_profile_repo)

        assert await service.delete_episode_profile("ep_prof:1") is True


class TestPodcastServiceSpeakerProfiles:

    @pytest.mark.asyncio
    async def test_get_speaker_profile(self, episode_repo, episode_profile_repo, speaker_profile_repo):
        sp = make_speaker_profile()
        speaker_profile_repo.get.return_value = sp
        service = PodcastService(episode_repo, episode_profile_repo, speaker_profile_repo)

        result = await service.get_speaker_profile("speaker_profile:test1")

        assert result == sp

    @pytest.mark.asyncio
    async def test_get_all_speaker_profiles(self, episode_repo, episode_profile_repo, speaker_profile_repo):
        speaker_profile_repo.get_all.return_value = [make_speaker_profile()]
        service = PodcastService(episode_repo, episode_profile_repo, speaker_profile_repo)

        result = await service.get_all_speaker_profiles()

        speaker_profile_repo.get_all.assert_called_once_with(order_by="name ASC")
        assert len(result) == 1
