"""Tests for SettingsService."""

import pytest

from app_main.services.settings_service import SettingsService
from tests.conftest import make_settings


class TestSettingsServiceGet:

    @pytest.mark.asyncio
    async def test_get(self, settings_repo):
        s = make_settings()
        settings_repo.get.return_value = s
        service = SettingsService(settings_repo)

        result = await service.get()

        settings_repo.get.assert_called_once()
        assert result == s


class TestSettingsServiceUpdate:

    @pytest.mark.asyncio
    async def test_update(self, settings_repo):
        updated = make_settings(default_embedding_option="always")
        settings_repo.update.return_value = updated
        service = SettingsService(settings_repo)

        result = await service.update({"default_embedding_option": "always"})

        settings_repo.update.assert_called_once_with({"default_embedding_option": "always"})
        assert result.default_embedding_option == "always"


class TestSettingsServicePatch:

    @pytest.mark.asyncio
    async def test_patch(self, settings_repo):
        patched = make_settings(auto_delete_files="no")
        settings_repo.patch.return_value = patched
        service = SettingsService(settings_repo)

        result = await service.patch({"auto_delete_files": "no"})

        settings_repo.patch.assert_called_once_with({"auto_delete_files": "no"})
        assert result.auto_delete_files == "no"
