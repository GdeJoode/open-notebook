"""Tests for ModelService."""

import pytest

from app_main.services.model_service import ModelService
from tests.conftest import make_default_models, make_model


class TestModelServiceGet:

    @pytest.mark.asyncio
    async def test_get(self, model_repo, defaults_repo, model_manager):
        m = make_model()
        model_repo.get.return_value = m
        service = ModelService(model_repo, defaults_repo, model_manager)

        result = await service.get("model:test1")

        model_repo.get.assert_called_once_with("model:test1")
        assert result == m

    @pytest.mark.asyncio
    async def test_get_returns_none(self, model_repo, defaults_repo, model_manager):
        model_repo.get.return_value = None
        service = ModelService(model_repo, defaults_repo, model_manager)

        assert await service.get("model:missing") is None


class TestModelServiceGetAll:

    @pytest.mark.asyncio
    async def test_get_all(self, model_repo, defaults_repo, model_manager):
        model_repo.get_all.return_value = [make_model(), make_model(id="model:2")]
        service = ModelService(model_repo, defaults_repo, model_manager)

        result = await service.get_all()

        assert len(result) == 2


class TestModelServiceGetByType:

    @pytest.mark.asyncio
    async def test_get_by_type(self, model_repo, defaults_repo, model_manager):
        model_repo.get_by_type.return_value = [make_model(type="embedding")]
        service = ModelService(model_repo, defaults_repo, model_manager)

        result = await service.get_by_type("embedding")

        model_repo.get_by_type.assert_called_once_with("embedding")
        assert len(result) == 1


class TestModelServiceCreate:

    @pytest.mark.asyncio
    async def test_create(self, model_repo, defaults_repo, model_manager):
        m = make_model()
        model_repo.create.return_value = m
        service = ModelService(model_repo, defaults_repo, model_manager)

        result = await service.create({"name": "gpt-4o-mini", "provider": "openai", "type": "language"})

        model_repo.create.assert_called_once()
        assert result == m


class TestModelServiceDelete:

    @pytest.mark.asyncio
    async def test_delete(self, model_repo, defaults_repo, model_manager):
        model_repo.delete.return_value = True
        service = ModelService(model_repo, defaults_repo, model_manager)

        assert await service.delete("model:1") is True


class TestModelServiceDefaults:

    @pytest.mark.asyncio
    async def test_get_defaults(self, model_repo, defaults_repo, model_manager):
        defs = make_default_models(default_chat_model="model:gpt4")
        defaults_repo.get.return_value = defs
        service = ModelService(model_repo, defaults_repo, model_manager)

        result = await service.get_defaults()

        assert result.default_chat_model == "model:gpt4"

    @pytest.mark.asyncio
    async def test_update_defaults_clears_cache(self, model_repo, defaults_repo, model_manager):
        updated = make_default_models(default_chat_model="model:new")
        defaults_repo.update.return_value = updated
        service = ModelService(model_repo, defaults_repo, model_manager)

        result = await service.update_defaults({"default_chat_model": "model:new"})

        defaults_repo.update.assert_called_once()
        model_manager.set_defaults.assert_called_once_with(updated)
        model_manager.clear_cache.assert_called_once()
        assert result.default_chat_model == "model:new"
