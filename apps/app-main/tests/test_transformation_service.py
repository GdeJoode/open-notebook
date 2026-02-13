"""Tests for TransformationService."""

import pytest

from app_main.services.transformation_service import TransformationService
from tests.conftest import make_default_prompts, make_transformation


class TestTransformationServiceGet:

    @pytest.mark.asyncio
    async def test_get(self, transformation_repo, default_prompts_repo):
        t = make_transformation()
        transformation_repo.get.return_value = t
        service = TransformationService(transformation_repo, default_prompts_repo)

        result = await service.get("transformation:test1")

        transformation_repo.get.assert_called_once_with("transformation:test1")
        assert result == t


class TestTransformationServiceGetAll:

    @pytest.mark.asyncio
    async def test_get_all(self, transformation_repo, default_prompts_repo):
        transformation_repo.get_all.return_value = [make_transformation()]
        service = TransformationService(transformation_repo, default_prompts_repo)

        result = await service.get_all()

        transformation_repo.get_all.assert_called_once_with(order_by="name asc")
        assert len(result) == 1


class TestTransformationServiceCreate:

    @pytest.mark.asyncio
    async def test_create(self, transformation_repo, default_prompts_repo):
        t = make_transformation()
        transformation_repo.create.return_value = t
        service = TransformationService(transformation_repo, default_prompts_repo)

        result = await service.create({"name": "summarize"})

        assert result == t


class TestTransformationServiceUpdate:

    @pytest.mark.asyncio
    async def test_update(self, transformation_repo, default_prompts_repo):
        t = make_transformation(title="Updated")
        transformation_repo.update.return_value = t
        service = TransformationService(transformation_repo, default_prompts_repo)

        result = await service.update("transformation:1", {"title": "Updated"})

        assert result.title == "Updated"


class TestTransformationServiceDelete:

    @pytest.mark.asyncio
    async def test_delete(self, transformation_repo, default_prompts_repo):
        transformation_repo.delete.return_value = True
        service = TransformationService(transformation_repo, default_prompts_repo)

        assert await service.delete("transformation:1") is True


class TestTransformationServiceDefaultPrompts:

    @pytest.mark.asyncio
    async def test_get_default_prompts(self, transformation_repo, default_prompts_repo):
        dp = make_default_prompts()
        default_prompts_repo.get.return_value = dp
        service = TransformationService(transformation_repo, default_prompts_repo)

        result = await service.get_default_prompts()

        assert result.transformation_instructions == "Default instructions"
