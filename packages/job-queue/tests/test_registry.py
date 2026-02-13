"""Tests for the handler registry."""

import pytest

from shared.types.enums import JobType

from job_queue.registry import HandlerRegistry


class TestHandlerRegistry:
    """Test handler registration and execution."""

    @pytest.fixture
    def registry(self):
        return HandlerRegistry()

    def test_register_decorator(self, registry):
        @registry.register(JobType.DOCUMENT_PARSE)
        async def handler(payload):
            return {"ok": True}

        assert registry.has_handler(JobType.DOCUMENT_PARSE)
        assert JobType.DOCUMENT_PARSE in registry.registered_types

    def test_register_handler_imperative(self, registry):
        async def handler(payload):
            return {"ok": True}

        registry.register_handler(JobType.AUDIO_TRANSCRIBE, handler)
        assert registry.has_handler(JobType.AUDIO_TRANSCRIBE)

    def test_has_handler_false(self, registry):
        assert not registry.has_handler(JobType.DOCUMENT_PARSE)

    def test_registered_types_empty(self, registry):
        assert registry.registered_types == []

    def test_registered_types(self, registry):
        @registry.register(JobType.DOCUMENT_PARSE)
        async def h1(payload):
            return {}

        @registry.register(JobType.BATCH_PROCESS)
        async def h2(payload):
            return {}

        types = registry.registered_types
        assert JobType.DOCUMENT_PARSE in types
        assert JobType.BATCH_PROCESS in types
        assert len(types) == 2

    @pytest.mark.asyncio
    async def test_execute_success(self, registry):
        @registry.register(JobType.CHUNK_EXTRACT)
        async def handler(payload):
            return {"chunks": payload.get("count", 0)}

        result = await registry.execute(JobType.CHUNK_EXTRACT, {"count": 5})
        assert result == {"chunks": 5}

    @pytest.mark.asyncio
    async def test_execute_no_handler_raises(self, registry):
        with pytest.raises(KeyError, match="No handler registered"):
            await registry.execute(JobType.DOCUMENT_PARSE, {})

    @pytest.mark.asyncio
    async def test_execute_handler_exception_propagates(self, registry):
        @registry.register(JobType.DOCUMENT_PARSE)
        async def bad_handler(payload):
            raise ValueError("something went wrong")

        with pytest.raises(ValueError, match="something went wrong"):
            await registry.execute(JobType.DOCUMENT_PARSE, {})

    def test_overwrite_warning(self, registry):
        """Registering the same job type twice should overwrite."""

        @registry.register(JobType.DOCUMENT_PARSE)
        async def first(payload):
            return {"v": 1}

        @registry.register(JobType.DOCUMENT_PARSE)
        async def second(payload):
            return {"v": 2}

        # The second handler should be active
        assert registry.has_handler(JobType.DOCUMENT_PARSE)
