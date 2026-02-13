"""Tests for the in-memory priority queue."""

import asyncio

import pytest

from shared.types.enums import JobPriority

from job_queue.queue import PRIORITY_MAP, JobQueue


class TestPriorityMap:
    """Test priority numeric mapping."""

    def test_critical_is_highest(self):
        assert PRIORITY_MAP[JobPriority.CRITICAL] < PRIORITY_MAP[JobPriority.HIGH]

    def test_ordering(self):
        assert (
            PRIORITY_MAP[JobPriority.CRITICAL]
            < PRIORITY_MAP[JobPriority.HIGH]
            < PRIORITY_MAP[JobPriority.NORMAL]
            < PRIORITY_MAP[JobPriority.LOW]
        )


class TestJobQueue:
    """Test JobQueue enqueue/dequeue behaviour."""

    @pytest.fixture
    def queue(self):
        return JobQueue(max_size=10)

    @pytest.mark.asyncio
    async def test_enqueue_dequeue(self, queue):
        await queue.enqueue("job:1")
        assert queue.size() == 1
        job_id = await queue.dequeue()
        assert job_id == "job:1"
        assert queue.is_empty()

    @pytest.mark.asyncio
    async def test_priority_ordering(self, queue):
        """Higher-priority jobs should dequeue first."""
        await queue.enqueue("low", JobPriority.LOW)
        await queue.enqueue("critical", JobPriority.CRITICAL)
        await queue.enqueue("normal", JobPriority.NORMAL)

        first = await queue.dequeue()
        second = await queue.dequeue()
        third = await queue.dequeue()

        assert first == "critical"
        assert second == "normal"
        assert third == "low"

    @pytest.mark.asyncio
    async def test_fifo_within_same_priority(self, queue):
        """Jobs with the same priority dequeue in FIFO order."""
        await queue.enqueue("a", JobPriority.NORMAL)
        await queue.enqueue("b", JobPriority.NORMAL)
        await queue.enqueue("c", JobPriority.NORMAL)

        assert await queue.dequeue() == "a"
        assert await queue.dequeue() == "b"
        assert await queue.dequeue() == "c"

    @pytest.mark.asyncio
    async def test_dequeue_blocks_until_item(self, queue):
        """Dequeue should block until an item is available."""
        result = []

        async def producer():
            await asyncio.sleep(0.05)
            await queue.enqueue("delayed")

        async def consumer():
            job_id = await queue.dequeue()
            result.append(job_id)

        await asyncio.gather(producer(), consumer())
        assert result == ["delayed"]

    @pytest.mark.asyncio
    async def test_is_empty(self, queue):
        assert queue.is_empty()
        await queue.enqueue("job:1")
        assert not queue.is_empty()
        await queue.dequeue()
        assert queue.is_empty()

    @pytest.mark.asyncio
    async def test_size(self, queue):
        assert queue.size() == 0
        await queue.enqueue("a")
        await queue.enqueue("b")
        assert queue.size() == 2
        await queue.dequeue()
        assert queue.size() == 1
