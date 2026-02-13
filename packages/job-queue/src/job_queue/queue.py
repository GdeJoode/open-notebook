"""
In-memory priority queue backed by asyncio.PriorityQueue.

Jobs are enqueued by ID with a priority value. Lower numeric priority
means higher importance (CRITICAL=0, HIGH=1, NORMAL=2, LOW=3).
"""

import asyncio
from dataclasses import dataclass, field

from shared.types.enums import JobPriority

# Lower value = higher priority for PriorityQueue
PRIORITY_MAP: dict[JobPriority, int] = {
    JobPriority.CRITICAL: 0,
    JobPriority.HIGH: 1,
    JobPriority.NORMAL: 2,
    JobPriority.LOW: 3,
}

_counter: int = 0


@dataclass(order=True)
class _QueueItem:
    """Sortable wrapper for priority queue entries."""

    priority: int
    sequence: int = field(compare=True)
    job_id: str = field(compare=False)


class JobQueue:
    """
    Async priority queue for job IDs.

    Wraps asyncio.PriorityQueue so that `dequeue` blocks until
    a job is available, making it ideal for a worker loop.
    """

    def __init__(self, max_size: int = 100):
        self._queue: asyncio.PriorityQueue[_QueueItem] = asyncio.PriorityQueue(
            maxsize=max_size,
        )
        self._counter = 0

    async def enqueue(self, job_id: str, priority: JobPriority = JobPriority.NORMAL) -> None:
        """Add a job ID to the queue with the given priority."""
        numeric = PRIORITY_MAP.get(priority, 2)
        item = _QueueItem(priority=numeric, sequence=self._counter, job_id=job_id)
        self._counter += 1
        await self._queue.put(item)

    async def dequeue(self) -> str:
        """Block until a job is available, then return its ID."""
        item = await self._queue.get()
        return item.job_id

    def size(self) -> int:
        """Current number of items in the queue."""
        return self._queue.qsize()

    def is_empty(self) -> bool:
        """Whether the queue is empty."""
        return self._queue.empty()
