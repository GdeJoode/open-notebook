"""WB.1: the Werkbak's shared read, without a database.

Groups share one read of the notebooks; each group runs under its own
deadline. When one group is cancelled (its deadline passed) while another
still waits, the read must go on for the other. The DB test times all
groups out in one tick and cannot separate the two; this pins the guard.
"""

import asyncio

import pytest

from app_main.services.attention_service import _shared

pytestmark = pytest.mark.asyncio


async def test_a_cancelled_waiter_does_not_cancel_the_shared_read() -> None:
    read = asyncio.ensure_future(asyncio.sleep(0.05, result="notebooks"))
    first = asyncio.ensure_future(_shared(read))
    second = asyncio.ensure_future(_shared(read))
    await asyncio.sleep(0)
    first.cancel()
    assert await second == "notebooks"
    assert first.cancelled() and not read.cancelled()
