"""The server-wide limit on every-frame decodes, and the stream clock it resets."""

import asyncio
import time

import pytest

from lib.model.dense_frames import DenseFrameStream
from lib.model.video_preprocessor import DenseDecodeSlots


@pytest.mark.asyncio
async def test_a_second_every_frame_decode_waits_for_the_first():
    slots = DenseDecodeSlots(1)
    waits = []
    release_first = await slots.acquire(waits.append)

    second = asyncio.ensure_future(slots.acquire(waits.append))
    await asyncio.sleep(0.05)
    assert not second.done()
    assert waits == [1]

    release_first()
    release_second = await asyncio.wait_for(second, 1)
    assert slots.running == 1
    release_second()
    assert slots.running == 0


@pytest.mark.asyncio
async def test_waiting_decodes_get_their_turn_in_arrival_order():
    """The shot-boundary model analyses one request at a time, in arrival
    order, so decodes must start in that order too."""
    slots = DenseDecodeSlots(1)
    release = await slots.acquire()
    started = []

    async def decode(name):
        release_slot = await slots.acquire()
        started.append(name)
        release_slot()

    waiting = [asyncio.ensure_future(decode(name)) for name in ("b", "c", "d")]
    await asyncio.sleep(0.01)
    release()
    await asyncio.wait_for(asyncio.gather(*waiting), 1)

    assert started == ["b", "c", "d"]


@pytest.mark.asyncio
async def test_freeing_a_slot_twice_frees_it_once():
    slots = DenseDecodeSlots(1)
    release = await slots.acquire()
    release()
    release()

    held = await slots.acquire()
    assert slots.running == 1
    blocked = asyncio.ensure_future(slots.acquire())
    await asyncio.sleep(0.05)
    assert not blocked.done()
    held()
    (await asyncio.wait_for(blocked, 1))()


@pytest.mark.asyncio
async def test_a_limit_of_zero_does_not_limit():
    slots = DenseDecodeSlots(0)
    releases = [await asyncio.wait_for(slots.acquire(), 1) for _ in range(3)]

    assert slots.running == 3
    for release in releases:
        release()


def test_a_stream_that_waited_for_its_decode_still_gives_its_consumer_time_to_attach():
    """A stream is created before its decode waits for a slot. Its consumer's
    attach timeout runs from when the decode starts, not from then."""
    stream = DenseFrameStream(4, 4, maxsize=1, attach_timeout=0.2)
    stream._created = time.monotonic() - 60  # as if it waited a minute for a slot
    stream.start_producing()

    assert stream.put(0, "first clip")  # fills the queue
    started = time.monotonic()
    assert not stream.put(0, "second clip")  # nobody attached: abandoned after the timeout
    assert time.monotonic() - started >= 0.15
