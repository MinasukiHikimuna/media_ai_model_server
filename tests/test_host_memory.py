"""The server-wide bound on decoded frames' host memory, and the video
preprocessor's use of it."""

import asyncio
import threading

import pytest
import torch

from lib.async_lib.async_processing import ItemFuture, QueueItem
from lib.model import video_preprocessor as vp
from lib.pipeline.preprocess_spec import PreprocessSpec
from lib.utils.host_memory import HeapTrimmer, HostMemoryBudget


def _in_thread(target):
    result = []
    thread = threading.Thread(target=lambda: result.append(target()), daemon=True)
    thread.start()
    return result, thread


def test_a_batch_waits_until_the_budget_has_room():
    budget = HostMemoryBudget(100)
    first = budget.acquire(60, 1)
    result, thread = _in_thread(lambda: budget.acquire(60, 1))
    thread.join(timeout=0.5)
    assert thread.is_alive()

    first.finish()
    thread.join(timeout=2)
    assert not thread.is_alive() and result[0] is not None
    assert budget.in_use == 60


def test_one_batch_larger_than_the_budget_still_goes_through():
    budget = HostMemoryBudget(100)
    lease = budget.acquire(500, 1)
    assert budget.in_use == 500
    lease.finish()
    assert budget.in_use == 0


def test_a_stopped_wait_gives_up():
    budget = HostMemoryBudget(100)
    budget.acquire(100, 1)
    stop = threading.Event()
    result, thread = _in_thread(lambda: budget.acquire(10, 1, stop))
    stop.set()
    thread.join(timeout=2)
    assert result == [None] and budget.in_use == 100


def test_a_lease_is_released_once_when_its_last_frame_is_done():
    budget = HostMemoryBudget(1000)
    lease = budget.acquire(300, 3)
    for _ in range(3):
        lease.deliver()
    lease.finish()
    lease.finish()
    assert budget.in_use == 300
    lease.finish()
    lease.finish()
    lease.release()
    assert budget.in_use == 0


def test_frames_never_handed_on_stop_counting_when_the_lease_is_sealed():
    budget = HostMemoryBudget(1000)
    lease = budget.acquire(300, 3)
    lease.deliver()
    lease.seal()
    assert budget.in_use == 300
    lease.finish()
    assert budget.in_use == 0


def test_no_limit_never_waits():
    budget = HostMemoryBudget(0)
    budget.acquire(10 ** 12, 1)
    assert budget.acquire(10 ** 12, 1) is not None


def test_the_heap_is_trimmed_at_most_once_per_interval(monkeypatch):
    calls = []
    monkeypatch.setattr("lib.utils.host_memory.trim_heap", lambda: calls.append(1) or True)
    trimmer = HeapTrimmer(min_interval=60)
    assert trimmer.maybe_trim()
    assert not trimmer.maybe_trim()
    assert calls == [1]


# ---- the video preprocessor -------------------------------------------------

SIZE = (64, 36)
BATCH = 8
INPUTS = ["video", "use_timestamps", "interval", "threshold", "return_confidence", "vr", "skip", "requested"]
OUTPUTS = ["children", "frame_index", "threshold", "return_confidence", "skipped", "requested", "frame"]


class _Source:
    """Stands in for a decode: yields CHW float frames, records its closing."""

    def __init__(self, frames):
        self.frames = frames
        self.emitted = 0
        self.closed_by = None

    def __iter__(self):
        return self._generate()

    def _generate(self):
        for index in range(self.frames):
            self.emitted = index + 1
            yield index, torch.full((3, SIZE[1], SIZE[0]), float(index % 255))

    def close(self):
        self.closed_by = threading.current_thread().name


def _model(monkeypatch, source, budget):
    model = vp.VideoPreprocessorModel({"preprocess_backend": "av", "preprocess_batch_size": BATCH})
    model.specs = [PreprocessSpec(width=32, height=18)]
    model._host_memory_budget = budget
    model._preprocess_callable = lambda *_args, **_kwargs: source
    monkeypatch.setattr(vp, "probe_video_dimensions", lambda _path: SIZE)
    return model


async def _request(handler):
    data = dict(zip(INPUTS, ["clip.mp4", True, 2.0, 0.5, False, False, None, None]))
    root = await ItemFuture.create(None, data, handler)
    return root, QueueItem(root, INPUTS, OUTPUTS)


def _batch_bytes():
    return BATCH * 3 * 18 * 32 * 4


@pytest.mark.asyncio
async def test_decoded_frames_never_hold_more_than_the_budget(monkeypatch):
    budget = HostMemoryBudget(int(_batch_bytes() * 2.5))
    peak = [0]
    acquire = budget.acquire

    def tracked(*args, **kwargs):
        lease = acquire(*args, **kwargs)
        peak[0] = max(peak[0], budget.in_use)
        return lease

    budget.acquire = tracked
    source = _Source(100)
    model = _model(monkeypatch, source, budget)

    async def handler(future, key):
        # Inference stand-in: a frame is done a little after it arrives.
        if key == "frame":
            asyncio.get_running_loop().call_later(0.005, future.close_future, None)

    root, item = await _request(handler)
    await model.worker_function([item])

    children = root["children"]
    assert len(children) == 100
    await asyncio.gather(*(child.future for child in children))
    assert 0 < peak[0] <= _batch_bytes() * 2.5
    assert budget.in_use == 0
    assert source.closed_by == "video-preprocess-producer"


@pytest.mark.asyncio
async def test_a_failed_request_stops_the_decode_and_gives_its_memory_back(monkeypatch):
    budget = HostMemoryBudget(_batch_bytes() * 4)
    source = _Source(1_000_000)
    model = _model(monkeypatch, source, budget)
    seen = [0]

    async def handler(future, key):
        if key == "frame":
            seen[0] += 1
            if seen[0] == 20:
                raise RuntimeError("inference failed")

    root, item = await _request(handler)
    await model.worker_function([item])

    with pytest.raises(RuntimeError, match="inference failed"):
        await root.future
    await asyncio.sleep(0)  # the root's done callbacks
    assert source.closed_by == "video-preprocess-producer"
    assert source.emitted < 1_000_000
    assert budget.in_use == 0
    assert not any(t.name == "video-preprocess-producer" for t in threading.enumerate())
