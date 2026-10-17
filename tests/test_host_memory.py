"""Host memory: the heap trimmer, the video preprocessor stopping its decode,
and the 8-bit region source."""

import asyncio
import threading

import pytest
import torch

from lib.async_lib.async_processing import ItemFuture, QueueItem
from lib.model import video_preprocessor as vp
from lib.pipeline.preprocess_spec import PreprocessSpec
from lib.utils.host_memory import HeapTrimmer


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


def _model(monkeypatch, source):
    model = vp.VideoPreprocessorModel({"preprocess_backend": "av", "preprocess_batch_size": BATCH})
    model.specs = [PreprocessSpec(width=32, height=18)]
    model._preprocess_callable = lambda *_args, **_kwargs: source
    monkeypatch.setattr(vp, "probe_video_dimensions", lambda _path: SIZE)
    return model


async def _request(handler):
    data = dict(zip(INPUTS, ["clip.mp4", True, 2.0, 0.5, False, False, None, None]))
    root = await ItemFuture.create(None, data, handler)
    return root, QueueItem(root, INPUTS, OUTPUTS)


@pytest.mark.asyncio
async def test_a_finished_decode_is_closed_by_its_own_thread(monkeypatch):
    source = _Source(100)
    model = _model(monkeypatch, source)

    async def handler(future, key):
        if key == "frame":
            asyncio.get_running_loop().call_later(0.005, future.close_future, None)

    root, item = await _request(handler)
    await model.worker_function([item])

    children = root["children"]
    assert len(children) == 100
    await asyncio.gather(*(child.future for child in children))
    assert source.closed_by == "video-preprocess-producer"


@pytest.mark.asyncio
async def test_a_failed_request_stops_the_decode(monkeypatch):
    source = _Source(1_000_000)
    model = _model(monkeypatch, source)
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
    assert source.closed_by == "video-preprocess-producer"
    assert source.emitted < 1_000_000
    assert not any(t.name == "video-preprocess-producer" for t in threading.enumerate())


# ---- the 8-bit region source --------------------------------------------------

def test_the_region_source_is_kept_as_8_bit_rounded():
    from lib.pipeline.preprocess_spec import apply_spec, apply_spec_batch

    spec = PreprocessSpec.for_region_source(max_long_edge=64)
    frame = torch.full((3, 36, 64), 100.6)
    frame[0, 0, 0] = 300.0

    single = apply_spec(frame, spec)
    batch = apply_spec_batch(frame.unsqueeze(0), spec)
    for out in (single, batch[0]):
        assert out.dtype == torch.uint8
        assert out[1, 1, 1] == 101 and out[0, 0, 0] == 255
    assert spec.key.endswith("_u8_max64")


def test_a_region_cut_from_an_8_bit_source_is_float():
    from lib.model.python_functions import _crop_and_resize_region

    source = torch.full((3, 36, 64), 200, dtype=torch.uint8)
    crop = _crop_and_resize_region(source, [4, 2, 20, 12])
    assert crop.dtype == torch.float32 and crop.shape == (3, 10, 16)
    assert float(crop.max()) == 200.0
    crop += 1
    assert int(source.max()) == 200


def _photo_like(width, height):
    """An 8-bit frame as decode produces it (float32 holding whole values):
    smooth gradients plus fine noise, so a resize lands between levels."""
    generator = torch.Generator().manual_seed(7)
    y = torch.linspace(0, 1, height).view(1, height, 1)
    x = torch.linspace(0, 1, width).view(1, 1, width)
    base = 255 * (0.5 + 0.25 * torch.sin(9 * x + torch.tensor([0.0, 1.0, 2.0]).view(3, 1, 1)) * torch.cos(7 * y))
    noise = torch.randint(-12, 13, (3, height, width), generator=generator).float()
    return (base + noise).round().clamp(0, 255)


@pytest.mark.parametrize("width, height, max_diff", [
    (1920, 1080, 0.0),   # kept at full size: the 8-bit source is the decoded frame itself
    (1280, 720, 0.0),
    (3840, 2160, 0.5),   # downscaled to 1920: each value rounds to the nearest level
])
def test_an_aligned_face_from_the_8_bit_source_matches_the_float_one(width, height, max_diff):
    from dataclasses import replace

    import numpy as np

    from lib.model.ai_face_embedding_model import _norm_crop_gpu
    from lib.pipeline.preprocess_spec import apply_spec

    frame = _photo_like(width, height)
    spec = PreprocessSpec.for_region_source()
    as_8_bit = apply_spec(frame, spec)
    as_float = apply_spec(frame, replace(spec, precision="float32"))
    assert as_8_bit.dtype == torch.uint8 and as_float.dtype == torch.float32

    # Five landmarks of a face filling about a tenth of the source's width.
    w = as_float.shape[-1]
    kps = np.array([[0.45, 0.40], [0.55, 0.40], [0.50, 0.46], [0.46, 0.52], [0.54, 0.52]], dtype=np.float32)
    kps[:, 0] *= w
    kps[:, 1] *= as_float.shape[-2]
    kps = (kps - kps.mean(0)) * 0.5 + kps.mean(0)

    assert float((as_8_bit.float() - as_float).abs().max()) <= max_diff
    aligned_8_bit = _norm_crop_gpu(as_8_bit, kps)
    aligned_float = _norm_crop_gpu(as_float, kps)
    assert float((aligned_8_bit - aligned_float).abs().max()) <= max_diff + 1e-4
