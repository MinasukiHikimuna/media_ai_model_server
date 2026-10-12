"""Every-frame decode on NVDEC beside the CPU interval decode.

The threading and failure handling run against fake sources without a GPU;
one test decodes a real clip with NVDEC when the host has one.
"""

import shutil
import subprocess
import threading

import numpy as np
import pytest

from lib.model.dense_frames import DenseFrameStream, DenseStreamError
from lib.model.preprocessing_python import ffmpeg_pipe
from lib.model.preprocessing_python import image_preprocessing as ip

SIZE = (32, 18)


class _Source:
    """Stands in for an opened ffmpeg_pipe.FrameSource."""

    backend_name = "ffmpeg_cuda"

    def __init__(self, frames, fail_after=None):
        self.width, self.height = SIZE
        self._frames = frames
        self._fail_after = fail_after
        self.frames_emitted = 0
        self.closed = False

    def __iter__(self):
        try:
            for index in range(self._frames):
                if self._fail_after is not None and index == self._fail_after:
                    raise ffmpeg_pipe.DecodeStreamError("ffmpeg exited 1: broken")
                self.frames_emitted = index + 1
                yield index, np.full((SIZE[1], SIZE[0], 3), index % 256, dtype=np.uint8)
        finally:
            self.closed = True


def _read(stream):
    """Read a stream to its end on a thread; returns (frames, errors)."""
    received, errors = [], []

    def consume():
        try:
            for segment, frames in stream:
                received.append((segment, len(frames)))
        except DenseStreamError as exc:
            errors.append(exc)

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    return received, errors, thread


def _run(sources, streams_per_size, interval=(), emit_interval=True):
    sinks = [(SIZE, streams) for streams in streams_per_size]
    return list(ip._preprocess_with_gpu_dense(
        "clip.mp4", sources, sinks, emit_interval, lambda: iter(interval), lambda frame: frame * 2))


def test_every_frame_arrives_as_one_segment_beside_the_interval_frames():
    stream = DenseFrameStream(*SIZE)
    received, errors, reader = _read(stream)

    out = _run([_Source(250)], [[stream]], interval=[(0.0, 1), (2.0, 2)])
    reader.join(timeout=10)

    assert out == [(0.0, 2), (2.0, 4)]
    assert errors == []
    assert received == [(0, 100), (0, 100), (0, 50)]
    assert stream.segment_counts == [250]
    assert stream.backend_name == "ffmpeg_cuda"


def test_a_dense_only_decode_yields_nothing_and_still_delivers_every_frame():
    stream = DenseFrameStream(*SIZE)
    received, errors, reader = _read(stream)

    out = _run([_Source(30)], [[stream]], interval=[(0.0, 1)], emit_interval=False)
    reader.join(timeout=10)

    assert out == []
    assert received == [(0, 30)]
    assert stream.segment_counts == [30]


def test_a_failed_gpu_decode_fails_the_stream_and_the_interval_frames_continue():
    stream = DenseFrameStream(*SIZE)
    received, errors, reader = _read(stream)
    source = _Source(250, fail_after=120)

    out = _run([source], [[stream]], interval=[(0.0, 1), (2.0, 2)])
    reader.join(timeout=10)

    assert out == [(0.0, 2), (2.0, 4)]
    assert len(errors) == 1 and "cannot decode every frame" in str(errors[0])
    assert stream.segment_counts is None
    assert source.closed


def test_stopping_the_interval_frames_early_stops_the_gpu_decode():
    stream = DenseFrameStream(*SIZE)
    received, errors, reader = _read(stream)
    source = _Source(10_000_000)

    frames = ip._preprocess_with_gpu_dense(
        "clip.mp4", [source], [(SIZE, [stream])], True,
        lambda: iter([(0.0, 1), (2.0, 2)]), lambda frame: frame)
    next(frames)
    frames.close()
    reader.join(timeout=10)

    assert source.closed
    assert len(errors) == 1
    assert source.frames_emitted < 10_000_000


def test_a_reader_that_stops_reading_stops_the_gpu_decode():
    stream = DenseFrameStream(*SIZE)
    stream.abandon()
    source = _Source(10_000_000)

    assert _run([source], [[stream]]) == []
    assert source.closed and source.frames_emitted <= 100


def test_the_cpu_decodes_every_frame_unless_nvdec_is_asked_for(monkeypatch):
    opened = []
    monkeypatch.setattr(ffmpeg_pipe, "probe_video", lambda path: "info")
    monkeypatch.setattr(ffmpeg_pipe, "make_video_frame_source",
                        lambda *args, **kwargs: opened.append(kwargs) or _Source(1))
    sinks = [(SIZE, [object()])]

    assert ip._open_gpu_dense_sources("clip.mp4", sinks, None, "pyav", "exact") is None
    assert ip._open_gpu_dense_sources("clip.mp4", [], None, "ffmpeg_cuda", "exact") is None
    assert opened == []

    sources = ip._open_gpu_dense_sources("clip.mp4", sinks, None, "ffmpeg_cuda", "exact")
    assert len(sources) == 1
    assert opened[0]["backend"] == "ffmpeg_cuda" and opened[0]["fallback"] is False


def test_a_cropped_source_is_decoded_on_the_cpu(monkeypatch):
    monkeypatch.setattr(ffmpeg_pipe, "make_video_frame_source",
                        lambda *args, **kwargs: pytest.fail("NVDEC was tried for a cropped source"))

    assert ip._open_gpu_dense_sources(
        "clip.mp4", [(SIZE, [object()])], (0, 0, 10, 10), "ffmpeg_cuda", "exact") is None


def test_a_source_nvdec_cannot_take_is_decoded_on_the_cpu(monkeypatch):
    def refuse(*args, **kwargs):
        raise ffmpeg_pipe.DecodeSetupError("pixel format yuv444p is not supported by NVDEC")

    monkeypatch.setattr(ffmpeg_pipe, "probe_video", lambda path: "info")
    monkeypatch.setattr(ffmpeg_pipe, "make_video_frame_source", refuse)

    assert ip._open_gpu_dense_sources("clip.mp4", [(SIZE, [object()])], None, "ffmpeg_cuda", "exact") is None


def test_without_fallback_an_unusable_backend_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(ffmpeg_pipe, "ffmpeg_features", lambda refresh=False: None)
    info = ffmpeg_pipe.VideoInfo(64, 36, "yuv420p", "h264", 30.0, 1.0, 30, "progressive", None, None)

    with pytest.raises(ffmpeg_pipe.DecodeSetupError, match="not found"):
        ffmpeg_pipe.make_video_frame_source(
            tmp_path / "clip.mp4", decode_size=SIZE, backend="ffmpeg_cuda", info=info, fallback=False)


def _has_nvdec():
    if shutil.which("ffmpeg") is None or shutil.which("nvidia-smi") is None:
        return False
    features = ffmpeg_pipe.ffmpeg_features()
    return bool(features and features.has_cuda and features.has_scale_cuda)


@pytest.mark.skipif(not _has_nvdec(), reason="needs ffmpeg with CUDA and an NVIDIA GPU")
def test_nvdec_delivers_the_same_frames_as_the_cpu_decode(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30", "-t", "4",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-g", "30", "-bf", "2", str(clip)],
        check=True,
    )

    def decode(backend):
        stream = DenseFrameStream(*SIZE)
        received = []

        def consume():
            for segment, frames in stream:
                received.append((segment, frames))

        reader = threading.Thread(target=consume, daemon=True)
        reader.start()
        interval = list(ip.preprocess_video_mp_pyav(
            str(clip), 0.5, 0, False, "cpu", True, norm_config=-1, decode_workers=2,
            dense_sinks=[(SIZE, [stream])], dense_backend=backend))
        reader.join(timeout=30)
        # The CPU decode's segments interleave; each arrives in order.
        ordered = [frames for _segment, frames in sorted(received, key=lambda item: item[0])]
        return stream, np.concatenate(ordered), sorted(target for target, _frame in interval)

    gpu_stream, gpu_frames, gpu_targets = decode("ffmpeg_cuda")
    cpu_stream, cpu_frames, cpu_targets = decode("pyav")

    assert gpu_stream.backend_name == "ffmpeg_cuda"
    assert gpu_stream.segment_counts == [sum(cpu_stream.segment_counts)] == [120]
    assert gpu_targets == cpu_targets
    assert np.abs(gpu_frames.astype(np.int16) - cpu_frames.astype(np.int16)).mean() < 1.0
