"""The per-video cap on in-flight preprocessed frames."""

from lib.model import video_preprocessor as vp

GB = 1024 ** 3


def _total_ram(monkeypatch, gb):
    import psutil

    class _Memory:
        total = gb * GB

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _Memory())


def test_a_byte_budget_bounds_a_pinned_frame_count(monkeypatch):
    _total_ram(monkeypatch, 62)
    # 1080p float32 frames: 19200 of them would be about 470 GB.
    assert vp.compute_pending_frames_cap(23.7, 19200, 4, 0.6, 3) == 172


def test_a_byte_budget_bounds_the_auto_cap(monkeypatch):
    _total_ram(monkeypatch, 62)
    auto = vp.compute_auto_pending_frames(23.7, 0.6, 3)
    assert vp.compute_pending_frames_cap(23.7, 0, 4, 0.6, 3) == 172 < auto


def test_small_frames_keep_the_frame_cap(monkeypatch):
    _total_ram(monkeypatch, 62)
    assert vp.compute_pending_frames_cap(0.6, 500, 4, 0.6, 3) == 500


def test_without_a_byte_budget_the_frame_cap_stands(monkeypatch):
    _total_ram(monkeypatch, 62)
    assert vp.compute_pending_frames_cap(23.7, 19200, 0, 0.6, 3) == 19200
    assert vp.compute_pending_frames_cap(23.7, 0, 0, 0.6, 3) == vp.compute_auto_pending_frames(23.7, 0.6, 3)


def test_huge_frames_still_leave_a_few_in_flight(monkeypatch):
    _total_ram(monkeypatch, 62)
    assert vp.compute_pending_frames_cap(1000.0, 0, 4, 0.6, 3) == 8
