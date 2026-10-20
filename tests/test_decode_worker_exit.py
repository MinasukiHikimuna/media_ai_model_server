import os

import pytest

from lib.model.preprocessing_python import mp_decode


def exit_without_sentinel(*_args):
    os._exit(17)


def exit_cleanly_without_sentinel(*_args):
    os._exit(0)


@pytest.mark.parametrize("target", [exit_without_sentinel, exit_cleanly_without_sentinel])
def test_native_worker_exit_fails_instead_of_waiting_forever(monkeypatch, target):
    monkeypatch.setattr(mp_decode, "probe_duration_seconds", lambda _: 2)
    monkeypatch.setattr(mp_decode, "_decode_chunk_worker", target)
    with pytest.raises(RuntimeError, match="exited without"):
        list(mp_decode.iter_parallel_frames("synthetic.mp4", 1, 0, 1, True))
