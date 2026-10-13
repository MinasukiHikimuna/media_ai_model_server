"""Keep the server's host memory bounded and give it back when it is freed.

``HostMemoryBudget`` bounds the bytes of decoded frames held anywhere in the
server: queued behind inference, in flight, or waiting for a batch to finish.
``trim_heap`` makes glibc return freed frame buffers to the system;
otherwise a server that once held 20 GB keeps most of it while idle.
"""

import ctypes
import ctypes.util
import logging
import threading
import time

logger = logging.getLogger("logger")

_libc = None
_libc_loaded = False


def _glibc():
    global _libc, _libc_loaded
    if not _libc_loaded:
        _libc_loaded = True
        name = ctypes.util.find_library("c")
        try:
            libc = ctypes.CDLL(name) if name else None
            if libc is not None and hasattr(libc, "malloc_trim"):
                _libc = libc
        except OSError:
            _libc = None
    return _libc


def trim_heap():
    """Return free heap memory to the system. Returns whether any was."""
    libc = _glibc()
    if libc is None:
        return False
    return bool(libc.malloc_trim(0))


class HostMemoryBudget:
    """Server-wide bound on the bytes of decoded frames.

    A decode takes a lease for each batch before handing its frames on and
    blocks while the budget is spent; the lease is released once every frame
    of the batch is done. One batch is always let through, so a batch larger
    than the budget cannot stall the server. A limit of 0 or less means no limit.
    """

    def __init__(self, limit_bytes):
        self.limit = int(limit_bytes)
        self.in_use = 0
        self._cond = threading.Condition()

    def acquire(self, nbytes, frames, stop=None):
        """Block until ``nbytes`` fit, then return a lease for ``frames`` frames.
        Returns None if ``stop`` (a threading.Event) is set while waiting."""
        nbytes = max(0, int(nbytes))
        with self._cond:
            while self.limit > 0 and self.in_use > 0 and self.in_use + nbytes > self.limit:
                if stop is not None and stop.is_set():
                    return None
                self._cond.wait(timeout=0.2)
            self.in_use += nbytes
        return BatchLease(self, nbytes, frames)

    def release(self, nbytes):
        with self._cond:
            self.in_use = max(0, self.in_use - nbytes)
            self._cond.notify_all()


class BatchLease:
    """The budget held by one batch of frames, each owning an equal share.

    The decode calls ``deliver`` for each frame it hands on and ``finish`` when
    that frame is done, which gives back the frame's share. ``seal`` says no
    more frames will be handed on, so frames never delivered (the decode
    stopped early) stop counting. ``release`` gives back everything left, for
    a request that ended anyway; later calls do nothing.
    """

    def __init__(self, budget, nbytes, frames):
        self._budget = budget
        self.nbytes = nbytes
        self.frames = max(1, int(frames))
        self.delivered = 0
        self.finished = 0
        self.sealed = False
        self.returned = 0

    @property
    def released(self):
        return self.returned >= self.nbytes

    def deliver(self):
        self.delivered += 1

    def finish(self):
        self.finished += 1
        self._return_up_to(self.finished)

    def seal(self):
        self.sealed = True
        self._return_up_to(self.frames - (self.delivered - self.finished))

    def release(self):
        self._return_up_to(self.frames)

    def _return_up_to(self, frames_done):
        """Return the shares of ``frames_done`` frames in all."""
        if self.sealed:
            frames_done = max(frames_done, self.frames - (self.delivered - self.finished))
        target = self.nbytes if frames_done >= self.frames else self.nbytes * frames_done // self.frames
        if target > self.returned:
            self._budget.release(target - self.returned)
            self.returned = target


_budget = None


def host_memory_budget(limit_gb):
    """The budget every video preprocessor of this server shares; the first
    preprocessor's configured limit applies."""
    global _budget
    if _budget is None:
        _budget = HostMemoryBudget(int(float(limit_gb or 0) * 1024 ** 3))
        if _budget.limit > 0:
            logger.info("Decoded frames may hold %.1f GB of host memory across the server", float(limit_gb))
    return _budget


class HeapTrimmer:
    """Trims the heap after a request, at most once per ``min_interval`` seconds,
    so memory a finished video freed goes back to the system."""

    def __init__(self, min_interval=10.0):
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def maybe_trim(self):
        now = time.monotonic()
        with self._lock:
            if now - self._last < self.min_interval:
                return False
            self._last = now
        return trim_heap()
