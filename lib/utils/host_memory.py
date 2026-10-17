"""Give freed host memory back to the system.

``trim_heap`` makes glibc return freed frame buffers to the system;
otherwise a server that once held 20 GB keeps most of it while idle.
It does nothing where there is no glibc ``malloc_trim``, such as Windows.
"""

import ctypes
import ctypes.util
import threading
import time

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
