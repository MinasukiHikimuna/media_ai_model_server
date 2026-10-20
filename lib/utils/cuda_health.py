"""A fatal CUDA execution error poisons the process until it is restarted."""
import logging
import threading
import weakref


class CudaUnavailableError(RuntimeError):
    pass


def is_fatal_cuda_error(exception):
    # OOM is recoverable; illegal addresses/assertions/launch failures are not.
    message = str(exception).lower()
    return "cuda" in message and any(text in message for text in (
        "illegal memory access", "illegal address", "device-side assert",
        "unspecified launch failure", "misaligned address",
        "launch timed out", "launch timeout",
    ))


class CudaHealth:
    def __init__(self):
        self._lock = threading.Lock()
        self._failure = None
        self._items = weakref.WeakSet()

    @property
    def failed(self):
        with self._lock:
            return self._failure is not None

    def raise_if_failed(self):
        if self.failed:
            raise CudaUnavailableError("CUDA execution failed; restart the server before submitting more work")

    def track(self, item):
        with self._lock:
            if self._failure is not None:
                raise CudaUnavailableError("CUDA execution failed; restart the server before submitting more work")
            self._items.add(item)

    def record(self, exception):
        if not is_fatal_cuda_error(exception):
            return
        with self._lock:
            if self._failure is not None:
                return
            self._failure = str(exception)
            items = list(self._items)
        logging.getLogger("logger").critical(
            "Fatal CUDA error; rejecting further work until process restart: %s", exception,
            exc_info=(type(exception), exception, exception.__traceback__))
        # A producer can report a failure from a native-worker thread. Asyncio
        # futures and their semaphore-release callbacks belong to their loop.
        for item in items:
            loop = item.future.get_loop()
            try:
                loop.call_soon_threadsafe(self._fail_item, item)
            except RuntimeError:  # its request loop has already closed
                pass

    def _fail_item(self, item):
        if not item.future.done():
            item.set_exception(CudaUnavailableError(
                "CUDA execution failed; restart the server before submitting more work"))
            # Some child futures are no longer awaited once their root fails.
            # Retrieving here suppresses warnings but preserves await behavior.
            item.future.exception()


cuda_health = CudaHealth()
