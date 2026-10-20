import asyncio
import json
import threading
import weakref

import pytest

from lib.async_lib.async_processing import ItemFuture, ModelProcessor, QueueItem
from lib.model.model import Model
from lib.utils.cuda_health import CudaUnavailableError, cuda_health, is_fatal_cuda_error


@pytest.fixture(autouse=True)
def isolated_health(monkeypatch):
    monkeypatch.setattr(cuda_health, "_failure", None)
    monkeypatch.setattr(cuda_health, "_items", weakref.WeakSet())


async def item(value="frame"):
    async def noop(*_):
        pass
    future = await ItemFuture.create(None, {"frame": value}, noop)
    return QueueItem(future, ["frame"], ["result"])


@pytest.mark.parametrize("message", [
    "CUDA error: an illegal memory access was encountered",
    "CUDA error: device-side assert triggered",
    "CUDA error: unspecified launch failure",
    "CUDA error: misaligned address",
    "CUDA error: the launch timed out and was terminated",
])
def test_fatal_cuda_execution_errors_are_recognized(message):
    assert is_fatal_cuda_error(RuntimeError(message))


@pytest.mark.parametrize("message", ["CUDA out of memory", "inference failed", "illegal address in a file"])
def test_recoverable_or_unrelated_errors_do_not_poison_cuda(message):
    cuda_health.record(RuntimeError(message))
    assert not cuda_health.failed


@pytest.mark.asyncio
async def test_fatal_failure_keeps_original_error_and_fails_other_requests():
    first, second = await item(), await item()
    failure = RuntimeError("CUDA error: an illegal memory access was encountered")
    first.item_future.set_exception(failure)
    await asyncio.sleep(0)
    assert first.item_future.future.exception() is failure
    with pytest.raises(CudaUnavailableError, match="restart"):
        await second.item_future
    with pytest.raises(CudaUnavailableError, match="restart"):
        await item()
    assert second.item_future.data is None


@pytest.mark.asyncio
async def test_a_producer_thread_can_report_a_fatal_failure():
    pending = await item()
    worker = threading.Thread(target=cuda_health.record,
        args=(RuntimeError("CUDA error: an illegal memory access was encountered"),))
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive()
    with pytest.raises(CudaUnavailableError):
        await asyncio.wait_for(pending.item_future.future, timeout=2)


@pytest.mark.asyncio
async def test_failed_items_are_discarded_before_enqueue_and_batching():
    failed = await item()
    failed.item_future.set_exception(RuntimeError("inference failed"))
    failed.item_future.future.exception()
    processor = ModelProcessor(Model({}))
    await processor.add_to_queue(failed)
    assert processor.queue.empty()
    batch = []
    assert await processor.batch_data_append_with_skips(batch, failed)
    assert batch == []


@pytest.mark.asyncio
async def test_a_sibling_failure_during_batch_collection_does_not_reach_inference():
    seen = []

    class Recorder(Model):
        async def worker_function(self, data):
            seen.extend(entry.item_future["frame"] for entry in data)
            for entry in data:
                entry.item_future.close_future(None)

    processor = ModelProcessor(Recorder({"max_batch_size": 2, "batch_collect_timeout_ms": 1000}))
    failed, healthy = await item("failed"), await item("healthy")
    await processor.add_to_queue(failed)
    worker = asyncio.create_task(processor.worker_process())
    try:
        await asyncio.sleep(0.02)
        failed.item_future.set_exception(RuntimeError("sibling inference failed"))
        failed.item_future.future.exception()
        await processor.add_to_queue(healthy)
        await asyncio.wait_for(processor.queue.join(), timeout=2)
        assert seen == ["healthy"]
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_poisoned_process_rejects_http_requests_without_dispatch():
    from lib.server.server_manager import add_process_time_header
    cuda_health.record(RuntimeError("CUDA error: an illegal memory access was encountered"))

    async def must_not_run(_request):
        pytest.fail("a poisoned process dispatched another request")

    response = await add_process_time_header(None, must_not_run)
    assert response.status_code == 503
    assert json.loads(response.body)["restart_required"] is True


@pytest.mark.asyncio
async def test_model_wrapper_does_not_invoke_model_after_fatal_error():
    class MustNotRun(Model):
        async def worker_function(self, data):
            pytest.fail("inference ran after a fatal CUDA error")

    pending = await item()
    cuda_health.record(RuntimeError("CUDA error: device-side assert triggered"))
    await MustNotRun({}).worker_function_wrapper([pending])
    with pytest.raises(CudaUnavailableError):
        await pending.item_future
    await asyncio.sleep(0)


@pytest.mark.parametrize("batched", [False, True])
def test_preprocessing_stops_before_launching_more_tensor_operations(batched, monkeypatch):
    import torch
    from lib.pipeline import preprocess_spec as prep
    source = torch.zeros((1, 3, 8, 8) if batched else (3, 8, 8))
    cuda_health.record(RuntimeError("CUDA error: an illegal memory access was encountered"))
    monkeypatch.setattr(prep.F, "interpolate", lambda *_args, **_kwargs: pytest.fail("another transform ran"))
    operation = prep.apply_spec_batch if batched else prep.apply_spec
    with pytest.raises(CudaUnavailableError):
        operation(source, prep.PreprocessSpec(width=16, height=16))


@pytest.mark.parametrize("fatal", [False, True])
def test_face_alignment_only_falls_back_for_recoverable_errors(monkeypatch, fatal):
    import torch
    from lib.model import ai_face_embedding_model as faces
    failure = RuntimeError("CUDA error: an illegal memory access was encountered") if fatal else ValueError("bad landmarks")
    monkeypatch.setattr(faces, "_try_extract_face_landmarks", lambda _: object())
    def fail(*_args, **_kwargs):
        raise failure
    monkeypatch.setattr(faces, "_norm_crop_gpu", fail)
    model = faces.AIFaceEmbeddingModel.__new__(faces.AIFaceEmbeddingModel)
    data = {"dynamic_region_source": torch.zeros((3, 8, 8))}
    if fatal:
        with pytest.raises(RuntimeError) as caught:
            model._try_aligned_face_image(data)
        assert caught.value is failure
        assert cuda_health.failed
    else:
        assert model._try_aligned_face_image(data) is None
        assert not cuda_health.failed
