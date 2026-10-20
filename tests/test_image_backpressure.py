import asyncio

import pytest

from lib.async_lib.async_processing import ItemFuture, QueueItem
from lib.model import image_preprocessor as images


@pytest.mark.asyncio
async def test_image_slots_are_shared_and_held_until_downstream_completion(monkeypatch):
    loaded = []
    def load(path, *_):
        loaded.append(path)
        return {"frame": object()}
    monkeypatch.setattr(images, "_load_and_apply_specs", load)
    model = images.ImagePreprocessorModel({"max_pending_images": 1})

    async def handler(future, key):
        if key == "frame" and future["path"] == "second":
            future.close_future(None)

    first = await ItemFuture.create(None, {"path": "first"}, handler)
    second = await ItemFuture.create(None, {"path": "second"}, handler)
    await model.worker_function([QueueItem(first, ["path"], ["frame"])])
    waiting = asyncio.create_task(model.worker_function([QueueItem(second, ["path"], ["frame"])]))
    try:
        await asyncio.sleep(0.05)
        assert loaded == ["first"]
        first.set_exception(RuntimeError("downstream failed"))
        first.future.exception()
        await asyncio.wait_for(waiting, timeout=2)
        assert loaded == ["first", "second"]
        assert second.future.done()
    finally:
        first.close_future(None)
        if not waiting.done():
            waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
