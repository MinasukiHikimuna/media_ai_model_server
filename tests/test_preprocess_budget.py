from types import SimpleNamespace

import pytest
import torch

from lib.model.video_preprocessor import VideoPreprocessorModel, MAX_AUTO_PENDING_FRAMES
from lib.pipeline.preprocess_spec import PreprocessSpec, apply_spec, apply_spec_batch
from lib.utils import vram_budget as budget


def video(pending="auto", workers=1):
    model = VideoPreprocessorModel({"preprocess_backend": "av", "max_pending_frames": pending,
                                    "instance_count": workers, "preprocess_batch_size": 32})
    model.specs = [PreprocessSpec(width=378, height=378, device="gpu", half_precision=True),
                   PreprocessSpec(width=512, height=512, device="cpu")]
    return model


def test_auto_budget_reserves_the_upper_bound_plus_staged_batches():
    model = video()
    per_frame = 378*378*3*2/2**20
    reserved = budget.estimate_preprocess_storage_mb([model])
    assert reserved > MAX_AUTO_PENDING_FRAMES * per_frame
    assert reserved > 525 * per_frame
    assert budget.estimate_preprocess_storage_mb([model, model]) == reserved


def test_explicit_backlogs_above_64_and_worker_concurrency_are_reserved():
    small, large, parallel = video(64), video(1024), video(1024, 2)
    assert budget.estimate_preprocess_storage_mb([large]) > budget.estimate_preprocess_storage_mb([small])
    assert budget.estimate_preprocess_storage_mb([parallel]) == 2*budget.estimate_preprocess_storage_mb([large])


def test_gpu_specs_use_actual_geometry_precision_and_device():
    gpu = PreprocessSpec(width=320, height=180, device="gpu", precision="float32")
    cpu = PreprocessSpec(width=4096, height=4096, device="cpu", precision="float32")
    preprocessor = SimpleNamespace(specs=[gpu, gpu, cpu], _max_pending_images=96)
    assert budget.estimate_preprocess_storage_mb([preprocessor]) == pytest.approx(96*320*180*3*4/2**20)
    preprocessor.specs = [cpu]
    preprocessor._max_pending_images = 0
    assert budget.estimate_preprocess_storage_mb([preprocessor]) == 0


def test_gpu_native_frames_and_unbounded_image_queues_require_a_bound():
    preprocessor = SimpleNamespace(specs=[PreprocessSpec(device="gpu")], _max_pending_images=96)
    with pytest.raises(ValueError, match="fixed size or max_long_edge"):
        budget.estimate_preprocess_storage_mb([preprocessor])
    preprocessor.specs = [PreprocessSpec(device="gpu", max_long_edge=1920, precision="uint8")]
    assert budget.estimate_preprocess_storage_mb([preprocessor]) == 96*1920*1920*3/2**20
    preprocessor._max_pending_images = 0
    with pytest.raises(ValueError, match="max_pending_images"):
        budget.estimate_preprocess_storage_mb([preprocessor])


def test_reserving_a_larger_real_backlog_reduces_the_activation_budget(monkeypatch):
    monkeypatch.setattr(budget, "get_total_vram_mb", lambda _: 10000)
    def model():
        return SimpleNamespace(model_file_name="test", _model_weight_mb_override=1000,
            activation_per_item_mb=100, _config_max_batch_cap=None,
            max_batch_size=1, max_model_batch_size=1, max_queue_size=1)
    small, large = model(), model()
    budget.compute_batch_sizes([small], "cuda", preprocessors=[video(64)])
    budget.compute_batch_sizes([large], "cuda", preprocessors=[video("auto")])
    assert large.max_batch_size < small.max_batch_size


@pytest.mark.parametrize("batched", [False, True])
def test_raw_center_crop_does_not_retain_an_oversized_backing_store(batched):
    spec = PreprocessSpec(width=16, height=16, center_crop=True, precision="float32")
    source = torch.ones((3, 32, 256))
    result = apply_spec_batch(source.unsqueeze(0), spec) if batched else apply_spec(source, spec)
    assert result.untyped_storage().nbytes() == result.numel()*result.element_size()
    assert result.shape[-2:] == (16, 16)
