"""The asset-scope stage: collection, response shaping, and decode skipping."""
import types

import pytest

from lib.async_lib.async_processing import ItemFuture, QueueItem, Skip
from lib.config.model_capabilities import ModelCapabilitiesConfig
from lib.model.python_functions import asset_result_collector, _apply_asset_results
from lib.model.video_preprocessor import VideoPreprocessorModel


async def _future(data):
    async def noop(item_future, key):
        return None
    return await ItemFuture.create(None, dict(data), noop)


# ── asset_result_collector ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_collector_completes_with_no_asset_models():
    """The common deployment installs no asset model; the stage must still finish."""
    future = await _future({"video_path": "/tmp/a.mp4"})
    await asset_result_collector([QueueItem(future, ["video_path"], ["assetResults"])])
    result = future["assetResults"]
    assert result is not None
    assert result["_outputs"] == []
    # The trigger input must not leak into the result.
    assert "video_path" not in result


@pytest.mark.asyncio
async def test_collector_gathers_results_and_excludes_trigger():
    future = await _future({"video_path": "/tmp/a.mp4", "shot_boundaries": {"boundaries": []}})
    await asset_result_collector(
        [QueueItem(future, ["video_path", "shot_boundaries"], ["assetResults"])]
    )
    result = future["assetResults"]
    assert result["shot_boundaries"] == {"boundaries": []}
    assert "video_path" not in result
    assert result["_outputs"] == [{"model_step": "shot_boundaries", "status": "ok"}]


@pytest.mark.asyncio
async def test_collector_records_skips_and_errors():
    future = await _future({
        "video_path": "/tmp/a.mp4",
        "skipped_thing": Skip(),
        "broken_thing": RuntimeError("boom"),
    })
    await asset_result_collector(
        [QueueItem(future, ["video_path", "skipped_thing", "broken_thing"], ["assetResults"])]
    )
    result = future["assetResults"]
    assert "skipped_thing" not in result and "broken_thing" not in result
    statuses = {o["model_step"]: o["status"] for o in result["_outputs"]}
    assert statuses == {"skipped_thing": "skipped", "broken_thing": "error"}
    assert result["_errors"][0]["model_step"] == "broken_thing"


# ── response shaping ─────────────────────────────────────────────────────

def test_temporal_segmentation_lands_in_analysis_other():
    """The shape must match the image pipeline's asset-level `analysis`, which is
    what clients already know how to parse."""
    payload = {"frames": []}
    _apply_asset_results(
        payload,
        {"shot_boundaries": {"boundaries": [1]}, "_outputs": [{"model_step": "x", "status": "ok"}]},
        {"shot_boundaries": "temporal_segmentation"},
    )
    assert payload["analysis"] == {"other": {"shot_boundaries": {"boundaries": [1]}}}
    assert payload["asset_outputs"]


def test_known_capabilities_bucket_under_analysis_capabilities():
    payload = {"frames": []}
    _apply_asset_results(payload, {"whole_clip_vibe": [0.1]}, {"whole_clip_vibe": "embedding"})
    assert payload["analysis"] == {"capabilities": {"embedding": {"whole_clip_vibe": [0.1]}}}


def test_absent_stage_adds_nothing():
    payload = {"frames": []}
    _apply_asset_results(payload, None, {})
    assert payload == {"frames": []}


def test_stage_that_ran_with_no_models_adds_nothing():
    payload = {"frames": []}
    _apply_asset_results(payload, {"_outputs": []}, {})
    assert "analysis" not in payload


def test_skipped_model_is_not_emitted():
    payload = {"frames": []}
    _apply_asset_results(payload, {"shot_boundaries": Skip()}, {"shot_boundaries": "temporal_segmentation"})
    assert "analysis" not in payload


# ── decode skipping ──────────────────────────────────────────────────────

def _preprocessor(downstream):
    model = VideoPreprocessorModel({"type": "video_preprocessor", "frame_interval": 2})
    model.downstream_model_names = downstream
    return model


class _Future:
    def __init__(self, data):
        self.data = data
    def __getitem__(self, key):
        return self.data.get(key)


@pytest.mark.parametrize("downstream,requested,expected", [
    (None, ["anything"], False),            # not wired by the dynamic manager
    (set(), None, True),                    # no frame-scope model active at all
    ({"tagger"}, None, False),              # no explicit request -> run everything
    ({"tagger"}, ["tagger"], False),        # requested model uses these frames
    ({"tagger"}, ["shot_boundaries"], True),  # only asset-scope models requested
    ({"tagger"}, [], False),                # empty request list means "no filter"
])
def test_should_skip_decode(downstream, requested, expected):
    model = _preprocessor(downstream)
    future = _Future({"requested_model_names": requested})
    inputs = ["video_path", "return_timestamps", "time_interval", "threshold",
              "return_confidence", "vr_video", "skipped_categories", "requested_model_names"]
    assert model._should_skip_decode(future, inputs) is expected


# ── model_capabilities asset stage ───────────────────────────────────────

def _capabilities_config(config):
    import logging
    return ModelCapabilitiesConfig(config, active_model_library=[], logger=logging.getLogger("test"))


def _model(name, capabilities, scopes):
    inner = types.SimpleNamespace(
        config_name=name, model_file_name=name,
        model_capabilities=capabilities, supported_target_scopes=scopes,
        model_type="ShotBoundary",
    )
    return types.SimpleNamespace(model=inner, config_name=name)


def test_asset_stage_all_selects_only_asset_scope_temporal_models():
    config = _capabilities_config({"p": {"asset_models": "ALL"}})
    models = [
        _model("boundaries", ["temporal_segmentation"], ["asset"]),
        _model("tagger", ["tagging"], ["frame"]),
        _model("embedder", ["embedding"], ["frame", "region"]),
    ]
    assert config.resolve_model_names_for_stage("p", "asset", models) == ["boundaries"]


def test_asset_stage_absent_means_no_constraint():
    config = _capabilities_config({"p": {"full_image_models": "ALL"}})
    assert config.resolve_model_names_for_stage("p", "asset", []) is None


def test_asset_stage_accepts_an_explicit_list():
    config = _capabilities_config({"p": {"asset_models": ["boundaries"]}})
    models = [_model("boundaries", ["temporal_segmentation"], ["asset"])]
    assert config.resolve_model_names_for_stage("p", "asset", models) == ["boundaries"]


# ── stage isolation ──────────────────────────────────────────────────────

def _manager_with(models):
    """A DynamicAIManager with model selection stubbed onto a fixed model set."""
    import logging
    from lib.pipeline.dynamic_ai_manager import DynamicAIManager

    manager = DynamicAIManager.__new__(DynamicAIManager)
    manager.models = models
    manager.logger = logging.getLogger("test")
    manager.model_capabilities = _capabilities_config({})
    return manager


def test_asset_only_model_is_never_selected_for_a_frame_stage():
    """A pipeline with no stage constraints (the v3 pipelines) must not wire a
    whole-asset model in as a per-frame model."""
    boundaries = _model("boundaries", ["temporal_segmentation"], ["asset"])
    tagger = _model("tagger", ["tagging"], ["frame"])
    manager = _manager_with([boundaries, tagger])

    frame_stage = manager._select_models(mode="video")
    assert [m.config_name for m in frame_stage] == ["tagger"]

    asset_stage = manager._select_models(mode="asset", required_scope="asset")
    assert [m.config_name for m in asset_stage] == ["boundaries"]


def test_a_model_supporting_both_scopes_still_serves_the_frame_stage():
    both = _model("both", ["tagging"], ["asset", "frame"])
    manager = _manager_with([both])
    assert [m.config_name for m in manager._select_models(mode="video")] == ["both"]
