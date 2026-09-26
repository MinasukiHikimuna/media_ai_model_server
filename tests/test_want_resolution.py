"""`want` must be able to select an asset-scope model on a video request.

/v4/analyze/video passes default_scope="frame". Before the asset stage existed
that was harmless, because every selectable model was frame-scoped. Applying it
to an explicit capability ask would now silently match nothing and 400.
"""
import types

import pytest

from lib.server import v4_service


CATALOG = [
    {"config_name": "tagger", "capabilities": ["tagging"],
     "categories": ["actions"], "supported_scopes": ["frame"]},
    {"config_name": "face_detector", "capabilities": ["detection"],
     "categories": ["face_detections"], "supported_scopes": ["frame"]},
    {"config_name": "boundaries", "capabilities": ["temporal_segmentation"],
     "categories": ["shot_boundaries"], "supported_scopes": ["asset"]},
]


@pytest.fixture(autouse=True)
def _catalog(monkeypatch):
    monkeypatch.setattr(v4_service, "get_model_catalog", lambda _sm: CATALOG)


def want(**kwargs):
    fields = dict(models=None, capability=None, capabilities=None,
                  category=None, categories=None, model_category=None,
                  scope=None, scopes=None)
    fields.update(kwargs)
    return types.SimpleNamespace(**fields)


def resolve(items, default_scope="frame"):
    return v4_service.resolve_want_model_names(None, items, default_scope)


def test_capability_ask_reaches_asset_scope_despite_frame_default():
    assert resolve([want(capability="temporal_segmentation")]) == ["boundaries"]


def test_category_ask_reaches_asset_scope_despite_frame_default():
    assert resolve([want(category="shot_boundaries")]) == ["boundaries"]


def test_explicit_scope_still_honoured():
    assert resolve([want(capability="temporal_segmentation", scope="asset")]) == ["boundaries"]
    with pytest.raises(ValueError):
        resolve([want(capability="temporal_segmentation", scope="frame")])


def test_bare_want_item_still_uses_the_default_scope():
    # No capability/category: the default scope is the only filter, and it must
    # still exclude asset-scope models from a plain frame request.
    assert resolve([want()]) == ["tagger", "face_detector"]


def test_tagging_ask_does_not_pull_in_the_asset_model():
    assert resolve([want(capability="tagging")]) == ["tagger"]


def test_explicit_model_names_bypass_scope_filtering():
    assert resolve([want(models=["boundaries"])]) == ["boundaries"]


def test_combined_request_selects_both_stages():
    got = resolve([want(capability="tagging"), want(capability="temporal_segmentation")])
    assert got == ["tagger", "boundaries"]
