"""The boundary contract consumers depend on: a gapless partition of the asset.

These are pure-function tests — no artifact, no GPU, no video.
"""
import pytest

from lib.model.ai_shot_boundary_model import (
    merge_ranges,
    normalize_boundaries,
    prune_non_context_ranges,
)


def assert_partition(boundaries, duration):
    assert boundaries, "expected at least one boundary"
    assert boundaries[0]["start_seconds"] == 0.0
    for index in range(1, len(boundaries)):
        assert boundaries[index]["start_seconds"] == boundaries[index - 1]["end_seconds"], \
            f"gap or overlap at index {index}"
    assert boundaries[-1]["end_seconds"] == round(duration, 3)
    for boundary in boundaries:
        for key in ("start_seconds", "end_seconds"):
            assert round(boundary[key], 3) == boundary[key], f"{key} is not ms-rounded"
        assert boundary["end_seconds"] > boundary["start_seconds"]


@pytest.mark.parametrize("ranges,duration", [
    ([[0, 30], [30, 60]], 2.0),                     # contiguous
    ([[0, 60]], 2.0),                               # single shot covering everything
    ([[0, 40], [30, 60]], 2.0),                     # overlapping -> midpoint split
    ([[0, 20], [40, 59]], 2.0),                     # gap -> filled
    ([[0, 30], [30, 600]], 2.0),                    # past duration -> clamped
    ([], 2.0),                                      # no predictions at all
    ([[0, 10]], 0.5),                               # short asset
])
def test_partition_invariants(ranges, duration):
    boundaries, _ = normalize_boundaries(
        ranges=ranges,
        intra_labels=["General"] * len(ranges),
        inter_labels=["Hard_Cut"] * len(ranges),
        fps=30.0,
        duration=duration,
    )
    assert_partition(boundaries, duration)


def test_zero_duration_yields_no_boundaries():
    boundaries, _ = normalize_boundaries([[0, 10]], ["General"], ["Hard_Cut"], fps=30.0, duration=0.0)
    assert boundaries == []


def test_ranges_are_half_open():
    # A shot [0, 30) at 30fps covers 0.0-1.0s.
    boundaries, _ = normalize_boundaries([[0, 30]], ["General"], ["Hard_Cut"], fps=30.0, duration=1.0)
    assert boundaries[0]["end_seconds"] == 1.0


def test_adjacent_shots_produce_one_boundary_each():
    """Half-open ranges must not overlap, or every cut becomes two boundaries
    separated by a one-frame sliver."""
    boundaries, _ = normalize_boundaries(
        [[0, 120], [120, 240], [240, 360]],
        ["General"] * 3, ["New_Start", "Hard_Cut", "Hard_Cut"],
        fps=30.0, duration=12.0,
    )
    assert [(b["start_seconds"], b["end_seconds"]) for b in boundaries] == [
        (0.0, 4.0), (4.0, 8.0), (8.0, 12.0)
    ]


def test_transition_attaches_within_tolerance_only():
    boundaries, _ = normalize_boundaries(
        [[0, 30], [30, 60]], ["General", "General"], ["New_Start", "Hard_Cut"],
        fps=30.0, duration=2.0,
    )
    assert boundaries[0]["transition_after"] == "New_Start"
    # The final boundary ends at the asset duration, which coincides with the
    # last shot's end, so it carries that shot's transition.
    assert boundaries[-1]["transition_after"] == "Hard_Cut"


def test_label_counts():
    _, counts = normalize_boundaries(
        [[0, 10], [10, 20]], ["General", "Dissolve"], ["Hard_Cut", "Hard_Cut"],
        fps=30.0, duration=1.0,
    )
    assert counts["intra"] == {"General": 1, "Dissolve": 1}
    assert counts["inter"] == {"Hard_Cut": 2}


# ── windowing helpers ────────────────────────────────────────────────────

def test_prune_is_identity_without_context():
    ranges = [[0, 10], [10, 40]]
    got = prune_non_context_ranges(ranges, ["A", "B"], ["X", "Y"], 100, 0)
    assert got == (ranges, ["A", "B"], ["X", "Y"])


def test_prune_drops_and_rebases_context_regions():
    # window 100, context 10 -> the usable region is local [10, 90)
    ranges = [[0, 5], [5, 50], [95, 100]]
    pruned, intra, inter = prune_non_context_ranges(
        ranges, ["A", "B", "C"], ["X", "Y", "Z"], 100, 10
    )
    assert pruned == [[0, 40]]        # [0,5] dropped (inside leading context), [5,50] rebased
    assert intra == ["B"] and inter == ["Y"]


def test_merge_offsets_subsequent_windows():
    full_r, full_i, full_e = [], [], []
    merge_ranges(full_r, full_i, full_e, [[0, 50], [50, 100]], [0, 0], [1, 1], new_start_inter_index=0)
    merge_ranges(full_r, full_i, full_e, [[0, 30]], [0], [1], new_start_inter_index=0)
    assert full_r == [[0, 50], [50, 100], [100, 130]]


def test_merge_joins_a_shot_continuing_across_the_seam():
    # Same intra label and a 'new_start' inter label means the next window's
    # first shot is a continuation, not a new shot.
    full_r, full_i, full_e = [[0, 100]], [7], [1]
    merge_ranges(full_r, full_i, full_e, [[0, 30], [30, 60]], [7, 7], [0, 1], new_start_inter_index=0)
    assert full_r[0] == [0, 130]
    assert len(full_r) == 2


def test_merge_does_not_join_across_different_intra_labels():
    full_r, full_i, full_e = [[0, 100]], [8], [1]
    merge_ranges(full_r, full_i, full_e, [[0, 30]], [7], [0], new_start_inter_index=0)
    assert full_r == [[0, 100], [100, 130]]
