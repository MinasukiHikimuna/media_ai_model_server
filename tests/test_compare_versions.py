"""scripts/compare_versions.py: comparing two server versions on a dataset."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "compare_versions.py"
_spec = importlib.util.spec_from_file_location("compare_versions", SCRIPT)
cv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cv)


def _analysis(faces=(), tags=()):
    return {
        "capabilities": {
            "tagging": {"actions": [list(tag) for tag in tags]},
            "detection": {"face_detections": [{"bbox": box, "score": score} for box, score, _ in faces]},
        },
        "region_branches": {"regions__face_detector": [
            {"capabilities": {"embedding": {"face_embeddings": [{"vector": vector}]}}} for *_, vector in faces
        ]},
    }


def _video(*frames, cuts=None):
    result = {"frames": [{"time_seconds": when, "analysis": _analysis(*rest)} for when, *rest in frames]}
    if cuts is not None:
        result["analysis"] = {"other": {"shot_boundaries": {
            "boundaries": [{"start_frame": 0}] + [{"start_frame": cut} for cut in cuts]}}}
    return result


FACE = ([0.1, 0.1, 0.3, 0.4], 0.9, [1.0, 0.0])


def test_identical_results_show_no_difference():
    result = _video((0.0, [FACE], [("Kiss", 0.8)]), cuts=[120, 300])
    stats = cv.compare_results(result, result)
    assert stats["faces"] == stats["embeddings"] == 1
    assert stats["faces_unmatched"] == stats["tag_flips"] == stats["shot_cuts_only_in_one"] == 0
    assert stats["shot_cuts"] == 2
    assert stats["cosine_min"] == 1.0 and stats["bbox_max_diff"] == 0.0


def test_faces_are_matched_by_overlap_whatever_their_order():
    near = ([0.6, 0.6, 0.8, 0.9], 0.7, [0.0, 1.0])
    moved = ([0.61, 0.6, 0.8, 0.9], 0.71, [0.1, 1.0])
    stats = cv.compare_results(_video((2.0, [near, FACE])), _video((2.0, [FACE, moved])))
    assert stats["faces_unmatched"] == 0 and stats["embeddings"] == 2
    assert abs(stats["bbox_max_diff"] - 0.01) < 1e-9
    assert 0.99 < stats["cosine_min"] < 1.0 and stats["embeddings_below_0_99"] == 0


def test_lost_faces_tags_frames_and_cuts_are_counted():
    a = _video((0.0, [FACE], [("Kiss", 0.8)]), (2.0,), cuts=[120, 300])
    b = _video((0.0, [], [("Kiss", 0.7), ("Hug", 0.6)]), cuts=[121, 300])
    stats = cv.compare_results(a, b)
    assert stats["faces_unmatched"] == 1
    assert stats["tag_flips"] == 1
    assert abs(stats["tag_conf_max_diff"] - 0.1) < 1e-9
    assert stats["frames_only_in_one"] == 1
    assert stats["shot_cuts_only_in_one"] == 2


def test_an_image_compares_as_one_frame():
    image = {"analysis": _analysis([FACE], [("Kiss", 0.8)])}
    stats = cv.compare_results(image, image)
    assert stats["frames"] == 1 and stats["faces"] == 1


def test_a_difference_is_flagged_only_beyond_the_run_to_run_one():
    noise = {"faces_unmatched": 2, "cosine_min": 0.90}
    assert not cv._exceeds("faces_unmatched", True, {"faces_unmatched": 3}, noise)
    assert cv._exceeds("faces_unmatched", True, {"faces_unmatched": 4}, noise)
    assert not cv._exceeds("cosine_min", False, {"cosine_min": 0.88}, noise)
    assert cv._exceeds("cosine_min", False, {"cosine_min": 0.80}, noise)
    assert cv._exceeds("tag_flips", True, {"tag_flips": 1}, {"tag_flips": 0})


def _install(tmp_path):
    install = tmp_path / "install"
    (install / "config" / "models").mkdir(parents=True)
    (install / "models").mkdir()
    (install / "config" / "config.yaml").write_text("loglevel: INFO\nport: 8001\n")
    (install / "config" / "active_ai.yaml").write_text("active_ai_models:\n- face_detector\n")
    (install / "config" / "models" / "face_detector.yaml").write_text("type: installed\n")
    (install / "config" / "models" / "video_preprocessor_dynamic.yaml").write_text("from: install\n")
    return install


def test_a_version_runs_its_own_code_with_the_installs_models_and_settings(tmp_path):
    install = _install(tmp_path)
    version = tmp_path / "version"
    (version / "config" / "models").mkdir(parents=True)
    (version / "lib").mkdir()
    (version / "server.py").write_text("# server\n")
    (version / "lib" / "code.py").write_text("VERSION = 2\n")
    (version / "config" / "config.yaml").write_text("port: 8000\n")
    (version / "config" / "models" / "video_preprocessor_dynamic.yaml").write_text("from: version\n")
    (version / ".venv").mkdir()

    tree = tmp_path / "tree"
    cv.materialize(str(version), tree, install, tmp_path, 18810)

    assert (tree / "lib" / "code.py").read_text() == "VERSION = 2\n"
    assert not (tree / ".venv").exists()
    assert (tree / "models").resolve() == install / "models"
    assert (tree / "config" / "models" / "face_detector.yaml").read_text() == "type: installed\n"
    assert (tree / "config" / "models" / "video_preprocessor_dynamic.yaml").read_text() == "from: version\n"
    assert (tree / "config" / "config.yaml").read_text() == "loglevel: INFO\nport: 18810\n"
    assert (tree / "config" / "active_ai.yaml").exists()


def test_a_git_version_is_the_code_at_that_commit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    (repo / "server.py").write_text("# one\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "one")
    (repo / "server.py").write_text("# two\n")

    tree = tmp_path / "tree"
    described = cv.materialize("git:HEAD", tree, _install(tmp_path), repo, 18811)
    assert (tree / "server.py").read_text() == "# one\n"
    assert described.startswith("git:HEAD (")


def test_the_report_reads_a_run(tmp_path, capsys):
    out = tmp_path / "out"
    image = {"analysis": _analysis([FACE], [("Kiss", 0.8)])}
    for name in ("base", "new"):
        for repeat in (1, 2):
            directory = out / "results" / name / f"r{repeat}"
            directory.mkdir(parents=True)
            (directory / "image-1.json").write_text(json.dumps(image))
    (out / "manifest.json").write_text(json.dumps(
        {"dataset": [["image", "1"]], "versions": {"base": "git:a", "new": "git:b"}}))
    (out / "perf.jsonl").write_text("".join(json.dumps({
        "version": name, "repeat": repeat, "kind": "image", "ids": ["1"], "seconds": seconds, "error": None,
        "peak_rss_bytes": 2 * 1024 ** 3, "peak_gpu_bytes": 1024 ** 3, "cpu_seconds": 1.0}) + "\n"
        for name, seconds in (("base", 2.0), ("new", 1.0)) for repeat in (1, 2)))

    cv.main(["report", str(out)])

    printed = capsys.readouterr().out
    assert "No result differs beyond run-to-run variation." in printed
    assert "-50%" in printed


@pytest.mark.parametrize("line", ["audio\t1\t/a.mp3", "video\t1"])
def test_a_bad_dataset_row_is_refused(tmp_path, line):
    dataset = tmp_path / "set.tsv"
    dataset.write_text(line + "\n")
    with pytest.raises(ValueError):
        cv._dataset(dataset)
