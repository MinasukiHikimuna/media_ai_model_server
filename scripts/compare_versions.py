"""Compare two versions of the server on a set of videos and images.

Runs every item of a dataset through each version and compares what the
models returned and what it cost::

    python scripts/compare_versions.py run --dataset set.tsv --out ab \\
        --version base=git:main --version new=git:HEAD --install /path/to/installed/server
    python scripts/compare_versions.py report ab

A version is ``git:<ref>`` (taken from this repository, or ``--repo``) or a
directory holding a server tree. It runs with the models, the installed model
configs and the local settings (``config.yaml``, ``active_ai.yaml``) of
``--install``, on its own port, with ``--install``'s Python.

Each repeat starts every version in a fresh server process, alternating
between them. GPU inference is not bit-exact across processes: the same code
in a new process can return slightly different boxes and embeddings, and a
small face can move a long way. So results are judged against that: the
report sets the difference between versions beside the difference between two
processes of the same version, and flags where it is larger. Two repeats is
the least that gives such a baseline.

The dataset is a TSV of ``kind<TAB>id<TAB>path`` rows, kind ``video`` or
``image``. Results are stored by id without paths, so an output directory can
be shared as it is; the dataset file names the files and should stay private.

Linux only: it samples the server's memory and CPU time from /proc.
"""

import argparse
import itertools
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IOU_MATCH = 0.5
_TREE_EXCLUDES = {".git", ".venv", "models", "logs", "__pycache__", ".pytest_cache"}
_LOCAL_SETTINGS = ("config.yaml", "active_ai.yaml")


# ---- versions -------------------------------------------------------------------

def materialize(spec, target, install, repo, port):
    """Build a runnable server tree for ``spec`` at ``target``; returns a
    description of the code it holds."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    if spec.startswith("git:"):
        ref = spec[4:]
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", f"{ref}^{{commit}}"],
                                check=True, capture_output=True, text=True).stdout.strip()
        archive = subprocess.Popen(["git", "-C", str(repo), "archive", commit], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=archive.stdout, mode="r|") as tar:
            tar.extractall(target, filter="data")
        if archive.wait() != 0:
            raise RuntimeError(f"git archive {ref} failed")
        described = f"{spec} ({commit[:12]})"
    else:
        source = Path(spec).resolve()
        if not (source / "server.py").exists():
            raise ValueError(f"{spec} is neither git:<ref> nor a server directory")
        def ignore(directory, names):
            # The install's models and the version's own environment stay out;
            # config/models is the version's.
            top = Path(directory) == source
            return [n for n in names if n in ("__pycache__", ".pytest_cache") or (top and n in _TREE_EXCLUDES)]

        shutil.copytree(source, target, dirs_exist_ok=True, ignore=ignore)
        described = str(source)

    # The install's model configs and local settings; the version's own
    # config files win otherwise.
    for path in (install / "config").rglob("*"):
        if path.is_file():
            relative = path.relative_to(install)
            if not (target / relative).exists() or path.name in _LOCAL_SETTINGS:
                (target / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target / relative)
    config = target / "config" / "config.yaml"
    text = config.read_text()
    text, found = re.subn(r"(?m)^port:.*$", f"port: {port}", text)
    config.write_text(text if found else text + f"\nport: {port}\n")
    (target / "models").symlink_to(install / "models")
    (target / "logs").mkdir(exist_ok=True)
    return described


class Server:
    def __init__(self, tree, python, port, log_path):
        self.tree, self.python, self.port, self.log_path = tree, python, port, log_path
        self.process = None

    def __enter__(self):
        self._log = open(self.log_path, "w")
        self.process = subprocess.Popen([str(self.python), "server.py"], cwd=self.tree, stdout=self._log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited with {self.process.returncode}; see {self.log_path}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v4/health", timeout=5) as response:
                    if json.load(response).get("status") == "ok":
                        return self
            except (OSError, ValueError):
                pass
            time.sleep(2)
        raise TimeoutError(f"server did not become healthy; see {self.log_path}")

    def __exit__(self, *_exc):
        if self.process and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGINT)
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        self._log.close()

    def get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=60) as response:
            return json.load(response)

    def post(self, path, body, timeout):
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)


# ---- resource sampling ------------------------------------------------------------

def _process_tree(pid):
    pids, queue = [], [pid]
    while queue:
        current = queue.pop()
        pids.append(current)
        for task in Path(f"/proc/{current}/task").glob("*"):
            try:
                queue.extend(int(child) for child in (task / "children").read_text().split())
            except OSError:
                pass
    return pids


def _usage(pids):
    """Anonymous memory, user and kernel CPU seconds of ``pids``. CPU time
    includes children that exited and were waited for (a video's decode
    workers), which the kernel folds into their parent's."""
    rss, user, system, tick = 0, 0.0, 0.0, os.sysconf("SC_CLK_TCK")
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("RssAnon:"):
                    rss += int(line.split()[1]) * 1024
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            user += (int(fields[11]) + int(fields[13])) / tick
            system += (int(fields[12]) + int(fields[14])) / tick
        except (OSError, IndexError, ValueError):
            pass
    return rss, user, system


def _gpu_bytes(pids):
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    wanted, total = set(pids), 0
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) in wanted and parts[1].isdigit():
            total += int(parts[1]) * 1024 ** 2
    return total


class Sampler:
    """Peak anonymous memory and GPU memory of a server and its children, and
    the CPU time they spent, while a request runs."""

    def __init__(self, pid):
        self.pid = pid

    def __enter__(self):
        self.peak_rss = self.peak_gpu = 0
        self._stop = threading.Event()
        self._cpu_start = _usage(_process_tree(self.pid))[1:]
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        last_gpu = 0.0
        while True:
            pids = _process_tree(self.pid)
            rss = _usage(pids)[0]
            self.peak_rss = max(self.peak_rss, rss)
            if time.monotonic() - last_gpu >= 1.0:
                self.peak_gpu = max(self.peak_gpu, _gpu_bytes(pids))
                last_gpu = time.monotonic()
            if self._stop.wait(0.25):
                break

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join()
        _rss, user, system = _usage(_process_tree(self.pid))
        self.user_seconds = max(0.0, user - self._cpu_start[0])
        self.system_seconds = max(0.0, system - self._cpu_start[1])
        self.cpu_seconds = self.user_seconds + self.system_seconds


# ---- run ------------------------------------------------------------------------------

def _dataset(path):
    items = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        kind, item_id, item_path = line.split("\t", 2)
        if kind not in ("video", "image"):
            raise ValueError(f"unknown kind {kind!r} in {path}")
        items.append((kind, item_id, item_path))
    return items


def _model_names(tree):
    import yaml  # the server's environment has it; the report does not need it
    active = yaml.safe_load((tree / "config" / "active_ai.yaml").read_text()) or {}
    return list(active.get("active_ai_models") or [])


def _run_items(server, items, out, args, want):
    pid = server.process.pid
    videos = [item for item in items if item[0] == "video"]
    images = [item for item in items if item[0] == "image"]
    batches = [("video", [item]) for item in videos]
    batches += [("image", images[i:i + args.image_batch]) for i in range(0, len(images), args.image_batch)]
    perf = []
    for kind, batch in batches:
        if kind == "video":
            body = {"path": batch[0][2], "frame_interval": args.interval, "return_confidence": True}
            route = "/v4/analyze/video"
        else:
            body = {"paths": [path for *_rest, path in batch], "return_confidence": True}
            route = "/v4/analyze/images"
        if want:
            body["want"] = [{"models": want}]
        started = time.monotonic()
        with Sampler(pid) as sample:
            try:
                response, error = server.post(route, body, args.timeout), None
            except (OSError, ValueError) as exc:
                response, error = None, repr(exc)
                if isinstance(exc, urllib.error.HTTPError):
                    error += " " + exc.read().decode(errors="replace")[:500]
        seconds = time.monotonic() - started
        results = [response] if kind == "video" else (response or {}).get("result", [None] * len(batch))
        for (_kind, item_id, _path), result in zip(batch, results):
            if result is not None:
                result.pop("asset_id", None)
            (out / f"{kind}-{item_id}.json").write_text(json.dumps(result if error is None else {"error": error}))
        metrics = (response or {}).get("metrics", {}) if kind == "video" else (response or {}).get("metrics", {})
        record = {
            "kind": kind, "ids": [item_id for _k, item_id, _p in batch], "seconds": seconds, "error": error,
            "peak_rss_bytes": sample.peak_rss, "peak_gpu_bytes": sample.peak_gpu, "cpu_seconds": sample.cpu_seconds,
            "cpu_user_seconds": sample.user_seconds, "cpu_system_seconds": sample.system_seconds,
            "server_preprocess_seconds": metrics.get("preprocess_seconds"),
            "server_inference_seconds": metrics.get("ai_inference_seconds"),
            "frames": (response or {}).get("frame_count") if kind == "video" else len(batch),
        }
        perf.append(record)
        label = record["ids"][0] if kind == "video" else f"{len(batch)} images"
        print(f"  {kind} {label}: {'error' if error else 'ok'} in {seconds:.1f} s, "
              f"peak {sample.peak_rss / 1024 ** 3:.1f} GB RAM, {sample.peak_gpu / 1024 ** 3:.1f} GB GPU", flush=True)
    return perf


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    install = Path(args.install).resolve()
    python = Path(args.python) if args.python else install / ".venv" / "bin" / "python"
    versions = []
    for index, entry in enumerate(args.version):
        name, _, spec = entry.partition("=")
        if not spec or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise SystemExit(f"--version wants NAME=SPEC, got {entry!r}")
        versions.append((name, spec, args.port + index))
    items = _dataset(args.dataset)
    manifest = {"dataset": [[kind, item_id] for kind, item_id, _ in items], "versions": {},
                "repeats": args.repeats, "interval": args.interval, "all_models": args.all_models}
    for name, spec, port in versions:
        manifest["versions"][name] = materialize(spec, out / "trees" / name, install, Path(args.repo), port)
        print(f"{name}: {manifest['versions'][name]}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    with open(out / "perf.jsonl", "a") as perf_log:
        for repeat in range(1, args.repeats + 1):
            order = versions if repeat % 2 else list(reversed(versions))
            for name, _spec, port in order:
                results = out / "results" / name / f"r{repeat}"
                if results.exists() and len(list(results.glob("*.json"))) == len(items):
                    print(f"{name} repeat {repeat}: already done", flush=True)
                    continue
                results.mkdir(parents=True, exist_ok=True)
                tree = out / "trees" / name
                print(f"{name} repeat {repeat}: starting a fresh server on port {port}", flush=True)
                with Server(tree, python, port, out / f"server-{name}-r{repeat}.log") as server:
                    want = _model_names(tree) if args.all_models else None
                    for record in _run_items(server, items, results, args, want):
                        perf_log.write(json.dumps({"version": name, "repeat": repeat, **record}) + "\n")
                        perf_log.flush()


# ---- compare results ------------------------------------------------------------------

def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _faces(analysis):
    """Face detections of one frame or image, each with its region's embedding."""
    detections = analysis.get("capabilities", {}).get("detection", {}).get("face_detections", [])
    regions = analysis.get("region_branches", {}).get("regions__face_detector", [])
    faces = []
    for index, detection in enumerate(detections):
        vector = None
        if index < len(regions):
            embeddings = regions[index].get("capabilities", {}).get("embedding", {}).get("face_embeddings", [])
            vector = embeddings[0]["vector"] if embeddings else None
        faces.append((detection["bbox"], detection["score"], vector))
    return faces


def _tags(analysis):
    tagging = analysis.get("capabilities", {}).get("tagging", {})
    return {(category, entry[0]): (entry[1] if len(entry) > 1 else 1.0)
            for category, entries in tagging.items() for entry in entries}


def _frames(result):
    """A result's analyses by time: a video's frames, or an image as one frame."""
    if "frames" in result:
        return {round(frame["time_seconds"], 3): frame.get("analysis", {}) for frame in result["frames"]}
    return {0.0: result.get("analysis", {})}


def _shot_cuts(result):
    """Frames where a shot starts, or None if the result has no shot boundaries."""
    shots = result.get("analysis", {}).get("other", {}).get("shot_boundaries")
    if not isinstance(shots, dict):
        return None
    return {shot["start_frame"] for shot in shots.get("boundaries", []) if shot.get("start_frame")}


def compare_results(a, b):
    """Differences between two results of one item, as a flat dict."""
    stats = {
        "frames": 0, "frames_only_in_one": 0, "tags": 0, "tag_flips": 0, "tag_conf_max_diff": 0.0,
        "faces": 0, "faces_unmatched": 0, "bbox_max_diff": 0.0, "score_max_diff": 0.0,
        "embeddings": 0, "cosine_min": 1.0, "cosine_sum": 0.0, "embeddings_below_0_99": 0,
        "shot_cuts": 0, "shot_cuts_only_in_one": 0,
    }
    frames_a, frames_b = _frames(a), _frames(b)
    stats["frames"] = len(frames_a.keys() & frames_b.keys())
    stats["frames_only_in_one"] = len(frames_a.keys() ^ frames_b.keys())
    for when in frames_a.keys() & frames_b.keys():
        tags_a, tags_b = _tags(frames_a[when]), _tags(frames_b[when])
        stats["tags"] += len(tags_a.keys() | tags_b.keys())
        stats["tag_flips"] += len(tags_a.keys() ^ tags_b.keys())
        for key in tags_a.keys() & tags_b.keys():
            stats["tag_conf_max_diff"] = max(stats["tag_conf_max_diff"], abs(tags_a[key] - tags_b[key]))

        faces_a, faces_b = _faces(frames_a[when]), _faces(frames_b[when])
        stats["faces"] += max(len(faces_a), len(faces_b))
        unmatched = list(faces_b)
        for box, score, vector in faces_a:
            best = max(unmatched, key=lambda face: _iou(box, face[0]), default=None)
            if best is None or _iou(box, best[0]) < IOU_MATCH:
                stats["faces_unmatched"] += 1
                continue
            unmatched.remove(best)
            stats["bbox_max_diff"] = max(stats["bbox_max_diff"], *(abs(p - q) for p, q in zip(box, best[0])))
            stats["score_max_diff"] = max(stats["score_max_diff"], abs(score - best[1]))
            if vector is not None and best[2] is not None:
                cosine = _cosine(vector, best[2])
                stats["embeddings"] += 1
                stats["cosine_min"] = min(stats["cosine_min"], cosine)
                stats["cosine_sum"] += cosine
                stats["embeddings_below_0_99"] += cosine < 0.99
        stats["faces_unmatched"] += len(unmatched)

    cuts_a, cuts_b = _shot_cuts(a), _shot_cuts(b)
    if cuts_a is not None or cuts_b is not None:
        cuts_a, cuts_b = cuts_a or set(), cuts_b or set()
        stats["shot_cuts"] = len(cuts_a | cuts_b)
        stats["shot_cuts_only_in_one"] = len(cuts_a ^ cuts_b)
    return stats


def _total(per_item):
    total = {"items": len(per_item), "cosine_min": 1.0}
    for stats in per_item.values():
        for key, value in stats.items():
            if key.endswith("max_diff"):
                total[key] = max(total.get(key, 0.0), value)
            elif key == "cosine_min":
                total[key] = min(total[key], value)
            else:
                total[key] = total.get(key, 0) + value
    total["cosine_mean"] = total["cosine_sum"] / total["embeddings"] if total.get("embeddings") else 1.0
    return total


def _load_run(directory):
    return {path.stem: json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))}


def _compare_runs(run_a, run_b):
    per_item, errors = {}, []
    for item in sorted(run_a.keys() | run_b.keys()):
        a, b = run_a.get(item), run_b.get(item)
        if not a or not b or "error" in a or "error" in b:
            errors.append(item)
            continue
        per_item[item] = compare_results(a, b)
    return per_item, errors


def _worst(totals):
    """The worst of several totals, field by field."""
    worst = {}
    for total in totals:
        for key, value in total.items():
            if key in ("cosine_min", "cosine_mean"):
                worst[key] = min(worst.get(key, 1.0), value)
            elif key.endswith("max_diff"):
                worst[key] = max(worst.get(key, 0.0), value)
            else:
                worst[key] = max(worst.get(key, 0), value)
    return worst


# (key, label, format, larger_is_worse)
RESULT_ROWS = [
    ("items", "items compared", "{:d}", None),
    ("frames", "frames and images compared", "{:d}", None),
    ("frames_only_in_one", "frames present in only one run", "{:d}", True),
    ("tag_flips", "tags present in only one run", "{:d}", True),
    ("tag_conf_max_diff", "largest tag confidence difference", "{:.3f}", True),
    ("faces", "face detections", "{:d}", None),
    ("faces_unmatched", "faces without a match (IoU < 0.5)", "{:d}", True),
    ("bbox_max_diff", "largest box edge move (fraction of frame)", "{:.4f}", True),
    ("score_max_diff", "largest detection score difference", "{:.4f}", True),
    ("embeddings", "embedding pairs compared", "{:d}", None),
    ("embeddings_below_0_99", "embedding pairs below cosine 0.99", "{:d}", True),
    ("cosine_min", "lowest embedding cosine", "{:.4f}", False),
    ("cosine_mean", "mean embedding cosine", "{:.6f}", False),
    ("shot_cuts", "shot cuts", "{:d}", None),
    ("shot_cuts_only_in_one", "shot cuts in only one run", "{:d}", True),
]


def _exceeds(key, larger_is_worse, change, noise):
    if larger_is_worse is None or key not in change or key not in noise:
        return False
    if larger_is_worse:
        return change[key] > noise[key] * 1.5 + (1e-9 if isinstance(change[key], float) else 0)
    return (1 - change[key]) > (1 - noise[key]) * 1.5 + 1e-9


# ---- report ---------------------------------------------------------------------------

def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def _per_repeat(group, field):
    """The median over repeats of ``field``, summed within each repeat (the
    images of a repeat come in several requests)."""
    repeats = {}
    for record in group:
        if record.get(field) is not None:
            repeats[record["repeat"]] = repeats.get(record["repeat"], 0.0) + record[field]
    return _median(list(repeats.values()))


def _perf_table(records, names):
    by = {}
    for record in records:
        if record.get("error"):
            continue
        key = (record["kind"], record["ids"][0] if record["kind"] == "video" else "images")
        by.setdefault((record["version"], key), []).append(record)
    keys = sorted({key for _version, key in by}, key=lambda k: (k[0] != "video", k[1]))
    pair = len(names) == 2
    lines = [f"{'item':<14}" + "".join(f"{name + ' s':>11}" for name in names) + (f"{'change':>9}" if pair else "")
             + "".join(f"{name + ' cpu s':>13}" for name in names) + "".join(f"{name + ' GB':>10}" for name in names)]
    fields = ("seconds", "cpu_seconds", "cpu_user_seconds", "cpu_system_seconds")
    totals = {name: dict.fromkeys(fields + ("rss", "gpu"), 0.0) for name in names}
    for key in keys:
        per = {}
        for name in names:
            group = by.get((name, key), [])
            per[name] = {field: _per_repeat(group, field) for field in fields}
            per[name]["rss"] = max((r["peak_rss_bytes"] for r in group), default=0)
            per[name]["gpu"] = max((r["peak_gpu_bytes"] for r in group), default=0)
            for field in fields:
                totals[name][field] += per[name][field] or 0.0
            for field in ("rss", "gpu"):
                totals[name][field] = max(totals[name][field], per[name][field])

        def cell(value, width, fmt):
            return f"{fmt.format(value):>{width}}" if value is not None else f"{'-':>{width}}"

        line = f"{key[0][0] + ':' + key[1]:<14}" + "".join(cell(per[n]["seconds"], 11, "{:.1f}") for n in names)
        if pair:
            a, b = (per[n]["seconds"] for n in names)
            line += f"{(b / a - 1) * 100:>+8.0f}%" if a and b else f"{'':>9}"
        line += "".join(cell(per[n]["cpu_seconds"], 13, "{:.0f}") for n in names)
        line += "".join(cell(per[n]["rss"] / 1024 ** 3, 10, "{:.1f}") for n in names)
        lines.append(line)
    lines.append("")
    lines.append(f"{'':<34}" + "".join(f"{name:>12}" for name in names))
    for field, label, fmt, scale in [("seconds", "wall time, s", "{:.0f}", 1),
                                     ("cpu_seconds", "server CPU time, s", "{:.0f}", 1),
                                     ("cpu_user_seconds", "  of it user, s", "{:.0f}", 1),
                                     ("cpu_system_seconds", "  of it kernel, s", "{:.0f}", 1),
                                     ("rss", "peak server RAM, GB", "{:.1f}", 1024 ** 3),
                                     ("gpu", "peak server GPU memory, GB", "{:.1f}", 1024 ** 3)]:
        if field.startswith("cpu_") and field != "cpu_seconds" and not any(totals[n][field] for n in names):
            continue  # runs from before the split was recorded
        lines.append(f"{label:<34}" + "".join(f"{fmt.format(totals[n][field] / scale):>12}" for n in names))
    return lines


def report(args):
    out = Path(args.out)
    manifest = json.loads((out / "manifest.json").read_text())
    names = list(manifest["versions"])
    runs = {name: {int(d.name[1:]): _load_run(d) for d in sorted((out / "results" / name).glob("r*"))}
            for name in names}
    print("Versions:")
    for name in names:
        print(f"  {name}: {manifest['versions'][name]} ({len(runs[name])} repeats)")
    print()

    noise_totals, all_errors = [], set()
    for name in names:
        for r1, r2 in itertools.combinations(sorted(runs[name]), 2):
            per_item, errors = _compare_runs(runs[name][r1], runs[name][r2])
            all_errors.update(errors)
            noise_totals.append(_total(per_item))
    change_per_item, change_totals = {}, []
    if len(names) >= 2:
        a, b = names[0], names[1]
        for repeat in sorted(runs[a].keys() & runs[b].keys()):
            per_item, errors = _compare_runs(runs[a][repeat], runs[b][repeat])
            all_errors.update(errors)
            change_totals.append(_total(per_item))
            for item, stats in per_item.items():
                change_per_item.setdefault(item, []).append(stats)
    noise = _worst(noise_totals) if noise_totals else None
    change = _worst(change_totals) if change_totals else None

    if change_per_item:
        print(f"Per item, {names[0]} vs {names[1]} (worst repeat):")
        print(f"{'item':<16}{'frames':>7}{'faces':>7}{'unmatched':>10}{'tag flips':>10}{'cuts diff':>10}{'min cos':>9}")
        for item, stats_list in sorted(change_per_item.items()):
            worst = _worst(stats_list)
            print(f"{item:<16}{worst['frames']:>7}{worst['faces']:>7}{worst['faces_unmatched']:>10}"
                  f"{worst['tag_flips']:>10}{worst['shot_cuts_only_in_one']:>10}{worst['cosine_min']:>9.4f}")
        print()

    print("Results" + (f": {names[0]} vs {names[1]}, beside two processes of the same version" if change else ""))
    header = f"{'':<44}"
    if change:
        header += f"{'between':>10}"
    if noise:
        header += f"{'same':>10}"
    print(header)
    flagged = []
    for key, label, fmt, larger_is_worse in RESULT_ROWS:
        line = f"{label:<44}"
        if change:
            line += f"{fmt.format(change.get(key, 0)):>10}"
        if noise:
            line += f"{fmt.format(noise.get(key, 0)):>10}"
        if change and noise and _exceeds(key, larger_is_worse, change, noise):
            line += "  <- beyond run-to-run"
            flagged.append(label)
        print(line)
    if all_errors:
        print(f"\nItems that failed in some run: {', '.join(sorted(all_errors))}")
    if change and noise:
        print("\n" + ("Results differ beyond run-to-run variation in: " + "; ".join(flagged) if flagged
                      else "No result differs beyond run-to-run variation."))
    elif change and not noise:
        print("\nOnly one repeat: there is no run-to-run baseline to judge the differences against.")

    records = [json.loads(line) for line in (out / "perf.jsonl").read_text().splitlines() if line.strip()]
    print("\nPerformance (median of repeats; memory is the peak over all repeats)")
    for line in _perf_table(records, names):
        print(line)
    failed = [r for r in records if r.get("error")]
    for record in failed:
        print(f"failed: {record['version']} r{record['repeat']} {record['kind']} {record['ids']}: {record['error']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="run the dataset through each version")
    run_parser.add_argument("--dataset", required=True, help="TSV of kind<TAB>id<TAB>path rows")
    run_parser.add_argument("--out", required=True, help="directory for trees, results, logs and timings")
    run_parser.add_argument("--version", action="append", required=True, metavar="NAME=SPEC",
                            help="git:<ref> or a server directory; give it twice, the first is the baseline")
    run_parser.add_argument("--install", required=True, help="server install with models/, config/ and .venv/")
    run_parser.add_argument("--repo", default=str(REPO), help="repository git: refs are taken from")
    run_parser.add_argument("--python", help="interpreter to run the servers (default: the install's .venv)")
    run_parser.add_argument("--repeats", type=int, default=2, help="fresh-process runs of each version")
    run_parser.add_argument("--port", type=int, default=18810, help="first version's port; the next use +1")
    run_parser.add_argument("--interval", type=float, default=2.0, help="seconds between analysed video frames")
    run_parser.add_argument("--image-batch", type=int, default=8, help="images per request")
    run_parser.add_argument("--timeout", type=float, default=7200)
    run_parser.add_argument("--all-models", action="store_true",
                            help="name every active model, so asset models such as shot boundaries run too")
    report_parser = commands.add_parser("report", help="compare the results and timings of a run")
    report_parser.add_argument("out")
    args = parser.parse_args(argv)
    {"run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
