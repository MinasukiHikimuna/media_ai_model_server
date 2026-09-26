"""Asset-scope temporal segmentation (shot-boundary detection).

Unlike every other AI model in this server, a shot-boundary model does not
consume preprocessed frames handed to it by the shared frame preprocessor: it
needs dense, *adjacent* frames across the whole asset, at its own resolution.
It therefore runs at ``asset`` scope, receives the video path directly, and
drives its own decode.

The artifact contract (see README) is a self-contained ``.pt2``/``.pt`` module::

    forward(clip: [1, T, 3, H, W]) -> (shot_logits, intra_logits, inter_logits)

with ``shot_logits[..., :-1]`` a per-query distribution over the clip-relative
end frame, and the other two heads per-query class distributions whose last
class is the "no object" slot.  ``T``/``H``/``W`` and the int -> label-name
maps come from a ``<model_file_name>.labels.json`` sidecar next to the
artifact, mirroring how tagging models read ``<model_file_name>.tags.txt``.

Everything outside the artifact — windowing, the greedy query walk, context
pruning, cross-window merging and boundary normalization — is generic and
model-independent.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from lib.model.ai_model import AIModel
from lib.model.preprocessing_python.ffmpeg_pipe import (
    make_video_frame_source,
    probe_video,
)
from lib.model.preprocessing_python.image_preprocessing import vr_permute
from lib.pipeline.preprocess_spec import PreprocessSpec, apply_spec_batch


class AIShotBoundaryModel(AIModel):
    """Runs a shot-boundary artifact over a whole video and emits segments."""

    def __init__(self, configValues):
        super().__init__(configValues, keep_on_device=False)

        # One video at a time: a single request already saturates the model,
        # and batching whole videos would multiply peak memory for no gain.
        self.max_batch_size = 1
        self.instance_count = 1
        self.fill_to_batch = False

        self.window_frames = int(configValues.get("window_frames", 0) or 0)
        self.context_frames = int(configValues.get("context_frames", 0) or 0)
        self.mode = str(configValues.get("mode", "default") or "default").lower()
        self.general_intra_label = str(configValues.get("general_intra_label", "general"))
        self.new_start_inter_label = str(configValues.get("new_start_inter_label", "new_start"))
        self.preprocess_config = configValues.get("preprocess_config", None)

        # Decoding dominates runtime on long, high-resolution sources. See
        # ffmpeg_pipe._ladder for the measured backend comparison.
        decode = configValues.get("decode", None) or {}
        self.decode_backend = str(decode.get("backend", "auto"))
        self.decode_quality = str(decode.get("quality", "exact"))
        self.decode_device_index = decode.get("device_index", None)

        self.labels: Optional[dict] = None
        self._spec: Optional[PreprocessSpec] = None
        self._intra_names: Dict[int, str] = {}
        self._inter_names: Dict[int, str] = {}
        self._general_intra_index: Optional[int] = None
        self._new_start_inter_index: Optional[int] = None

    @property
    def _name(self) -> str:
        """Display name: the config yaml's stem once the manager attaches it, else the artifact."""
        return getattr(self, "config_name", None) or self.model_file_name

    # ── lifecycle ────────────────────────────────────────────────────────

    async def load(self):
        await super().load()
        if self.labels is None:
            self._load_labels()
        if self._spec is None:
            self._spec = PreprocessSpec.for_model(self)
            if self._spec.width <= 0 or self._spec.height <= 0:
                raise ValueError(
                    f"{self._name}: shot-boundary models need an explicit "
                    f"preprocess_config width/height (the artifact's trained clip geometry)"
                )
        self.logger.info(
            f"Shot-boundary model ready: window={self.window_frames} frames, "
            f"clip={self._spec.width}x{self._spec.height}, context={self.context_frames}, mode={self.mode}"
        )

    def _load_labels(self):
        sidecar = Path(f"./models/{self.model_file_name}.labels.json")
        if not sidecar.exists():
            raise FileNotFoundError(
                f"Shot-boundary label sidecar not found: {sidecar}. "
                f"It ships alongside the {self.model_file_name} artifact."
            )
        with open(sidecar, "r", encoding="utf-8") as handle:
            labels = json.load(handle)

        self.labels = labels
        self._intra_names = {int(k): str(v) for k, v in (labels.get("intra") or {}).items()}
        self._inter_names = {int(k): str(v) for k, v in (labels.get("inter") or {}).items()}

        intra_index = labels.get("intra_name_to_index") or {}
        inter_index = labels.get("inter_name_to_index") or {}
        if self.general_intra_label in intra_index:
            self._general_intra_index = int(intra_index[self.general_intra_label])
        if self.new_start_inter_label in inter_index:
            self._new_start_inter_index = int(inter_index[self.new_start_inter_label])

        # The sidecar carries the artifact's trained geometry; the yaml may override.
        if self.window_frames <= 0:
            self.window_frames = int(labels.get("window_frames", 0) or 0)
        if self.window_frames <= 0:
            raise ValueError(
                f"{self._name}: window_frames is unknown "
                f"(set it in the model yaml or in {sidecar})"
            )
        if self.mode == "clean_shot" and self._general_intra_index is None:
            raise ValueError(
                f"{self._name}: mode=clean_shot needs intra label "
                f"'{self.general_intra_label}' in {sidecar}"
            )

    # ── worker ───────────────────────────────────────────────────────────

    async def worker_function(self, data):
        loop = asyncio.get_running_loop()
        for item in data:
            try:
                video_path = item.item_future[item.input_names[0]]
                vr_video = False
                if len(item.input_names) > 1:
                    raw_vr = item.item_future[item.input_names[1]]
                    vr_video = bool(raw_vr) if raw_vr is not None else False

                # Decode + inference take minutes, not milliseconds. Running them
                # inline would block the event loop and stall every other model —
                # including the tagging pass over this same video.
                result = await loop.run_in_executor(None, self._analyze, video_path, vr_video)
                await item.item_future.set_data(item.output_names[0], result)
            except Exception as exception:  # noqa: BLE001 - reported per item
                item.item_future.set_exception(exception)

    # ── inference ────────────────────────────────────────────────────────

    def _analyze(self, video_path, vr_video: bool) -> dict:
        window = self.window_frames
        context = self.context_frames
        stride = window - 2 * context
        if stride <= 0:
            raise ValueError(
                f"context_frames={context} is too large for window_frames={window} "
                f"(stride would be {stride})"
            )

        spec = self._spec
        info = probe_video(video_path)
        duration = float(info.duration or 0.0)
        fps = float(info.fps or 30.0)

        ranges_full: List[List[int]] = []
        intra_full: List[int] = []
        inter_full: List[int] = []

        # Rolling window over the padded frame stream. Upstream prepends
        # `context` black frames to the whole video, then walks windows of
        # `window` frames at `stride`. Holding only one window at a time keeps
        # peak memory at window*C*H*W regardless of video length.
        blank = torch.zeros((3, spec.height, spec.width), dtype=torch.float32)
        buffer: List[torch.Tensor] = [blank.clone() for _ in range(context)]
        real_frames = 0

        # vr_permute crops AFTER decode, so cropping a frame already squashed to
        # the clip geometry is not the same as cropping then resizing. Decode at
        # native size and let the CPU path crop when VR is in play.
        decode_size = None if vr_video else (spec.width, spec.height)
        frame_source = make_video_frame_source(
            video_path,
            decode_size=decode_size,
            frame_step=1,              # every frame
            backend=self.decode_backend,
            quality=self.decode_quality,
            device_index=self.decode_device_index,
            info=info,
        )
        analyze_started = time.perf_counter()
        inference_seconds = 0.0

        for _index, frame_np in frame_source:
            buffer.append(self._to_frame_tensor(frame_np, vr_video))
            real_frames += 1
            while len(buffer) >= window:
                inference_seconds += self._run_window(
                    buffer[:window], 0, ranges_full, intra_full, inter_full)
                del buffer[:stride]

        # Tail: pad the remainder out to a full window. `len(buffer) > context`
        # is exactly "the next window would still start inside real footage",
        # since buffer holds padded[base:] and padding adds `context` frames.
        if len(buffer) > context:
            num_pad = window - len(buffer)
            clip = list(buffer)
            if num_pad > 0:
                clip.extend(blank.clone() for _ in range(num_pad))
            inference_seconds += self._run_window(
                clip[:window], max(0, num_pad), ranges_full, intra_full, inter_full)

        # Decode and inference interleave, so time the one we can bound exactly
        # and attribute the remainder to decoding.
        total_seconds = time.perf_counter() - analyze_started
        decode_seconds = max(0.0, total_seconds - inference_seconds)
        self.logger.info(
            f"Shot boundaries for {real_frames} frames: "
            f"decode {decode_seconds:.1f}s ({real_frames / decode_seconds:.0f} fps) / "
            f"inference {inference_seconds:.1f}s / backend {frame_source.backend_name}"
            if decode_seconds > 0 else
            f"Shot boundaries for {real_frames} frames (backend {frame_source.backend_name})"
        )

        if self.mode == "clean_shot":
            keep = [i for i, label in enumerate(intra_full) if label == self._general_intra_index]
            ranges_full = [ranges_full[i] for i in keep]
            intra_full = [intra_full[i] for i in keep]
            inter_full = [inter_full[i] for i in keep]

        if duration <= 0 and fps > 0 and real_frames:
            duration = real_frames / fps

        boundaries, label_counts = normalize_boundaries(
            ranges=ranges_full,
            intra_labels=[self._intra_names.get(x, f"Unknown_{x}") for x in intra_full],
            inter_labels=[self._inter_names.get(x, f"Unknown_{x}") for x in inter_full],
            fps=fps,
            duration=duration,
        )

        return {
            "model": self._name,
            "mode": self.mode,
            "fps": fps,
            "duration_seconds": round(duration, 3),
            "source_frame_count": real_frames,
            "window_frames": window,
            "context_frames": context,
            "decode_backend": frame_source.backend_name,
            "decode_seconds": round(decode_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "boundaries": boundaries,
            "label_counts": label_counts,
            # Frame-accurate shots, before the conversion to a time partition.
            # Carries the per-shot intra label, which `boundaries` does not.
            "shots": [
                {
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "intra": self._intra_names.get(intra_full[i], f"Unknown_{intra_full[i]}"),
                    "inter": self._inter_names.get(inter_full[i], f"Unknown_{inter_full[i]}"),
                }
                for i, (start, end) in enumerate(ranges_full)
            ],
        }

    def _to_frame_tensor(self, frame_np, vr_video: bool) -> torch.Tensor:
        """HxWx3 uint8 -> CHW float32 in [0,255], matching apply_spec_batch's input."""
        tensor = torch.from_numpy(frame_np.copy())
        if vr_video:
            tensor = vr_permute(tensor)
        return tensor.permute(2, 0, 1).float()

    def _run_window(self, frames: List[torch.Tensor], num_pad_frames: int,
                    ranges_full: List[List[int]], intra_full: List[int],
                    inter_full: List[int]) -> float:
        """Run one clip, merge its predictions, and return seconds spent."""
        started = time.perf_counter()
        clip = apply_spec_batch(torch.stack(frames), self._spec).unsqueeze(0)
        shot_logits, intra_logits, inter_logits = self.model.run_raw_multi_output(clip, use_half=False)

        # [..., :-1] drops the trailing "no object" slot on every head.
        range_idx = torch.from_numpy(shot_logits).softmax(-1)[0, :, :-1].argmax(dim=-1)
        intra_idx = torch.from_numpy(intra_logits).softmax(-1)[0, :, :-1].argmax(dim=-1)
        inter_idx = torch.from_numpy(inter_logits).softmax(-1)[0, :, :-1].argmax(dim=-1)

        ranges, intra, inter = [], [], []
        start_frame_idx = 0
        for query in range(len(intra_idx)):
            intra_label = int(intra_idx[query])
            inter_label = int(inter_idx[query])
            end_frame_idx = int(range_idx[query])
            if start_frame_idx >= end_frame_idx:
                continue
            ranges.append([start_frame_idx, end_frame_idx])
            intra.append(intra_label)
            inter.append(inter_label)
            start_frame_idx = end_frame_idx
            if end_frame_idx >= self.window_frames - num_pad_frames:
                break

        ranges, intra, inter = prune_non_context_ranges(
            ranges, intra, inter, self.window_frames, self.context_frames
        )
        merge_ranges(
            ranges_full, intra_full, inter_full, ranges, intra, inter,
            new_start_inter_index=self._new_start_inter_index,
        )
        return time.perf_counter() - started


# ── generic helpers (no model-specific knowledge) ────────────────────────


def prune_non_context_ranges(ranges, intra_labels, inter_labels, window_frames, context_frames):
    """Drop predictions that fall inside a window's context padding and re-base
    the survivors onto the window's own coordinate system."""
    new_ranges, new_intra, new_inter = [], [], []
    for shot_idx in range(len(ranges)):
        start_frame_idx, end_frame_idx = ranges[shot_idx]
        if end_frame_idx <= context_frames:
            continue
        if start_frame_idx >= window_frames - context_frames:
            break
        aligned_start = max(start_frame_idx, context_frames) - context_frames
        aligned_end = min(end_frame_idx, window_frames - context_frames) - context_frames
        new_ranges.append([aligned_start, aligned_end])
        new_intra.append(intra_labels[shot_idx])
        new_inter.append(inter_labels[shot_idx])
    return new_ranges, new_intra, new_inter


def merge_ranges(ranges_full, intra_full, inter_full, ranges, intra_labels, inter_labels,
                 new_start_inter_index=None):
    """Append one window's predictions to the whole-video result, joining a shot
    that continues across the window seam."""
    last_frame_idx = ranges_full[-1][-1] if len(intra_full) != 0 else 0

    if (
        len(intra_full) != 0
        and len(intra_labels) != 0
        and intra_full[-1] == intra_labels[0]
        and new_start_inter_index is not None
        and inter_labels[0] == new_start_inter_index
    ):
        ranges_full[-1][-1] = last_frame_idx + ranges[0][-1]
        ranges = ranges[1:]
        intra_labels = intra_labels[1:]
        inter_labels = inter_labels[1:]

    for idx in range(len(ranges)):
        start_frame_idx, end_frame_idx = ranges[idx]
        ranges_full.append([last_frame_idx + start_frame_idx, last_frame_idx + end_frame_idx])
        intra_full.append(intra_labels[idx])
        inter_full.append(inter_labels[idx])

    return ranges_full, intra_full, inter_full


def normalize_boundaries(ranges, intra_labels, inter_labels, fps: float, duration: float):
    """Turn frame ranges into a contiguous, gapless partition of the asset.

    Consumers rely on this exactly: boundaries are ordered, ``boundaries[0]``
    starts at 0.0, each boundary starts where the previous ended, the last ends
    at ``duration``, and every value is rounded to milliseconds.
    """
    from collections import Counter

    fps = float(fps) or 1.0
    duration = float(duration)

    raw = []
    for index, frame_range in enumerate(ranges):
        if not isinstance(frame_range, (list, tuple)) or len(frame_range) != 2:
            continue
        # Ranges are half-open: the greedy query walk starts each shot at the
        # previous shot's end, and the final end equals the total frame count
        # (one past the last frame index). Treating end as inclusive would
        # overlap every adjacent pair by one frame, and the overlap rule below
        # would then split each cut into a sliver boundary.
        start = max(0.0, min(duration, float(frame_range[0]) / fps))
        end = max(0.0, min(duration, float(frame_range[1]) / fps))
        if end > start:
            raw.append({
                "start": start,
                "end": end,
                "transition": str(inter_labels[index]) if index < len(inter_labels) else None,
            })
    raw.sort(key=lambda item: (item["start"], item["end"]))

    cuts = {0.0, duration}
    transitions: Dict[float, str] = {}
    previous_end = 0.0
    for item in raw:
        start = float(item["start"])
        end = float(item["end"])
        if start <= previous_end:
            cut = (start + previous_end) / 2 if start != previous_end else start
            cuts.add(max(0.0, min(duration, cut)))
        elif start > previous_end:
            cuts.add(previous_end)
            cuts.add(start)
        cuts.add(end)
        previous_end = max(previous_end, end)
        if item["transition"]:
            transitions[end] = str(item["transition"])

    rounded = sorted({round(cut, 3) for cut in cuts if 0 <= cut <= duration})
    boundaries = []
    if rounded:
        rounded[0] = 0.0
        rounded[-1] = round(duration, 3)
        for start, end in zip(rounded, rounded[1:]):
            if end <= start:
                continue
            boundaries.append({
                "start_seconds": start,
                "end_seconds": end,
                "transition_after": _nearest_transition(end, transitions),
            })
        for index in range(1, len(boundaries)):
            boundaries[index]["start_seconds"] = boundaries[index - 1]["end_seconds"]

    counts = {
        "intra": dict(Counter(str(value) for value in intra_labels)),
        "inter": dict(Counter(str(value) for value in inter_labels)),
    }
    return boundaries, counts


def _nearest_transition(end: float, transitions: Dict[float, str]) -> Optional[str]:
    if not transitions:
        return None
    cut = min(transitions, key=lambda value: abs(value - end))
    return transitions[cut] if abs(cut - end) <= 0.002 else None
