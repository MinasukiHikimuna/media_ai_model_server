# Asset-scope models

Most models in this server run **per frame**: the shared video preprocessor
decodes at `frame_interval`, fans each frame out into its own `ItemFuture`, and
every active model scores frames independently. Some models cannot work that
way — they need dense, *adjacent* frames across the whole asset, at their own
resolution. Shot-boundary detection is the motivating example.

`asset` has always been one of the three declarable target scopes
(`asset`, `frame`, `region`) but nothing consumed it. The `dynamic_asset_ai`
stage fills that gap.

## How the stage works

`config/pipelines/video_pipeline_dynamic_v4.yaml` runs two stages in parallel
off the same request:

```yaml
  - name: dynamic_video_ai          # per-frame fan-out
    outputs: [childrenResults]
  - name: dynamic_asset_ai          # whole-asset models
    inputs:  [video_path, vr_video, skipped_categories, requested_model_names]
    outputs: [assetResults]
  - name: video_result_postprocessor_v4
    inputs:  [childrenResults, video_path, time_interval, assetResults]
```

The pipeline is a data-flow DAG, so the postprocessor simply waits until both
stages have produced their key.

An asset-scope model receives the stage's declared inputs directly (the asset
path first) and returns one result for the whole asset. There is no
preprocessor and no child fan-out. `asset_result_collector` gathers the models'
outputs into `assetResults`; it is wired on the asset path alone, so the stage
still completes — with an empty result — on the common deployment that has **no
asset-scope model installed at all**.

Selection follows the same rules as every other stage:

| Request | Effect |
|---|---|
| `want: [{capability: "temporal_segmentation"}]` | runs asset models with that capability |
| `want: [{category: "shot_boundaries"}]` | runs models declaring that category |
| `want: [{models: ["<config name>"]}]` | runs exactly that model |
| `categories_to_skip: ["shot_boundaries"]` | suppresses it |
| a request naming only frame-scope models | suppresses it |

Asking for *only* asset-scope models also skips the frame decode entirely,
rather than decoding the video to build children that all resolve to `Skip()`.

`config/model_capabilities.yaml` selects which models the stage may use:

```yaml
video_pipeline_dynamic_v4:
  asset_models: ALL     # every active asset-scope temporal_segmentation model
```

## Response

Asset-scope output is emitted as a top-level `analysis` node with exactly the
shape the **image** pipeline already uses for its asset-level analysis, so a
client needs one code path for both. Known capabilities bucket under
`analysis.capabilities`; anything else — temporal segmentation included — under
`analysis.other`, keyed by category.

The key is absent when the stage did not run, so a caller can tell "not
requested" from "ran and found nothing".

```jsonc
{
  "frames": [ /* unchanged per-frame results */ ],
  "analysis": {
    "other": {
      "shot_boundaries": {
        "model": "omnishotcut_shot_boundaries",
        "mode": "clean_shot",
        "fps": 30.0,
        "duration_seconds": 60.0,
        "source_frame_count": 1800,
        "window_frames": 100,
        "context_frames": 0,
        "boundaries": [
          { "start_seconds": 0.0,   "end_seconds": 2.017, "transition_after": null },
          { "start_seconds": 2.017, "end_seconds": 5.4,   "transition_after": "Hard_Cut" }
        ],
        "shots": [
          { "start_frame": 0, "end_frame": 60, "intra": "General", "inter": "New_Start" }
        ],
        "label_counts": { "intra": {...}, "inter": {...} }
      }
    }
  },
  "asset_outputs": [ { "model_step": "shot_boundaries", "status": "ok" } ]
}
```

`boundaries` is guaranteed to be a **contiguous, gapless partition of the whole
asset**: ordered, first `start_seconds` is `0.0`, each entry starts where the
previous ended, the last ends at `duration_seconds`, all values rounded to
milliseconds. Consumers rely on this; see `tests/test_shot_boundary_contract.py`.

`shots` is the frame-accurate view before the conversion to a time partition,
and additionally carries each shot's `intra` label.

# The `shot_boundary` model type

The artifact is a self-contained `.pt2` (`torch.export`) or `.pt` (TorchScript)
module under `./models/`, exactly like every other model here — the server
contains no model-specific code.

**Signature**

```
forward(clip: [1, T, 3, H, W]) -> (shot_logits, intra_logits, inter_logits)
```

* `clip` is `T` consecutive frames, ImageNet-normalised, at the model's trained
  geometry.
* `shot_logits` is `[1, Q, T + 2]`: per query, a distribution over the
  clip-relative end frame. The final class is a "no object" slot and is dropped.
* `intra_logits` / `inter_logits` are `[1, Q, C]`: per query, the shot-type and
  transition-type class distributions, again with a trailing "no object" slot.

All three are dense and fixed-shape, which is what keeps the artifact
exportable. Everything else — windowing, the greedy query walk, context
pruning, cross-window merging, `clean_shot` filtering and boundary
normalization — is generic and lives in
`lib/model/ai_shot_boundary_model.py`.

**Label sidecar** — `./models/<model_file_name>.labels.json`, alongside the
artifact (mirroring how tagging models read `<model_file_name>.tags.txt`):

```json
{
  "window_frames": 100,
  "frame_height": 96,
  "frame_width": 128,
  "num_queries": 24,
  "intra": { "0": "General", "1": "Dissolve" },
  "inter": { "0": "New_Start", "1": "Hard_Cut" },
  "intra_name_to_index": { "general": 0 },
  "inter_name_to_index": { "new_start": 0 }
}
```

`intra_name_to_index` / `inter_name_to_index` let the generic code find the two
labels it must reason about without hard-coding them: the "general" shot type
used by `clean_shot`, and the "new start" transition used to join a shot that
continues across a window seam.

**Model yaml** (`config/models/<name>.yaml`, supplied by the user like all
model configs — this directory is gitignored):

```yaml
type: shot_boundary
model_file_name: omnishotcut_shot_boundaries
model_category: [shot_boundaries]
model_capabilities: [temporal_segmentation]
supported_target_scopes: [asset]
full_image_model: false
model_type: ShotBoundary
model_version: 1.0
model_image_size: null        # asset models declare no shared-preprocessor geometry
model_precision: float32
max_batch_size: 1
instance_count: 1

mode: clean_shot              # or "default" to keep transition shots
context_frames: 0             # frames of overlap between windows
preprocess_config:
  width: 128
  height: 96
  normalization: 0            # ImageNet
  device: cpu
  precision: float32
```

Optional keys: `window_frames` (defaults to the sidecar's value),
`general_intra_label` / `new_start_inter_label` (default `general` /
`new_start`), and `use_frame_timestamps` (default `false`; `index / fps` is used
for frame-to-time conversion).

An optional `decode:` block selects how frames are produced. Every key may be
omitted; the defaults are what the model runs with when the block is absent:

```yaml
decode:
  backend: auto        # auto | ffmpeg_cpu | ffmpeg_cuda | av
  quality: exact       # exact | fast
  device_index: null   # CUDA device for ffmpeg_cuda; null = default device
```

`auto` tries `ffmpeg_cpu` and falls back to `av`; it never selects NVDEC, for
the reasons under [Decoding](#decoding) below — set `backend: ffmpeg_cuda`
explicitly to use it. Naming a backend is a preference, not a pin: if it fails
its pre-flight checks the ladder still falls through to the next one, and the
backend actually used is reported as `decode_backend` in the result.

`quality: fast` replaces the GPU scaling pyramid with a single `scale_cuda`
step. It measured no faster and it aliases, so it exists only for diagnosis;
leave it at `exact`.

## Decoding

An asset-scope model that needs every frame is decode-bound, not compute-bound:
a 4K source costs far more to decode than to score. `ffmpeg_pipe.py` runs one
ffmpeg and reads raw RGB frames from its stdout, which is substantially faster
than decoding frame-by-frame through PyAV.

Measured on a 32-core host with an RTX 4090, against a 3840x2160 60fps H.264
source scaled to 128x96, and compared against the PyAV path the model was
validated on:

| backend | throughput | max \|delta\| vs PyAV |
|---|---|---|
| `ffmpeg_cpu` (default) | 417 fps | 3 |
| `ffmpeg_cuda` (NVDEC) | 275 fps | 40 |
| `av` (PyAV) | 108 fps | - |

Two results worth knowing before reaching for the GPU:

* **NVDEC is not automatically faster**, and which one wins is a property of
  the host. Same host and card, a heavier 4096x2160 60fps H.264 source, 3000
  frames, with the decode pinned to a varying number of cores via `taskset`
  (absolute numbers move with the file; the shape does not):

  | cores | `ffmpeg_cpu` | `ffmpeg_cuda` |
  |---|---|---|
  | 2 | 91 fps | 247 fps |
  | 4 | 165 fps | 248 fps |
  | 8 | 260 fps | 248 fps |
  | 16 | 313 fps | 247 fps |
  | 32 | 324 fps | 247 fps |

  NVDEC is flat: the work sits on the card's decoder engine, so host cores buy
  nothing. Software decode scales, steeply at first and then flattening as it
  approaches its single-stream limit. They cross at roughly **8 cores**. Pin
  `ffmpeg_cuda` below that; above it, prefer the default. The other reason to pin
  it is that its 247 fps costs almost no CPU, which matters when analysis runs
  alongside transcodes or scans rather than on an idle box.
* **NVDEC's frames differ.** `scale_cuda` uses a fixed 4x4 kernel with no
  ratio-adaptive widening, unlike swscale's bicubic, so its output is not
  interchangeable with the software path and can move a borderline boundary.

For that reason `backend: auto` never selects NVDEC; it must be requested
explicitly. When it is, the descent to the target size is done as a sequence of
GPU halvings followed by one small swscale step, so no single step exceeds 2x and
the final kernel matches the software path. That pyramid is free — it measured
274.5 fps against 275.8 fps for a single `scale_cuda`.

Frame count must match a software decode exactly, because consumers map frame
index to time as `index / fps`. The pipeline therefore uses `-fps_mode
passthrough` and a `select` frame modulo, never the `fps` filter, which
resamples to a wall-clock rate. A short read is raised as an error rather than
treated as a short video: a truncated decode yields results that look entirely
plausible and are wrong.

Backends are tried in order and every check — binary features, codec and pixel
format, and a two-frame smoke decode with the real command — runs before the
first frame is produced, so a failure falls through to the next backend rather
than surfacing mid-stream.

## Notes for implementers

* Decoding happens at the exact clip geometry without preserving aspect ratio,
  reproducing an `ffmpeg -s WxH -pix_fmt rgb24` pipeline — this matters when
  matching a reference implementation's output. The conversion to RGB is left to
  the output pixel format rather than a separate `format` filter, so scaling and
  colour conversion happen in one swscale pass, as PyAV's `reformat` does.
* Only one window of frames is held at a time, so peak memory is
  `window_frames x 3 x H x W` regardless of video length.
* Decode and inference run in an executor. They take minutes, and running them
  inline would block the event loop and stall every other model — including the
  tagging pass over the same video.
* Inference runs in float32. A transformer with sinusoidal position encodings
  overflows in fp16.
