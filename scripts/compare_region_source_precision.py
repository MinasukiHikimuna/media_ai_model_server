"""Measure what keeping the region source as 8-bit does to face embeddings.

Face embeddings are cut from the region source: the frame at up to 1920 px,
which the server keeps as 8-bit instead of float32. This runs the real face
detector and embedder on keyframes of the given videos, in one process and
one frame at a time, and embeds every detected face from both a float32 and
an 8-bit region source. Nothing else differs between the two, so any
difference is the 8-bit source's alone::

    python scripts/compare_region_source_precision.py --videos videos.tsv

Frames are decoded at their full size and scaled down to the region source by
the server's own ``apply_spec`` (the case where 8-bit rounding can matter);
at 1920 px or less the 8-bit source is the decoded frame itself. The videos
file is a TSV of ``id<TAB>path`` rows. Run it from the server directory, which
has ``models/`` and ``config/``.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import av  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402

from lib.model.face_torch_export_model import _norm_crop_gpu, run_detection  # noqa: E402
from lib.pipeline.preprocess_spec import PreprocessSpec, apply_spec  # noqa: E402
from lib.utils.torch_export_loader import load_exported_module  # noqa: E402


def _keyframes(path, limit):
    """Up to ``limit`` keyframes as float32 CHW [0, 255] tensors, as decoded."""
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONKEY"
        for count, frame in enumerate(container.decode(stream)):
            if count >= limit:
                break
            yield torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1).float()


def _embed(model, source, kps, device):
    aligned = _norm_crop_gpu(source.to(device), kps, image_size=112).unsqueeze(0)
    aligned = F.interpolate(aligned, size=(112, 112), mode="bilinear", align_corners=False)
    with torch.no_grad():
        feat = model((aligned - 127.5) / 127.5)
    if isinstance(feat, (list, tuple)):
        feat = feat[0]
    return feat.detach().float().cpu().reshape(-1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", required=True, help="TSV of id<TAB>path rows")
    parser.add_argument("--frames", type=int, default=60, help="keyframes per video")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    detector_config = yaml.safe_load(Path("config/models/face_detector.yaml").read_text())
    device = torch.device(args.device)
    detector = load_exported_module("models/face_rec_detector.pt2", device)
    embedder = load_exported_module("models/face_rec_embedder.pt2", device)
    det_size = tuple(detector_config.get("det_size", [640, 640]))
    detector_spec = PreprocessSpec(width=0, height=0, normalization=-1, device="cpu",
                                   half_precision=False, max_long_edge=detector_config.get("model_image_size", 640))
    source_8_bit = PreprocessSpec.for_region_source()
    source_float = replace(source_8_bit, precision="float32")

    print(f"{'video':>8} {'size':>11} {'frames':>6} {'faces':>6} {'max px diff':>11} "
          f"{'min cos':>9} {'mean cos':>9}")
    all_cosines = []
    for row in Path(args.videos).read_text().splitlines():
        if not row.strip():
            continue
        video_id, path = row.split("\t", 1)
        cosines, pixel_diff, frames, size = [], 0.0, 0, "?"
        for frame in _keyframes(path, args.frames):
            frames += 1
            size = f"{frame.shape[2]}x{frame.shape[1]}"
            detector_input = apply_spec(frame, detector_spec)
            as_float, as_8_bit = apply_spec(frame, source_float), apply_spec(frame, source_8_bit)
            pixel_diff = max(pixel_diff, float((as_8_bit.float() - as_float).abs().max()))
            det, kpss = run_detection(detector, detector_input, det_size=det_size,
                                      det_thresh=float(detector_config.get("model_threshold", 0.5)),
                                      nms_thresh=float(detector_config.get("det_nms_thresh", 0.4)), device=device)
            if kpss is None:
                continue
            scale_x = as_float.shape[2] / detector_input.shape[2]
            scale_y = as_float.shape[1] / detector_input.shape[1]
            for kps in kpss:
                kps = np.asarray(kps, dtype=np.float32) * np.array([scale_x, scale_y], dtype=np.float32)
                a, b = _embed(embedder, as_float, kps, device), _embed(embedder, as_8_bit, kps, device)
                cosines.append(float(F.cosine_similarity(a, b, dim=0)))
        all_cosines += cosines
        low = min(cosines) if cosines else float("nan")
        mean = sum(cosines) / len(cosines) if cosines else float("nan")
        print(f"{video_id:>8} {size:>11} {frames:>6} {len(cosines):>6} {pixel_diff:>11.3f} {low:>9.6f} {mean:>9.6f}",
              flush=True)
    if all_cosines:
        print(f"\n{len(all_cosines)} faces: lowest cosine {min(all_cosines):.6f}, "
              f"mean {sum(all_cosines) / len(all_cosines):.6f}")


if __name__ == "__main__":
    sys.exit(main())
