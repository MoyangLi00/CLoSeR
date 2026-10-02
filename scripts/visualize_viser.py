"""Standalone viser launcher for bundles saved by `demo_viser.py --save_viser_data`.

Loads `<stem>.pt` (predictions) and `<stem>_viser_config.json` (wrapper kwargs)
written side-by-side in `--output_folder`, then calls `viser_wrapper` exactly
the way demo_viser would have at the end of inference. No PyTorch model is
loaded, no inference is run — purely a replay of the saved point clouds.

Usage:
    # by config json (recommended):
    python scripts/visualize_viser.py \
        --config /path/to/results/<stem>_viser_config.json

    # or by .pt directly (auto-discovers the sibling _viser_config.json if any):
    python scripts/visualize_viser.py \
        --predictions /path/to/results/<stem>.pt

CLI overrides take precedence over the JSON values: e.g. `--port 8181` to
launch on a different port without editing the file.
"""
import argparse
import json
import os
import sys

import torch

# Allow `from loger.utils.viser_utils import viser_wrapper` when run from any cwd.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from loger.utils.viser_utils import viser_wrapper  # noqa: E402

# Built-in default for --subsample. Edit here to change the launcher-wide
# default. Values from the JSON config are intentionally ignored for this
# field — subsample is a viewing preference, not a property of the saved data.
DEFAULT_SUBSAMPLE = 1

# 55-340 for campus_train0

def load_predictions(pt_path):
    """Load <stem>.pt and convert tensors to numpy (mirrors demo_viser's --load path)."""
    print(f"Loading predictions from {pt_path}")
    pred = torch.load(pt_path, map_location="cpu", weights_only=False)
    for k, v in list(pred.items()):
        if isinstance(v, torch.Tensor):
            pred[k] = v.numpy()
        elif isinstance(v, dict):  # multi-cam: cam01..cam05
            for sk, sv in list(v.items()):
                if isinstance(sv, torch.Tensor):
                    pred[k][sk] = sv.numpy()
    return pred


# Per-frame fields shared by every camera (cam0 + optional cam01..cam05).
_PER_FRAME_KEYS = ("images", "points", "conf", "camera_poses", "camera_poses_pre_pgo")


def trim_predictions(pred, start, end):
    """Slice every per-frame field of `pred` to ``[start:end]`` (in-place).

    Stacks before `frame_stride`: this trims the loaded sequence to a window
    of interest, and `viser_wrapper`'s `frame_stride` then strides whatever's
    left. Trajectory length printed for sanity.
    """
    # Pre-trim length (any per-frame field will do).
    n_pre = None
    for k in _PER_FRAME_KEYS:
        v = pred.get(k)
        if v is not None and hasattr(v, "shape") and len(v.shape) >= 1:
            n_pre = int(v.shape[0])
            break
    if n_pre is None:
        return pred  # nothing to trim

    # Resolve None-end to "all" and clamp to valid range; support negative idx.
    s = 0 if start is None else int(start)
    e = n_pre if end is None or int(end) < 0 else int(end)
    s = max(0, min(s, n_pre))
    e = max(s, min(e, n_pre))
    if s == 0 and e == n_pre:
        return pred  # no-op

    for k in _PER_FRAME_KEYS:
        v = pred.get(k)
        if v is not None and hasattr(v, "shape") and len(v.shape) >= 1:
            pred[k] = v[s:e]
    for cam_key in (f"cam{i:02d}" for i in range(1, 6)):
        sub = pred.get(cam_key)
        if isinstance(sub, dict):
            for k in _PER_FRAME_KEYS:
                v = sub.get(k)
                if v is not None and hasattr(v, "shape") and len(v.shape) >= 1:
                    sub[k] = v[s:e]
    print(f"Trimmed predictions to [{s}:{e}] → {e - s}/{n_pre} frames")
    return pred


def resolve_paths(args):
    """Return (pt_path, cfg_dict). cfg_dict is {} if no JSON found."""
    cfg = {}
    cfg_path = None

    if args.config:
        cfg_path = args.config
    elif args.predictions:
        # Auto-discover sibling _viser_config.json next to the .pt
        stem, _ = os.path.splitext(args.predictions)
        candidate = stem + "_viser_config.json"
        if os.path.isfile(candidate):
            cfg_path = candidate

    if cfg_path is not None:
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"viser config not found: {cfg_path}")
        print(f"Loading viser config from {cfg_path}")
        with open(cfg_path) as f:
            cfg = json.load(f)

    if args.predictions:
        pt_path = args.predictions
    elif "predictions_pt" in cfg:
        pt_path = os.path.join(os.path.dirname(cfg_path), cfg["predictions_pt"])
    else:
        raise ValueError("Provide --predictions or a --config that includes 'predictions_pt'.")
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"predictions .pt not found: {pt_path}")
    return pt_path, cfg


def merge_kwargs(cfg, args):
    """JSON values are defaults; CLI overrides win when explicitly set."""
    def _pick(key, cli_value, default):
        if cli_value is not None:
            return cli_value
        if key in cfg:
            return cfg[key]
        return default

    return dict(
        port=int(_pick("port", args.port, 8080)),
        init_conf_threshold=float(_pick("init_conf_threshold", args.conf_threshold, 50.0)),
        background_mode=bool(_pick("background_mode", args.background_mode, False)),
        mask_sky=bool(_pick("mask_sky", args.mask_sky, False)),
        image_folder_for_sky_mask=_pick(
            "image_folder_for_sky_mask", args.image_folder_for_sky_mask, None
        ),
        # `subsample` deliberately ignores the JSON config — it's a viewing
        # preference, not a property of the saved data. Take it from the CLI
        # if explicitly passed, otherwise the launcher-wide DEFAULT_SUBSAMPLE.
        subsample=int(args.subsample),
        frame_stride=int(args.frame_stride),
        video_width=int(_pick("video_width", args.video_width, 320)),
        share=bool(_pick("share", args.share, False)),
        canonical_first_frame=bool(
            _pick("canonical_first_frame", args.canonical_first_frame, True)
        ),
        point_size=float(_pick("point_size", args.point_size, 0.001)),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=str, default=None,
                     help="Path to <stem>_viser_config.json. The sibling .pt is "
                          "found via the JSON's 'predictions_pt' field.")
    src.add_argument("--predictions", type=str, default=None,
                     help="Path to <stem>.pt directly. Auto-loads the sibling "
                          "<stem>_viser_config.json if it exists.")
    # Optional CLI overrides (None = take from JSON / wrapper default).
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--conf_threshold", type=float, default=None)
    ap.add_argument("--background_mode", action="store_true", default=None)
    ap.add_argument("--mask_sky", action="store_true", default=None)
    ap.add_argument("--image_folder_for_sky_mask", type=str, default=None)
    ap.add_argument("--subsample", type=int, default=DEFAULT_SUBSAMPLE,
                    help=f"Spatial subsample factor for point clouds (drops every Nth "
                         f"pixel along H and W). Launcher default = {DEFAULT_SUBSAMPLE}; "
                         f"the JSON config's 'subsample' is intentionally ignored.")
    ap.add_argument("--frame_stride", type=int, default=1,
                    help="Keep every Nth frame for viser display (temporal subsample). "
                         "Default 1 — long sequences over an SSH tunnel are otherwise "
                         "unusable. Set 1 to upload every frame.")
    ap.add_argument("--start_load_index", type=int, default=0,
                    help="First frame index (inclusive) to load from the .pt. "
                         "Trims the trajectory before `frame_stride` is applied. "
                         "Default 0.")
    ap.add_argument("--end_load_index", type=int, default=-1,
                    help="Last frame index (exclusive) to load from the .pt; "
                         "-1 means load through the final frame. Default -1.")
    ap.add_argument("--video_width", type=int, default=None)
    ap.add_argument("--share", action="store_true", default=None)
    ap.add_argument("--canonical_first_frame", action="store_true", default=None)
    ap.add_argument("--point_size", type=float, default=None)
    args = ap.parse_args()

    pt_path, cfg = resolve_paths(args)
    pred_dict = load_predictions(pt_path)
    # Informational: report any save-time stride recorded in the JSON. The
    # bundle on disk is already strided by this factor — visualize_viser.py
    # does NOT re-apply it. Combine with `--frame_stride N` to further sparsen
    # at viser display time.
    if "save_stride" in cfg:
        _ss = int(cfg["save_stride"])
        if _ss > 1:
            print(f"Bundle was saved with save_stride={_ss} (per-frame fields "
                  f"already strided in the .pt; no re-application).")
    pred_dict = trim_predictions(pred_dict, args.start_load_index, args.end_load_index)
    kwargs = merge_kwargs(cfg, args)

    print(f"Launching viser_wrapper(port={kwargs['port']}, "
          f"subsample={kwargs['subsample']}, mask_sky={kwargs['mask_sky']}, "
          f"share={kwargs['share']})")
    viser_wrapper(pred_dict, **kwargs)


if __name__ == "__main__":
    main()
