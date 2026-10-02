import os
import sys
import glob
import time
import threading
import argparse
import inspect
import math
import tempfile
import shutil
import yaml
import torch
import cv2
from PIL import Image
from torchvision import transforms
from natsort import natsorted
from typing import List, Optional

from pathlib import Path
from loger.utils.rotation import mat_to_quat
from loger.utils.geometry import depth_edge
from loger.models.pi3 import Pi3
from loger.utils.viser_utils import viser_wrapper


# Helper function to check if a path is a video file
def is_video_file(path):
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in video_extensions

# Helper function to extract frames from video
def extract_frames_from_video(video_path, output_dir, start_frame, end_frame, stride):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame_idx = 0
    saved_frame_count = 0
    image_paths = []

    actual_end_frame = total_frames -1 if end_frame == -1 else end_frame
    if actual_end_frame >= total_frames:
        print(f"Warning: end_frame ({actual_end_frame}) is beyond total frames ({total_frames-1}). Adjusting to last frame.")
        actual_end_frame = total_frames - 1

    print(f"Extracting frames from {video_path}: start={start_frame}, end={actual_end_frame}, stride={stride}")

    while True:
        ret, frame = cap.read()
        if not ret or current_frame_idx > actual_end_frame:
            break

        if current_frame_idx >= start_frame and (current_frame_idx - start_frame) % stride == 0:
            frame_filename = f"frame_{saved_frame_count:06d}.png"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            image_paths.append(frame_path)
            saved_frame_count += 1
        
        current_frame_idx += 1

    cap.release()
    print(f"Successfully extracted {saved_frame_count} frames to {output_dir}.")
    return natsorted(image_paths)

parser = argparse.ArgumentParser(description="Pi3 demo with viser for 3D visualization")
parser.add_argument(
    "--input", type=str, default="data/examples/office", help="Path to input (folder of images or a video file)"
)
parser.add_argument(
    "--input2", type=str, default=None, help="Path to input (folder of images or a video file)"
)
parser.add_argument(
    "--input3", type=str, default=None, help="Path to input (folder of images or a video file)"
)
parser.add_argument(
    "--input4", type=str, default=None, help="Path to input (folder of images or a video file)"
)
parser.add_argument(
    "--input5", type=str, default=None, help="Path to input (folder of images or a video file)"
)
parser.add_argument("--start_frame", type=int, default=0, help="Start frame for video processing")
parser.add_argument("--end_frame", type=int, default=-1, help="End frame for video processing (-1 for last frame)")
parser.add_argument("--stride", type=int, default=1, help="Stride for frame extraction/loading")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
parser.add_argument("--share", action="store_true", help="Share the viser server with others")
parser.add_argument(
    "--conf_threshold", type=float, default=20.0, help="Initial confidence threshold (percentage)"
)
parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
parser.add_argument(
    "--output_folder", type=str, default='./results_pi3',
    help="Folder for the --save_viser_data bundle (and the default loop_detection/ dir)"
)
parser.add_argument(
    "--load", type=str, default=None, help="Path to folder or .pt file to load pre-computed inference results from"
)
parser.add_argument("--seq_name", type=str, default=None, help="Name of the sequence for saving results")
parser.add_argument("--subsample", type=int, default=2, help="Subsample the point cloud for visualization by this factor")
parser.add_argument("--frame_stride", type=int, default=10,
                    help="Keep every Nth frame for viser display (temporal subsample). "
                         "Default 10 — long sequences over an SSH tunnel are otherwise "
                         "unusable. Set 1 to upload every frame.")
parser.add_argument("--video_width", type=int, default=320, help="Width of the video display in the GUI")
parser.add_argument("--skip_viser", action="store_true", help="Skip viser visualization and only run inference")
parser.add_argument("--save_viser_data", action="store_true",
                    help="Save a bundle for offline viser playback (scripts/visualize_viser.py) in "
                         "--output_folder: <stem>.pt with points/conf/camera poses/images, plus "
                         "<stem>_viser_config.json with the viser_wrapper kwargs. This is the only "
                         "way a predictions .pt is written.")
parser.add_argument("--save_stride", type=int, default=10,
                    help="--save_viser_data only: keep every Nth frame of every per-frame field "
                         "(points, conf, camera poses, images) in the bundle. The live viser and the "
                         "trajectory .txt always use every frame (thin the live display with --frame_stride).")
parser.add_argument(
    "--model_name",
    type=str,
    default="ckpts/LoGeR_star/latest.pt",
    help="Name of the model to load from Hugging Face Hub or a local path to a checkpoint."
)
parser.add_argument("--config", type=str, default="ckpts/LoGeR_star/original_config.yaml", help="Path to a yaml config file for model initialization.")
parser.add_argument("--resolution", type=list, default=None, help="Target resolution for input images (shorter side).")
parser.add_argument("--window_size", type=int, default=32, help="Window size for non-causal inference (-1 for full sequence).")
parser.add_argument("--overlap_size", type=int, default=3, help="Overlap size for sliding window inference.")
parser.add_argument("--sim3", action="store_true", help="Use sim3 transformation for TTT.")
parser.add_argument("--sim3_scale_mode", type=str, default="median", choices=["median", "trimmed_mean", "median_all", "trimmed_mean_all", "sim3_avg1", "translation_magnitude", "pointcloud_umeyama", "w0_reference", "max_reference", "p75_reference", "median_reference", "run_max_reference"], help="Scale estimation mode for Sim3.")
parser.add_argument("--reset_every", type=int, default=None, help="Reset TTT / adapter state every N windows (0 disables).")
parser.add_argument("--output_txt", type=str, default=None, help="Output trajectory txt file path.")
parser.add_argument("--se3", action="store_true", default=None, help="Use se3 transformation for TTT. If omitted, fallback to config value, then False.")
parser.add_argument("--sim3_on_reset", action="store_true", default=False, help="When reset_every>0 and neither sim3/se3 is set, align each reset block with one Sim(3) instead of SE(3). Default False.")
parser.add_argument("--gt_scale", type=str, default=None, help="DEBUG: path to KITTI-format GT poses (12 floats/line). When set together with --sim3 or --sim3_on_reset, the per-window relative scale is forced to s_k/s_0 where s_k is the Umeyama-Sim3 scale of window k vs GT. Window 0 stays unchanged.")
parser.add_argument("--no_ttt", action="store_true", help="Disable TTT.")
parser.add_argument("--no_swa", action="store_true", help="Disable SWA.")
parser.add_argument("--pgo", action="store_true",
                    help="Run PGO with sequential edges even without loop-closure pairs.")
parser.add_argument("--pgo_dist_thresh", type=int, default=5, help="Max frame-index gap for PGO sequential edges.")
parser.add_argument("--pgo_sigma_seq",   type=float, default=0.01, help="Legacy isotropic sigma for sequential PGO edges (overridden by --pgo_sigma_R_seq/--pgo_sigma_t_seq).")
parser.add_argument("--pgo_sigma_lc",    type=float, default=0.1,  help="Legacy isotropic sigma for LC PGO edges (overridden by --pgo_sigma_R_lc/--pgo_sigma_t_lc).")
parser.add_argument("--pgo_sigma_R_seq", type=float, default=None, help="Per-axis sigma (rotation) for sequential PGO edges. None = use --pgo_sigma_seq.")
parser.add_argument("--pgo_sigma_t_seq", type=float, default=None, help="Per-axis sigma (translation) for sequential PGO edges.")
parser.add_argument("--pgo_sigma_R_lc",  type=float, default=0.005, help="Per-axis sigma (rotation) for LC PGO edges. Tight by default — bridge R is reliable.")
parser.add_argument("--pgo_sigma_t_lc",  type=float, default=0.1,   help="Per-axis sigma (translation) for LC PGO edges. Loose by default — bridge t magnitude is noisy.")
parser.add_argument("--pgo_lc_robust",   type=str, default="huber", choices=["huber", "cauchy", "none"], help="Robust kernel on LC noise. 'none' to disable.")
parser.add_argument("--pgo_lc_robust_k", type=float, default=1.345, help="Threshold for the LC robust kernel (Huber/Cauchy).")
parser.add_argument("--pgo_debug_save",  type=str, default=None,
                    help="Save PGO debug edges npz alongside output_txt (auto) or to this explicit path.")
parser.add_argument("--pgo_sigma_match", type=float, default=0.01,
                    help="Noise sigma for GT frame-match PGO constraints (identity relative pose).")
parser.add_argument("--pgo_no_match_constraints", action="store_true",
                    help="Disable GT frame-match identity constraints in PGO.")
parser.add_argument("--pgo_adj_constraints", action="store_true",
                    help="Enable bridge-derived adjacent constraints in PGO (disabled by default).")
parser.add_argument("--pgo_block_constraints", action="store_true",
                    help="Enable block-consistency constraints: within each reset block of `reset_every` windows, "
                         "add relative-pose edges between the middle frames of every pair of windows in the "
                         "block (relative pose taken from merged world-frame init; regulariser only).")
parser.add_argument("--pgo_sigma_block", type=float, default=0.05,
                    help="Noise sigma for block-consistency PGO edges.")
parser.add_argument("--pgo_block_middle_count", type=int, default=4,
                    help="Number of middle frames per window used for block-consistency edges.")
parser.add_argument("--pgo_lc_check_t", type=float, default=1.0,
                    help="Online-PGO loop consistency check: after each optimisation, reject a bridge "
                         "(all its LC edges) whose median translation residual exceeds this, in "
                         "trajectory units (the model's scale, not metres), then re-optimise. "
                         "Default 1.0 (best on KITTI / DROID-W / VBR / Oxford Spires); 0 = off.")
parser.add_argument("--pgo_lc_check_R", type=float, default=0.0,
                    help="Same check on the median rotation residual [deg]. Default 0 = off.")
parser.add_argument("--pgo_lc_check_iters", type=int, default=3,
                    help="Max reject + re-optimise rounds per online PGO call.")
parser.add_argument("--save_debug_poses", action="store_true",
                    help="Besides --output_txt, also write debug outputs next to it: <stem>_pre_pgo.txt, "
                         "<stem>_chunk_*.npy, <stem>_pgo_debug.npz (offline PGO) and "
                         "<stem>_pgo_inputs.pt (online PGO, for scripts/replay_online_pgo.py).")
parser.add_argument("--save_pgo_inputs", type=str, default=None,
                    help="Save all online-PGO inputs to this .pt (with --save_debug_poses the default is "
                         "<output_txt stem>_pgo_inputs.pt). Replay offline with "
                         "scripts/replay_online_pgo.py, e.g. to tune --pgo_lc_check_*.")
parser.add_argument("--online_pgo", action="store_true",
                    help="Run PGO inline: after each batch of bridge windows for one normal window "
                         "completes, run merge + PGO once with everything accumulated so far. The "
                         "post-loop offline PGO is skipped; intermediate PGO results overlay merged "
                         "camera_poses for the frames they cover. Output-only — corrected poses are "
                         "not fed back into subsequent window inference.")
parser.add_argument("--pi3x", action="store_true", help="Use Pi3X model.")
parser.add_argument("--pi3x_metric", action="store_true", default=True, help="Use metric scaling for Pi3X (default: True).")
parser.add_argument("--no_pi3x_metric", action="store_false", dest='pi3x_metric', help="Disable metric scaling for Pi3X.")
parser.add_argument("--canonical_first_frame", action="store_true", default=True, help="Use first frame as canonical frame (identity pose) for visualization.")
parser.add_argument("--no_canonical_first_frame", action="store_false", dest='canonical_first_frame', help="Do not use first frame as canonical frame.")
parser.add_argument("--warmup", action="store_true", help="Run a warmup inference pass to trigger torch.compile before timing.")
parser.add_argument("--benchmark", action="store_true", help="Run multiple inference passes and report timing statistics.")
parser.add_argument("--loop_detect", action="store_true",
                    help="Auto-detect loop closures from input images using SALAD (or a grayscale fallback). "
                         "Detected pairs are passed as lc_frame_pairs and PGO is enabled automatically.")
parser.add_argument("--loop_ckpt", type=str, default="ckpts/SALAD/dino_salad.ckpt",
                    help="Path to a SALAD checkpoint for loop detection. "
                         "If empty, falls back to grayscale-hash based detection.")
parser.add_argument("--loop_result_dir", type=str, default=None,
                    help="Directory to save loop detection results (default: <output_folder>/loop_detection).")
parser.add_argument("--loop_nms_threshold", type=int, default=25,
                    help="NMS suppression radius (in frames) for loop pair deduplication (default: 25).")
parser.add_argument("--loop_min_frame_gap", type=int, default=None,
                    help="Minimum frame index gap to consider a loop pair (default: 4*window_size).")
parser.add_argument("--loop_sim_thresh", type=float, default=0.7,
                    help="SALAD descriptor cosine-similarity threshold for accepting a loop pair "
                         "(higher => stricter, fewer pairs). Default 0.7.")
parser.add_argument("--loop_top_k", type=int, default=5,
                    help="Number of nearest descriptor neighbours retrieved per query frame "
                         "before thresholding (higher => more candidate pairs). Default 5.")
parser.add_argument("--gt_traj_path", type=str, default=None,
                    help="Path to a ground-truth trajectory txt file (timestamp tx ty tz qx qy qz qw) for evaluation. "
                         "Timestamps are matched to input frames using the timestamp loading logic in _try_load_timestamps_for_images().")
parser.add_argument("--lc_detect_use_nms", action="store_true", default=False,
                    help="Enable NMS for loop detection.")
parser.add_argument("--lc_bridge_nms", type=int, default=2,
                    help="Per window, run a bridge only for the loop window with the most frame "
                         "matches, drop candidate windows within +-N of it, and repeat (NMS). "
                         "Default 2 drops the 4 neighbouring windows. 0 = off (a bridge per candidate).")
parser.add_argument("--lc_bridge_keep_state", action="store_true",
                    help="Loop-closure bridge windows do not update the TTT/SWA state used by the "
                         "following windows (bridges only provide camera poses for PGO).")
parser.add_argument("--lc_chord_arc_thresh", type=float, default=0.85,
                    help="Inline chord/arc filter threshold for loop pairs: keep pair iff "
                         "chord/arc < thresh (lower => stricter, more pairs dropped). "
                         "Set to a value >=1.0 (e.g. 1.5) to disable the filter. "
                         "Default 0.80 matches pi3.py default.")
parser.add_argument("--loop_save_pair_vis", action="store_true", default=False,
                    help="Save side-by-side PNGs for every newly detected loop pair "
                         "(under <loop_result_dir>/loop_vis/pairs/). Off by default — "
                         "matplotlib + disk IO on the main thread can stall the GPU "
                         "for several seconds per window when many pairs land at once.")
parser.add_argument("--lc_pts_folder", type=str, default=None,
                    help="Directory to save per-bridge-window .pt files (points, conf, camera_poses). "
                         "Only saves LC bridge windows, not the full sequence. Much smaller than --output_folder.")
parser.add_argument("--save_online_pgo_poses_dir", type=str, default=None,
                    help="Directory to dump per-frame camera_poses snapshots taken before AND after each "
                         "inline online-PGO call. Files: pgo_call_<NNNN>_pre.pt / _post.pt. "
                         "Requires --online_pgo. Off by default.")
parser.add_argument("--save_window_local_pts_dir", type=str, default=None,
                    help="Directory to dump per-window local pointcloud .pt files (local_points, conf, "
                         "camera_poses, start_idx, end_idx) — one file window_<NNNN>.pt per normal "
                         "window. Off by default.")

def load_pi3_model(model_name: str, config_path: Optional[str] = None, pi3x: bool = False, pi3x_metric: bool = True):
    """Initializes the Pi3 model and loads weights."""
    print(f"Initializing Pi3 model...")

    model_kwargs = {}
    if config_path:
        print(f"Loading model configuration from: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            model_config = config.get('model', {})
            pi3_signature = inspect.signature(Pi3.__init__)
            valid_kwargs = {
                name
                for name, param in pi3_signature.parameters.items()
                if name not in {"self", "args", "kwargs"}
                and param.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }

            def _maybe_parse_sequence(value):
                if isinstance(value, str):
                    stripped = value.strip()
                    if stripped.startswith("[") and stripped.endswith("]"):
                        try:
                            parsed = yaml.safe_load(stripped)
                            if isinstance(parsed, (list, tuple)):
                                return list(parsed)
                        except Exception:
                            pass
                return value

            for key in sorted(valid_kwargs):
                if key in model_config:
                    value = model_config[key]
                    if key in {"ttt_insert_after", "attn_insert_after"}:
                        value = _maybe_parse_sequence(value)
                    model_kwargs[key] = value

            print("Model parameters from config:", model_kwargs)
        except Exception as e:
            print(f"Error loading or parsing config file {config_path}: {e}")
            print("Falling back to default model parameters.")
            model_kwargs = {}

    if pi3x:
        model_kwargs['pi3x'] = True
        model_kwargs['pi3x_metric'] = pi3x_metric
        if model_name == "yyfz233/Pi3":
            print("Switching default model to yyfz233/Pi3X because --pi3x is set.")
            model_name = "yyfz233/Pi3X"



    try:
        # Initialize model with parameters from config
        model = Pi3(**model_kwargs)

        if model_name.startswith("yyfz233/"):
            print("Loading pre-trained weights from Hugging Face Hub...")
            model = model.from_pretrained(model_name, strict=False if pi3x else True, **model_kwargs)
            print("Model loaded successfully from Hugging Face Hub.")
            return model
        
        # Load pre-trained weights
        print(f"Loading pre-trained weights from: {model_name}")
        # Use strict=False to allow for architecture mismatches when loading weights
        # This is useful when the config defines a different architecture than the saved checkpoint
        checkpoint = torch.load(model_name, map_location='cpu', weights_only=False)
        # If the checkpoint is a state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Adjust state_dict keys if they are prefixed (e.g., by DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # remove `module.`
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=True)
        
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Could not load model. Error: {e}")
        return None
        
    return model


def run_core_inference(
    model_obj: Pi3,
    input_paths: List[str],
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    target_resolution: List[int] = [504, 280],
):
    """
    Handles data preparation and runs the core model inference for Pi3.
    """
    model_obj.eval()
    model_obj = model_obj.to(device)
    
    temp_frame_dirs = {}
    input_indices = {}
    all_image_names = []
    
    for i, input_path in enumerate(input_paths):
        input_key = f"input{i+1}"
        if i > 0:
            input_indices[f"cam{i:02d}"] = len(all_image_names)
        
        if is_video_file(input_path):
            temp_dir = tempfile.mkdtemp(prefix=f"pi3_frames_{input_key}_")
            temp_frame_dirs[input_key] = temp_dir
            image_names_current_input = extract_frames_from_video(input_path, temp_dir, start_frame, end_frame, stride)
        elif os.path.isdir(input_path):
            image_names_current_input = natsorted(glob.glob(os.path.join(input_path, "*.png"))+glob.glob(os.path.join(input_path, "*.jpg"))+glob.glob(os.path.join(input_path, "*.jpeg")))
            end_idx = end_frame if end_frame != -1 else None
            image_names_current_input = image_names_current_input[start_frame:end_idx:stride]
        else:
            print(f"Warning: Input path {input_path} is not a valid video file or directory. Skipping.")
            image_names_current_input = []

        if not image_names_current_input:
            if input_key in temp_frame_dirs:
                if os.path.exists(temp_frame_dirs[input_key]): shutil.rmtree(temp_frame_dirs[input_key])
                del temp_frame_dirs[input_key]
            if f"cam{i:02d}" in input_indices: del input_indices[f"cam{i:02d}"]
        else:
            all_image_names.extend(image_names_current_input)

    if not all_image_names:
        print("Error: No images found from any input.")
        return None, [], {}, {}
        
    print(f"Loading images from combined inputs ({len(all_image_names)} images found)...")
    tw, th = int(target_resolution[0]), int(target_resolution[1])
    print(f"All images will be resized to a uniform size: ({tw}, {th})")
    images_tensor = LazyImageTensor(all_image_names, tw, th)
    print(f"Preprocessed images tensor shape: {images_tensor.shape}")

    print("Running inference...")    
    dtype = select_autocast_dtype(device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
        raw_model_predictions = model_obj(images_tensor[None]) # Add batch dimension
    
    # Post-process predictions
    raw_model_predictions['images'] = images_tensor[None].permute(0, 1, 3, 4, 2) # B, S, H, W, C
    raw_model_predictions['conf'] = torch.sigmoid(raw_model_predictions['conf'])
    edge = depth_edge(raw_model_predictions['local_points'][..., 2], rtol=0.03)
    raw_model_predictions['conf'][edge] = 0.0
    if 'local_points' in raw_model_predictions:
        del raw_model_predictions['local_points']

    return raw_model_predictions, all_image_names, input_indices, temp_frame_dirs


def _try_load_timestamps_for_images(image_paths, input_rgb_dir: Path):
    """Best-effort timestamp loader.

    Priority:
    1) <parent>/rgb.txt (TUM style: "timestamp rgb/xxxxx.png")
    2) <input_rgb_dir>/timestamps.txt (one timestamp per line)
    3) Fallback to sequential indices starting at 0
    """
    # 1) TUM-format rgb.txt in the parent directory
    if input_rgb_dir.is_file(): # if input is a video file
        return [float(i) for i in range(len(image_paths))]

    rgb_txt_path = input_rgb_dir.parent / "rgb.txt"
    if rgb_txt_path.exists():
        name_to_ts = {}
        with open(rgb_txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    ts = float(parts[0])
                except ValueError:
                    continue
                img_rel = parts[1]
                # Map by basename for robustness
                name_to_ts[Path(img_rel).name] = ts

        ts_list = []
        for p in image_paths:
            ts_list.append(name_to_ts.get(Path(p).name, None))
        if all(t is not None for t in ts_list) and len(ts_list) == len(image_paths):
            return ts_list
        # If partial or mismatch, fall through to next option

    # 2) timestamps.txt alongside images
    timestamps_txt = input_rgb_dir / "timestamps.txt"
    if timestamps_txt.exists():
        with open(timestamps_txt, "r") as f:
            raw_lines = [l.strip() for l in f.readlines() if l.strip() and not l.strip().startswith("#")]
        # Take as many as needed in order
        ts_list = []
        for i in range(min(len(raw_lines), len(image_paths))):
            try:
                ts_list.append(float(raw_lines[i]))
            except ValueError:
                ts_list.append(float(i))
        # If fewer timestamps than images, pad with indices
        for i in range(len(ts_list), len(image_paths)):
            ts_list.append(float(i))
        return ts_list

    # 3) Fallback: sequential indices as timestamps
    return [float(i) for i in range(len(image_paths))]


def write_trajectory_txt(output_path: Path, timestamps, translations, quaternions):
    """Write trajectory file with lines: ts tx ty tz qx qy qz qw"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, t, q in zip(timestamps, translations, quaternions):
            f.write(
                f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n"
            )


def load_images_from_paths(image_paths, PIXEL_LIMIT=255000, Target_W=None, Target_H=None, verbose=True):
    sources = []
    for img_path in image_paths:
        try:
            sources.append(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")

    if not sources:
        print("No images found or loaded.")
        return torch.empty(0)

    if Target_W is None and Target_H is None:
        first_img = sources[0]
        W_orig, H_orig = first_img.size
        scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
        W_target, H_target = W_orig * scale, H_orig * scale
        k, m = round(W_target / 14), round(H_target / 14)
        while (k * 14) * (m * 14) > PIXEL_LIMIT:
            if k / m > W_target / H_target: k -= 1
            else: m -= 1
        TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    else:
        TARGET_W, TARGET_H = Target_W, Target_H
    
    if verbose:
        print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    tensor_list = []
    to_tensor_transform = transforms.ToTensor()
    
    for img_pil in sources:
        try:
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        return torch.empty(0)

    return torch.stack(tensor_list, dim=0)


def _infer_target_resolution(first_image_path, pixel_limit=255000):
    """Peek at one image to determine the target (W, H) used by load_images_from_paths."""
    img = Image.open(first_image_path).convert('RGB')
    W_orig, H_orig = img.size
    scale = math.sqrt(pixel_limit / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    k, m = round(W_orig * scale / 14), round(H_orig * scale / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > W_orig / H_orig:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


class _LazyBatch:
    """Wraps LazyImageTensor with a prepended batch dimension of 1.

    Supports the ``imgs[:, start:end]`` slicing pattern used inside the model's
    forward pass so that only the frames required for a given window are loaded
    from disk at a time.
    """

    def __init__(self, lazy: "LazyImageTensor"):
        self._lazy = lazy
        N, C, H, W = lazy.shape
        self.shape = (1, N, C, H, W)

    def dim(self):
        return len(self.shape)

    def __getitem__(self, idx):
        if isinstance(idx, tuple) and len(idx) == 2:
            _, frame_idx = idx
            if isinstance(frame_idx, slice):
                N = self._lazy.shape[0]
                start = frame_idx.start if frame_idx.start is not None else 0
                stop = frame_idx.stop if frame_idx.stop is not None else N
                stop = min(stop, N)
                return self._lazy._load(start, stop).unsqueeze(0)
            if isinstance(frame_idx, (torch.Tensor, list)):
                return self._lazy._load_indices(frame_idx).unsqueeze(0)
        raise NotImplementedError(f"_LazyBatch: unsupported index {idx!r}")

    def permute(self, *dims):
        """Load all frames and permute — only call after inference when GPU memory is free."""
        tensor = self._lazy._load(0, self._lazy.shape[0])
        return tensor.unsqueeze(0).permute(*dims)


class LazyImageTensor:
    """Stores image paths and loads frames from disk only when a window is requested.

    Avoids keeping the full sequence in CPU or GPU memory.  The model's forward
    pass accesses frames via ``imgs[:, start:end]`` which triggers ``_LazyBatch.__getitem__``
    and loads just those frames.
    """

    def __init__(self, image_paths, target_w: int, target_h: int):
        self.image_paths = list(image_paths)
        self.target_w = target_w
        self.target_h = target_h
        self.shape = (len(image_paths), 3, target_h, target_w)

    def numel(self):
        N, C, H, W = self.shape
        return N * C * H * W

    def _load(self, start: int, stop: int) -> torch.Tensor:
        paths = self.image_paths[start:stop]
        if not paths:
            return torch.empty(0, 3, self.target_h, self.target_w)
        return load_images_from_paths(
            paths, Target_W=self.target_w, Target_H=self.target_h, verbose=False
        )

    def _load_indices(self, indices) -> torch.Tensor:
        """Load specific frames by index list/tensor."""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        if not indices:
            return torch.empty(0, 3, self.target_h, self.target_w)
        paths = [self.image_paths[i] for i in indices]
        return load_images_from_paths(
            paths, Target_W=self.target_w, Target_H=self.target_h, verbose=False
        )

    def __getitem__(self, idx):
        if idx is None:
            return _LazyBatch(self)
        raise NotImplementedError(f"LazyImageTensor: unsupported index {idx!r}")


def select_autocast_dtype(device):
    """Autocast dtype for inference: BF16, which requires compute capability >= 8.

    FP16 is not supported: its range (max 65504) overflows in the TTT layers and the
    camera poses become NaN after the first window.
    """
    if not torch.cuda.is_available():
        return torch.float16
    major, minor = torch.cuda.get_device_capability(device)
    if major < 8:
        red, reset = ("", "") if os.environ.get("NO_COLOR") else ("\033[91m", "\033[0m")
        sys.exit(f"{red}Error: {torch.cuda.get_device_name(device)} (compute capability {major}.{minor}) "
                 f"does not support BF16. FP16 inference overflows and produces NaN poses. "
                 f"Use a GPU with compute capability >= 8 (e.g. RTX 4090, A100).{reset}")
    return torch.bfloat16


def main():
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Output plan --------------------------------------------------------
    #   live viser (default)   : full, frame-aligned predictions; display thinning via --frame_stride
    #   --save_viser_data      : bundle <output_folder>/<stem>.pt + <stem>_viser_config.json,
    #                            every per-frame field strided by --save_stride
    #   --skip_viser alone     : camera poses only -> --output_txt
    # World-space points / conf / images are only computed when one of the first two needs them.
    args.need_points = (not args.skip_viser) or args.save_viser_data
    # Frames kept in the bundle; applied already during inference when there is no live viser.
    args.bundle_stride = max(1, int(args.save_stride)) if args.save_viser_data else 1
    if args.save_viser_data and not args.output_folder:
        sys.exit("--save_viser_data needs --output_folder (where the bundle is written).")
    print(f"Using device: {device}")

    # Fail fast for local LoGeR checkpoints/configs when files are missing.
    if args.config and not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not os.path.isfile(args.model_name):
        raise FileNotFoundError(f"Checkpoint file not found: {args.model_name}")

    predictions_dict = None
    _full_cam_poses_np = _full_cam_poses_pre_pgo_np = None   # full-resolution poses for the .txt
    temp_frame_dirs = {}
    input_indices = {}
    image_folder_for_sky = None
    target_resolution = args.resolution if args.resolution and len(args.resolution) == 2 else None
    
    # Generate seq_name automatically if not provided, similar to demo_viser.py
    if args.seq_name is None:
        args.seq_name = os.path.basename(os.path.dirname(args.input)) + "_" + os.path.basename(args.input)
        if args.input2:
            args.seq_name += f"_{os.path.basename(os.path.dirname(args.input2))}_{os.path.basename(args.input2)}"
        if args.input3:
            args.seq_name += f"_{os.path.basename(os.path.dirname(args.input3))}_{os.path.basename(args.input3)}"
        if args.input4:
            args.seq_name += f"_{os.path.basename(os.path.dirname(args.input4))}_{os.path.basename(args.input4)}"
        if args.input5:
            args.seq_name += f"_{os.path.basename(os.path.dirname(args.input5))}_{os.path.basename(args.input5)}"
    
    if args.load:
        saved_predictions_path = args.load
        if os.path.isdir(saved_predictions_path):
            if args.seq_name and os.path.exists(os.path.join(saved_predictions_path, f"{args.seq_name}.pt")):
                saved_predictions_path = os.path.join(saved_predictions_path, f"{args.seq_name}.pt")
            else:
                saved_predictions_path = os.path.join(saved_predictions_path, "predictions.pt")
        
        if os.path.exists(saved_predictions_path):
            print(f"Loading pre-computed results from {saved_predictions_path}...")
            try:
                predictions_dict = torch.load(saved_predictions_path, map_location="cpu", weights_only=False)
                print("Successfully loaded pre-computed results.")
                # Convert loaded tensors to numpy format for compatibility
                for key, value in predictions_dict.items():
                    if isinstance(value, torch.Tensor):
                        predictions_dict[key] = value.numpy()
                    elif isinstance(value, dict):  # Handle nested dictionaries (e.g., cam01, cam02)
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, torch.Tensor):
                                predictions_dict[key][subkey] = subvalue.numpy()
                image_folder_for_sky = args.input # Assume first input is the reference for sky mask
            except Exception as e:
                print(f"Error loading {saved_predictions_path}: {e}. Proceeding with inference.")
                predictions_dict = None 
        else:
            print(f"No pre-computed results found at {saved_predictions_path}. Proceeding with inference.")

    if predictions_dict is None:
        model = load_pi3_model(args.model_name, args.config, args.pi3x, args.pi3x_metric)
        if model is None:
            print("Failed to load model. Exiting.")
            return

        # Move model to device early
        model = model.to(device)
        #model.eval()
        model = model.eval()

        input_paths = [p for p in [args.input, args.input2, args.input3, args.input4, args.input5] if p is not None]
        
        all_image_names_collected = []
        input_indices = {}
        
        for i, input_path in enumerate(input_paths):
            if i > 0: input_indices[f"cam{i:02d}"] = len(all_image_names_collected)

            if is_video_file(input_path):
                # Video frames for each input are extracted to its own sub-folder
                temp_dir = tempfile.mkdtemp(prefix=f"pi3_frames_input{i+1}_")
                temp_frame_dirs[f"input{i+1}"] = temp_dir
                current_frames = extract_frames_from_video(input_path, temp_dir, args.start_frame, args.end_frame, args.stride)
                all_image_names_collected.extend(current_frames)
            elif os.path.isdir(input_path):
                current_frames = natsorted(glob.glob(os.path.join(input_path, "*.png"))+glob.glob(os.path.join(input_path, "*.jpg"))+glob.glob(os.path.join(input_path, "*.jpeg")))
                # remove the files that has depth in the name
                current_frames = [f for f in current_frames if "depth" not in os.path.basename(f).lower()]
                end_idx = args.end_frame if args.end_frame != -1 else None
                current_frames = current_frames[args.start_frame:end_idx:args.stride]
                all_image_names_collected.extend(current_frames)
        
        if not all_image_names_collected:
            print("No images to process. Exiting.")
            return
            
        print(f"Found {len(all_image_names_collected)} images to process.")
        if target_resolution is not None:
            target_w, target_h = int(target_resolution[0]), int(target_resolution[1])
        else:
            target_w, target_h = _infer_target_resolution(all_image_names_collected[0])
        print(f"All images will be resized to a uniform size: ({target_w}, {target_h})")
        images_tensor = LazyImageTensor(all_image_names_collected, target_w, target_h)

        image_folder_for_sky = os.path.dirname(all_image_names_collected[0]) if all_image_names_collected else None

        if not all_image_names_collected:
            print("Error: No images were loaded successfully. Check image paths and formats.")
            return

        print("Running inference...")
        dtype = select_autocast_dtype(device)
        num_frames = images_tensor.shape[0]
        
        forward_kwargs = {}
        if args.config:
            try:
                with open(args.config, 'r') as f:
                    config = yaml.safe_load(f)

                training_settings = config.get('training_settings', {})
                model_settings = config.get('model', {})
                se3_from_config = model_settings.get('se3', config.get('se3', False))
                se3_value = args.se3 if args.se3 is not None else bool(se3_from_config)
                forward_kwargs.update({
                    'window_size': args.window_size if args.window_size is not None else training_settings.get('window_size', -1),
                    'overlap_size': args.overlap_size if args.overlap_size is not None else training_settings.get('overlap_size', 0),
                    'reset_every': args.reset_every if args.reset_every is not None else training_settings.get('reset_every', 0),
                    'num_iterations': config.get('num_iterations', 1), # Or from training_settings
                    'sim3': config.get('sim3', False) or args.sim3,
                    'sim3_scale_mode': args.sim3_scale_mode,
                    'se3': se3_value,
                    'sim3_on_reset': args.sim3_on_reset or bool(config.get('sim3_on_reset', False)),
                    'turn_off_ttt': args.no_ttt,
                    'turn_off_swa': args.no_swa,
                })
                print(f"Forward pass kwargs from config: {forward_kwargs}")

            except Exception as e:
                print(f"Could not read config for forward pass arguments: {e}")
        elif args.window_size or args.overlap_size or args.sim3 or args.reset_every is not None:
            forward_kwargs.update({
                'window_size': args.window_size,
                'overlap_size': args.overlap_size,
                'window_size': args.window_size,
                'overlap_size': args.overlap_size,
                'sim3': args.sim3,
                'pi3x': args.pi3x,
                'pi3x_metric': args.pi3x_metric,
                'se3': bool(args.se3) if args.se3 is not None else False,
                'sim3_scale_mode': args.sim3_scale_mode,
                'sim3_on_reset': args.sim3_on_reset,
                'reset_every': args.reset_every if args.reset_every is not None else 0
            })

        if args.gt_scale:
            try:
                import numpy as _np
                gt_raw = _np.loadtxt(args.gt_scale)
                if gt_raw.ndim == 2 and gt_raw.shape[1] == 12:
                    gt_mat = gt_raw.reshape(-1, 3, 4).astype(_np.float32)
                    gt_pose = _np.tile(_np.eye(4, dtype=_np.float32), (gt_mat.shape[0], 1, 1))
                    gt_pose[:, :3, :4] = gt_mat
                    forward_kwargs['gt_poses_for_scale'] = torch.from_numpy(gt_pose)
                    print(f"[gt_scale] Loaded {gt_pose.shape[0]} GT poses from {args.gt_scale}")
                else:
                    print(f"[gt_scale] Expected (N,12) KITTI format, got shape {gt_raw.shape}; ignoring.")
            except Exception as e:
                print(f"[gt_scale] Failed to load {args.gt_scale}: {e}")

        forward_kwargs['run_pgo'] = args.pgo or args.loop_detect
        forward_kwargs['pgo_dist_thresh'] = args.pgo_dist_thresh
        forward_kwargs['pgo_sigma_seq']   = args.pgo_sigma_seq
        forward_kwargs['pgo_sigma_lc']    = args.pgo_sigma_lc
        forward_kwargs['pgo_sigma_R_seq'] = args.pgo_sigma_R_seq
        forward_kwargs['pgo_sigma_t_seq'] = args.pgo_sigma_t_seq
        forward_kwargs['pgo_sigma_R_lc']  = args.pgo_sigma_R_lc
        forward_kwargs['pgo_sigma_t_lc']  = args.pgo_sigma_t_lc
        forward_kwargs['pgo_lc_robust']   = None if args.pgo_lc_robust == "none" else args.pgo_lc_robust
        forward_kwargs['pgo_lc_robust_k'] = args.pgo_lc_robust_k
        forward_kwargs['pgo_sigma_match']        = args.pgo_sigma_match
        forward_kwargs['pgo_add_match_constraints'] = not args.pgo_no_match_constraints
        forward_kwargs['pgo_add_adj_constraints']   = args.pgo_adj_constraints
        forward_kwargs['pgo_add_block_constraints'] = args.pgo_block_constraints
        forward_kwargs['pgo_sigma_block']           = args.pgo_sigma_block
        forward_kwargs['pgo_block_middle_count']    = args.pgo_block_middle_count
        forward_kwargs['online_pgo']                = args.online_pgo
        forward_kwargs['pgo_lc_check_t']            = args.pgo_lc_check_t
        forward_kwargs['pgo_lc_check_R']            = args.pgo_lc_check_R
        forward_kwargs['pgo_lc_check_iters']        = args.pgo_lc_check_iters
        _spi = args.save_pgo_inputs
        if _spi is None and args.save_debug_poses and args.online_pgo and args.output_txt:
            _spi = str(Path(args.output_txt).with_suffix('')) + '_pgo_inputs.pt'
        if _spi:
            forward_kwargs['save_pgo_inputs'] = _spi
        if args.save_online_pgo_poses_dir:
            forward_kwargs['save_online_pgo_poses_dir'] = args.save_online_pgo_poses_dir
        if args.save_window_local_pts_dir:
            forward_kwargs['save_window_local_pts_dir'] = args.save_window_local_pts_dir
        # Bundle without a live viser: stride points/conf per window already inside
        # pi3 (drops ~12 GB of intermediate memory on long VBR-style runs). With a
        # live viser the predictions stay full and the bundle is strided at save time.
        if args.save_viser_data and args.skip_viser and args.bundle_stride > 1:
            forward_kwargs['inference_save_stride'] = args.bundle_stride
        # With --save_debug_poses: save debug npz alongside output_txt as <stem>_pgo_debug.npz
        _pgo_dbg = args.pgo_debug_save
        if _pgo_dbg is None and args.save_debug_poses and args.output_txt:
            _pgo_dbg = str(Path(args.output_txt).with_suffix('')) + '_pgo_debug.npz'
        if _pgo_dbg is not None:
            forward_kwargs['pgo_debug_save'] = _pgo_dbg

        # Pass loop-detection kwargs; loop_image_paths is populated from all_image_names
        # (the same paths used to build the image tensor).
        if args.loop_detect:
            _loop_result_dir = args.loop_result_dir
            if _loop_result_dir is None:
                _loop_result_dir = os.path.join(args.output_folder or './results_pi3', 'loop_detection')
            forward_kwargs['loop_detect'] = True
            forward_kwargs['loop_image_paths'] = all_image_names_collected
            forward_kwargs['loop_ckpt'] = args.loop_ckpt
            forward_kwargs['loop_result_dir'] = _loop_result_dir
            forward_kwargs['loop_nms_threshold'] = args.loop_nms_threshold
            _min_gap = args.loop_min_frame_gap
            if _min_gap is None:
                _min_gap = args.window_size * 4
            forward_kwargs['loop_min_frame_gap'] = _min_gap
            forward_kwargs['loop_sim_thresh'] = args.loop_sim_thresh
            forward_kwargs['loop_top_k'] = args.loop_top_k
            forward_kwargs['gt_traj_path'] = args.gt_traj_path
            forward_kwargs['lc_detect_use_nms'] = args.lc_detect_use_nms
            forward_kwargs['loop_save_pair_vis'] = args.loop_save_pair_vis
            forward_kwargs['lc_chord_arc_thresh'] = args.lc_chord_arc_thresh
            forward_kwargs['lc_bridge_keep_state'] = args.lc_bridge_keep_state
            forward_kwargs['lc_bridge_nms'] = args.lc_bridge_nms
            print(f"[LC] Auto loop detection enabled (ckpt='{args.loop_ckpt or 'fallback'}', "
                  f"result_dir={_loop_result_dir}),"
                  f"min_frame_gap={_min_gap}, "
                  f"sim_thresh={args.loop_sim_thresh}, "
                  f"top_k={args.loop_top_k}, "
                  f"use_nms={args.lc_detect_use_nms})")

        if args.lc_pts_folder:
            forward_kwargs['save_bridge_pts'] = True

        # For trajectory-only eval, skip accumulating large point/conf tensors per window
        # (~220 MB/window × 157 windows for a 4541-frame sequence = ~34 GB otherwise).
        # Exception: Sim3 scale modes that need per-pixel depth (all except
        # translation_magnitude) require local_points — otherwise they silently
        # fall back to scale=1.0 and produce no actual scale correction.
        # Read sim3 off forward_kwargs, not args: it can also be switched on by the
        # config file (`sim3: true`), in which case args.sim3 is False and local_points
        # would be dropped here — leaving Sim(3) alignment stuck at scale=1.0.
        _sim3_effective = bool(forward_kwargs.get('sim3', args.sim3)) or bool(forward_kwargs.get('sim3_on_reset', args.sim3_on_reset))
        _sim3_scale_mode_effective = forward_kwargs.get('sim3_scale_mode', args.sim3_scale_mode)
        _sim3_needs_local_pts = _sim3_effective and _sim3_scale_mode_effective != 'translation_magnitude'
        if not args.need_points:
            forward_kwargs['output_keys'] = (
                {'camera_poses', 'local_points'} if _sim3_needs_local_pts else {'camera_poses'})

        # Per-window striding of `local_points` and Sim(3) scale estimation are
        # incompatible: the estimator reads the depth of the overlap frame at local
        # index `Nw - overlap_size`, which the stride throws away, so every relative
        # scale silently comes back as 1.0 and Sim(3) degenerates to SE(3).
        if _sim3_needs_local_pts and 'inference_save_stride' in forward_kwargs:
            print(f"[sim3] dropping inference_save_stride={forward_kwargs['inference_save_stride']}: "
                  f"sim3_scale_mode='{_sim3_scale_mode_effective}' needs unstrided per-window local_points "
                  f"(pass --sim3_scale_mode translation_magnitude to keep the stride — it only uses camera poses)")
            forward_kwargs.pop('inference_save_stride')

        # Warmup run to trigger torch.compile (first run has compilation overhead)
        if args.warmup or args.benchmark:
            print("Running warmup inference (to trigger torch.compile)...")
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
                _ = model(images_tensor[None], **forward_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print("Warmup complete.")

        # Benchmark mode: run multiple times and report statistics
        if args.benchmark:
            num_runs = 3
            print(f"\nRunning benchmark with {num_runs} inference passes...")
            inference_times = []
            for run_idx in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_start = time.time()
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
                    raw_model_predictions = model(images_tensor[None], **forward_kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_end = time.time()
                inference_times.append(t_end - t_start)
                print(f"  Run {run_idx + 1}/{num_runs}: {t_end - t_start:.3f}s")
            
            avg_time = sum(inference_times) / len(inference_times)
            min_time = min(inference_times)
            max_time = max(inference_times)
            std_time = (sum((t - avg_time) ** 2 for t in inference_times) / len(inference_times)) ** 0.5
            
            print(f"\n{'='*50}")
            print(f"Benchmark Results ({num_runs} runs):")
            print(f"  Total frames: {num_frames}")
            print(f"  Avg inference time: {avg_time:.3f}s (std: {std_time:.3f}s)")
            print(f"  Min/Max: {min_time:.3f}s / {max_time:.3f}s")
            print(f"  Avg FPS: {num_frames / avg_time:.2f}")
            print(f"  Avg time per frame: {(avg_time / num_frames) * 1000:.2f} ms")
            print(f"{'='*50}\n")
            inference_time = avg_time
        else:
            # Single timed inference
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            inference_start_time = time.time()

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
                raw_model_predictions = model(images_tensor[None], **forward_kwargs) # Add batch dimension

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_end_time = time.time()

            peak_alloc_gb = peak_reserved_gb = float('nan')
            if torch.cuda.is_available():
                peak_alloc_gb    = torch.cuda.max_memory_allocated() / (1024**3)
                peak_reserved_gb = torch.cuda.max_memory_reserved()  / (1024**3)

            # Calculate and display timing
            inference_time = inference_end_time - inference_start_time
            fps = num_frames / inference_time
            ms_per_frame = (inference_time / num_frames) * 1000
            print(f"\n{'='*50}")
            print(f"Inference Timing Results:")
            print(f"  Total frames: {num_frames}")
            print(f"  Inference time: {inference_time:.3f} seconds")
            print(f"  FPS: {fps:.2f}")
            print(f"  Time per frame: {ms_per_frame:.2f} ms")
            print(f"  Peak GPU memory: {peak_alloc_gb:.2f} GiB allocated, "
                  f"{peak_reserved_gb:.2f} GiB reserved")
            if not args.warmup:
                print(f"  (Note: First run includes torch.compile overhead. Use --warmup for accurate timing)")
            # Per-section breakdown (populated by Pi3.forward when loop_detect /
            # PGO actually run; both report 0.0 otherwise).
            _timing = raw_model_predictions.get("_timing") if isinstance(raw_model_predictions, dict) else None
            if _timing is not None:
                _t_ld = float(_timing.get("loop_detect", 0.0))
                _t_pg = float(_timing.get("pgo", 0.0))
                _t_other = max(inference_time - _t_ld - _t_pg, 0.0)
                print(f"  Breakdown:")
                print(f"    Loop detection : {_t_ld:7.3f} s ({100.0 * _t_ld / inference_time:5.1f}%)")
                print(f"    PGO            : {_t_pg:7.3f} s ({100.0 * _t_pg / inference_time:5.1f}%)")
                print(f"    Inference+TTT  : {_t_other:7.3f} s ({100.0 * _t_other / inference_time:5.1f}%)")
                if _t_other > 0:
                    print(f"    Pure-inference FPS (excl. PGO+LC): {num_frames / _t_other:.2f}")
            print(f"{'='*50}\n")

        # Post-process predictions. Images are only needed for the live viser / bundle;
        # for a bundle without live viser, read only the frames the bundle keeps.
        if args.need_points:
            _stride_imgs = (args.skip_viser and args.bundle_stride > 1
                            and hasattr(images_tensor, "_load_indices"))
            if _stride_imgs:
                _idx = list(range(0, num_frames, args.bundle_stride))
                print(f"[post-process] loading {len(_idx)}/{num_frames} input images "
                      f"(save_stride={args.bundle_stride})...", flush=True)
                raw_model_predictions['images'] = images_tensor._load_indices(_idx).unsqueeze(0).permute(0, 1, 3, 4, 2)
            else:
                print(f"[post-process] loading all {num_frames} input images from disk...", flush=True)
                raw_model_predictions['images'] = images_tensor[None].permute(0, 1, 3, 4, 2)
        if raw_model_predictions.get('conf') is not None:
            raw_model_predictions['conf'] = torch.sigmoid(raw_model_predictions['conf'])
        raw_model_predictions.pop('local_points', None)
        if not args.need_points:
            raw_model_predictions.pop('points', None)

        # Per-frame fields strided in the bundle (all together, so they stay frame-aligned).
        BUNDLE_KEYS = ("points", "conf", "camera_poses", "camera_poses_pre_pgo", "images")

        # Capture FULL-resolution camera poses BEFORE any striding, so the
        # trajectory .txt always carries every frame.
        _cp_t = raw_model_predictions.get("camera_poses", None)
        if torch.is_tensor(_cp_t):
            _full_cam_poses_np = _cp_t.squeeze(0).detach().cpu().float().numpy()
        _cp_pre_t = raw_model_predictions.get("camera_poses_pre_pgo", None)
        if torch.is_tensor(_cp_pre_t):
            _full_cam_poses_pre_pgo_np = _cp_pre_t.squeeze(0).detach().cpu().float().numpy()

        # Convert all tensors to numpy and remove the batch dimension. Without a live
        # viser the bundle stride is applied right here, so the full per-pixel arrays
        # are never materialised; with a live viser everything stays full-resolution.
        _build_stride = args.bundle_stride if args.skip_viser else 1
        print("[post-process] building predictions_dict (numpy conversion)...", flush=True)
        predictions_dict = {}
        for k, v in raw_model_predictions.items():
            if v is None or not torch.is_tensor(v):
                continue
            v_sq = v.squeeze(0)
            if _build_stride > 1 and k in BUNDLE_KEYS and v_sq.dim() > 0 and v_sq.shape[0] == num_frames:
                v_sq = v_sq[::_build_stride]
            # Preserve integer dtypes (e.g. window_frame_starts) — only float-cast floats.
            predictions_dict[k] = v_sq.cpu().float().numpy() if v_sq.is_floating_point() else v_sq.cpu().numpy()
            raw_model_predictions[k] = None      # free the tensor (non-tensor entries such as lc_bridge_pts stay)
        import gc; gc.collect()

        if args.save_viser_data:
            os.makedirs(args.output_folder, exist_ok=True)
            seq_name_to_use = f"{args.seq_name}_{str(args.start_frame)}_{str(args.end_frame)}_{str(args.stride)}"
            
            # Count number of inputs processed
            input_paths = [p for p in [args.input, args.input2, args.input3, args.input4, args.input5] if p is not None]
            num_inputs_processed = len(input_paths)
            if num_inputs_processed > 1:
                seq_name_to_use += f"_x{num_inputs_processed}"
            
            output_filename = f"{seq_name_to_use}.pt" if seq_name_to_use else "predictions.pt"
            output_path = os.path.join(args.output_folder, output_filename)
            _bytes_total = 0
            for _v in predictions_dict.values():
                if hasattr(_v, "nbytes"):
                    _bytes_total += int(_v.nbytes)
            print(f"Saving inference results to {output_path} (~{_bytes_total / 1024**2:.1f} MB)...", flush=True)
            try:
                # Stride every per-frame field by the bundle stride (a no-op for fields that
                # are already strided, i.e. without a live viser).
                torch.save({k: torch.from_numpy(v[::args.bundle_stride] if (
                                args.bundle_stride > 1 and k in BUNDLE_KEYS and v.ndim > 0
                                and v.shape[0] == num_frames) else v)
                            for k, v in predictions_dict.items()}, output_path)
                print("Successfully saved inference results.", flush=True)
            except Exception as e:
                print(f"Error saving results to {output_path}: {e}", flush=True)

            # Dump the viser_wrapper kwargs alongside the .pt so scripts/visualize_viser.py
            # can replay the visualisation without re-running inference.
            import json
            viser_cfg_path = os.path.join(
                args.output_folder, f"{seq_name_to_use}_viser_config.json"
            )
            viser_cfg = {
                "predictions_pt": output_filename,        # relative to JSON's directory
                "port": int(args.port),
                "init_conf_threshold": float(args.conf_threshold),
                "background_mode": bool(args.background_mode),
                "mask_sky": bool(args.mask_sky),
                "image_folder_for_sky_mask": image_folder_for_sky,
                "subsample": int(args.subsample),
                "frame_stride": int(args.frame_stride),
                # save_stride records what was applied at save time. Per-frame
                # fields in the .pt are already strided by this amount; the
                # launcher reads it for documentation / startup print but
                # does NOT re-apply it (no double stride).
                "save_stride": int(args.bundle_stride),
                "video_width": int(args.video_width),
                "share": bool(args.share),
                "canonical_first_frame": bool(args.canonical_first_frame),
                # point_size is a viser_wrapper default that demo_viser doesn't
                # currently expose; persist the wrapper default so the launcher
                # remains a one-to-one replay.
                "point_size": 0.001,
            }
            try:
                with open(viser_cfg_path, "w") as _f:
                    json.dump(viser_cfg, _f, indent=2)
                print(f"Saved viser config to {viser_cfg_path}")
            except Exception as _e:
                print(f"Error saving viser config to {viser_cfg_path}: {_e}")

        if args.lc_pts_folder and "lc_bridge_pts" in raw_model_predictions:
            os.makedirs(args.lc_pts_folder, exist_ok=True)
            bridge_pts = raw_model_predictions["lc_bridge_pts"]
            for (j_idx, i_idx), br in bridge_pts.items():
                out_path = os.path.join(args.lc_pts_folder, f"bridge_{j_idx}_{i_idx}.pt")
                torch.save(br, out_path)
            print(f"Saved {len(bridge_pts)} bridge window .pt files to {args.lc_pts_folder}")

    if args.output_txt and predictions_dict is not None and "camera_poses" in predictions_dict:
        print(f"Saving trajectory to {args.output_txt}...")
        try:
            # 1) Prepare timestamps
            # We use the first input path to try finding timestamps
            input_path_for_ts = Path(args.input)
            # If we have multiple inputs, we might need to be careful, but usually we evaluate on the first sequence or combined.
            # Here we use all_image_names_collected which corresponds to the inference frames.
            # Note: all_image_names_collected might be temp paths if we copied them.
            # If we copied them, we lost the original path connection for timestamp lookup if we rely on temp dir.
            # However, _try_load_timestamps_for_images uses input_rgb_dir to find rgb.txt.
            # If we pass the original input path as input_rgb_dir, it might work if filenames match.
            # But filenames in temp dir are frame_xxxxxx.png.
            # So we should probably use the original filenames if possible, or just fallback to indices if using temp dir.
            
            # If we used temp dir (which we did for combined inputs), filenames are frame_000000.png.
            # This breaks mapping to rgb.txt which uses original filenames.
            # So for now, if we used temp dir, we might have to fallback to indices unless we tracked original names.
            # In run_core_inference or main, we didn't track original names in a way that maps easily back for rgb.txt lookup 
            # unless we parse them.
            
            # However, if args.input is a directory and we are just processing it (and maybe others), 
            # and if we want to evaluate, we usually care about the timestamps of the frames we processed.
            
            # Let's try to use indices as timestamps if we can't easily map back, 
            # OR if the user provided a single input folder, we can try to be smarter.
            
            timestamps = None
            if len(input_paths) == 1 and os.path.isdir(args.input) and not is_video_file(args.input):
                 # If single folder input, we can try to load timestamps using the original filenames
                 # Re-glob to get original paths
                 # import glob # Already imported globally
                 # from natsort import natsorted # Already imported globally
                 current_frames = natsorted(glob.glob(os.path.join(args.input, "*.png"))+glob.glob(os.path.join(args.input, "*.jpg"))+glob.glob(os.path.join(args.input, "*.jpeg")))
                 current_frames = [f for f in current_frames if "depth" not in os.path.basename(f).lower()]
                 end_idx = args.end_frame if args.end_frame != -1 else None
                 current_frames = current_frames[args.start_frame:end_idx:args.stride]

                 timestamps = _try_load_timestamps_for_images(current_frames, Path(args.input))
            else:
                 # Fallback to indices
                 timestamps = [float(i) for i in range(len(all_image_names_collected))]

            # The .txt carries the full-resolution trajectory captured before any striding.
            # (With --load it falls back to the loaded, possibly bundle-strided poses.)

            # 2) Extract poses (prefer the full-res copy)
            # predictions_dict['camera_poses'] is (N, 4, 4) numpy array
            _cam_for_txt = (
                _full_cam_poses_np
                if _full_cam_poses_np is not None
                else predictions_dict['camera_poses']
            )
            camera_poses = torch.from_numpy(_cam_for_txt)
            
            # Pi3 outputs Twc (Camera to World) directly
            Twc = camera_poses
            Rwc = Twc[..., :3, :3]
            twc = Twc[..., :3, 3]
            
            qwc = mat_to_quat(Rwc) # XYZW
            
            # 3) Write
            # Ensure lengths match
            S = min(len(timestamps), twc.shape[0], qwc.shape[0])
            write_trajectory_txt(Path(args.output_txt), timestamps[:S], twc[:S].tolist(), qwc[:S].tolist())
            print(f"Successfully saved trajectory to {args.output_txt}")

            # Pre-PGO trajectory (if PGO was run, pi3.py keeps a snapshot).
            # Same rule as above: prefer the full-res copy captured before any
            # striding; fall back to the possibly-strided predictions_dict.
            _has_pre = (
                _full_cam_poses_pre_pgo_np is not None
                or "camera_poses_pre_pgo" in predictions_dict
            )
            if _has_pre and args.save_debug_poses:
                pre_path = str(Path(args.output_txt).with_suffix('')) + "_pre_pgo.txt"
                _cam_pre_for_txt = (
                    _full_cam_poses_pre_pgo_np
                    if _full_cam_poses_pre_pgo_np is not None
                    else predictions_dict['camera_poses_pre_pgo']
                )
                Twc_pre  = torch.from_numpy(_cam_pre_for_txt)
                Rwc_pre  = Twc_pre[..., :3, :3]
                twc_pre  = Twc_pre[..., :3, 3]
                qwc_pre  = mat_to_quat(Rwc_pre)
                S_pre = min(len(timestamps), twc_pre.shape[0], qwc_pre.shape[0])
                write_trajectory_txt(Path(pre_path), timestamps[:S_pre],
                                     twc_pre[:S_pre].tolist(), qwc_pre[:S_pre].tolist())
                print(f"Successfully saved pre-PGO trajectory to {pre_path}")

            import numpy as _np_dump
            scales_stem = str(Path(args.output_txt).with_suffix(''))
            dumped = []
            for key in ("chunk_sim3_scales", "chunk_sim3_poses", "chunk_se3_poses", "alignment_mode"):
                if args.save_debug_poses and key in predictions_dict:
                    _np_dump.save(scales_stem + f"_{key}.npy", predictions_dict[key])
                    dumped.append(key)
            if dumped:
                print(f"Dumped chunk-alignment arrays: {dumped}  (stem: {scales_stem})")
        except Exception as e:
            print(f"Error saving trajectory to {args.output_txt}: {e}")

    if predictions_dict is None:
        print("Error: Predictions are not available. Exiting.")
        for temp_dir_path in temp_frame_dirs.values():
            if os.path.exists(temp_dir_path): shutil.rmtree(temp_dir_path)
        return

    if args.skip_viser:
        print("Skipping viser visualization.")
        return

    print("Starting viser visualization...")
    viser_wrapper(
        predictions_dict,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder_for_sky_mask=image_folder_for_sky,
        subsample=args.subsample,
        frame_stride=args.frame_stride,
        video_width=args.video_width,
        share=args.share,
        canonical_first_frame=args.canonical_first_frame,
    )
    
    for temp_dir_path in temp_frame_dirs.values():
        if os.path.exists(temp_dir_path):
            print(f"Cleaning up temporary directory: {temp_dir_path}")
            shutil.rmtree(temp_dir_path)

    print("Visualization setup complete. Server is running.")

if __name__ == "__main__":
    main()
