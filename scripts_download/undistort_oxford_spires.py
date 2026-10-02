#!/usr/bin/env python3
"""Undistort (rectify) Oxford Spires cam0 fisheye images to pinhole.

Follows the official rectifier (ori-drs/oxford_spires_dataset,
scripts/rectify_image_from_distorted_image.py):
  new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (W, H), I,
                                                                  balance=0.0, new_size=(W, H))
  maps  = cv2.fisheye.initUndistortRectifyMap(K, D, I, new_K, (W, H), CV_32FC1)
  image = cv2.remap(image, *maps, INTER_LINEAR, BORDER_CONSTANT)
For cam0 this gives fx~532.4 fy~532.7 cx~701.3 cy~571.8 at 1440x1080 (~107 x 91 deg).

--method lingbot instead follows lingbot-map_new/preprocess/oxford.py: new_K = K
with the principal point moved to the image centre (fx~698.2 fy~698.5 cx=720
cy=540, ~92 x 75 deg). It keeps the raw centre resolution and crops more of the
fisheye border; remap is the same.

K and D = [k1..k4] come from <root>/calibration/cam0.yaml (Kalibr, equidistant);
the official configs/sensor.yaml has identical values for cam_front and can be
passed with --calib as well.

Input -> output. The output tree mirrors sequences/, with the same filenames
(so rgb.txt timestamps and GT matching still work) and the GT linked, not copied:
  <root>/sequences/<SEQ>/raw/cam0/<ts>.jpg                   (from download_oxford_spires.py)
  <root>/processed_sequences/<SEQ>/raw/cam0/<ts>.jpg         rectified
  <root>/processed_sequences/<SEQ>/raw/rgb.txt               image timestamps
  <root>/processed_sequences/<SEQ>/raw/cam0_intrinsics.txt   "fx fy cx cy width height"
  <root>/processed_sequences/<SEQ>/processed -> ../../sequences/<SEQ>/processed   (GT)

Run LoGeR / plot on the result with:
  OXFORD_SEQS_DIR=processed_sequences bash scripts/run_oxford_spires.sh ...
  python scripts/oxford_spires_traj_plot.py --gt_dir data/oxford_spires/processed_sequences ...

Usage:
  python scripts_download/undistort_oxford_spires.py                # all extracted sequences
  python scripts_download/undistort_oxford_spires.py --seq 2024-03-12-keble-college-02
"""
import argparse
import logging
import os
from multiprocessing import Pool

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_ROOT = os.path.join(REPO_ROOT, "data", "oxford_spires")
IMG_EXTS = (".jpg", ".jpeg", ".png")


def convert_to_intrinsic_matrix(intrinsics):
    if len(intrinsics) != 4:
        raise ValueError(f"Intrinsics must have exactly 4 elements [fx, fy, cx, cy], got {len(intrinsics)}")
    fx, fy, cx, cy = intrinsics
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def load_calib(path, cam_label="cam_front"):
    """K, D, width, height from a Kalibr cam<n>.yaml or the official sensor.yaml."""
    with open(path) as f:
        c = yaml.safe_load(f)
    if "sensor" in c:   # official configs/sensor.yaml
        cam = {x["label"]: x for x in c["sensor"]["cameras"]}[cam_label]
        K, D = convert_to_intrinsic_matrix(cam["intrinsics"]), cam["extra_params"]
    else:               # Kalibr calibration/cam0.yaml
        assert c["distortion_model"] == "equidistant", c["distortion_model"]
        K, D = np.array(c["camera_matrix"]["data"]).reshape(3, 3), c["distortion_coefficients"]["data"]
        cam = c
    return K, np.array(D, dtype=np.float64), int(cam["image_width"]), int(cam["image_height"])


class ImageRectifier:
    """Fisheye -> pinhole rectification (one camera).

    method="official": new camera from estimateNewCameraMatrixForUndistortRectify
                       (official script, widest view without black borders);
    method="lingbot":  new camera = K with a centred principal point
                       (lingbot-map, keeps the raw focal length)."""

    def __init__(self, K, D, width, height, method="official", balance=0.0, scale_factor=1.0):
        new_size = (int(width * scale_factor), int(height * scale_factor))
        if method == "official":
            self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (width, height), np.eye(3), balance=balance, new_size=new_size)
        elif method == "lingbot":
            self.new_K = K.copy()
            self.new_K[0, 2], self.new_K[1, 2] = new_size[0] / 2.0, new_size[1] / 2.0
        else:
            raise ValueError(method)
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), self.new_K, new_size, cv2.CV_32FC1)
        self.width, self.height = new_size

    def process_image(self, image):
        return cv2.remap(image, self.map1, self.map2,
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


_RECTIFIER = None  # one per worker process


def _init_worker(K, D, width, height, method):
    global _RECTIFIER
    _RECTIFIER = ImageRectifier(K, D, width, height, method)


def _rectify_file(job):
    src, dst = job
    image = cv2.imread(src)
    if image is None:
        return src
    tmp = dst + ".tmp" + os.path.splitext(dst)[1]
    if not cv2.imwrite(tmp, _RECTIFIER.process_image(image)):
        return src
    os.replace(tmp, dst)   # never leave a half-written image behind
    return None


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Rectify Oxford Spires cam0 fisheye images (official method).")
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="dataset root with sequences/ and calibration/ (default: data/oxford_spires)")
    ap.add_argument("--seq", nargs="+", default=None,
                    help="sequences (default: every sequence with raw/cam0)")
    ap.add_argument("--calib", default=None,
                    help="calibration yaml: Kalibr cam0.yaml or official sensor.yaml "
                         "(default: <root>/calibration/cam0.yaml)")
    ap.add_argument("--out_dir", default=None,
                    help="output sequences root (default: <root>/processed_sequences)")
    ap.add_argument("--method", default="lingbot", choices=["official", "lingbot"],
                    help="new pinhole camera: official (fx~532, ~107 deg) or lingbot (fx~698, ~92 deg)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    K, D, W, H = load_calib(args.calib or os.path.join(args.root, "calibration", "cam0.yaml"))
    new_K = ImageRectifier(K, D, W, H, args.method).new_K
    logger.info(f"[{args.method}] cam0 {W}x{H} fisheye -> pinhole fx={new_K[0, 0]:.3f} fy={new_K[1, 1]:.3f} "
                f"cx={new_K[0, 2]:.3f} cy={new_K[1, 2]:.3f}")

    seq_root = os.path.join(args.root, "sequences")
    out_root = args.out_dir or os.path.join(args.root, "processed_sequences")
    # Resolve name links (2024-03-18-christ-church-05 -> 2024-03-20-...) to real dirs.
    seqs = sorted({os.path.basename(os.path.realpath(os.path.join(seq_root, s)))
                   for s in (args.seq or os.listdir(seq_root))
                   if os.path.isdir(os.path.join(seq_root, s, "raw", "cam0"))})

    with Pool(args.workers, initializer=_init_worker, initargs=(K, D, W, H, args.method)) as pool:
        for seq in seqs:
            logger.info(f"=== Rectifying {seq} ===")
            src_dir = os.path.join(seq_root, seq, "raw", "cam0")
            out_seq = os.path.join(out_root, seq)
            raw = os.path.join(out_seq, "raw")
            dst_dir = os.path.join(raw, "cam0")
            os.makedirs(dst_dir, exist_ok=True)
            # GT and other processed files: link the original sequence's processed/.
            gt_link = os.path.join(out_seq, "processed")
            if not os.path.lexists(gt_link):
                os.symlink(os.path.relpath(os.path.join(seq_root, seq, "processed"), out_seq), gt_link)
            names = sorted(n for n in os.listdir(src_dir) if n.lower().endswith(IMG_EXTS))
            jobs = [(os.path.join(src_dir, n), os.path.join(dst_dir, n)) for n in names
                    if not os.path.exists(os.path.join(dst_dir, n))]   # resume
            failed = [r for r in pool.imap_unordered(_rectify_file, jobs, chunksize=16) if r]
            # Only list successfully rectified frames, including resumed output.
            with open(os.path.join(raw, "rgb.txt"), "w") as f:
                for name in names:
                    if os.path.isfile(os.path.join(dst_dir, name)):
                        f.write(f"{os.path.splitext(name)[0]} cam0/{name}\n")
            with open(os.path.join(raw, "cam0_intrinsics.txt"), "w") as f:
                f.write("# fx fy cx cy width height (rectified pinhole)\n")
                f.write(f"{new_K[0, 0]:.6f} {new_K[1, 1]:.6f} {new_K[0, 2]:.6f} {new_K[1, 2]:.6f} {W} {H}\n")
            logger.info(f"  {seq}: {len(names)} images ({len(names) - len(jobs)} already done)"
                        f"{f', {len(failed)} FAILED' if failed else ''} -> {dst_dir}")

    # Mirror the name links of sequences/ (e.g. the christ-church-05 rename).
    for s in os.listdir(seq_root):
        src = os.path.join(seq_root, s)
        dst = os.path.join(out_root, s)
        if os.path.islink(src) and os.path.isdir(os.path.join(out_root, os.readlink(src))) \
                and not os.path.lexists(dst):
            os.symlink(os.readlink(src), dst)

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
