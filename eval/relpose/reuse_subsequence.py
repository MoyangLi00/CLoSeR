#!/usr/bin/env python3
"""Reuse a subset of an existing evaluation run by truncating first N frames."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from eval.relpose.metadata import dataset_metadata
from eval.relpose.evo_utils import (
    load_traj,
    eval_metrics,
    plot_trajectory,
    process_directory,
    calculate_averages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reuse cached pi3 outputs for a subset run")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., scannet_s3_100)")
    parser.add_argument("--frames", type=int, required=True, help="Number of frames to keep")
    parser.add_argument(
        "--base-dataset",
        required=True,
        help="Dataset name that already has a finished evaluation (typically *_1000)",
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Path to the finished base dataset tag directory (contains seq folders and _error_log.txt)",
    )
    parser.add_argument(
        "--target-dir",
        required=True,
        help="Directory where the truncated results should be written",
    )
    parser.add_argument(
        "--num-seqs",
        type=int,
        default=-1,
        help="Limit how many sequences to reuse (matches --num-seqs from launch)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Pose evaluation stride used in the base run (default: 1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the previous target directory before writing",
    )
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sorted_sequence_dirs(base_dir: str, limit: int) -> List[str]:
    entries = [
        name
        for name in sorted(os.listdir(base_dir))
        if os.path.isdir(os.path.join(base_dir, name))
    ]
    if limit > 0:
        entries = entries[:limit]
    return entries


def _slice_numeric_file(path: str, frames: int, allow_short: bool = False) -> Tuple[np.ndarray, int]:
    data = np.loadtxt(path)
    if data.ndim == 0:
        data = np.expand_dims(data, axis=0)
    available = data.shape[0]
    if available < frames:
        if not allow_short:
            raise ValueError(
                f"File {path} only has {available} rows, but {frames} were requested"
            )
        frames = available
    if frames <= 0:
        raise ValueError(f"File {path} does not contain enough rows to reuse")
    return data[:frames], frames


def _prepare_pred_traj(pred_matrix: np.ndarray) -> List[np.ndarray]:
    timestamps = pred_matrix[:, 0]
    tum_poses = pred_matrix[:, 1:]
    return [tum_poses, timestamps]


def _write_numeric_file(path: str, data: np.ndarray) -> None:
    np.savetxt(path, data, fmt="%.9f")


def reuse_subset(args: argparse.Namespace) -> None:
    metadata = dataset_metadata.get(args.dataset)
    _require(metadata is not None, f"Unknown dataset {args.dataset}")

    base_dir = Path(args.base_dir)
    target_dir = Path(args.target_dir)
    _require(base_dir.is_dir(), f"Base directory {base_dir} was not found")

    if args.overwrite and target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    sequences = _sorted_sequence_dirs(str(base_dir), args.num_seqs)
    _require(len(sequences) > 0, f"No sequences found under {base_dir}")

    img_path = metadata.get("img_path")
    anno_path = metadata.get("anno_path")
    traj_format = metadata.get("traj_format")
    gt_traj_func = metadata.get("gt_traj_func")

    _require(
        traj_format is not None and gt_traj_func is not None,
        f"Dataset {args.dataset} is missing pose metadata for evaluation",
    )

    log_entries: List[str] = []

    for seq in sequences:
        src_seq_dir = base_dir / seq
        dst_seq_dir = target_dir / seq
        dst_seq_dir.mkdir(parents=True, exist_ok=True)

        pred_traj_src = src_seq_dir / "pred_traj.txt"
        pred_focal_src = src_seq_dir / "pred_focal.txt"
        pred_intr_src = src_seq_dir / "pred_intrinsics.txt"

        for required_file in (pred_traj_src, pred_focal_src, pred_intr_src):
            _require(required_file.exists(), f"Missing {required_file}")

        pred_matrix, frames_used = _slice_numeric_file(
            str(pred_traj_src), args.frames, allow_short=True
        )
        if frames_used < args.frames:
            print(
                f"Warning: {seq} only has {frames_used} frames but {args.frames} were requested; using available frames instead."
            )
        _write_numeric_file(str(dst_seq_dir / "pred_traj.txt"), pred_matrix)

        focal_values, _ = _slice_numeric_file(str(pred_focal_src), frames_used)
        _write_numeric_file(str(dst_seq_dir / "pred_focal.txt"), focal_values)

        intr_values, _ = _slice_numeric_file(str(pred_intr_src), frames_used)
        _write_numeric_file(str(dst_seq_dir / "pred_intrinsics.txt"), intr_values)

        pred_traj = _prepare_pred_traj(pred_matrix)

        gt_traj_file = gt_traj_func(img_path, anno_path, seq)  # type: ignore[misc]
        _require(gt_traj_file and os.path.exists(gt_traj_file), f"GT pose not found for {seq}")

        eval_file = target_dir / f"{seq}_eval_metric.txt"
        plot_file = target_dir / f"{seq}.png"

        try:
            gt_traj = load_traj(
                gt_traj_file=gt_traj_file,
                traj_format=traj_format,
                stride=args.stride,
                num_frames=frames_used,
            )

            ate, rpe_trans, rpe_rot = eval_metrics(
                pred_traj,
                gt_traj,
                seq=seq,
                filename=str(eval_file),
            )
            plot_trajectory(pred_traj, gt_traj, title=seq, filename=str(plot_file))

            log_entries.append(
                f"{args.dataset}-{seq:<16} | Frames: {frames_used:>4} | ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}\n"
            )
            log_entries.append(f"{ate:.5f}\n")
            log_entries.append(f"{rpe_trans:.5f}\n")
            log_entries.append(f"{rpe_rot:.5f}\n")
        except Exception as e:
            error_msg = str(e)
            if "Degenerate covariance rank" in error_msg or "Eigenvalues did not converge" in error_msg:
                warning = (
                    f"Trajectory evaluation error in sequence {seq}, skipping."
                )
                print(warning)
                log_entries.append(
                    f"{args.dataset}-{seq:<16} | Frames: {frames_used:>4} | Skipped due to trajectory eval error: {error_msg}\n"
                )
                continue
            raise

    error_log_local = target_dir / "_error_log_0.txt"
    with error_log_local.open("w") as f:
        for line in log_entries:
            f.write(line)

    results = process_directory(str(target_dir))
    avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)

    error_log_main = target_dir / "_error_log.txt"
    with error_log_main.open("w") as f:
        f.write(error_log_local.read_text())
        f.write(
            f"Average ATE: {avg_ate:.5f}, Average RPE trans: {avg_rpe_trans:.5f}, Average RPE rot: {avg_rpe_rot:.5f}\n"
        )

    print(
        f"Reused {args.frames} frames for {args.dataset} across {len(sequences)} sequences."
    )


def main() -> None:
    args = parse_args()
    reuse_subset(args)


if __name__ == "__main__":
    main()
