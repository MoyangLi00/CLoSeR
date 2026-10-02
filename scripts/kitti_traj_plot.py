"""KITTI Odometry: per-sequence trajectory plots and ATE table.

Prediction: <results_dir>/<seq>.txt (TUM; timestamp = frame index).
GT:         <gt_dir>/<seq>.txt (KITTI 3x4 pose per line; row = frame index).
Plots the x-z plane (KITTI camera y points down). Shared logic:
scripts/_traj_plot_common.py.

Usage:
  python scripts/kitti_traj_plot.py --results_dir results/kitti/LoGeR
"""
from _traj_plot_common import Dataset, main

KITTI = Dataset(
    name="KITTI",
    pred_suffix=".txt",
    gt_path="{seq}.txt",
    gt_format="kitti",
    match="index",
    plane=(0, 2),
    default_gt_dir="data/kitti/dataset/poses",
    order=[f"{i:02d}" for i in range(11)],
)

if __name__ == "__main__":
    main(KITTI)
