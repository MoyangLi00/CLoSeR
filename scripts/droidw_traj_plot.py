"""DROID-W: per-sequence trajectory plots and ATE table.

Prediction: <results_dir>/<seq>.txt (TUM; timestamp = image time in seconds).
GT:         <gt_dir>/<seq>/poses.txt (TUM; sparser than the images).
Frames are matched by nearest timestamp within 0.1 s. Plots the x-y plane.
Shared logic: scripts/_traj_plot_common.py.

Usage:
  python scripts/droidw_traj_plot.py --results_dir results/droidw/LoGeR
"""
from _traj_plot_common import Dataset, main

DROIDW = Dataset(
    name="DROID-W",
    pred_suffix=".txt",
    gt_path="{seq}/poses.txt",
    gt_format="tum",
    match="time",
    max_time_diff=0.1,
    plane=(0, 1),
    default_gt_dir="data/DROID-W",
)

if __name__ == "__main__":
    main(DROIDW)
