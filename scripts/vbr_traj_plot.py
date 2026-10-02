"""VBR: per-sequence trajectory plots and ATE table.

Prediction: <results_dir>/<seq>_es.txt (TUM; timestamp = frame index).
GT:         <gt_dir>/<seq>_gt.txt (TUM; timestamp = frame index).
Plots the x-y plane. Shared logic: scripts/_traj_plot_common.py.

Usage:
  python scripts/vbr_traj_plot.py --results_dir results/vbr/LoGeR
"""
from _traj_plot_common import Dataset, main

VBR = Dataset(
    name="VBR",
    pred_suffix="_es.txt",
    gt_path="{seq}_gt.txt",
    gt_format="tum",
    match="index",
    plane=(0, 1),
    default_gt_dir="data/vbr/processed_gt",
    order=["colosseo_train0", "campus_train0", "campus_train1", "pincio_train0",
           "spagna_train0", "diag_train0", "ciampino_train1"],
    labels={"colosseo_train0": "colosseo_0", "campus_train0": "campus_0",
            "campus_train1": "campus_1", "pincio_train0": "pincio_0",
            "spagna_train0": "spagna_0", "diag_train0": "diag_0",
            "ciampino_train1": "ciampino_1"},
)

if __name__ == "__main__":
    main(VBR)
