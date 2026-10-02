"""Oxford Spires: per-sequence trajectory plots and ATE table.

Prediction: <results_dir>/<seq>_es.txt (TUM; timestamp = image time in seconds,
            written by scripts/run_oxford_spires.sh via raw/rgb.txt).
GT:         <gt_dir>/<seq>/processed/trajectory/gt-tum.txt (TUM, seconds); the default
            processed_sequences/ links it from sequences/, so either root works.
Frames are matched by nearest timestamp within 0.05 s. Plots the x-y plane.
Shared logic: scripts/_traj_plot_common.py.

Usage:
  python scripts/oxford_spires_traj_plot.py --results_dir results/oxford_spires/LoGeR
  # predictions named <seq>.txt, e.g. Baseline_results/<method>/oxford_spires:
  python scripts/oxford_spires_traj_plot.py --results_dir Baseline_results/loger/oxford_spires --pred_suffix .txt
"""
from _traj_plot_common import Dataset, main

SEQS = ["2024-03-12-keble-college-02", "2024-03-12-keble-college-03",
        "2024-03-12-keble-college-04", "2024-03-12-keble-college-05",
        "2024-03-13-observatory-quarter-01", "2024-03-13-observatory-quarter-02",
        "2024-03-14-blenheim-palace-01", "2024-03-14-blenheim-palace-02",
        "2024-03-14-blenheim-palace-05",
        "2024-03-18-christ-church-01", "2024-03-18-christ-church-02",
        "2024-03-18-christ-church-03", "2024-03-18-christ-church-05",
        "2024-05-20-bodleian-library-02"]

OXFORD_SPIRES = Dataset(
    name="Oxford Spires",
    pred_suffix="_es.txt",
    gt_path="{seq}/processed/trajectory/gt-tum.txt",
    gt_format="tum",
    match="time",
    max_time_diff=0.05,
    plane=(0, 1),
    default_gt_dir="data/oxford_spires/processed_sequences",   # GT linked from sequences/
    order=SEQS,
    # "2024-03-12-keble-college-02" -> "keble_02"
    labels={q: q.split("-")[3] + "_" + q.split("-")[-1] for q in SEQS},
)

if __name__ == "__main__":
    main(OXFORD_SPIRES)
