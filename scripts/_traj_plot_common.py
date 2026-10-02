"""Shared code for scripts/<dataset>_traj_plot.py.

Each dataset script only describes where its files live and how its frames
are matched to GT (see `Dataset`); everything else — loading, Sim(3)
alignment, ATE, loop-pair counts, the plot and the LaTeX table — is here, so
all datasets produce the same kind of figure and table.

For each sequence with both a prediction and a GT file:
  - match predicted frames to GT poses (by frame index or by timestamp),
  - Sim(3)-align the prediction to GT on the matched frames (Umeyama),
  - write <out_dir>/<seq>_traj.png: top-down view of GT vs aligned prediction,
    titled with ATE RMSE and, if the run used --loopdetect, the loop pairs
    kept / detected (from <results_dir>/loop_detection/<seq>/),
then print a LaTeX table of per-sequence ATE RMSE plus the average.
"""
import argparse
import glob
import os
import re
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Dataset:
    name: str                 # shown in the help text
    pred_suffix: str          # prediction file = <results_dir>/<seq><pred_suffix>
    gt_path: str              # GT file, relative to --gt_dir, with {seq}
    gt_format: str            # "kitti" (3x4 per line, row = frame) or "tum"
    match: str                # "index" (frame-index timestamps) or "time"
    plane: tuple              # top-down axes, e.g. (0, 2) for x-z
    default_gt_dir: str
    max_time_diff: float = 0.1            # seconds, for match="time"
    order: list = field(default_factory=list)   # table order (rest alphabetical)
    labels: dict = field(default_factory=dict)  # short table labels


# ------------------------------------------------------------------ loading

def load_tum(path):
    """(timestamps (N,), xyz (N,3)) from a TUM file `ts tx ty tz qx qy qz qw`."""
    data = np.loadtxt(path, comments="#", ndmin=2)
    return data[:, 0], data[:, 1:4]


def load_kitti(path):
    """(frame indices (N,), xyz (N,3)) from a KITTI Odometry pose file."""
    data = np.loadtxt(path, ndmin=2)
    return np.arange(len(data), dtype=float), data[:, [3, 7, 11]]


# ----------------------------------------------------------------- matching

def match_by_index(pred_ts, gt_ts):
    """Pair frames whose integer frame-index timestamps agree."""
    gt_pos = {int(v): i for i, v in enumerate(np.rint(gt_ts))}
    pairs = [(i, gt_pos[int(v)]) for i, v in enumerate(np.rint(pred_ts))
             if int(v) in gt_pos]
    return np.array(pairs, dtype=int).reshape(-1, 2)


def match_by_time(pred_ts, gt_ts, max_diff):
    """Greedy one-to-one matching of the closest timestamps within max_diff
    (same result as eval_droidw_full.associate)."""
    order = np.argsort(gt_ts)
    sorted_gt = gt_ts[order]
    cands = []
    for i, t in enumerate(pred_ts):
        lo = np.searchsorted(sorted_gt, t - max_diff, side="right")
        hi = np.searchsorted(sorted_gt, t + max_diff, side="left")
        for k in range(lo, hi):
            gap = abs(t - sorted_gt[k])
            if gap < max_diff:
                cands.append((gap, i, int(order[k])))
    cands.sort()
    used_p, used_g, pairs = set(), set(), []
    for _, i, j in cands:
        if i not in used_p and j not in used_g:
            used_p.add(i); used_g.add(j); pairs.append((i, j))
    return np.array(sorted(pairs), dtype=int).reshape(-1, 2)


# ---------------------------------------------------------------- alignment

def umeyama_sim3(src, tgt):
    """Scale s, rotation R, translation t minimising |s R src + t - tgt|."""
    mu_s, mu_t = src.mean(0), tgt.mean(0)
    sc, tc = src - mu_s, tgt - mu_t
    U, D, Vt = np.linalg.svd(sc.T @ tc / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = (U @ S @ Vt).T
    s = (D * np.diag(S)).sum() / max((sc ** 2).sum() / len(src), 1e-12)
    return s, R, mu_t - s * R @ mu_s


# -------------------------------------------------------------------- loops

def count_pairs(path):
    """Data rows in a loop_closures*.txt, or None if the file is missing."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return sum(1 for l in f if l.strip() and not l.startswith("#"))


def loop_label(results_dir, seq):
    d = os.path.join(results_dir, "loop_detection", seq)
    kept = count_pairs(os.path.join(d, "detected_loops.txt"))
    det = None
    if kept is not None:   # header: "# Detected loops (...): K of N SALAD pairs kept"
        m = re.search(r"of (\d+) SALAD pairs", open(os.path.join(d, "detected_loops.txt")).readline())
        det = int(m.group(1)) if m else None
    else:                  # results from before detected_loops.txt
        det = count_pairs(os.path.join(d, "loop_closures.txt"))
        kept = count_pairs(os.path.join(d, "loop_closures_filtered.txt"))
    if det is None and kept is None:
        return ""
    return f"loops {'?' if kept is None else kept}/{'?' if det is None else det} kept"


# --------------------------------------------------------------------- plot

def plot_traj(out_path, seq, gt_xy, pred_xy, plane, title, subtitle):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="#888", lw=1.2, label="GT (matched)")
    ax.plot(pred_xy[:, 0], pred_xy[:, 1], color="#1d7aff", lw=1.0, label="pred")
    ax.scatter(*gt_xy[0], s=22, color="#2a9d8f", zorder=5, label="GT start")
    ax.scatter(*gt_xy[-1], s=22, color="#e63946", zorder=5, label="GT end")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel(f"{'xyz'[plane[0]]} [m]"); ax.set_ylabel(f"{'xyz'[plane[1]]} [m]")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"{seq}\n{subtitle}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------- main

def evaluate_seq(ds, seq, results_dir, gt_dir, out_dir, pred_suffix):
    """Plot one sequence; return its ATE RMSE [m], or None if skipped."""
    pred_path = os.path.join(results_dir, seq + pred_suffix)
    gt_path = os.path.join(gt_dir, ds.gt_path.format(seq=seq))
    for kind, p in (("GT", gt_path), ("pred", pred_path)):
        if not os.path.isfile(p):
            print(f"[skip] {seq}: {kind} not found at {p}")
            return None

    gt_ts, gt = (load_kitti if ds.gt_format == "kitti" else load_tum)(gt_path)
    pred_ts, pred = load_tum(pred_path)
    n_nan = int((~np.isfinite(pred)).any(axis=1).sum())
    if n_nan:
        print(f"[skip] {seq}: {n_nan}/{len(pred)} predicted poses are NaN/inf "
              f"(typically an FP16 GPU such as Titan RTX / Quadro RTX 6000; re-run on a BF16 GPU, e.g. RTX 4090)")
        return None
    pairs = (match_by_index(pred_ts, gt_ts) if ds.match == "index"
             else match_by_time(pred_ts, gt_ts, ds.max_time_diff))
    if len(pairs) < 4:
        hint = ""
        if ds.match == "time" and pred_ts.max() < 1e6:
            hint = (" (prediction timestamps look like frame indices, not seconds: "
                    "re-run so the trajectory stores image timestamps)")
        print(f"[skip] {seq}: only {len(pairs)} matched frames{hint}")
        return None
    pi, gi = pairs[:, 0], pairs[:, 1]

    s, R, t = umeyama_sim3(pred[pi], gt[gi])
    pred_a = (s * (R @ pred.T)).T + t
    ate = float(np.sqrt((np.linalg.norm(pred_a[pi] - gt[gi], axis=1) ** 2).mean()))

    loops = loop_label(results_dir, seq)
    a, b = ds.plane
    out_path = os.path.join(out_dir, f"{seq}_traj.png")
    plot_traj(out_path, seq, gt[gi][:, [a, b]], pred_a[:, [a, b]], ds.plane,
              f"ATE RMSE = {ate:.3f} m   (Sim3 s={s:.3f}, K={len(pairs)})",
              f"N_pred={len(pred)}  N_gt={len(gt)}  matched={len(pairs)}  {loops}")
    print(f"[ok]   {seq}: ATE={ate:.3f}m  K={len(pairs)}  {loops}  -> {out_path}")
    return ate


def main(ds):
    ap = argparse.ArgumentParser(description=f"{ds.name}: per-sequence trajectory "
                                 "plots (GT vs Sim3-aligned prediction) and ATE table.")
    ap.add_argument("--results_dir", required=True,
                    help="dir with the <seq><pred_suffix> predictions (TUM format)")
    ap.add_argument("--gt_dir", default=ds.default_gt_dir,
                    help=f"GT root; file = <gt_dir>/{ds.gt_path} (default: {ds.default_gt_dir})")
    ap.add_argument("--out_dir", default=None, help="where to write PNGs (default: results_dir)")
    ap.add_argument("--seq", default=None, help="only this sequence")
    ap.add_argument("--pred_suffix", default=ds.pred_suffix,
                    help=f"prediction file = <seq><pred_suffix> (default: {ds.pred_suffix})")
    args = ap.parse_args()
    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.seq:
        seqs = [args.seq]
    else:
        suffix = args.pred_suffix
        seqs = sorted(os.path.basename(p)[:-len(suffix)]
                      for p in glob.glob(os.path.join(args.results_dir, "*" + suffix)))
        seqs = [q for q in seqs if not q.endswith("_pre_pgo") and not q.startswith("results_")]
    if not seqs:
        print(f"No <seq>{args.pred_suffix} files in {args.results_dir}")
        return

    print(f"Rendering {len(seqs)} sequence(s) -> {out_dir}  (gt_dir={args.gt_dir})")
    rows = [(q, ate) for q in seqs
            if (ate := evaluate_seq(ds, q, args.results_dir, args.gt_dir, out_dir,
                                    args.pred_suffix)) is not None]
    if not rows:
        return
    rank = {q: i for i, q in enumerate(ds.order)}
    rows.sort(key=lambda r: (rank.get(r[0], len(rank)), r[0]))
    avg = sum(ate for _, ate in rows) / len(rows)
    print("\n% ----- LaTeX table (row 1: sequence, row 2: ATE RMSE [m]) -----")
    print(" & ".join([ds.labels.get(q, q) for q, _ in rows] + ["avg"]) + " \\\\")
    print(" & ".join([f"{ate:.2f}" for _, ate in rows] + [f"{avg:.3f}"]) + " \\\\")
