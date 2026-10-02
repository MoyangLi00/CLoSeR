#!/bin/bash
# Run LoGeR on the 14 Oxford Spires sequences.
# All flags are shared with the other dataset runners: see scripts/_run_common.sh
# or `bash scripts/run_oxford_spires.sh --help`.
# Evaluate with: python scripts/oxford_spires_traj_plot.py --results_dir <OUTPUT_DIR>/LoGeR
#
# Images: data/oxford_spires/processed_sequences/<seq>/raw/cam0, the rectified
# pinhole images from scripts_download/undistort_oxford_spires.py (run it once
# after download_oxford_spires.py). For the raw fisheye images as released:
#   OXFORD_SEQS_DIR=sequences bash scripts/run_oxford_spires.sh ...

MODE=oxford_spires
ALL_SEQS="2024-03-12-keble-college-02 2024-03-12-keble-college-03
          2024-03-12-keble-college-04 2024-03-12-keble-college-05
          2024-03-13-observatory-quarter-01 2024-03-13-observatory-quarter-02
          2024-03-14-blenheim-palace-01 2024-03-14-blenheim-palace-02
          2024-03-14-blenheim-palace-05
          2024-03-18-christ-church-01 2024-03-18-christ-church-02
          2024-03-18-christ-church-03 2024-03-18-christ-church-05
          2024-05-20-bodleian-library-02"
SEQS_ENV=OXFORD_SEQS
WIN_LOGER=32; WIN_STAR=64; WIN_PI3=32

seq_input_dir() { echo "$REPO_ROOT/data/oxford_spires/${OXFORD_SEQS_DIR:-processed_sequences}/$1/raw/cam0"; }

source "$(dirname "${BASH_SOURCE[0]}")/_run_common.sh"
