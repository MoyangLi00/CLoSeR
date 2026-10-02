#!/bin/bash
# Run LoGeR on the KITTI Odometry sequences 00-10.
# All flags are shared with the other dataset runners: see scripts/_run_common.sh
# or `bash scripts/run_kitti.sh --help`.
#
# Example:
#   bash scripts/run_kitti.sh --pgo --loopdetect --online_pgo --output_dir results/kitti

MODE=kitti
ALL_SEQS="00 01 02 03 04 05 06 07 08 09 10"
SEQS_ENV=KITTI_SEQS
WIN_LOGER=32; WIN_STAR=64; WIN_PI3=32

seq_input_dir() { echo "$REPO_ROOT/data/kitti/dataset/sequences/$1/image_2"; }

source "$(dirname "${BASH_SOURCE[0]}")/_run_common.sh"
