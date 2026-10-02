#!/bin/bash
# Run LoGeR on the 7 VBR sequences (download: scripts_download/download_vbr.sh).
# All flags are shared with the other dataset runners: see scripts/_run_common.sh
# or `bash scripts/run_vbr.sh --help`.
#
# Example:
#   bash scripts/run_vbr.sh --pgo --loopdetect --online_pgo --output_dir results/vbr
#   VBR_SEQS="pincio_train0 spagna_train0" bash scripts/run_vbr.sh

MODE=vbr
ALL_SEQS="campus_train0 campus_train1 ciampino_train1 colosseo_train0 diag_train0 pincio_train0 spagna_train0"
SEQS_ENV=VBR_SEQS
WIN_LOGER=48; WIN_STAR=64; WIN_PI3=32

seq_input_dir() { echo "$REPO_ROOT/data/vbr/$1_processed_aligned/rgb"; }

source "$(dirname "${BASH_SOURCE[0]}")/_run_common.sh"
