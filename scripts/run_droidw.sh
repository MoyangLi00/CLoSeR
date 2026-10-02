#!/bin/bash
# Run LoGeR on the 7 DROID-W downtown sequences.
# All flags are shared with the other dataset runners: see scripts/_run_common.sh
# or `bash scripts/run_droidw.sh --help`.
#
# Example:
#   bash scripts/run_droidw.sh --pgo --loopdetect --online_pgo --output_dir results/droidw

MODE=droidw
ALL_SEQS="downtown1 downtown2 downtown3 downtown4 downtown5 downtown6 downtown7"
SEQS_ENV=DROIDW_SEQS
WIN_LOGER=32; WIN_STAR=64; WIN_PI3=32

seq_input_dir() { echo "${DROIDW_DATA_ROOT:-$REPO_ROOT/data/DROID-W}/$1/images_anonymized"; }

source "$(dirname "${BASH_SOURCE[0]}")/_run_common.sh"
