#!/bin/bash
# Download the processed VBR benchmark (LoGeR release, huggingface.co/datasets/Junyi42/vbr_processed)
# into data/vbr, one sequence at a time.
#
# Resulting layout (what eval/demo_run_longeval.sh, scripts/run_vbr.sh and
# scripts/vbr_traj_plot.py expect):
#   data/vbr/<SEQ>_processed_aligned/{rgb/, camera_pose/, camera_pose.txt, intrinsics.txt}
#   data/vbr/processed_gt/<SEQ>_gt.txt
#
# Usage:
#   bash scripts_download/download_vbr.sh [--seq "SEQ ..."] [--root DIR] [--keep_tar]
#
#   --seq       space-separated subset (default: all 7 sequences, ~121 GB of tar.gz)
#               campus_train0 campus_train1 ciampino_train1 colosseo_train0
#               diag_train0 pincio_train0 spagna_train0
#   --root      target dir (default: data/vbr, which the runners read from)
#   --keep_tar  keep the downloaded .tar.gz files after extraction
#
# Downloads are resumable (wget -c) and finished sequences are skipped, so just
# re-run the script if it is interrupted.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL_BASE="https://huggingface.co/datasets/Junyi42/vbr_processed/resolve/main"
ALL_SEQS="campus_train0 campus_train1 ciampino_train1 colosseo_train0 diag_train0 pincio_train0 spagna_train0"

SEQS="$ALL_SEQS"
ROOT="$REPO_ROOT/data/vbr"
KEEP_TAR=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seq)      [[ $# -ge 2 ]] || { echo "Error: --seq requires a value."; exit 1; }
                    SEQS="$2"; shift 2 ;;
        --root)     [[ $# -ge 2 ]] || { echo "Error: --root requires a value."; exit 1; }
                    ROOT="$2"; shift 2 ;;
        --keep_tar) KEEP_TAR=1; shift ;;
        -h|--help)  sed -n 2,20p "$0"; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

for seq in $SEQS; do
    [[ " $ALL_SEQS " == *" $seq "* ]] || { echo "Error: unknown sequence '$seq' (valid: $ALL_SEQS)"; exit 1; }
done

TAR_DIR="$ROOT/tars"
mkdir -p "$TAR_DIR" "$ROOT/processed_gt"

for seq in $SEQS; do
    dst="$ROOT/${seq}_processed_aligned"
    if [[ -f "$dst/.done" && -f "$ROOT/processed_gt/${seq}_gt.txt" ]]; then
        echo "=== $seq: already done, skipping ==="
        continue
    fi

    echo "=== Downloading $seq ==="
    wget -c --show-progress -O "$TAR_DIR/$seq.tar.gz" "$URL_BASE/$seq.tar.gz"

    # The per-sequence tar unpacks to <seq>/{rgb, camera_pose, camera_pose.txt,
    # intrinsics.txt, <seq>_gt.txt}; move it into the layout the eval code uses.
    echo "=== Extracting $seq ==="
    tmp="$ROOT/_extract_$seq"
    rm -rf "$tmp"
    mkdir -p "$tmp"
    tar -xzf "$TAR_DIR/$seq.tar.gz" -C "$tmp"
    [[ -d "$tmp/$seq/rgb" ]] || { echo "Error: $seq tar has no $seq/rgb/"; exit 1; }

    mv "$tmp/$seq/${seq}_gt.txt" "$ROOT/processed_gt/${seq}_gt.txt"
    mkdir -p "$dst"
    for item in "$tmp/$seq"/*; do
        name="$(basename "$item")"
        # Replace stale entries (e.g. dangling rgb symlinks left by a scratch purge).
        rm -rf "${dst:?}/$name"
        mv "$item" "$dst/$name"
    done
    rm -rf "$tmp"
    touch "$dst/.done"

    if [[ "$KEEP_TAR" -eq 0 ]]; then
        rm -f "$TAR_DIR/$seq.tar.gz"
    fi
done

[[ "$KEEP_TAR" -eq 1 ]] || rmdir "$TAR_DIR" 2>/dev/null || true

echo "=== Done ==="
for seq in $SEQS; do
    n=$(ls "$ROOT/${seq}_processed_aligned/rgb" 2>/dev/null | wc -l)
    gt=$([[ -f "$ROOT/processed_gt/${seq}_gt.txt" ]] && grep -vc '^#' "$ROOT/processed_gt/${seq}_gt.txt" || echo 0)
    echo "  $seq: $n images, $gt GT poses"
done
