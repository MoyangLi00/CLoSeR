#!/bin/bash
# Download and prepare data/DROID-W/ (~17 GB).
# DROIDW_DATA_ROOT overrides the dataset location.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DROIDW_DATA_ROOT:-$REPO_ROOT/data/DROID-W}"

mkdir -p "$DATA_ROOT"
# The archive contains a top-level DROID-W directory.
(
    cd "$(dirname "$DATA_ROOT")"
    echo "=== Downloading DROID-W.zip ==="
    wget -c https://cvg-data.inf.ethz.ch/DROID-W/DROID-W.zip
    echo "=== Extracting DROID-W.zip ==="
    unzip -q -n DROID-W.zip -x 'DROID-W/videos_mp4/*'
    [[ "$(basename "$DATA_ROOT")" == DROID-W ]] || cp -a DROID-W/. "$(basename "$DATA_ROOT")/"
    rm DROID-W.zip
)

echo "=== Preparing sequences ==="
for seq in "$DATA_ROOT"/downtown{1..7}; do
    if [[ ! -d "$seq/images_anonymized" ]]; then
        echo "=== $(basename "$seq"): no images, skipping ==="; continue
    fi
    (
        cd "$seq"
        gt=traj_gt_fastlivo.txt
        [[ -f "$gt" ]] || gt=traj_gt.txt
        [[ -f "$gt" ]] || { echo "=== $(basename "$seq"): no ground truth, skipping ==="; exit 0; }
        ln -sfn "$gt" poses.txt
        printf '%s\n' images_anonymized/*.jpg | sort | \
            awk '{ts=$0; sub(/^.*\//, "", ts); sub(/\.jpg$/, "", ts); print ts, $0}' > rgb.txt
    )
done
echo "=== Done ==="
for seq in "$DATA_ROOT"/downtown{1..7}; do
    [[ -f "$seq/rgb.txt" && -f "$seq/poses.txt" ]] || continue
    n=$(wc -l < "$seq/rgb.txt")
    gt=$(grep -vc '^#' "$seq/poses.txt" || true)
    echo "  $(basename "$seq"): $n images, $gt GT poses"
done
