#!/bin/bash
# Download the KITTI Odometry (SLAM) benchmark into data/kitti/dataset.
#
# Resulting layout:
#   data/kitti/dataset/sequences/<SEQ>/{image_2/, calib.txt, times.txt}
#   data/kitti/dataset/poses/<SEQ>.txt          (GT, sequences 00-10 only)
#
# Usage:
#   bash scripts_download/download_kitti.sh [--both_cams] [--keep_zip]
#
#   --both_cams   also extract image_3 (right camera); default extracts image_2 only
#   --keep_zip    keep the downloaded .zip files after extraction
#
# Downloads are resumable (wget -c): just re-run the script if it is interrupted.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL_BASE="https://s3.eu-central-1.amazonaws.com/avg-kitti"
ROOT="$REPO_ROOT/data/kitti"
ZIP_DIR="$ROOT/zips"

BOTH_CAMS=0
KEEP_ZIP=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --both_cams) BOTH_CAMS=1; shift ;;
        --keep_zip)  KEEP_ZIP=1; shift ;;
        -h|--help)   sed -n 2,14p "$0"; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$ZIP_DIR"

download() {
    local name="$1"
    echo "=== Downloading $name ==="
    wget -c --show-progress -O "$ZIP_DIR/$name" "$URL_BASE/$name"
}

extract() {
    local name="$1"; shift
    echo "=== Extracting $name ==="
    # -n: never overwrite, so re-running after a partial extract is cheap
    unzip -q -n "$ZIP_DIR/$name" "$@" -d "$ROOT"
}

# 1) Calibration (calib.txt per sequence, tiny)
download data_odometry_calib.zip
extract  data_odometry_calib.zip

# 2) GT poses for 00-10 (tiny)
download data_odometry_poses.zip
extract  data_odometry_poses.zip

# 3) Color images (~69 GB zip). times.txt is included here.
download data_odometry_color.zip
if [[ "$BOTH_CAMS" -eq 1 ]]; then
    extract data_odometry_color.zip
else
    extract data_odometry_color.zip "dataset/sequences/*/image_2/*" "dataset/sequences/*/times.txt"
fi

if [[ "$KEEP_ZIP" -eq 0 ]]; then
    rm -rf "$ZIP_DIR"
fi

echo "=== Done ==="
for seq in 00 01 02 03 04 05 06 07 08 09 10; do
    n=$(ls "$ROOT/dataset/sequences/$seq/image_2" 2>/dev/null | wc -l)
    gt=$([[ -f "$ROOT/dataset/poses/$seq.txt" ]] && grep -vc '^#' "$ROOT/dataset/poses/$seq.txt" || echo 0)
    echo "  $seq: $n images, $gt GT poses"
done
