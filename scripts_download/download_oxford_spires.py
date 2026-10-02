#!/usr/bin/env python3
"""Download the Oxford Spires sequences LoGeR evaluates on.

Follows the official downloader (ori-drs/oxford_spires_dataset,
scripts/dataset_download.py): huggingface_hub.snapshot_download per pattern,
then unpack the zips and delete them, then check the result. Only what LoGeR
needs is fetched: per sequence raw/images.zip and the GT trajectory, plus the
calibration folder.

Resulting layout (what scripts/run_oxford_spires.sh and the eval/plot scripts expect):
  <local_dir>/calibration/cam0.yaml ...
  <local_dir>/sequences/<SEQ>/raw/cam0/<timestamp>.jpg          (cam1/cam2 with --all_cams)
  <local_dir>/sequences/<SEQ>/processed/trajectory/gt-tum.txt

snapshot_download skips files that are already complete and resumes partial
ones, so just re-run the script if it is interrupted. On a compute node, load
the proxy first (`module load eth_proxy`); login nodes have direct access.

Release notes that matter here (see the dataset CHANGELOG.md):
  - 2026-05-11: christ-church-05 was renamed 2024-03-18-* -> 2024-03-20-*. It is
    downloaded under the new name, with a 2024-03-18-christ-church-05 symlink so
    older scripts and results keep working. GT trajectories were also pruned to
    the rosbag time range, so numbers can differ slightly from older copies.
  - 2026-05-09: the christ-church-01 GT trajectory was removed (images only).

Dataset license: CC BY-NC-SA 4.0 (non-commercial academic use).

Usage:
  python scripts_download/download_oxford_spires.py                  # 14 eval sequences
  python scripts_download/download_oxford_spires.py --seq 2024-03-12-keble-college-02
  python scripts_download/download_oxford_spires.py --all_cams --keep_zip
  python scripts_download/download_oxford_spires.py --dry_run        # list what would be fetched
"""
import argparse
import logging
import os
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

REPO_ID = "ori-drs/oxford_spires_dataset"
REPO_TYPE = "dataset"
BRANCH = "main"
DEFAULT_LOCAL_DIR = str(Path(__file__).resolve().parents[1] / "data" / "oxford_spires")

EVAL_SEQS = [
    "2024-03-12-keble-college-02", "2024-03-12-keble-college-03",
    "2024-03-12-keble-college-04", "2024-03-12-keble-college-05",
    "2024-03-13-observatory-quarter-01", "2024-03-13-observatory-quarter-02",
    "2024-03-14-blenheim-palace-01", "2024-03-14-blenheim-palace-02",
    "2024-03-14-blenheim-palace-05",
    "2024-03-18-christ-church-01", "2024-03-18-christ-church-02",
    "2024-03-18-christ-church-03", "2024-03-18-christ-church-05",
    "2024-05-20-bodleian-library-02",
]
# Name used by LoGeR's scripts/results -> current name in the release.
RENAMED = {
    "2024-03-18-christ-church-05": "2024-03-20-christ-church-05",
    "2024-03-18-christ-church-06": "2024-03-20-christ-church-06",
}


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def unpacked(local_dir, name):
    """images.zip was already extracted (and removed) for this sequence."""
    raw = Path(local_dir, "sequences", name, "raw")
    cam0 = raw / "cam0"
    return (not (raw / "images.zip").exists() and cam0.is_dir()
            and any(p.suffix == ".jpg" for p in cam0.iterdir()))


def build_patterns(seqs, local_dir):
    """Only the files LoGeR needs (the official config downloads whole sequences).
    images.zip is skipped for sequences already unpacked, since the zip is deleted
    after extraction and snapshot_download would otherwise fetch it again."""
    # Camera/IMU yamls + README only; calibration/* would also pull ~38 GB of
    # calibration-recording zips. ('*' also matches '/', so subfolders are included.)
    patterns = ["calibration/*.yaml", "calibration/*.md", "calibration/*.txt"]
    for seq in seqs:
        name = RENAMED.get(seq, seq)
        if not unpacked(local_dir, name):
            patterns.append(f"sequences/{name}/raw/images.zip")
        patterns.append(f"sequences/{name}/processed/trajectory/gt-tum.txt")
    return patterns


def download_patterns(patterns, local_dir, dry_run=False):
    """Download each pattern from the HuggingFace repository (as the official script)."""
    logger.info(f"Repository: {REPO_ID}  ->  {local_dir}")
    for pattern in patterns:
        logger.info(f"=== Downloading {pattern} ===")
        out = snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            revision=BRANCH,
            allow_patterns=pattern,
            local_dir=local_dir,
            token=False,            # public dataset (official script: use_auth_token=False)
            dry_run=dry_run,
        )
        if dry_run:
            for f in out:
                logger.info(f"    would download {f.filename}  ({f.file_size / 1e9:.2f} GB)"
                            f"{'' if f.will_download else '  [already complete]'}")


def unpack_zip_files(local_dir, seqs, all_cams=False, keep_zip=False):
    """Unpack each sequence's raw/images.zip next to itself (official:
    shutil.unpack_archive + unlink). By default only cam0 is kept, the camera
    LoGeR uses; --all_cams also extracts cam1/cam2."""
    for seq in seqs:
        zip_path = Path(local_dir, "sequences", RENAMED.get(seq, seq), "raw", "images.zip")
        if not zip_path.is_file():
            continue
        logger.info(f"=== Extracting {seq} ===")
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if all_cams or m.startswith("cam0/")]
            zf.extractall(zip_path.parent, members=members)
        if not keep_zip:
            zip_path.unlink()


def prepare_timestamps(local_dir, seqs):
    """Write cam0 timestamps for trajectory estimation on the raw images."""
    for seq in seqs:
        raw = Path(local_dir, "sequences", RENAMED.get(seq, seq), "raw")
        cam0 = raw / "cam0"
        if cam0.is_dir():
            images = sorted(p for p in cam0.iterdir()
                            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            with (raw / "rgb.txt").open("w") as f:
                for image in images:
                    f.write(f"{image.stem} cam0/{image.name}\n")


def link_renamed(local_dir, seqs):
    for seq in seqs:
        new = RENAMED.get(seq)
        if new and Path(local_dir, "sequences", new).is_dir():
            link = Path(local_dir, "sequences", seq)
            if link.is_symlink() or not link.exists():
                if link.is_symlink():
                    link.unlink()
                link.symlink_to(new)
                logger.info(f"=== Linking {seq} -> {new} ===")


def check(local_dir, seqs):
    """Per sequence: number of cam0 images and GT poses (replaces the official
    image-lidar sync check, which needs the lidar data LoGeR does not use)."""
    ok = True
    summaries = []
    for seq in seqs:
        d = Path(local_dir, "sequences", seq)
        cam0 = d / "raw" / "cam0"
        gt = d / "processed" / "trajectory" / "gt-tum.txt"
        n_img = len([p for p in os.listdir(cam0) if p.endswith(".jpg")]) if cam0.is_dir() else 0
        n_gt = sum(1 for l in open(gt) if l.strip() and not l.startswith("#")) if gt.is_file() else 0
        ok &= n_img > 0
        summaries.append(f"  {seq}: {n_img} images, {n_gt} GT poses")
    if ok:
        logger.info("=== Done ===")
    for summary in summaries:
        logger.info(summary)
    return ok


def main():
    setup_logging()
    ap = argparse.ArgumentParser(description="Download Oxford Spires (images + GT) for LoGeR.")
    ap.add_argument("--local_dir", default=DEFAULT_LOCAL_DIR,
                    help=f"download root (default: {DEFAULT_LOCAL_DIR})")
    ap.add_argument("--seq", nargs="+", default=EVAL_SEQS,
                    help="sequences (default: the 14 LoGeR evaluation sequences)")
    ap.add_argument("--all_cams", action="store_true", help="also extract cam1 and cam2")
    ap.add_argument("--keep_zip", action="store_true", help="keep raw/images.zip after unpacking")
    ap.add_argument("--dry_run", action="store_true", help="only list what would be downloaded")
    args = ap.parse_args()

    Path(args.local_dir).mkdir(parents=True, exist_ok=True)
    download_patterns(build_patterns(args.seq, args.local_dir), args.local_dir, dry_run=args.dry_run)
    if args.dry_run:
        return
    unpack_zip_files(args.local_dir, args.seq, all_cams=args.all_cams, keep_zip=args.keep_zip)
    logger.info("=== Preparing sequences ===")
    prepare_timestamps(args.local_dir, args.seq)
    link_renamed(args.local_dir, args.seq)
    if not check(args.local_dir, args.seq):
        raise SystemExit("Some sequences have no images; re-run to resume the download.")
    cache = Path(args.local_dir, ".cache")
    if cache.exists():
        shutil.rmtree(cache)
        logger.info(f"=== Removed cache: {cache} ===")


if __name__ == "__main__":
    main()
