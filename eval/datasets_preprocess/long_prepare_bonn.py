import glob
import os
import shutil
import numpy as np

START_FRAME = 30
TARGET_FRAME_CHOICES = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 10000]
BASE_TARGET_FRAMES = max(TARGET_FRAME_CHOICES)
OTHER_TARGETS = [t for t in TARGET_FRAME_CHOICES if t < BASE_TARGET_FRAMES]

DATA_ROOT = "data/bonn/rgbd_bonn_dataset"
OUTPUT_ROOT = "data/long_bonn_s1/rgbd_bonn_dataset"


def remove_path(path: str):
    if os.path.islink(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def save_gt_file(path: str, gt_values: np.ndarray, frame_count: int):
    if os.path.islink(path):
        os.unlink(path)
    np.savetxt(path, gt_values[:frame_count])


def create_relative_symlinks(base_dir: str, target_dir: str, filenames: list, count: int):
    if count == 0:
        return

    base_dir_abs = os.path.abspath(base_dir)
    target_dir_abs = os.path.abspath(target_dir)

    for filename in filenames[:count]:
        source_file = os.path.join(base_dir_abs, filename)
        dest_file = os.path.join(target_dir_abs, filename)
        rel_source = os.path.relpath(source_file, start=os.path.dirname(dest_file))
        os.symlink(rel_source, dest_file)


SELECTED_SEQUENCES = {
    "rgbd_bonn_balloon",
    "rgbd_bonn_balloon2",
    "rgbd_bonn_crowd",
    "rgbd_bonn_crowd2",
    "rgbd_bonn_person_tracking",
    "rgbd_bonn_person_tracking2",
    "rgbd_bonn_removing_nonobstructing_box",
    "rgbd_bonn_removing_nonobstructing_box2",
}

os.makedirs(OUTPUT_ROOT, exist_ok=True)

dirs = sorted(glob.glob(f"{DATA_ROOT}/*/"))
dirs = [d for d in dirs if os.path.basename(os.path.dirname(d)) in SELECTED_SEQUENCES]
for seq_dir in dirs:
    dir_name = os.path.basename(os.path.dirname(seq_dir))

    rgb_frames = sorted(glob.glob(os.path.join(seq_dir, "rgb", "*.png")))
    depth_frames = sorted(glob.glob(os.path.join(seq_dir, "depth", "*.png")))
    gt_path = os.path.join(seq_dir, "groundtruth.txt")

    if not rgb_frames or not depth_frames or not os.path.exists(gt_path):
        print(f"{dir_name}: missing rgb/depth/groundtruth, skip")
        continue

    gt_values = np.loadtxt(gt_path)
    if gt_values.ndim == 1:
        gt_values = gt_values.reshape(1, -1)

    available_rgb_frames = max(0, len(rgb_frames) - START_FRAME)
    available_depth_frames = max(0, len(depth_frames) - START_FRAME)
    available_gt_frames = max(0, len(gt_values) - START_FRAME)
    available_frames = min(available_rgb_frames, available_depth_frames, available_gt_frames)

    if available_frames == 0:
        print(f"{dir_name}: insufficient frames after START_FRAME={START_FRAME}, skip")
        continue

    rgb_frames = rgb_frames[START_FRAME:START_FRAME + available_frames]
    depth_frames = depth_frames[START_FRAME:START_FRAME + available_frames]
    gt_values = gt_values[START_FRAME:START_FRAME + available_frames]

    seq_root = os.path.join(OUTPUT_ROOT, dir_name)
    os.makedirs(seq_root, exist_ok=True)

    base_actual_frames = min(BASE_TARGET_FRAMES, available_frames)
    print(
        f"{dir_name}: available after start {available_frames}, "
        f"target frames {BASE_TARGET_FRAMES}, actual frames {base_actual_frames}"
    )

    base_rgb_dir = os.path.join(seq_root, f"rgb_{BASE_TARGET_FRAMES}")
    base_depth_dir = os.path.join(seq_root, f"depth_{BASE_TARGET_FRAMES}")
    remove_path(base_rgb_dir)
    remove_path(base_depth_dir)
    os.makedirs(base_rgb_dir, exist_ok=True)
    os.makedirs(base_depth_dir, exist_ok=True)

    base_rgb_filenames = []
    for frame in rgb_frames[:base_actual_frames]:
        filename = os.path.basename(frame)
        base_rgb_filenames.append(filename)
        shutil.copy(frame, os.path.join(base_rgb_dir, filename))

    base_depth_filenames = []
    for frame in depth_frames[:base_actual_frames]:
        filename = os.path.basename(frame)
        base_depth_filenames.append(filename)
        shutil.copy(frame, os.path.join(base_depth_dir, filename))

    save_gt_file(
        os.path.join(seq_root, f"groundtruth_{BASE_TARGET_FRAMES}.txt"),
        gt_values,
        base_actual_frames,
    )

    for target_frames in OTHER_TARGETS:
        actual_target_frames = min(target_frames, available_frames)
        print(
            f"{dir_name}: available after start {available_frames}, "
            f"target frames {target_frames}, actual frames {actual_target_frames}"
        )

        if actual_target_frames == 0:
            continue

        rgb_dir = os.path.join(seq_root, f"rgb_{target_frames}")
        depth_dir = os.path.join(seq_root, f"depth_{target_frames}")
        remove_path(rgb_dir)
        remove_path(depth_dir)
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        create_relative_symlinks(base_rgb_dir, rgb_dir, base_rgb_filenames, actual_target_frames)
        create_relative_symlinks(base_depth_dir, depth_dir, base_depth_filenames, actual_target_frames)

        save_gt_file(
            os.path.join(seq_root, f"groundtruth_{target_frames}.txt"),
            gt_values,
            actual_target_frames,
        )