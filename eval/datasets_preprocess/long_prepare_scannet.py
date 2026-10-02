import glob
import os
import shutil

import numpy as np

SAMPLE_INTERVAL = 3  # sampling interval, take 1 frame every N frames (originally 3)
BASE_TARGET_FRAMES = 1000
TARGET_FRAME_CHOICES = [50, 90, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, BASE_TARGET_FRAMES]
OTHER_TARGETS = [t for t in TARGET_FRAME_CHOICES if t < BASE_TARGET_FRAMES]

DATA_ROOT = "data/scannetv2"
OUTPUT_ROOT = f"data/long_scannet_s{SAMPLE_INTERVAL}"


def remove_path(path: str):
    if os.path.islink(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def save_pose_file(path: str, pose_lines, frame_count: int):
    if os.path.islink(path):
        os.unlink(path)
    with open(path, 'w') as f:
        for line in pose_lines[:frame_count]:
            f.write(f"{line}\n")


def create_relative_symlinks(base_dir: str, target_dir: str, frame_count: int, suffix: str):
    if frame_count == 0:
        return
    base_dir_abs = os.path.abspath(base_dir)
    target_dir_abs = os.path.abspath(target_dir)
    for i in range(frame_count):
        source_file = os.path.join(base_dir_abs, f"frame_{i:04d}.{suffix}")
        dest_file = os.path.join(target_dir_abs, f"frame_{i:04d}.{suffix}")
        rel_source = os.path.relpath(source_file, start=os.path.dirname(dest_file))
        os.symlink(rel_source, dest_file)


seq_list = sorted([seq for seq in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, seq))])

for seq in seq_list:
    color_paths = sorted(glob.glob(f"{DATA_ROOT}/{seq}/color/*.jpg"), key=lambda x: int(os.path.basename(x).split('.')[0]))
    depth_paths = sorted(glob.glob(f"{DATA_ROOT}/{seq}/depth/*.png"), key=lambda x: int(os.path.basename(x).split('.')[0]))
    pose_paths = sorted(glob.glob(f"{DATA_ROOT}/{seq}/pose/*.txt"), key=lambda x: int(os.path.basename(x).split('.')[0]))

    if not color_paths or not depth_paths or not pose_paths:
        print(f"{seq}: missing color/depth/pose files, skip")
        continue

    total_frames = min(len(color_paths), len(depth_paths), len(pose_paths))
    available_frames = total_frames // SAMPLE_INTERVAL
    if available_frames == 0:
        print(f"{seq}: insufficient frames (total {total_frames})")
        continue

    seq_root = os.path.join(OUTPUT_ROOT, seq)
    os.makedirs(seq_root, exist_ok=True)

    base_actual_frames = min(BASE_TARGET_FRAMES, available_frames)
    print(f"{seq}: original frame count {total_frames}, target frames {BASE_TARGET_FRAMES}, actual frames {base_actual_frames}")

    base_slice = base_actual_frames * SAMPLE_INTERVAL
    base_img_paths = color_paths[:base_slice:SAMPLE_INTERVAL]
    base_depth_paths = depth_paths[:base_slice:SAMPLE_INTERVAL]
    base_pose_paths = pose_paths[:base_slice:SAMPLE_INTERVAL]

    pose_lines = []
    for pose_path in base_pose_paths:
        pose = np.loadtxt(pose_path)
        pose_lines.append(' '.join(map(str, pose.reshape(-1))))

    base_color_dir = os.path.join(seq_root, f"color_{BASE_TARGET_FRAMES}")
    base_depth_dir = os.path.join(seq_root, f"depth_{BASE_TARGET_FRAMES}")

    remove_path(base_color_dir)
    remove_path(base_depth_dir)
    os.makedirs(base_color_dir, exist_ok=True)
    os.makedirs(base_depth_dir, exist_ok=True)

    for i, (img_path, depth_path) in enumerate(zip(base_img_paths, base_depth_paths)):
        shutil.copy(img_path, os.path.join(base_color_dir, f"frame_{i:04d}.jpg"))
        shutil.copy(depth_path, os.path.join(base_depth_dir, f"frame_{i:04d}.png"))

    save_pose_file(os.path.join(seq_root, f"pose_{BASE_TARGET_FRAMES}.txt"), pose_lines, base_actual_frames)

    for target_frames in OTHER_TARGETS:
        actual_target_frames = min(target_frames, available_frames)
        print(f"{seq}: original frame count {total_frames}, target frames {target_frames}, actual frames {actual_target_frames}")

        if actual_target_frames == 0:
            continue

        color_dir = os.path.join(seq_root, f"color_{target_frames}")
        depth_dir = os.path.join(seq_root, f"depth_{target_frames}")

        remove_path(color_dir)
        remove_path(depth_dir)
        os.makedirs(color_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        create_relative_symlinks(base_color_dir, color_dir, actual_target_frames, "jpg")
        create_relative_symlinks(base_depth_dir, depth_dir, actual_target_frames, "png")

        save_pose_file(os.path.join(seq_root, f"pose_{target_frames}.txt"), pose_lines, actual_target_frames)
