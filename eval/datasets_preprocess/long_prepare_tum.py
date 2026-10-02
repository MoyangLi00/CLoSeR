import glob
import os
import shutil
import numpy as np

SAMPLE_INTERVAL = 1  # sampling interval, take 1 frame every N frames (originally 3)
BASE_TARGET_FRAMES = 10000
TARGET_FRAME_CHOICES = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, BASE_TARGET_FRAMES]
OTHER_TARGETS = [t for t in TARGET_FRAME_CHOICES if t < BASE_TARGET_FRAMES]

DATA_ROOT = "data/tum"
OUTPUT_ROOT = f"data/long_tum_s{SAMPLE_INTERVAL}"


def remove_path(path: str):
    if os.path.islink(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def save_gt_file(path: str, gt_lines, frame_count: int):
    if os.path.islink(path):
        os.unlink(path)
    with open(path, 'w') as f:
        for line in gt_lines[:frame_count]:
            f.write(f"{' '.join(map(str, line))}\n")


def create_relative_symlinks(base_dir: str, target_dir: str, filenames: list, count: int):
    """Create relative symlinks from target_dir pointing to files in base_dir."""
    if count == 0:
        return
    base_dir_abs = os.path.abspath(base_dir)
    target_dir_abs = os.path.abspath(target_dir)
    for filename in filenames[:count]:
        source_file = os.path.join(base_dir_abs, filename)
        dest_file = os.path.join(target_dir_abs, filename)
        rel_source = os.path.relpath(source_file, start=os.path.dirname(dest_file))
        os.symlink(rel_source, dest_file)


def read_file_list(filename):
    """
    Reads a trajectory from a text file. 
    
    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 
    
    Input:
    filename -- File name
    
    Output:
    dict -- dictionary of (stamp,data) tuples
    
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)

def associate(first_list, second_list, offset, max_difference):
    """
    Associate two dictionaries of (stamp, data). As the time stamps never match exactly, we aim 
    to find the closest match for every input tuple.
    
    Input:
    first_list -- first dictionary of (stamp, data) tuples
    second_list -- second dictionary of (stamp, data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1, data1), (stamp2, data2))
    """
    # Convert keys to sets for efficient removal
    first_keys = set(first_list.keys())
    second_keys = set(second_list.keys())
    
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    
    matches.sort()
    return matches


os.makedirs(OUTPUT_ROOT, exist_ok=True)

dirs = glob.glob(f"{DATA_ROOT}/*/")
dirs = sorted(dirs)

for dir in dirs:
    frames = []
    gt = []
    first_file = dir + 'rgb.txt'
    second_file = dir + 'groundtruth.txt'

    if not os.path.exists(first_file) or not os.path.exists(second_file):
        print(f"{dir}: missing rgb.txt or groundtruth.txt, skip")
        continue

    first_list = read_file_list(first_file)
    second_list = read_file_list(second_file)
    matches = associate(first_list, second_list, 0.0, 0.02)

    for a, b in matches:
        frames.append(dir + first_list[a][0])
        gt.append([b] + second_list[b])

    total_frames = len(frames)
    available_frames = total_frames // SAMPLE_INTERVAL
    if available_frames == 0:
        print(f"{dir}: insufficient frames (total {total_frames})")
        continue

    dir_name = os.path.basename(os.path.dirname(dir))
    seq_root = os.path.join(OUTPUT_ROOT, dir_name)
    os.makedirs(seq_root, exist_ok=True)

    # --- Step 1: Process BASE_TARGET_FRAMES (1000) with real copies ---
    base_actual_frames = min(BASE_TARGET_FRAMES, available_frames)
    print(f"{dir_name}: original frame count {total_frames}, target frames {BASE_TARGET_FRAMES}, actual frames {base_actual_frames}")

    base_frames = frames[::SAMPLE_INTERVAL][:base_actual_frames]
    base_gt = gt[::SAMPLE_INTERVAL][:base_actual_frames]

    base_rgb_dir = os.path.join(seq_root, f"rgb_{BASE_TARGET_FRAMES}")
    remove_path(base_rgb_dir)
    os.makedirs(base_rgb_dir, exist_ok=True)

    # Copy files and record the new filenames
    base_filenames = []
    for frame in base_frames:
        filename = os.path.basename(frame)
        base_filenames.append(filename)
        shutil.copy(frame, os.path.join(base_rgb_dir, filename))

    save_gt_file(os.path.join(seq_root, f"groundtruth_{BASE_TARGET_FRAMES}.txt"), base_gt, base_actual_frames)

    # --- Step 2: Process OTHER_TARGETS with symlinks ---
    for target_frames in OTHER_TARGETS:
        actual_target_frames = min(target_frames, available_frames)
        print(f"{dir_name}: original frame count {total_frames}, target frames {target_frames}, actual frames {actual_target_frames}")

        if actual_target_frames == 0:
            continue

        rgb_dir = os.path.join(seq_root, f"rgb_{target_frames}")
        remove_path(rgb_dir)
        os.makedirs(rgb_dir, exist_ok=True)

        create_relative_symlinks(base_rgb_dir, rgb_dir, base_filenames, actual_target_frames)

        save_gt_file(os.path.join(seq_root, f"groundtruth_{target_frames}.txt"), base_gt, actual_target_frames)

    print(f"Finished {dir_name} with base {base_actual_frames} frames")

