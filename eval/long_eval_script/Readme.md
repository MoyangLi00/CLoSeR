## VBR Benchmark

Run

```bash
./vbr_benchmark /path_to/vbr_gt/ /path_to/method/ [--plot]
```

Example:
```bash
./vbr_eval/vbr_benchmark data/vbr/processed_gt data/data/vbr/processed_test --plot
```

Compile:
```bash
g++ -o vbr_benchmark vbr_benchmark.cpp -I /usr/include/eigen3 -O3 -std=c++17
```

Trajectory format: `**_es.txt` with TUM format (timestamp tx ty tz qx qy qz qw)

You can enable the --plot flag to get aligned trajectory plots and error files for your training estimates.

## KITTI Odometry Benchmark

Run

```bash
./kitti_benchmark /path_to/kitti_gt/ /path_to/method/ [--plot] [--se3]
```

Example:
```bash
./kitti_benchmark /path/to/kitti/dataset/poses /path/to/results --plot
```

Compile:
```bash
g++ -o kitti_benchmark kitti_benchmark.cpp -I /usr/include/eigen3 -O3 -std=c++17
```

Options:
- `--plot`: Generate SVG trajectory plots and error dumps
- `--se3`: Use SE(3) alignment instead of Sim(3) (for stereo/LiDAR methods)

**GT format (KITTI)**: 3x4 matrix per line (r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz)
**ES format**: TUM format with timestamp (timestamp tx ty tz qx qy qz qw)

Output files in `results_sim3/` or `results_se3/`:
- `*_top.svg`, `*_front.svg`, `*_side.svg`: Trajectory visualization
- `*_ate.txt`, `*_rpe.txt`: Per-frame error dumps
- `results_ate.txt`, `results_rpe.txt`: Summary statistics