import argparse
import os
import subprocess
import sys
from pathlib import Path


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    v = str(v).lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Python equivalent of eval/relpose/run_bonn.sh"
    )

    parser.add_argument(
        "ckpt_name",
        nargs="?",
        default="LoGeR",
        help="Checkpoint name under ckpts/ (default: LoGeR)",
    )
    parser.add_argument(
        "datasets_positional",
        nargs="*",
        help="Optional dataset names as positional args after ckpt_name",
    )

    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--revisit", type=int, default=1)
    parser.add_argument("--freeze-state", action="store_true")
    parser.add_argument("--solve-pose", action="store_true")
    parser.add_argument("--window-size", default="48")
    parser.add_argument("--overlap-size", default="3")
    parser.add_argument("--num-iterations", default="")
    parser.add_argument("--causal", default="")
    parser.add_argument("--sim3", nargs="?", const="true", default="")
    parser.add_argument("--sim3_mean", nargs="?", const="true", default="")
    parser.add_argument("--se3", nargs="?", const="true", default="")
    parser.add_argument("--pi3x", nargs="?", const="true", default="")
    parser.add_argument("--pi3x-metric", nargs="?", const="true", default="")
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--port", type=int, default=29551)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--tag", default="pi3")
    parser.add_argument("--weights-path", default="")
    parser.add_argument("--pi3-config", default="")
    parser.add_argument("--no-pi3-config", action="store_true")

    return parser.parse_args()


def is_true_string(v: str) -> bool:
    return str(v).lower() in {"true", "1", "yes", "y"}


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    os.chdir(repo_root)

    ckpt_run = args.ckpt if args.ckpt else args.ckpt_name
    if not ckpt_run:
        ckpt_run = "LoGeR"

    datasets = []
    if args.datasets:
        datasets.extend([x for x in args.datasets.split(",") if x.strip()])
    if args.datasets_positional:
        datasets.extend(args.datasets_positional)
    if not datasets:
        datasets = ["bonn_s1_500"]

    run_name = Path(ckpt_run).name if Path(ckpt_run).name else "LoGeR"

    pi3x_is_true = is_true_string(args.pi3x) if args.pi3x else False
    pi3x_metric_is_true = is_true_string(args.pi3x_metric) if args.pi3x_metric else False

    if pi3x_metric_is_true:
        run_name = f"{run_name}_pi3x_metric"
    elif pi3x_is_true:
        run_name = f"{run_name}_pi3x"

    sim3_is_true = is_true_string(args.sim3) if args.sim3 else False
    se3_is_true = is_true_string(args.se3) if args.se3 else False
    sim3_mean_is_true = is_true_string(args.sim3_mean) if args.sim3_mean else False

    sim3_suffix = ""
    se3_suffix = ""

    if sim3_is_true:
        sim3_suffix = "_sim3"
    if se3_is_true:
        se3_suffix = "_se3"
    if sim3_mean_is_true:
        sim3_is_true = True
        sim3_suffix = "_sim3_mean"

    if sim3_is_true and se3_is_true:
        print("Error: --sim3 and --se3 cannot both be true simultaneously.", file=sys.stderr)
        sys.exit(1)

    window_tag = f"win{args.window_size}o{args.overlap_size}{sim3_suffix}{se3_suffix}"

    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = repo_root / "eval_results" / run_name / window_tag / "relpose"

    ckpt_dir = repo_root / "ckpts" / ckpt_run
    default_config_path = ckpt_dir / "original_config.yaml"
    weights_path = Path(args.weights_path) if args.weights_path else ckpt_dir / "latest.pt"

    pi3_config_enabled = not args.no_pi3_config
    if pi3_config_enabled:
        if args.pi3_config:
            pi3_config = Path(args.pi3_config)
        elif default_config_path.is_file():
            pi3_config = default_config_path
        else:
            pi3_config = None
    else:
        pi3_config = None

    if not weights_path.is_file():
        if not args.weights_path and ckpt_dir.is_dir():
            candidates = sorted(ckpt_dir.glob("*.pt"))
            if candidates:
                weights_path = candidates[0]
                print(f"Using checkpoint file {weights_path}")

    if not weights_path.is_file():
        print(f"Error: checkpoint weights not found ({weights_path})", file=sys.stderr)
        print(f"Expected default path: {ckpt_dir / 'latest.pt'}", file=sys.stderr)
        sys.exit(1)

    if pi3_config is not None and not pi3_config.is_file():
        print(f"Warning: Pi3 config not found ({pi3_config}); continuing without it.", file=sys.stderr)
        pi3_config = None

    output_root.mkdir(parents=True, exist_ok=True)

    forward_args = []
    if args.window_size:
        forward_args += ["--window_size", str(args.window_size)]
    if args.overlap_size:
        forward_args += ["--overlap_size", str(args.overlap_size)]
    if args.num_iterations:
        forward_args += ["--num_iterations", str(args.num_iterations)]
    if args.causal:
        forward_args += ["--causal", str(args.causal)]
    if args.sim3:
        forward_args += ["--sim3", str(args.sim3)]
    if args.sim3_mean:
        forward_args += ["--sim3_mean", str(args.sim3_mean)]
    if args.se3:
        forward_args += ["--se3", str(args.se3)]
    if args.pi3x:
        forward_args += ["--pi3x", str(args.pi3x)]
    if args.pi3x_metric:
        forward_args += ["--pi3x_metric", str(args.pi3x_metric)]

    freeze_flag = ["--freeze_state"] if args.freeze_state else []
    solve_flag = ["--solve_pose"] if args.solve_pose else []

    pi3_flag = []
    if pi3_config is not None:
        pi3_flag = ["--pi3_config", str(pi3_config)]

    for dataset in datasets:
        output_dir = output_root / dataset / args.tag
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> Evaluating {dataset} with checkpoint {ckpt_run}")
        print(f"Results -> {output_dir}")

        cmd = [
            "accelerate",
            "launch",
            "--num_processes",
            str(args.num_processes),
            "--main_process_port",
            str(args.port),
            "eval/relpose/launch.py",
            "--weights",
            str(weights_path),
            "--output_dir",
            str(output_dir),
            "--eval_dataset",
            str(dataset),
            "--size",
            str(args.size),
            "--revisit",
            str(args.revisit),
            *freeze_flag,
            *solve_flag,
            *pi3_flag,
            *forward_args,
        ]

        print("Running command:")
        print(" ".join(cmd))

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()