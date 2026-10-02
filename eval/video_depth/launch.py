import numpy as np
import torch
import argparse
import os
import sys
import cv2
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from copy import deepcopy
from eval.video_depth.metadata import dataset_metadata
from eval.video_depth.utils import save_depth_maps
from eval.pi3_adapter import load_pi3_model, merge_forward_kwargs, run_pi3_inference_on_views
from accelerate import PartialState
import time
from tqdm import tqdm

PATCH_ALIGN = 14


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        help="path to the model weights",
        default="",
    )

    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument(
        "--no_crop", type=bool, default=True, help="whether to crop input data"
    )

    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="sintel",
        choices=list(dataset_metadata.keys()),
    )
    parser.add_argument("--size", type=int, default="224")

    parser.add_argument(
        "--pose_eval_stride", default=1, type=int, help="stride for pose evaluation"
    )
    parser.add_argument(
        "--full_seq",
        action="store_true",
        default=False,
        help="use full sequence for pose evaluation",
    )
    parser.add_argument(
        "--seq_list",
        nargs="+",
        default=None,
        help="list of sequences for pose evaluation",
    )
    parser.add_argument(
        "--pi3_config",
        type=str,
        default=None,
        help="optional path to Pi3 training config yaml",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="override Pi3 window size (-1 keeps config)",
    )
    parser.add_argument(
        "--overlap_size",
        type=int,
        default=None,
        help="override Pi3 overlap size",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=None,
        help="override Pi3 decoding iterations",
    )
    parser.add_argument(
        "--causal",
        type=str,
        default=None,
        choices=["true", "false"],
        help="override causal flag (true/false)",
    )
    parser.add_argument(
        "--sim3",
        type=str,
        default=None,
        choices=["true", "false"],
        help="override Sim(3) merge flag",
    )
    parser.add_argument(
        "--sim3_mean",
        type=str,
        default=None,
        choices=["true", "false"],
        help="override Sim(3) merge flag with trimmed mean scale",
    )
    parser.add_argument(
        "--se3",
        type=str,
        default=None,
        choices=["true", "false"],
        help="override SE(3) merge flag",
    )
    parser.add_argument(
        "--pi3x",
        type=str,
        default=None,
        choices=["true", "false"],
        help="override Pi3X flag",
    )
    parser.add_argument(
        "--pi3x_metric",
        type=str,
        default=None,
        choices=["true", "false"],
        help="override Pi3X Metric flag",
    )
    return parser


def eval_pose_estimation(args, save_dir=None):
    metadata = dataset_metadata.get(args.eval_dataset)
    img_path = metadata["img_path"]
    mask_path = metadata["mask_path"]

    ate_mean, rpe_trans_mean, rpe_rot_mean = eval_pose_estimation_dist(
        args, save_dir=save_dir, img_path=img_path, mask_path=mask_path
    )
    return ate_mean, rpe_trans_mean, rpe_rot_mean


def eval_pose_estimation_dist(args, img_path, save_dir=None, mask_path=None):
    metadata = dataset_metadata.get(args.eval_dataset)
    anno_path = metadata.get("anno_path", None)

    seq_list = args.seq_list
    if seq_list is None:
        if metadata.get("full_seq", False):
            args.full_seq = True
        else:
            seq_list = metadata.get("seq_list", [])
        if args.full_seq:
            seq_list = os.listdir(img_path)
            seq_list = [
                seq for seq in seq_list if os.path.isdir(os.path.join(img_path, seq))
            ]
        seq_list = sorted(seq_list)

    if save_dir is None:
        save_dir = args.output_dir

    distributed_state = PartialState()
    requested_device = str(args.device).lower() if args.device is not None else "cuda"
    if requested_device.startswith("cpu"):
        device = torch.device("cpu")
    else:
        device = distributed_state.device

    model_overrides = {}
    if args.pi3x is not None:
        model_overrides["pi3x"] = args.pi3x.lower() == "true"
    if args.pi3x_metric is not None:
        model_overrides["pi3x_metric"] = args.pi3x_metric.lower() == "true"

    model, base_forward_kwargs = load_pi3_model(
        args.weights,
        config_path=args.pi3_config,
        device=device,
        **model_overrides,
    )

    override_kwargs = {
        "window_size": args.window_size,
        "overlap_size": args.overlap_size,
        "num_iterations": args.num_iterations,
    }
    if args.causal is not None:
        override_kwargs["causal"] = args.causal.lower() == "true"
    if args.sim3 is not None:
        override_kwargs["sim3"] = args.sim3.lower() == "true"
    if args.sim3_mean is not None:
        if args.sim3_mean.lower() == "true":
            override_kwargs["sim3"] = True
            override_kwargs["sim3_scale_mode"] = "trimmed_mean"
    if args.se3 is not None:
        override_kwargs["se3"] = args.se3.lower() == "true"
    if args.pi3x is not None:
        override_kwargs["pi3x"] = args.pi3x.lower() == "true"
    forward_kwargs = merge_forward_kwargs(base_forward_kwargs, override_kwargs)

    with distributed_state.split_between_processes(seq_list) as seqs:
        ate_list = []
        rpe_trans_list = []
        rpe_rot_list = []
        load_img_size = args.size
        assert load_img_size == 512
        error_log_path = f"{save_dir}/_error_log_{distributed_state.process_index}.txt"  # Unique log file per process
        bug = False
        for seq in tqdm(seqs):
            try:
                dir_path = metadata["dir_path_func"](img_path, seq)

                # Handle skip_condition
                skip_condition = metadata.get("skip_condition", None)
                if skip_condition is not None and skip_condition(save_dir, seq):
                    continue

                mask_path_seq_func = metadata.get(
                    "mask_path_seq_func", lambda mask_path, seq: None
                )
                mask_path_seq = mask_path_seq_func(mask_path, seq)

                filelist = [
                    os.path.join(dir_path, name) for name in os.listdir(dir_path)
                ]
                filelist.sort()
                filelist = filelist[:: args.pose_eval_stride]

                views = prepare_input(
                    filelist,
                    [True for _ in filelist],
                    size=load_img_size,
                    crop=not args.no_crop,
                )
                start = time.time()
                outputs, _ = run_pi3_inference_on_views(
                    model,
                    views,
                    forward_kwargs=forward_kwargs,
                    device=device,
                )
                end = time.time()
                fps = len(filelist) / max(end - start, 1e-6)

                (
                    colors,
                    pts3ds_self,
                    pts3ds_other,
                    conf_self,
                    conf_other,
                    cam_dict,
                    pr_poses,
                ) = prepare_output(outputs)

                os.makedirs(f"{save_dir}/{seq}", exist_ok=True)
                save_depth_maps(pts3ds_self, f"{save_dir}/{seq}", conf_self=conf_self)

            except Exception as e:
                if "out of memory" in str(e):
                    # Handle OOM
                    if device.type == "cuda":
                        torch.cuda.empty_cache()  # Clear the CUDA memory
                    with open(error_log_path, "a") as f:
                        f.write(
                            f"OOM error in sequence {seq}, skipping this sequence.\n"
                        )
                    print(f"OOM error in sequence {seq}, skipping...")
                elif "Degenerate covariance rank" in str(
                    e
                ) or "Eigenvalues did not converge" in str(e):
                    # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                    with open(error_log_path, "a") as f:
                        f.write(f"Exception in sequence {seq}: {str(e)}\n")
                    print(f"Traj evaluation error in sequence {seq}, skipping.")
                else:
                    raise e  # Rethrow if it's not an expected exception
    return None, None, None


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    try:
        from dust3r.utils.image import load_images_for_eval as load_images
    except ModuleNotFoundError:
        load_images = None

    try:
        from dust3r.post_process import estimate_focal_knowing_depth
    except ModuleNotFoundError:
        estimate_focal_knowing_depth = None

    try:
        from dust3r.utils.camera import pose_encoding_to_camera
    except ModuleNotFoundError:
        pose_encoding_to_camera = None

    def _load_images_basic(img_paths, size=224, crop=True, verbose=False):
        images = []
        if isinstance(size, (list, tuple)):
            target_w, target_h = int(size[0]), int(size[1])
        else:
            target_w = target_h = int(size)

        for path in img_paths:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Unable to read image at {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]

            if target_w > 0 and target_h > 0:
                if crop:
                    scale = min(target_w / orig_w, target_h / orig_h)
                else:
                    scale = max(target_w / orig_w, target_h / orig_h)
                new_w = max(1, int(round(orig_w * scale)))
                new_h = max(1, int(round(orig_h * scale)))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                if crop:
                    start_x = max(0, (new_w - target_w) // 2)
                    start_y = max(0, (new_h - target_h) // 2)
                    img = img[start_y:start_y + target_h, start_x:start_x + target_w]
                else:
                    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

            if PATCH_ALIGN > 1:
                aligned_h = (img.shape[0] // PATCH_ALIGN) * PATCH_ALIGN
                aligned_w = (img.shape[1] // PATCH_ALIGN) * PATCH_ALIGN

                if aligned_h > 0 and aligned_w > 0:
                    if aligned_h != img.shape[0] or aligned_w != img.shape[1]:
                        start_y = max(0, (img.shape[0] - aligned_h) // 2)
                        start_x = max(0, (img.shape[1] - aligned_w) // 2)
                        img = img[start_y:start_y + aligned_h, start_x:start_x + aligned_w]

            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            tensor = tensor * 2.0 - 1.0  # map to [-1, 1]

            images.append({
                "img": tensor,
                "true_shape": np.int32([orig_h, orig_w])
            })

        if verbose:
            print(f"Loaded {len(images)} images with fallback loader.")
        return images

    if load_images is None:
        load_images = _load_images_basic

    def _estimate_focal_fallback(pts3ds_self, pp, focal_mode="weiszfeld"):
        B, H, W, _ = pts3ds_self.shape
        max_dim = float(max(H, W))
        return torch.full((B,), max_dim, dtype=pts3ds_self.dtype, device=pts3ds_self.device)

    if estimate_focal_knowing_depth is None:
        estimate_focal_knowing_depth = _estimate_focal_fallback

    def _pose_encoding_to_camera_fallback(pose_tensor):
        if torch.is_tensor(pose_tensor):
            data = pose_tensor
        else:
            data = torch.as_tensor(pose_tensor)
        if data.ndim == 2:
            return data
        if data.ndim == 3:
            return data.squeeze(0)
        raise ValueError(f"Unsupported pose encoding shape: {data.shape}")

    if pose_encoding_to_camera is None:
        pose_encoding_to_camera = _pose_encoding_to_camera_fallback

    if args.eval_dataset == "sintel":
        args.full_seq = True
    else:
        args.full_seq = False
    args.no_crop = True

    def prepare_input(
        img_paths,
        img_mask,
        size,
        raymaps=None,
        raymap_mask=None,
        revisit=1,
        update=True,
        crop=True,
    ):
        images = load_images(img_paths, size=size, crop=crop)
        views = []
        if raymaps is None and raymap_mask is None:
            num_views = len(images)

            for i in range(num_views):
                view = {
                    "img": images[i]["img"],
                    "ray_map": torch.full(
                        (
                            images[i]["img"].shape[0],
                            6,
                            images[i]["img"].shape[-2],
                            images[i]["img"].shape[-1],
                        ),
                        torch.nan,
                    ),
                    "true_shape": torch.from_numpy(images[i]["true_shape"]),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(
                        np.eye(4).astype(np.float32)
                    ).unsqueeze(0),
                    "img_mask": torch.tensor(True).unsqueeze(0),
                    "ray_mask": torch.tensor(False).unsqueeze(0),
                    "update": torch.tensor(True).unsqueeze(0),
                    "reset": torch.tensor(False).unsqueeze(0),
                }
                views.append(view)
        else:

            num_views = len(images) + len(raymaps)
            assert len(img_mask) == len(raymap_mask) == num_views
            assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

            j = 0
            k = 0
            for i in range(num_views):
                view = {
                    "img": (
                        images[j]["img"]
                        if img_mask[i]
                        else torch.full_like(images[0]["img"], torch.nan)
                    ),
                    "ray_map": (
                        raymaps[k]
                        if raymap_mask[i]
                        else torch.full_like(raymaps[0], torch.nan)
                    ),
                    "true_shape": (
                        torch.from_numpy(images[j]["true_shape"])
                        if img_mask[i]
                        else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                    ),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(
                        np.eye(4).astype(np.float32)
                    ).unsqueeze(0),
                    "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                    "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                    "update": torch.tensor(img_mask[i]).unsqueeze(0),
                    "reset": torch.tensor(False).unsqueeze(0),
                }
                if img_mask[i]:
                    j += 1
                if raymap_mask[i]:
                    k += 1
                views.append(view)
            assert j == len(images) and k == len(raymaps)

        if revisit > 1:
            # repeat input for 'revisit' times
            new_views = []
            for r in range(revisit):
                for i in range(len(views)):
                    new_view = deepcopy(views[i])
                    new_view["idx"] = r * len(views) + i
                    new_view["instance"] = str(r * len(views) + i)
                    if r > 0:
                        if not update:
                            new_view["update"] = torch.tensor(False).unsqueeze(0)
                    new_views.append(new_view)
            return new_views
        return views

    def prepare_output(outputs, revisit=1):
        valid_length = len(outputs["pred"]) // max(revisit, 1)
        outputs["pred"] = outputs["pred"][-valid_length:]
        outputs["views"] = outputs["views"][-valid_length:]

        pts3ds_self = [entry["pts3d_in_self_view"].cpu() for entry in outputs["pred"]]
        pts3ds_other = []
        conf_self = []
        conf_other = []
        pr_poses = []

        for entry in outputs["pred"]:
            pts3ds_other.append(entry.get("pts3d_in_other_view", entry["pts3d_in_self_view"]).cpu())

            conf_self_entry = entry.get(
                "conf_self",
                torch.ones_like(entry["pts3d_in_self_view"][..., :1]),
            )
            if not torch.is_tensor(conf_self_entry):
                conf_self_entry = torch.as_tensor(conf_self_entry)
            conf_self_entry = conf_self_entry.float()
            conf_self.append(conf_self_entry.cpu())

            conf_other_entry = entry.get("conf", conf_self_entry.squeeze(-1))
            if not torch.is_tensor(conf_other_entry):
                conf_other_entry = torch.as_tensor(conf_other_entry)
            conf_other.append(conf_other_entry.float().cpu())

            cam_pose = entry.get("camera_pose")
            if not torch.is_tensor(cam_pose):
                cam_pose = torch.as_tensor(cam_pose)
            cam_pose = cam_pose.float()
            if cam_pose.ndim == 3 and cam_pose.shape[1:] == (4, 4):
                pose_matrix = cam_pose[0].cpu()
            elif cam_pose.ndim == 2 and cam_pose.shape == (4, 4):
                pose_matrix = cam_pose.cpu()
            else:
                # from dust3r.utils.camera import pose_encoding_to_camera  # lazy import

                converted = pose_encoding_to_camera(cam_pose.clone())
                if torch.is_tensor(converted):
                    pose_matrix = converted.squeeze(0).cpu().float()
                else:
                    pose_matrix = torch.from_numpy(np.asarray(converted)).cpu().float()

            pr_poses.append(pose_matrix.unsqueeze(0))

        pts3ds_self = torch.cat(pts3ds_self, 0)
        pts3ds_other = torch.cat(pts3ds_other, 0)
        pr_poses = torch.cat(pr_poses, 0)

        B, H, W, _ = pts3ds_self.shape
        pp = (
            torch.tensor([W // 2, H // 2], device=pts3ds_self.device)
            .float()
            .repeat(B, 1)
            .reshape(B, 2)
        )
        focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

        colors = [0.5 * (output["rgb"][0] + 1.0) for output in outputs["pred"]]
        cam_dict = {
            "focal": focal.cpu().numpy(),
            "pp": pp.cpu().numpy(),
        }
        return (
            colors,
            pts3ds_self,
            pts3ds_other,
            conf_self,
            conf_other,
            cam_dict,
            pr_poses,
        )

    eval_pose_estimation(args, save_dir=args.output_dir)
