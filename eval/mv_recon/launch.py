import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import time
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
from accelerate import Accelerator
from torch.utils.data._utils.collate import default_collate
import tempfile
from tqdm import tqdm
from typing import Any, Dict, Optional

try:
    from utils.geometry import geotrf as _dust3r_geotrf  # type: ignore
except ModuleNotFoundError:
    _dust3r_geotrf = None


def _fallback_geotrf(transform: Any, pts: Any) -> np.ndarray:
    if torch.is_tensor(transform):
        transform_np = transform.detach().cpu().float().numpy()
    else:
        transform_np = np.asarray(transform, dtype=np.float32)

    pts_np = np.asarray(pts, dtype=np.float32)
    original_shape = pts_np.shape
    pts_flat = pts_np.reshape(-1, 3)
    ones = np.ones((pts_flat.shape[0], 1), dtype=pts_flat.dtype)
    pts_h = np.concatenate([pts_flat, ones], axis=1)
    transformed = (transform_np @ pts_h.T).T[..., :3]
    return transformed.reshape(original_shape)


def geotrf(transform: Any, pts: Any):
    if _dust3r_geotrf is not None:
        return _dust3r_geotrf(transform, pts)
    return _fallback_geotrf(transform, pts)

from eval.pi3_adapter import (
    load_pi3_model,
    merge_forward_kwargs,
    run_pi3_inference_on_views,
)


def _parse_bool_arg(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered not in {"true", "false"}:
        return None
    return lowered == "true"


def wait_for_rank_logs(save_path, num_processes, timeout_s=1800, poll_interval=5):
    if num_processes <= 1:
        return
    deadline = time.time() + timeout_s
    done_paths = [osp.join(save_path, f"logs_{i}.done") for i in range(num_processes)]
    while True:
        missing = [path for path in done_paths if not os.path.exists(path)]
        if not missing:
            return
        if time.time() > deadline:
            raise TimeoutError(
                f"Timed out waiting for rank completion markers: {missing}"
            )
        time.sleep(poll_interval)


def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="ckpt name",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument(
        "--conf_thresh", type=float, default=0.0, help="confidence threshold"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None, help="max frames limit")
    parser.add_argument(
        "--frame_sampling",
        type=str,
        default="max",
        choices=["max", "uniform"],
        help="Frame selection mode when max_frames is set",
    )
    parser.add_argument("--model_update_type", type=str, default="cut3r", help="model update type")
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.0,
        help="voxel size for voxel grid downsampling prior to metrics, 0 means no downsampling",
    )
    parser.add_argument(
        "--icp_downsample_ratio",
        type=float,
        default=0.1,
        help="uniform random downsample ratio applied only during ICP alignment (0 keeps all points)",
    )
    parser.add_argument(
        "--downsample_ratio",
        type=float,
        default=1.0,
        help="Downsample ratio for point clouds (pred and gt) for debugging. 1.0 means no downsampling. If > 1, takes every Nth point.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="save intermediate numpy dumps and point clouds for inspection",
    )
    parser.add_argument(
        "--log_merge_timeout",
        type=int,
        default=1800,
        help="Seconds for the main process to wait for per-rank logs before merging",
    )
    parser.add_argument(
        "--pi3_config",
        type=str,
        default=None,
        help="Path to Pi3 original_config.yaml (optional)",
    )
    parser.add_argument(
        "--pi3_window_size",
        type=int,
        default=None,
        help="Override Pi3 window size",
    )
    parser.add_argument(
        "--pi3_overlap_size",
        type=int,
        default=None,
        help="Override Pi3 overlap size",
    )
    parser.add_argument(
        "--pi3_num_iterations",
        type=int,
        default=None,
        help="Override Pi3 decoding iterations",
    )
    parser.add_argument(
        "--pi3_causal",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Override Pi3 causal flag",
    )
    parser.add_argument(
        "--pi3_sim3",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Override Pi3 Sim(3) merge flag",
    )
    parser.add_argument(
        "--pi3_se3",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Override Pi3 SE(3) merge flag",
    )
    parser.add_argument(
        "--pi3_reset_every",
        type=int,
        default=None,
        help="Override Pi3 reset_every",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset to evaluate on (7scenes, long3d, etc.)",
    )
    return parser


def main(args):
    from eval.mv_recon.data import SevenScenes, NRGBD, Long3D
    from eval.mv_recon.metric_utils import accuracy, completion

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    else:
        raise NotImplementedError
    datasets_all = {}
    
    if args.dataset is None or args.dataset == "7scenes":
        seven_scenes_kf_every = 1 if args.frame_sampling == "uniform" else 2
        datasets_all["7scenes"] = SevenScenes(
            split="test",
            ROOT="data/7scenes",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=seven_scenes_kf_every,
            max_frames=args.max_frames,
            frame_sampling=args.frame_sampling,
        )
    

    normalized_model_name = (args.model_name or "").lower()
    use_pi3 = bool(args.pi3_config) or normalized_model_name.startswith("pi3") or "pi3" in normalized_model_name


    if _dust3r_geotrf is None:
        print(
            "[mv_recon] Running Pi3-only evaluation without dust3r installed (geotrf fallback in use).",
            flush=True,
        )

    # Filter datasets
    if args.dataset is not None:
        datasets_all = {k: v for k, v in datasets_all.items() if k == args.dataset}
        if not datasets_all:
            raise ValueError(f"Dataset {args.dataset} not found in datasets_all")

    print("\n=== number of views for each scene ===")
    for name_data, dataset in datasets_all.items():
        print(f"\n{name_data} dataset:")
        for scene_id in dataset.scene_list:
            if name_data == "NRGBD":
                # NRGBD dataset file structure
                data_path = osp.join(dataset.ROOT, scene_id, "images")
                num_files = len([name for name in os.listdir(data_path) if name.endswith('.png')])
                view_count = len([f"{i}" for i in range(num_files)][::dataset.kf_every])
            elif "long3d" in name_data:
                # Long3D dataset file structure
                data_path = osp.join(dataset.ROOT, scene_id, "images", "scan_images")
                if not os.path.exists(data_path):
                    data_path = osp.join(dataset.ROOT, scene_id, "images")
                
                if os.path.exists(data_path):
                    num_files = len([name for name in os.listdir(data_path) if name.endswith('.jpg') or name.endswith('.png')])
                    view_count = len(range(num_files)[::dataset.kf_every])
                else:
                    view_count = 0
            else:
                # SevenScenes dataset file structure
                data_path = osp.join(dataset.ROOT, scene_id)
                num_files = len([name for name in os.listdir(data_path) if "color" in name])
                view_count = len([f"{i:06d}" for i in range(num_files)][::dataset.kf_every])
            
            # consider max_frames limit
            if dataset.max_frames is not None:
                actual_view_count = min(view_count, dataset.max_frames)
                print(f"  {scene_id}: {actual_view_count} views (original: {view_count}, limit: {dataset.max_frames})")
            else:
                print(f"  {scene_id}: {view_count} views")
    print("================================\n")
    # ====== print end ======

    accelerator = Accelerator()
    device = accelerator.device
    o3d_tensor_available = hasattr(o3d, "t") and hasattr(o3d, "core")
    o3d_cuda_available = (
        o3d_tensor_available
        and hasattr(o3d.core, "cuda")
        and callable(getattr(o3d.core.cuda, "is_available", None))
        and o3d.core.cuda.is_available()
    )
    use_tensor_icp = o3d_tensor_available and device.type == "cuda" and o3d_cuda_available
    if use_tensor_icp:
        local_rank = getattr(accelerator.state, "local_process_index", accelerator.process_index)
        o3d_device = o3d.core.Device(f"CUDA:{local_rank}")
    else:
        o3d_device = o3d.core.Device("CPU:0")

    from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
    from copy import deepcopy

    model = None
    pi3_model = None
    pi3_forward_kwargs: Optional[Dict[str, Any]] = None

    if use_pi3:
        pi3_model_overrides = {}
        pi3_model, pi3_forward_base = load_pi3_model(
            args.weights,
            config_path=args.pi3_config,
            device=device,
            **pi3_model_overrides,
        )
        overrides = {
            "window_size": args.pi3_window_size,
            "overlap_size": args.pi3_overlap_size,
            "num_iterations": args.pi3_num_iterations,
            "reset_every": args.pi3_reset_every,
            "causal": _parse_bool_arg(args.pi3_causal),
            "sim3": _parse_bool_arg(args.pi3_sim3),
            "se3": _parse_bool_arg(args.pi3_se3),
        }
        pi3_forward_kwargs = merge_forward_kwargs(pi3_forward_base, overrides)
    else:
        from dust3r.model import ARCroco3DStereo

        model = ARCroco3DStereo.from_pretrained(args.weights).to(device)
        model.config.model_update_type = args.model_update_type
        model.eval()
    os.makedirs(args.output_dir, exist_ok=True)

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)
    icp_backend_logged = False

    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(args.output_dir, name_data)
            os.makedirs(save_path, exist_ok=True)
            log_file = osp.join(save_path, f"logs_{accelerator.process_index}.txt")
            log_done_flag = osp.join(save_path, f"logs_{accelerator.process_index}.done")
            if os.path.exists(log_done_flag):
                os.remove(log_done_flag)
            with open(log_file, "w") as _:
                pass

            acc_all = 0
            acc_all_med = 0
            comp_all = 0
            comp_all_med = 0
            nc1_all = 0
            nc1_all_med = 0
            nc2_all = 0
            nc2_all_med = 0

            fps_all = []
            time_all = []

            with accelerator.split_between_processes(list(range(len(dataset)))) as idxs:
                for data_idx in tqdm(idxs):
                    batch = default_collate([dataset[data_idx]])
                    ignore_keys = set(
                        [
                            "depthmap",
                            "dataset",
                            "label",
                            "instance",
                            "idx",
                            "true_shape",
                            "rng",
                        ]
                    )
                    if use_pi3:
                        ignore_keys.add("img")
                    for view in batch:
                        for name in view.keys():  # pseudo_focal
                            if name in ignore_keys:
                                continue
                            if isinstance(view[name], tuple) or isinstance(
                                view[name], list
                            ):
                                view[name] = [
                                    x.to(device, non_blocking=True) for x in view[name]
                                ]
                            else:
                                view[name] = view[name].to(device, non_blocking=True)

                    # if model_name == "ours" or model_name == "cut3r":
                    revisit = args.revisit
                    if use_pi3 and revisit > 1:
                        accelerator.print(
                            "Pi3 backend does not support revisit>1; forcing revisit=1"
                        )
                        revisit = 1
                    update = not args.freeze
                    if revisit > 1:
                        # repeat input for 'revisit' times
                        new_views = []
                        for r in range(revisit):
                            for i in range(len(batch)):
                                new_view = deepcopy(batch[i])
                                new_view["idx"] = [
                                    (r * len(batch) + i)
                                    for _ in range(len(batch[i]["idx"]))
                                ]
                                new_view["instance"] = [
                                    str(r * len(batch) + i)
                                    for _ in range(len(batch[i]["instance"]))
                                ]
                                if r > 0:
                                    if not update:
                                        new_view["update"] = torch.zeros_like(
                                            batch[i]["update"]
                                        ).bool()
                                new_views.append(new_view)
                        batch = new_views
                    start = time.time()
                    pi3_sequence = None
                    if use_pi3:
                        assert pi3_model is not None and pi3_forward_kwargs is not None
                        output, pi3_sequence = run_pi3_inference_on_views(
                            pi3_model,
                            batch,
                            forward_kwargs=pi3_forward_kwargs,
                            device=device,
                            output_device=torch.device("cpu"),
                        )
                        preds = output["pred"]
                        batch = output["views"]
                        end = time.time()
                    else:
                        with torch.cuda.amp.autocast(enabled=False):
                            output = model(batch)
                        end = time.time()
                        preds, batch = output.ress, output.views
                    valid_length = len(preds) // revisit
                    preds = preds[-valid_length:]
                    batch = batch[-valid_length:]
                    fps = len(batch) / (end - start)
                    print(
                        f"Finished reconstruction for {name_data} {data_idx+1}/{len(dataset)}, FPS: {fps:.2f}"
                    )
                    # continue
                    fps_all.append(fps)
                    time_all.append(end - start)

                    # Evaluation
                    print(f"Evaluation for {name_data} {data_idx+1}/{len(dataset)}")
                    if "long3d" in name_data:
                        # Skip criterion for Long3D to avoid OOM / device mismatch
                        # Create dummy placeholders
                        gt_pts = [torch.zeros_like(p['pts3d']) for p in preds]
                        pred_pts = [p['pts3d'] for p in preds]
                        gt_factor = torch.ones(len(preds))
                        pr_factor = torch.ones(len(preds))
                        # Create valid mask from depthmap if available, else ones
                        masks = []
                        for i, view in enumerate(batch):
                            if "valid_mask" in view:
                                masks.append(view["valid_mask"])
                            elif "depthmap" in view:
                                masks.append((view["depthmap"] > 0).float())
                            else:
                                H, W = view["img"].shape[-2:]
                                masks.append(torch.ones((1, H, W), device=view["img"].device))
                        masks = torch.stack(masks)
                        
                        monitoring = {
                            "pred_scale": torch.ones(len(preds)),
                            "gt_scale": torch.ones(len(preds)),
                            "pred_shift_z": torch.zeros(len(preds)),
                            "gt_shift_z": torch.zeros(len(preds)),
                        }
                    else:
                        # Ensure batch is on CPU for criterion to match preds
                        for view in batch:
                            for k, v in view.items():
                                if isinstance(v, torch.Tensor):
                                    view[k] = v.cpu()
                        gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
                            criterion.get_all_pts3d_t(batch, preds)
                        )
                    
                    pred_scale, gt_scale, pred_shift_z, gt_shift_z = (
                        monitoring["pred_scale"],
                        monitoring["gt_scale"],
                        monitoring["pred_shift_z"],
                        monitoring["gt_shift_z"],
                    )

                    in_camera1 = None
                    pts_all = []
                    pts_gt_all = []
                    images_all = []
                    masks_all = []
                    conf_all = []

                    for j, view in enumerate(batch):
                        if in_camera1 is None:
                            in_camera1 = view["camera_pose"][0].cpu()

                        image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                        mask = view["valid_mask"].cpu().numpy()[0]

                        if "long3d" in name_data:
                            # Use raw predictions for Long3D
                            pts = preds[j]['pts3d' if j==0 else 'pts3d_in_other_view'].detach().cpu().numpy()[0]
                            # No alignment to dummy GT
                        else:
                            # pts = preds[j]['pts3d' if j==0 else 'pts3d_in_other_view'].detach().cpu().numpy()[0]
                            pts = pred_pts[j].cpu().numpy()[0]
                        
                        conf = preds[j]["conf"].cpu().data.numpy()[0]
                        # mask = mask & (conf > 1.8)

                        pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                        H, W = image.shape[:2]
                        cx = W // 2
                        cy = H // 2
                        l, t = cx - 112, cy - 112
                        r, b = cx + 112, cy + 112
                        image = image[t:b, l:r]
                        mask = mask[t:b, l:r]
                        pts = pts[t:b, l:r]
                        pts_gt = pts_gt[t:b, l:r]

                        #### Align predicted 3D points to the ground truth
                        if "long3d" not in name_data:
                            pts[..., -1] += gt_shift_z.cpu().numpy().item()
                            pts = geotrf(in_camera1, pts)

                            pts_gt[..., -1] += gt_shift_z.cpu().numpy().item()
                            pts_gt = geotrf(in_camera1, pts_gt)
                        else:
                            # For Long3D, just apply camera pose if needed (but it is identity)
                            # Actually, Pi3 outputs in camera1 frame usually.
                            # If we want to align to PCD, we might need ICP later (which is done later).
                            # But we shouldn't align to dummy GT here.
                            pass

                        images_all.append((image[None, ...] + 1.0) / 2.0)
                        pts_all.append(pts[None, ...])
                        pts_gt_all.append(pts_gt[None, ...])
                        masks_all.append(mask[None, ...])
                        conf_all.append(conf[None, ...])

                    images_all = np.concatenate(images_all, axis=0)
                    pts_all = np.concatenate(pts_all, axis=0)
                    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                    masks_all = np.concatenate(masks_all, axis=0)

                    scene_id = view["label"][0].rsplit("/", 1)[0]

                    save_params = {}

                    save_params["images_all"] = images_all
                    save_params["pts_all"] = pts_all
                    save_params["pts_gt_all"] = pts_gt_all
                    save_params["masks_all"] = masks_all
                    if pi3_sequence is not None and pi3_sequence.avg_gate_scale is not None:
                        save_params["pi3_avg_gate_scale"] = pi3_sequence.avg_gate_scale

                    if args.save:
                        np.save(
                            os.path.join(save_path, f"{scene_id.replace('/', '_')}.npy"),
                            save_params,
                        )

                    if "DTU" in name_data:
                        threshold = 100
                    else:
                        threshold = 0.1

                    pts_all_masked = pts_all[masks_all > 0]
                    pts_gt_all_masked = pts_gt_all[masks_all > 0]
                    images_all_masked = images_all[masks_all > 0]

                    if args.downsample_ratio > 1.0:
                        ratio = int(args.downsample_ratio)
                        pts_all_masked = pts_all_masked[::ratio]
                        pts_gt_all_masked = pts_gt_all_masked[::ratio]
                        images_all_masked = images_all_masked[::ratio]

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts_all_masked.reshape(-1, 3))
                    
                    if "long3d" in name_data:
                        # Load GT from PCD file
                        gt_pcd_path = os.path.join(dataset.ROOT, scene_id, "dense_cloud_map.pcd")
                        if os.path.exists(gt_pcd_path):
                            pcd_gt = o3d.io.read_point_cloud(gt_pcd_path)
                            
                            # Apply T_cam_wrt_lidar
                            T_cam_wrt_lidar = np.array([
                                [-0.012577, -0.999915, -0.003397, -0.03],
                                [0.345639, -0.001159, -0.938367, -0.04],
                                [0.938283, -0.012976, 0.345625, -0.03],
                                [0,  0,  0,  1.0]
                            ])
                            pcd_gt.transform(T_cam_wrt_lidar)
                            
                            print(f"Loaded GT PCD from {gt_pcd_path} with {len(pcd_gt.points)} points (applied T_cam_wrt_lidar)")
                            if args.downsample_ratio > 1.0:
                                pcd_gt = pcd_gt.uniform_down_sample(int(args.downsample_ratio))
                                print(f"Downsampled GT PCD to {len(pcd_gt.points)} points (ratio {args.downsample_ratio})")
                        else:
                            print(f"Warning: GT PCD not found at {gt_pcd_path}")
                            pcd_gt = o3d.geometry.PointCloud()
                            pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_all_masked.reshape(-1, 3))
                    else:
                        pcd_gt = o3d.geometry.PointCloud()
                        pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_all_masked.reshape(-1, 3))

                    if "long3d" in name_data and len(pcd.points) > 0 and len(pcd_gt.points) > 0:
                        # Coarse alignment: center and scale
                        pcd_center = pcd.get_center()
                        gt_center = pcd_gt.get_center()
                        
                        # Translate to center
                        pcd.translate(gt_center - pcd_center)
                        
                        # Scale
                        pcd_points = np.asarray(pcd.points)
                        gt_points = np.asarray(pcd_gt.points)
                        
                        dist_pcd = np.linalg.norm(pcd_points - gt_center, axis=1).mean()
                        dist_gt = np.linalg.norm(gt_points - gt_center, axis=1).mean()
                        
                        scale = dist_gt / (dist_pcd + 1e-8)
                        pcd.scale(scale, center=gt_center)
                        
                        print(f"Coarse alignment: translated by {gt_center - pcd_center}, scaled by {scale}")

                        # ==========================================
                        # [New] 1. Prep: Estimate normals and compute FPFH features
                        # ==========================================
                        if args.voxel_size > 0:
                            voxel_size = args.voxel_size
                        else:
                            # Dynamic voxel size based on bounding box
                            max_bound = pcd.get_max_bound()
                            min_bound = pcd.get_min_bound()
                            bbox_diag = np.linalg.norm(max_bound - min_bound)
                            voxel_size = bbox_diag / 50.0
                            print(f"Dynamic voxel_size: {voxel_size:.4f} (bbox diag: {bbox_diag:.4f})")
                        
                        # Downsample for feature computation
                        pcd_down = pcd.voxel_down_sample(voxel_size)
                        pcd_gt_down = pcd_gt.voxel_down_sample(voxel_size)

                        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
                        pcd_gt_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

                        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                            pcd_down,
                            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
                        )
                        pcd_gt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                            pcd_gt_down,
                            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
                        )

                        # ==========================================
                        # [New] 2. Run RANSAC Global Registration
                        # ==========================================
                        print(f"Running Global Registration (RANSAC) for {scene_id}...")
                        distance_threshold = voxel_size * 1.5
                        
                        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                            pcd_down, pcd_gt_down, pcd_fpfh, pcd_gt_fpfh,
                            True, distance_threshold,
                            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                            3, [
                                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
                        )
                        
                        trans_init = result_ransac.transformation
                        print(f"Global Registration Fitness: {result_ransac.fitness}")

                    if args.save:
                        colors = o3d.utility.Vector3dVector(
                            images_all_masked.reshape(-1, 3)
                        )
                        pcd.colors = colors
                        if len(pcd_gt.points) == len(colors):
                            pcd_gt.colors = colors

                    # ====== voxel grid downsampling ======
                    if args.voxel_size > 0:
                        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
                        pcd_gt = pcd_gt.voxel_down_sample(voxel_size=args.voxel_size)
                    # ===========================

                    if args.save:
                        o3d.io.write_point_cloud(
                            os.path.join(
                                save_path, f"{scene_id.replace('/', '_')}-mask.ply"
                            ),
                            pcd,
                        )

                        # Save combined for debug
                        pcd_combined = o3d.geometry.PointCloud()
                        pcd_temp = deepcopy(pcd)
                        # Apply RANSAC transform to visualization
                        pcd_temp.transform(trans_init)
                        pcd_temp.paint_uniform_color([1, 0, 0]) # Red for Pred
                        pcd_gt_temp = deepcopy(pcd_gt)
                        pcd_gt_temp.paint_uniform_color([0, 1, 0]) # Green for GT
                        pcd_combined += pcd_temp
                        pcd_combined += pcd_gt_temp
                        
                        o3d.io.write_point_cloud(
                            os.path.join(
                                save_path, f"{scene_id.replace('/', '_')}-combined.ply"
                            ),
                            pcd_combined,
                        )

                        o3d.io.write_point_cloud(
                            os.path.join(
                                save_path, f"{scene_id.replace('/', '_')}-gt.ply"
                            ),
                            pcd_gt,
                        )

                    if "long3d" not in name_data:
                        trans_init = np.eye(4)

                    pcd_icp = pcd
                    pcd_gt_icp = pcd_gt
                    if 0.0 < args.icp_downsample_ratio < 1.0:
                        rng = np.random.default_rng()
                        num_pts = len(pcd.points)
                        num_pts_gt = len(pcd_gt.points)
                        keep_pred = max(1, int(num_pts * args.icp_downsample_ratio / (args.max_frames / 100)))
                        keep_gt = max(1, int(num_pts_gt * args.icp_downsample_ratio  / (args.max_frames / 100)))
                        idx_pred = rng.choice(num_pts, keep_pred, replace=False)
                        idx_gt = rng.choice(num_pts_gt, keep_gt, replace=False)
                        pcd_icp = pcd.select_by_index(idx_pred)
                        pcd_gt_icp = pcd_gt.select_by_index(idx_gt)

                    transformation = None
                    if use_tensor_icp:
                        src_np = tgt_np = None
                        src_t = tgt_t = None
                        try:
                            src_np = np.asarray(pcd_icp.points, dtype=np.float32)
                            tgt_np = np.asarray(pcd_gt_icp.points, dtype=np.float32)
                            src_t = o3d.t.geometry.PointCloud(device=o3d_device)
                            tgt_t = o3d.t.geometry.PointCloud(device=o3d_device)
                            src_t.point["positions"] = o3d.core.Tensor(
                                src_np, dtype=o3d.core.Dtype.Float32, device=o3d_device
                            )
                            tgt_t.point["positions"] = o3d.core.Tensor(
                                tgt_np, dtype=o3d.core.Dtype.Float32, device=o3d_device
                            )
                            reg_p2p = o3d.t.pipelines.registration.icp(
                                src_t,
                                tgt_t,
                                threshold,
                                init_source_to_target=o3d.core.Tensor(
                                    trans_init,
                                    dtype=o3d.core.Dtype.Float32,
                                    device=o3d_device,
                                ),
                                estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
                            )
                            transformation = reg_p2p.transformation.cpu().numpy()
                            if not icp_backend_logged:
                                accelerator.print(
                                    f"[Rank {accelerator.process_index}] Using tensor ICP on {o3d_device} with {len(src_np)} src pts / {len(tgt_np)} tgt pts"
                                )
                                icp_backend_logged = True
                        except Exception as exc:
                            accelerator.print(
                                f"[Rank {accelerator.process_index}] Tensor ICP failed ({exc}), falling back to CPU ICP."
                            )
                        finally:
                            del src_t, tgt_t, src_np, tgt_np

                    if transformation is None:
                        reg_p2p = o3d.pipelines.registration.registration_icp(
                            pcd_icp,
                            pcd_gt_icp,
                            threshold,
                            trans_init,
                            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        )
                        transformation = reg_p2p.transformation

                    pcd = pcd.transform(transformation)
                    pcd.estimate_normals()
                    pcd_gt.estimate_normals()

                    gt_normal = np.asarray(pcd_gt.normals)
                    pred_normal = np.asarray(pcd.normals)

                    acc, acc_med, nc1, nc1_med = accuracy(
                        pcd_gt.points, pcd.points, gt_normal, pred_normal
                    )
                    comp, comp_med, nc2, nc2_med = completion(
                        pcd_gt.points, pcd.points, gt_normal, pred_normal
                    )
                    cd = 0.5 * (acc + comp)
                    cd_med = 0.5 * (acc_med + comp_med)
                    nc = 0.5 * (nc1 + nc2)
                    nc_med = 0.5 * (nc1_med + nc2_med)
                    print(
                        f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, CD: {cd}, NC1: {nc1}, NC2: {nc2}, NC: {nc} - Acc_med: {acc_med}, Compc_med: {comp_med}, CD_med: {cd_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}, NC_med: {nc_med}"
                    )
                    print(
                        f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, CD: {cd}, NC1: {nc1}, NC2: {nc2}, NC: {nc} - Acc_med: {acc_med}, Compc_med: {comp_med}, CD_med: {cd_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}, NC_med: {nc_med}",
                        file=open(log_file, "a"),
                    )

                    acc_all += acc
                    comp_all += comp
                    nc1_all += nc1
                    nc2_all += nc2

                    acc_all_med += acc_med
                    comp_all_med += comp_med
                    nc1_all_med += nc1_med
                    nc2_all_med += nc2_med

                    # release cuda memory
                    torch.cuda.empty_cache()

            with open(log_done_flag, "w") as f_done:
                f_done.write(str(time.time()))

            # Get depth from pcd and run TSDFusion
            if accelerator.is_main_process:
                try:
                    wait_for_rank_logs(
                        save_path,
                        accelerator.num_processes,
                        timeout_s=args.log_merge_timeout,
                    )
                except TimeoutError as exc:
                    print(
                        (
                            f"[Main process] Warning: {exc}. Proceeding with any available logs.\n"
                            "If some ranks finished later, rerun log aggregation manually."
                        ),
                        flush=True,
                    )
                to_write = ""
                # Copy the error log from each process to the main error log
                for i in range(8):
                    if not os.path.exists(osp.join(save_path, f"logs_{i}.txt")):
                        break
                    with open(osp.join(save_path, f"logs_{i}.txt"), "r") as f_sub:
                        to_write += f_sub.read()

                with open(osp.join(save_path, f"logs_all.txt"), "w") as f:
                    log_data = to_write
                    metrics = defaultdict(list)
                    for line in log_data.strip().split("\n"):
                        match = regex.match(line)
                        if match:
                            data = match.groupdict()
                            # Exclude 'scene_id' from metrics as it's an identifier
                            for key, value in data.items():
                                if key != "scene_id" and value is not None:
                                    metrics[key].append(float(value))
                            if data.get("nc") is None:
                                metrics["nc"].append(
                                    (float(data["nc1"]) + float(data["nc2"])) / 2
                                )
                            if data.get("nc_med") is None:
                                metrics["nc_med"].append(
                                    (float(data["nc1_med"]) + float(data["nc2_med"])) / 2
                                )
                    mean_metrics = {
                        metric: sum(values) / len(values)
                        for metric, values in metrics.items()
                    }

                    c_name = "mean"
                    print_str = f"{c_name.ljust(20)}: "
                    for m_name in mean_metrics:
                        print_num = np.mean(mean_metrics[m_name])
                        print_str = print_str + f"{m_name}: {print_num:.3f} | "
                    print_str = print_str + "\n"
                    f.write(to_write + print_str)

                for i in range(accelerator.num_processes):
                    done_path = osp.join(save_path, f"logs_{i}.done")
                    if os.path.exists(done_path):
                        os.remove(done_path)


from collections import defaultdict
import re

pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    (?:CD:\s*(?P<cd>[^,]+),\s*)?
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)(?:,\s*NC:\s*(?P<nc>[^,]+))?\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    (?:CD_med:\s*(?P<cd_med>[^,]+),\s*)?
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)(?:,\s*NC_med:\s*(?P<nc_med>[^,]+))?
"""

regex = re.compile(pattern, re.VERBOSE)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
