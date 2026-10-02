import os
import logging
from typing import List, Optional, Tuple
from os.path import join
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch
import numpy as np

from torch import nn
from PIL import Image

from .models.helper import get_aggregator, get_backbone

logger = logging.getLogger("loger.utils.loop.detector")


class SALADLoopModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = get_backbone(
            "dinov2_vitb14",
            {
                "num_trainable_blocks": 4,
                "return_token": True,
                "norm_layer": True,
            },
        )
        self.aggregator = get_aggregator(
            "SALAD",
            {
                "num_channels": 768,
                "num_clusters": 64,
                "cluster_dim": 128,
                "token_dim": 256,
            },
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.aggregator(self.backbone(inputs))


class LoopDetector:
    """Release-local wrapper of the original SALAD loop detector."""

    def __init__(
        self,
        image_list: List[str],
        gt_traj_path: str,
        result_dir: str,
        ckpt_path: str,
        image_size: Tuple[int, int] = (336, 336),
        batch_size: int = 64,
        similarity_threshold: float = 0.7,
        top_k: int = 5,
        use_nms: bool = True,
        nms_threshold: int = 25,
        min_frame_gap: int = 10,
        save_pair_vis: bool = False,
    ):
        self.image_list = image_list
        self.gt_traj_path = gt_traj_path
        self.result_dir = result_dir

        self.ckpt_path = ckpt_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.use_nms = use_nms
        self.nms_threshold = nms_threshold
        self.min_frame_gap = min_frame_gap
        self.save_pair_vis = bool(save_pair_vis)

        self.model = None
        self.device = None
        self.descriptors = None             # cumulative SALAD features, (N_cached, D)
        self.loop_closures = None

        # Streaming bookkeeping: frame_idx -> row in self.descriptors.
        self._frame_to_row: dict[int, int] = {}
        self._queried_frames: set[int] = set()
        self._streaming_pairs: set[tuple[int, int]] = set()
        self._streaming_loops: list = []

        # Streaming visualization: running counter across all visualize_new_pairs calls.
        self._vis_pair_counter: int = 0

    def _input_transform(self, image_size: Optional[Tuple[int, int]] = None):
        import torchvision.transforms as transforms

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if image_size:
            return transforms.Compose(
                [
                    transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def load_model(self):
        if not self.ckpt_path:
            raise ValueError("LoopDetector requires a SALAD checkpoint path.")

        self.model = SALADLoopModel()
        state_dict = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state_dict)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

    def extract_window_descriptors(
        self,
        frame_indices,
        image_paths,
    ) -> None:
        """Streaming: extract SALAD features for frames not yet seen.

        The extracted rows are appended to ``self.descriptors`` and indexed by
        absolute frame index in ``self._frame_to_row`` so that overlapping
        windows do not recompute already-seen frames.
        """
        if self.model is None or self.device is None:
            self.load_model()

        frame_indices = [int(fi) for fi in frame_indices]
        image_paths = list(image_paths)
        assert len(frame_indices) == len(image_paths), (
            "frame_indices and image_paths must align element-wise"
        )

        missing = [(fi, p) for fi, p in zip(frame_indices, image_paths)
                   if fi not in self._frame_to_row]
        if not missing:
            return

        transform = self._input_transform(self.image_size)
        height, width = self.image_size
        device = self.device
        amp_enabled = device.type == "cuda"

        new_rows: list[torch.Tensor] = []
        for b0 in range(0, len(missing), self.batch_size):
            batch = missing[b0:b0 + self.batch_size]

            def _load_one(item):
                _, path = item
                try:
                    with Image.open(path) as im:
                        image = im.convert("RGB")
                    return transform(image)
                except Exception as exc:
                    logger.warning("failed to process loop image %s: %s", path, exc)
                    return torch.zeros(3, height, width)

            max_workers = min(8, len(batch))
            if max_workers <= 1:
                batch_imgs = [_load_one(it) for it in batch]
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    batch_imgs = list(ex.map(_load_one, batch))

            batch_tensor = torch.stack(batch_imgs).to(device)
            with torch.no_grad(), torch.autocast(
                enabled=amp_enabled, device_type=device.type, dtype=torch.float16,
            ):
                descs = self.model(batch_tensor).cpu()
            new_rows.append(descs)

        new_mat = torch.cat(new_rows, dim=0)    # [N, 8448]

        if self.descriptors is None:
            self.descriptors, base_row = new_mat, 0
        else:
            base_row = self.descriptors.shape[0]
            self.descriptors = torch.cat([self.descriptors, new_mat], dim=0)
        for k, (fi, _) in enumerate(missing):
            self._frame_to_row[int(fi)] = base_row + k

    def detect_window_loops(self, query_indices) -> list:
        """Streaming: find new loop pairs for the given query frames.

        For each query frame, candidates are all cached frames (excluding the
        query itself, and excluding candidates within ``min_frame_gap``). Query
        frames already processed on a previous call are skipped. Newly-found
        pairs are accumulated in ``self._streaming_loops`` and returned.

        Pairs are ``(fi, fj, sim)`` with ``fi > fj``, matching ``self.loop_closures``.
        """
        if self.descriptors is None or not self._frame_to_row:
            return []

        q_idx = sorted(
            {int(fi) for fi in query_indices if int(fi) in self._frame_to_row}
            - self._queried_frames
        )
        if not q_idx:
            return []

        cand_idx = sorted(self._frame_to_row.keys())
        if len(cand_idx) <= 1:
            self._queried_frames.update(q_idx)
            return []

        desc_np = self.descriptors.numpy()
        q_rows = np.asarray([self._frame_to_row[fi] for fi in q_idx], dtype=np.int64)
        c_rows = np.asarray([self._frame_to_row[fi] for fi in cand_idx], dtype=np.int64)

        q_mat = desc_np[q_rows]           # (Q, D)
        c_mat = desc_np[c_rows]           # (C, D)
        sim = (q_mat @ c_mat.T).astype(np.float32, copy=False)  # [shape of query, shape of candidate]

        q_arr = np.asarray(q_idx, dtype=np.int64)[:, None]
        c_arr = np.asarray(cand_idx, dtype=np.int64)[None, :]
        gap = int(self.min_frame_gap)
        # Only keep candidates that lie strictly before the query by more than the gap.
        sim[(q_arr - c_arr) <= gap] = -1.0

        k = min(self.top_k, sim.shape[1])
        if k <= 0:
            self._queried_frames.update(q_idx)
            return []

        idxs = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        sims = np.take_along_axis(sim, idxs, axis=1)
        order = np.argsort(-sims, axis=1)
        idxs = np.take_along_axis(idxs, order, axis=1)
        sims = np.take_along_axis(sims, order, axis=1)

        thr = float(self.similarity_threshold)
        new: list = []
        for qp, qi in enumerate(q_idx):
            for kk in range(k):
                s = float(sims[qp, kk])
                if s <= thr:
                    continue
                cj = int(cand_idx[int(idxs[qp, kk])])
                fi, fj = (qi, cj) if qi > cj else (cj, qi)
                key = (fi, fj)
                if key in self._streaming_pairs:
                    continue
                self._streaming_pairs.add(key)
                entry = (fi, fj, s)
                self._streaming_loops.append(entry)
                new.append(entry)

        self._queried_frames.update(q_idx)
        return new

    def finalize_streaming_loops(self) -> None:
        """Populate ``self.loop_closures`` from streaming detections (sorted by sim)."""
        loops = sorted(self._streaming_loops, key=lambda item: item[2], reverse=True)
        if self.use_nms and self.nms_threshold > 0 and loops:
            loops = apply_loop_nms_filter(loops, self.nms_threshold)
        self.loop_closures = loops

    @staticmethod
    def _load_positions_any(gt_traj_path: Optional[str]) -> Optional[np.ndarray]:
        """Load (N, 3) positions from KITTI / TUM-VBR / plain-XYZ trajectory files."""
        if not gt_traj_path or not isinstance(gt_traj_path, str):
            return None
        if not os.path.isfile(gt_traj_path):
            logger.warning("gt_traj_path not found: %s", gt_traj_path)
            return None

        first = None
        with open(gt_traj_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                first = s
                break
        if first is None:
            logger.warning("gt_traj_path is empty: %s", gt_traj_path)
            return None

        try:
            parts = [float(x) for x in first.split()]
        except Exception:
            logger.warning("cannot parse gt_traj_path as floats: %s", gt_traj_path)
            return None

        try:
            data = np.loadtxt(gt_traj_path, comments="#")
        except Exception as exc:
            logger.warning("failed to load gt_traj_path via np.loadtxt (%s): %s", gt_traj_path, exc)
            return None

        if data.ndim == 1:
            data = data[None, :]

        if data.shape[1] >= 12 and len(parts) >= 12:
            mats = data[:, :12].reshape(-1, 3, 4)
            return mats[:, :, 3].astype(np.float32, copy=False)
        if data.shape[1] >= 8 and len(parts) >= 8:
            return data[:, 1:4].astype(np.float32, copy=False)
        if data.shape[1] >= 3:
            return data[:, 0:3].astype(np.float32, copy=False)

        logger.warning("unrecognized gt_traj_path format (shape=%s): %s", data.shape, gt_traj_path)
        return None

    def _frame_to_gt_rows(self) -> Optional[np.ndarray]:
        """Map frame index -> GT row via nearest timestamp, or None for identity.

        Needed when a TUM-format GT (``timestamp tx ty tz qx qy qz qw``) is
        sampled differently from the images, e.g. DROID-W (~20 Hz images,
        ~10 Hz GT starting later). Frame timestamps come from the image
        filename stems. Frames without a GT pose within 2x the median GT
        spacing map to -1. Returns None (identity) when the GT is not
        timestamped, the stems are not timestamps, or counts already match.
        """
        path, images = self.gt_traj_path, getattr(self, "image_list", None)
        if not path or not images or not os.path.isfile(path):
            return None
        try:
            data = np.loadtxt(path, comments="#")
        except Exception:
            return None
        if data.ndim != 2 or data.shape[1] != 8 or len(data) == len(images):
            return None
        try:
            frame_ts = np.array([float(os.path.splitext(os.path.basename(str(p)))[0])
                                 for p in images])
        except ValueError:
            return None
        gt_ts = data[:, 0]
        order = np.argsort(gt_ts)
        gt_ts = gt_ts[order]
        in_range = (frame_ts >= gt_ts[0] - 1.0) & (frame_ts <= gt_ts[-1] + 1.0)
        if len(gt_ts) < 2 or in_range.mean() < 0.5:
            return None   # stems are not on the GT clock (e.g. 000000.png)
        k = np.clip(np.searchsorted(gt_ts, frame_ts), 1, len(gt_ts) - 1)
        k -= (frame_ts - gt_ts[k - 1]) < (gt_ts[k] - frame_ts)
        max_dt = 2.0 * float(np.median(np.diff(gt_ts)))
        rows = order[k].astype(np.int64)
        rows[np.abs(gt_ts[k] - frame_ts) > max_dt] = -1
        return rows

    @staticmethod
    def _best_2d_axes(positions: np.ndarray) -> Tuple[int, int]:
        var = positions.var(axis=0)
        axes = np.argsort(var)[::-1][:2]
        return int(axes[0]), int(axes[1])

    @staticmethod
    def _safe_load_rgb(path: str) -> Optional[np.ndarray]:
        try:
            return np.asarray(Image.open(path).convert("RGB"))
        except Exception:
            try:
                bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
            except Exception:
                return None

    def _save_traj_plot(self, loops, out_path: str,
                        title: str = "Loop detection visualization",
                        rejected=None, rejected_label: str = "") -> bool:
        """Render the 2-panel trajectory + loop-edges figure to ``out_path``.

        ``rejected``: optional frame pairs (loop edges) dropped by the PGO loop
        consistency check; drawn in red and removed from the kept (blue) pairs.

        Returns True on success, False if matplotlib or GT traj are unavailable.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            from matplotlib.collections import LineCollection
        except Exception as exc:
            logger.warning("matplotlib not available, skipping trajectory plot: %s", exc)
            return False

        positions = self._load_positions_any(self.gt_traj_path)
        if positions is None:
            logger.warning("no GT trajectory available; skipping trajectory plot.")
            return False

        n_pos = positions.shape[0]
        a0, a1 = self._best_2d_axes(positions)
        px = positions[:, a0]
        py = positions[:, a1]

        pairs = [(int(t[0]), int(t[1]), float(t[2])) for t in loops if len(t) >= 3]
        rej_set = {tuple(sorted((int(a), int(b)))) for a, b in (rejected or [])}
        rej = [(i, j, 0.0) for i, j in sorted(rej_set)]
        pairs = [p for p in pairs if tuple(sorted(p[:2])) not in rej_set]
        # Loop pairs are frame indices; map them onto GT rows. Identity unless
        # the GT is timestamped and sampled differently from the images.
        frame_to_row = self._frame_to_gt_rows()

        def _to_rows(prs):
            if frame_to_row is not None:
                n_fr = len(frame_to_row)
                prs = [(int(frame_to_row[i]), int(frame_to_row[j]), sim)
                       for i, j, sim in prs
                       if i < n_fr and j < n_fr
                       and frame_to_row[i] >= 0 and frame_to_row[j] >= 0]
            return [(i, j, sim) for i, j, sim in prs if i < n_pos and j < n_pos]
        pairs, rej = _to_rows(pairs), _to_rows(rej)

        points = np.stack([px, py], axis=1)
        segments = np.stack([points[:-1], points[1:]], axis=1) if len(points) >= 2 else None

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        ax = axes[0]
        if segments is not None:
            t_param = np.linspace(0, 1, len(segments))
            ax.add_collection(LineCollection(segments, colors=cm.plasma(t_param),
                                             linewidths=1.2, zorder=2))
            sm = plt.cm.ScalarMappable(cmap="plasma",
                                       norm=plt.Normalize(0, max(n_pos - 1, 1)))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("Frame index", fontsize=9)
        else:
            ax.plot(px, py, "-", color="#666666", linewidth=1.0)
        if pairs:
            a_frames = sorted({i for i, _, _ in pairs})
            ax.scatter([px[k] for k in a_frames], [py[k] for k in a_frames],
                       c="#e63946", s=10, zorder=4,
                       label=f"LC side A ({len(a_frames)} fr)")
            ax.legend(fontsize=8, loc="best")
        ax.set_title("Trajectory coloured by time", fontsize=11)

        ax = axes[1]
        if segments is not None:
            ax.add_collection(LineCollection(segments, colors=["#cccccc"],
                                             linewidths=1.0, zorder=2))
        else:
            ax.plot(px, py, "-", color="#cccccc", linewidth=1.0)
        if pairs:
            max_lines = 5000
            draw_pairs = pairs
            if len(draw_pairs) > max_lines:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(draw_pairs), max_lines, replace=False)
                draw_pairs = [draw_pairs[k] for k in idx]
            for i, j, _ in draw_pairs:
                ax.plot([px[i], px[j]], [py[i], py[j]],
                        color="#457b9d", alpha=0.12, linewidth=0.4, zorder=1)
            b_frames = sorted({j for _, j, _ in pairs})
            ax.scatter([px[k] for k in b_frames], [py[k] for k in b_frames],
                       c="#1d7aff", s=10, zorder=3,
                       label=f"LC side B ({len(b_frames)} fr)")
        if rej:
            ax.add_collection(LineCollection(
                [[(px[i], py[i]), (px[j], py[j])] for i, j, _ in rej],
                colors="#e63946", alpha=0.7, linewidths=0.9, zorder=4,
                label=f"rejected by PGO check ({rejected_label or f'{len(rej)} edges'})"))
        if pairs or rej:
            ax.legend(fontsize=8, loc="best")
        ax.set_title(f"Loop closures ({len(pairs)} pairs kept"
                     + (f", {len(rej)} loop edges rejected by PGO check" if rej else "") + ")",
                     fontsize=11)

        if pairs:
            top_pairs = sorted(pairs, key=lambda t: t[2], reverse=True)[:50]
            dx = 0.01 * (float(px.max() - px.min()) + 1e-6)
            dy = 0.01 * (float(py.max() - py.min()) + 1e-6)

            def _annot(ax_, x, y, text, color, ha):
                ax_.text(x, y, text, fontsize=7, color=color, ha=ha, va="center", zorder=10,
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.65))

            for order, (i, j, _) in enumerate(top_pairs, start=1):
                _annot(axes[0], float(px[i] - dx), float(py[i] + dy), str(order), "#e63946", "right")
                _annot(axes[1], float(px[j] + dx), float(py[j] - dy), str(order), "#1d7aff", "left")

        axis_label = {0: "X", 1: "Y", 2: "Z"}
        for ax in axes:
            if n_pos > 0:
                ax.scatter(px[0], py[0], c="green", s=80, zorder=6, marker="^")
                ax.scatter(px[-1], py[-1], c="darkorange", s=80, zorder=6, marker="s")
                ax.annotate("start", (px[0], py[0]), textcoords="offset points",
                            xytext=(5, 5), fontsize=8, color="green")
                ax.annotate("end", (px[-1], py[-1]), textcoords="offset points",
                            xytext=(5, 5), fontsize=8, color="darkorange")
            ax.set_aspect("equal")
            ax.autoscale()
            ax.set_xlabel(f"{axis_label.get(a0, str(a0))} (m)")
            ax.set_ylabel(f"{axis_label.get(a1, str(a1))} (m)")
            ax.grid(True, linestyle="--", alpha=0.4)

        fig.suptitle(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        return True

    def _save_pair_image(self, rank: int, fi: int, fj: int, sim: float,
                         pairs_dir: str) -> bool:
        """Render and save a single side-by-side loop pair image."""
        if fi < 0 or fj < 0 or fi >= len(self.image_list) or fj >= len(self.image_list):
            return False
        im_i = self._safe_load_rgb(self.image_list[fi])
        im_j = self._safe_load_rgb(self.image_list[fj])
        if im_i is None or im_j is None:
            logger.warning("failed to load pair images (%d, %d)", fi, fj)
            return False

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(im_i); axes[0].set_title(f"frame {fi}", fontsize=10); axes[0].axis("off")
        axes[1].imshow(im_j); axes[1].set_title(f"frame {fj}", fontsize=10); axes[1].axis("off")
        fig.suptitle(f"Loop pair #{rank:04d}: ({fi}, {fj})  sim={sim:.4f}", fontsize=12)
        plt.tight_layout()
        plt.savefig(join(pairs_dir, f"pair_{rank:04d}_{fi}_{fj}_sim{sim:.4f}.png"), dpi=150)
        plt.close(fig)
        return True

    def visualize_new_pairs(self, new_pairs: list) -> None:
        """Save side-by-side images for newly detected loop pairs (streaming).

        Pair format: ``(fi, fj, sim)`` — same as ``detect_window_loops`` output. Files are numbered with a running
        counter so successive calls produce consistently ranked filenames:
          ``<result_dir>/loop_vis/pairs/pair_XXXX_i_j_simS.png``

        The cumulative loop_closures.txt and trajectory plot are NOT written
        here — call ``save_streaming_visualization()`` once at end of stream.
        """
        if not new_pairs:
            return
        if not self.save_pair_vis:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
        except Exception as exc:
            logger.warning("matplotlib not available, skipping loop visualization: %s", exc)
            return

        pairs_dir = join(self.result_dir, "loop_vis", "pairs")
        os.makedirs(pairs_dir, exist_ok=True)

        # Highest-sim new pairs get the lowest running indices.
        sorted_new = sorted(new_pairs, key=lambda t: t[2], reverse=True)
        saved = 0
        for fi, fj, sim in sorted_new:
            rank = self._vis_pair_counter
            if self._save_pair_image(rank, int(fi), int(fj), float(sim), pairs_dir):
                self._vis_pair_counter += 1
                saved += 1

        logger.info(
            "Streaming LC vis: +%d new pair imgs (running total %d) -> %s",
            saved, self._vis_pair_counter, pairs_dir,
        )

    def save_streaming_visualization(self, plot_loops=None, rejected_pairs=None,
                                     n_rejected_bridges: int = 0, n_bridges: int = 0) -> None:
        """End-of-stream finalizer: write ``loop_vis/loop_traj.png``.

        ``plot_loops``, when given, is the subset drawn in ``loop_traj.png``
        (e.g. the pairs that survived the chord/arc filter); otherwise all
        detected pairs are drawn. ``rejected_pairs`` (frame pairs of the loop
        edges dropped by the PGO consistency check) are drawn in red.

        Pair PNGs are saved incrementally by ``visualize_new_pairs`` and are
        not (re)written here.
        """
        if self.loop_closures is None:
            self.finalize_streaming_loops()

        loops = self.loop_closures or []
        out_dir = join(self.result_dir, "loop_vis")
        os.makedirs(out_dir, exist_ok=True)
        traj_path = join(out_dir, "loop_traj.png")
        if plot_loops is None:
            plot_loops = loops
            title = f"Loop detection (streaming, {len(loops)} pairs)"
        else:
            plot_loops = list(plot_loops)
            title = (f"Loop detection (streaming, {len(plot_loops)} filtered "
                     f"of {len(loops)} detected pairs)")
        rej_label = ""
        if rejected_pairs:
            rej_label = f"{n_rejected_bridges}/{n_bridges} bridges, {len(rejected_pairs)} edges"
            title += f"; PGO check rejected {rej_label}"
        self._save_traj_plot(plot_loops, traj_path, title=title,
                             rejected=rejected_pairs, rejected_label=rej_label)

        logger.info("Streaming finalize: %d loop pairs; traj -> %s", len(loops), traj_path)

def apply_loop_nms_filter(loop_closures, nms_threshold: int):
    if not loop_closures or nms_threshold <= 0:
        return loop_closures

    sorted_loops = sorted(loop_closures, key=lambda item: item[2], reverse=True)
    filtered = []
    max_frame = max(max(idx1, idx2) for idx1, idx2, _ in loop_closures)
    suppressed = set()
    for idx1, idx2, sim in sorted_loops:
        if idx1 in suppressed or idx2 in suppressed:
            continue

        suppress_range = set()
        filtered.append((idx1, idx2, sim))
        sidx1 = max(0, idx1 - nms_threshold)
        eidx1 = min(idx1 + nms_threshold + 1, idx2)
        suppress_range.update(range(sidx1, eidx1))
        sidx2 = max(idx1 + 1, idx2 - nms_threshold)
        eidx2 = min(idx2 + nms_threshold + 1, max_frame + 1)
        suppress_range.update(range(sidx2, eidx2))
        suppressed.update(suppress_range)

    return filtered
