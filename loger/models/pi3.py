import time
import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy
from typing import Optional, Union, List
from collections import deque
import contextlib
from tqdm import tqdm

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points, robust_scale_estimation
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d, ContextOnlyTransformerDecoder
from .layers.camera_head import CameraHead
from .layers.conv_head import ConvHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from huggingface_hub import PyTorchModelHubMixin
from loger.models.ttt import FastWeightGluMLPMultihead, TTTOperator
from ..utils.loop import LoopDetector
import os as _os


class Pi3(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            ttt_insert_after: Union[int, List[int]] = None,
            ttt_head_dim: int = 512,
            ttt_inter_multi: int = 2,
            num_muon_update_steps: int = 5,
            use_momentum: bool = False,
            ttt_update_steps: int = 1,
            conf: bool = True,
            attn_insert_after: Union[int, List[int], None] = None,
            ttt_pre_norm: bool = False,
            pi3x: bool = False,
            pi3x_metric: bool = True,
            num_pe_tokens: int = 3,
        ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        def _normalize_insert_positions(value: Union[int, List[int], None]) -> List[int]:
            if isinstance(value, (int, float)):
                return [int(value)]
            if isinstance(value, (list, tuple)):
                return [int(x) for x in value]
            return []

        parsed_ttt_insert_after = _normalize_insert_positions(ttt_insert_after)
        parsed_attn_insert_after = _normalize_insert_positions(attn_insert_after)

        if not parsed_attn_insert_after:
            parsed_attn_insert_after = parsed_ttt_insert_after.copy()

        self.ttt_insert_after = parsed_ttt_insert_after
        self.attn_insert_after = parsed_attn_insert_after
        self.detach_swa_history = False
        self.initialize_swa_from_global = True
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        self.num_muon_update_steps = int(num_muon_update_steps)
        # Three learned tokens mark previous overlap, interior, and next overlap.
        # Set num_pe_tokens=0 for vanilla Pi3 checkpoints.
        self.num_pe_tokens = int(num_pe_tokens)
        self.use_momentum = use_momentum
        self.ttt_update_steps = int(ttt_update_steps)
        self.use_conf = bool(conf)
        self.ttt_pre_norm = ttt_pre_norm
        self.pi3x = pi3x
        self.pi3x_metric = pi3x_metric
        del self.encoder.mask_token

        # ----------------------
        # Positional encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features        # 1024
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        if self.num_pe_tokens > 0:
            for i in range(self.num_pe_tokens):
                pe_token = nn.Parameter(torch.randn(1, 1, 1, self.dec_embed_dim))
                nn.init.normal_(pe_token, std=1e-6)
                self.register_parameter(f'pe_token_{i}', pe_token)
            self.patch_start_idx += 1

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim,    # 2 * dino feats dimension
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        if self.pi3x:
            self.point_head = ConvHead(
                num_features=4, 
                dim_in=1024,
                projects=nn.Identity(),
                dim_out=[2, 1], 
                dim_proj=1024,
                dim_upsample=[256, 128, 64],
                dim_times_res_block_hidden=2,
                num_res_blocks=2,
                res_block_norm='group_norm',
                last_res_blocks=0,
                last_conv_channels=32,
                last_conv_size=1,
                using_uv=True
            )
        else:
            self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #     Conf Decoder
        # ----------------------
        if self.use_conf:
            self.conf_decoder = deepcopy(self.point_decoder)
            self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)
        else:
            self.conf_decoder = None
            self.conf_head = None

        # ----------------------
        #     Metric Decoder
        # ----------------------
        if self.pi3x and self.pi3x_metric:
            self.metric_token = nn.Parameter(torch.randn(1, 1, 2*self.dec_embed_dim))
            self.metric_decoder = ContextOnlyTransformerDecoder(
                in_dim=2*self.dec_embed_dim, 
                dec_embed_dim=512,
                dec_num_heads=8,                # 8
                out_dim=512,
                rope=self.rope,
            )
            self.metric_head = nn.Linear(512, 1)
            nn.init.normal_(self.metric_token, std=1e-6)
        else:
            self.metric_token = None
            self.metric_decoder = None
            self.metric_head = None

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512, output_quat=False)

        # ImageNet normalisation constants.
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        # ----------------------
        #            TTT
        # ----------------------

        self.ttt_layers = None
        self.ttt_gate_projs = None
        self.ttt_op_order = None

        self.ttt_layers = nn.ModuleList([
            FastWeightGluMLPMultihead(
                dim=dec_embed_dim,
                head_dim=ttt_head_dim,
                inter_multi=ttt_inter_multi,
                bias=False,
                base_lr=0.01,
                muon_update_steps=self.num_muon_update_steps,
                use_momentum=self.use_momentum,
                ttt_update_steps=self.ttt_update_steps,
                ttt_pre_norm=self.ttt_pre_norm,
            )
            for _ in self.ttt_insert_after
        ])
        self.ttt_gate_projs = nn.ModuleList([
            nn.Linear(dec_embed_dim, 1)
            for _ in self.ttt_insert_after
        ])

        for gate_proj in self.ttt_gate_projs:
            torch.nn.init.zeros_(gate_proj.weight)
            if gate_proj.bias is not None:
                torch.nn.init.zeros_(gate_proj.bias)

        self.ttt_op_order = [
            TTTOperator(start=0, end=None, update=False, apply=True),
            TTTOperator(start=0, end=None, update=True, apply=False),
        ]

        # ----------------------
        #   Attention Adapters
        # ----------------------
        self.swa_layers = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=ttt_inter_multi,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope,
            )
            for _ in self.attn_insert_after
        ])
        self.swa_gate_projs = nn.ModuleList([
            nn.Linear(dec_embed_dim, 1)
            for _ in self.attn_insert_after
        ])

        for gate_proj in self.swa_gate_projs:
            torch.nn.init.zeros_(gate_proj.weight)
            if gate_proj.bias is not None:
                torch.nn.init.zeros_(gate_proj.bias)
    
    def _initialize_ttt_layers_from_global(
        self,
        layers: Optional[nn.ModuleList],
        kind: str,
        insert_after: Optional[List[int]] = None,
    ) -> None:
        """Initialise adapter layers from decoder global-attention weights."""
        if layers is None or len(layers) == 0:
            print(f"{kind} initialization skipped: no target layers defined.")
            return

        insert_positions = insert_after if insert_after is not None else self.ttt_insert_after
        if not insert_positions:
            print(f"{kind} initialization skipped: no insert positions defined.")
            return

        num_decoder_layers = len(self.decoder)
        print(f"Initializing {len(layers)} {kind} layers from decoder attention blocks")
        print(f"  Insert positions: {insert_positions}")


        for layer_idx, insert_idx in enumerate(insert_positions):
            decoder_idx = int(insert_idx)
            if decoder_idx % 2 == 0:
                decoder_idx += 1  # move to the subsequent global-attention layer

            if decoder_idx >= num_decoder_layers:
                raise IndexError(
                    f"Decoder index {decoder_idx} out of range for {kind} initialization (decoder has {num_decoder_layers} layers)."
                )

            if decoder_idx % 2 == 0:
                raise AssertionError(
                    f"Decoder index {decoder_idx} is not a global-attention layer after adjustment."
                )

            source_layer = self.decoder[decoder_idx]
            target_layer = layers[layer_idx]
            target_layer.load_state_dict(source_layer.state_dict())

            print(f"  Initialized {kind}_layer[{layer_idx}] from decoder[{decoder_idx}]")

    def _initialize_swa_from_global(self):
        if self.swa_layers is None:
            return
        self._initialize_ttt_layers_from_global(self.swa_layers, "swa", self.attn_insert_after)

    def decode(self, hidden, N, H, W, ttt_dict: Optional[dict] = None, window_size: Optional[int] = None, overlap_size: Optional[int] = None, is_first_window: bool = False,
               turn_off_ttt=False, turn_off_swa=False) -> torch.Tensor:
        BN, hw, _ = hidden.shape
        B = BN // N
        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        if self.num_pe_tokens > 0:
            pe_token_0 = getattr(self, 'pe_token_0')  # (1, 1, 1, dim)
            pe_token_1 = getattr(self, 'pe_token_1')  # (1, 1, 1, dim)
            pe_token_2 = getattr(self, 'pe_token_2')  # (1, 1, 1, dim)
            if overlap_size is None or window_size is None:
                raise ValueError("overlap_size and window_size must be provided when num_pe_tokens > 0")
            num_overlap_with_previous = min(overlap_size, N)
            num_other_frames = min(max(window_size - 2 * overlap_size, 0), N - num_overlap_with_previous)
            num_overlap_with_later = max(min(overlap_size, N, N - num_overlap_with_previous - num_other_frames), 0)
            pe_tokens = torch.cat([
                pe_token_0.repeat(B, num_overlap_with_previous, 1, 1),
                pe_token_1.repeat(B, num_other_frames, 1, 1),
                pe_token_2.repeat(B, num_overlap_with_later, 1, 1)
            ], dim=1).to(hidden.device).to(hidden.dtype).reshape(B*N, *pe_token_0.shape[-2:])  # (B*N, 1, dim)
            hidden = torch.cat([pe_tokens, hidden], dim=1)

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]


        if self.pos_type.startswith('rope'):    # add rope positional encoding only for frame patches
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # Use zero positions for special tokens.
            pos = pos + torch.ones_like(pos)
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        ttt_output_info = None
        ttt_state = ttt_dict.get("ttt") if ttt_dict is not None else None
        attn_state = ttt_dict.get("attn") if ttt_dict is not None else None
        gate_scales: List[torch.Tensor] = []
        attn_gate_scales: List[torch.Tensor] = []

        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                # frame attention
                pos_reshaped = pos.reshape(B*N, hw, -1) if pos is not None else None
                hidden = hidden.reshape(B*N, hw, -1)
                hidden_for_block = hidden
                pos_for_block = pos_reshaped
            else:
                # global attention
                pos_reshaped = pos.reshape(B, N*hw, -1) if pos is not None else None
                hidden = hidden.reshape(B, N*hw, -1)
                hidden_for_block = hidden
                pos_for_block = pos_reshaped

            # Keep the pre-block features for the residual update.
            layer_skip0 = (
                len(self.ttt_insert_after) == 36
                and i in self.ttt_insert_after
                and self.ttt_insert_after.index(i) % 2 == 0
            )
            
            if i % 2 == 1 and not layer_skip0:
                hidden_before_block = hidden_for_block
            elif i % 2 == 0 and layer_skip0:
                hidden_before_block = hidden_for_block
            else:
                hidden_before_block = hidden_for_block # dummy

            hidden = blk(hidden_for_block, xpos=pos_for_block)

            if ttt_state is not None and i in ttt_state.get("insert_after", []):
                # Help static analyzers: ensure non-None
                assert self.ttt_gate_projs is not None and self.ttt_layers is not None
                insert_after_list = ttt_state.get("insert_after", [])
                layer_idx = insert_after_list.index(i)

                x_for_residual = hidden.view(B, N, hw, -1)
                tokens_post = x_for_residual
                tokens_in = tokens_post

                gate_scale = torch.nn.functional.silu(self.ttt_gate_projs[layer_idx](tokens_in))
                # Disable TTT by setting its gate to zero.
                if turn_off_ttt: gate_scale = torch.zeros_like(gate_scale)  # Disable TTT.
                gate_scales.append(gate_scale)
                info = {
                    "ttt_op_order": ttt_state.get("ttt_op_order", []),
                    "w0": ttt_state["w0"][layer_idx],
                    "w1": ttt_state["w1"][layer_idx],
                    "w2": ttt_state["w2"][layer_idx],
                }
                ttt_output, output = self.ttt_layers[layer_idx](tokens_in, info)
                
                update_term = ttt_output * gate_scale

                tokens_out = update_term + tokens_post

                hidden = tokens_out

                if ttt_output_info is None:
                    ttt_output_info = {
                        "w0": [None] * len(insert_after_list),
                        "w1": [None] * len(insert_after_list),
                        "w2": [None] * len(insert_after_list),
                    }
                ttt_output_info["w0"][layer_idx] = output["w0"]
                ttt_output_info["w1"][layer_idx] = output["w1"]
                ttt_output_info["w2"][layer_idx] = output["w2"]

            # Sliding Window Attention (SWA)
            if attn_state is not None and i in attn_state.get("insert_after", []):
                assert self.swa_gate_projs is not None and self.swa_layers is not None
                insert_after_list = attn_state.get("insert_after", [])
                layer_idx = insert_after_list.index(i)

                patch_tokens_post_block = hidden    # [B, N, hw, dim]
                x_for_residual = patch_tokens_post_block.view(B, N, hw, -1)
                x_in = x_for_residual

                history_list = attn_state.get("history", [None] * len(insert_after_list))
                history = history_list[layer_idx]
                x_in_for_layer = x_in

                # Prepare position embeddings for current tokens
                if pos is not None:
                    pos_current = pos.reshape(B, N, hw, -1).reshape(B, N * hw, -1)    # [B, N*hw, 2]
                else:
                    pos_current = None

                # Check if we have KV cache from history
                use_kv_cache = (
                    history is not None 
                    and isinstance(history, dict) 
                    and "k" in history
                )

                if use_kv_cache:
                    # Use KV cache path
                    k_cache = history["k"]  # [B, num_heads, N_hist * hw, head_dim]
                    v_cache = history["v"]  # [B, num_heads, N_hist * hw, head_dim]
                    # Forward with KV cache
                    x_curr_flat = x_in_for_layer.reshape(B, N * hw, -1)

                    swa_output_flat = self.swa_layers[layer_idx].forward_with_kv_cache(
                        x_curr_flat, k_cache, v_cache,
                        xpos=pos_current,
                    )
                    swa_output = swa_output_flat.reshape(B, N, hw, -1)
                else:
                    # No cache or legacy tensor history.
                    # Handle legacy history format (raw tensor instead of dict)
                    history_raw = history if history is not None and not isinstance(history, dict) else None

                    if history_raw is not None:
                        x_with_history = torch.cat([history_raw, x_in_for_layer], dim=1)
                    else:
                        x_with_history = x_in_for_layer     # for the first window, use input as history => self-chunk attention

                    N_total = x_with_history.shape[1]
                    x_swa = x_with_history.reshape(B, N_total * hw, -1)

                    if pos is not None:
                        pos_swa = pos.reshape(B, N, hw, -1)
                        if history_raw is not None:
                            N_hist = history_raw.shape[1]
                            pos_hist = pos_swa[:, :1].repeat(1, N_hist, 1, 1)
                            pos_swa = torch.cat([pos_hist, pos_swa], dim=1)
                        pos_swa = pos_swa.reshape(B, N_total * hw, -1)
                    else:
                        pos_swa = None

                    swa_output_full = self.swa_layers[layer_idx](
                        x_swa, 
                        xpos=pos_swa, 
                    )
                    swa_output_full = swa_output_full.reshape(B, N_total, hw, x_in.shape[-1])
                    if history_raw is not None:
                        N_hist = history_raw.shape[1]
                        swa_output = swa_output_full[:, N_hist:, :, :]
                    else:
                        swa_output = swa_output_full

                gate_scale = torch.nn.functional.silu(self.swa_gate_projs[layer_idx](swa_output))
                if turn_off_swa: gate_scale = torch.zeros_like(gate_scale)
                attn_gate_scales.append(gate_scale)

                update_term = swa_output * gate_scale
                x_out_patch = update_term + x_for_residual
                x_out_patch_flat = x_out_patch.reshape(B, N * hw, -1)
                hidden = x_out_patch_flat.reshape(B * N, hw, -1)

                # Store KV cache for next window
                # Compute KV for current x_in with history_pe (since it will be history next time)
                if ttt_output_info is None:
                    ttt_output_info = {"history": [None] * len(insert_after_list)}
                elif "history" not in ttt_output_info:
                    ttt_output_info["history"] = [None] * len(insert_after_list)

                x_for_cache = x_in
                x_for_cache_flat = x_for_cache.reshape(B, N * hw, -1)
                
                # Position for cache: use first frame's position repeated (same as original logic)
                if pos is not None:
                    pos_for_cache = pos.reshape(B, N, hw, -1)[:, :1].repeat(1, N, 1, 1).reshape(B, N * hw, -1)
                else:
                    pos_for_cache = None

                k_new, v_new = self.swa_layers[layer_idx].compute_kv_cache(x_for_cache_flat, xpos=pos_for_cache)
                
                if getattr(self, "detach_swa_history", False):
                    k_new = k_new.detach()      # [B, num_heads, N*hw, head_dim]
                    v_new = v_new.detach()
                
                ttt_output_info["history"][layer_idx] = {"k": k_new, "v": v_new}

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:     # use final frame attention output and chunk global attention output, so 2*dino feats dimension
                final_output.append(hidden.reshape(B*N, hw, -1))

        avg_gate_scale = torch.tensor(0.0, device=hidden.device, dtype=torch.float32)
        avg_attn_gate_scale: Optional[torch.Tensor] = None
        if gate_scales:
            all_gate_scales = torch.cat([g.flatten() for g in gate_scales])
            if all_gate_scales.numel() > 0:
                avg_gate_scale = all_gate_scales.abs().mean()
        if attn_gate_scales:
            all_attn_gate_scales = torch.cat([g.flatten() for g in attn_gate_scales])
            if all_attn_gate_scales.numel() > 0:
                avg_attn_gate_scale = all_attn_gate_scales.abs().mean()

        if len(final_output) < 2:
            raise RuntimeError(
                f"Decoder expected to collect two final outputs but got {len(final_output)}."
            )

        return (
            torch.cat([final_output[0], final_output[1]], dim=-1),
            (pos.reshape(B*N, hw, -1) if pos is not None else None),
            ttt_output_info,
            avg_gate_scale,
            avg_attn_gate_scale,
            gate_scales,
        )
    
    def forward(self, imgs, *args, **kwargs):
        # Windowing controls (optional)
        window_size = kwargs.pop('window_size', -1)
        overlap_size = kwargs.pop('overlap_size', 1)
        num_iterations = kwargs.pop('num_iterations', 1)
        no_detach = kwargs.pop('no_detach', False)
        sim3 = kwargs.pop('sim3', False)
        se3 = kwargs.pop('se3', False)
        sim3_on_reset = kwargs.pop('sim3_on_reset', False)  # when reset_every>0 and neither sim3/se3 is set, align each reset block with one Sim(3) instead of SE(3)
        reset_every = kwargs.pop('reset_every', 0)  # reset TTT / adapter state every N windows (0 disables)
        turn_off_ttt = kwargs.pop('turn_off_ttt', False)
        turn_off_swa = kwargs.pop('turn_off_swa', False)
        sim3_scale_mode = kwargs.pop('sim3_scale_mode', 'median')
        output_keys = kwargs.pop('output_keys', None)  # if set, only accumulate these keys per window
        # Loop-closure state, filled only by the streaming loop detector below:
        # lc_frame_pairs: list of detected (frame_i, frame_j) pairs;
        # lc_pairs: dict window_j -> [window_i, ...] of bridges to run;
        # lc_frame_matches: dict (j_win, i_win) -> [[fi, fj], ...].
        lc_frame_pairs = None
        lc_pairs = None
        lc_frame_matches = None
        pgo_dist_thresh = kwargs.pop('pgo_dist_thresh', 5)    # max frame-index gap for sequential edges
        # Per-axis sigmas fall back to the legacy isotropic sigmas when unset.
        pgo_sigma_seq   = kwargs.pop('pgo_sigma_seq',   0.01) # legacy isotropic sigma for seq edges
        pgo_sigma_lc    = kwargs.pop('pgo_sigma_lc',    0.1)  # legacy isotropic sigma for LC edges
        pgo_sigma_R_seq = kwargs.pop('pgo_sigma_R_seq', None) # if None → fallback to pgo_sigma_seq
        pgo_sigma_t_seq = kwargs.pop('pgo_sigma_t_seq', None)
        pgo_sigma_R_lc  = kwargs.pop('pgo_sigma_R_lc',  0.005) # tight on rotation: bridge R is reliable
        pgo_sigma_t_lc  = kwargs.pop('pgo_sigma_t_lc',  0.1)   # loose on translation: bridge t magnitude is noisy
        pgo_lc_robust   = kwargs.pop('pgo_lc_robust',   'huber') # 'huber'|'cauchy'|None
        pgo_lc_robust_k = kwargs.pop('pgo_lc_robust_k', 1.345) # Huber threshold
        pgo_sigma_match      = kwargs.pop('pgo_sigma_match', 0.01)
        pgo_add_match_constraints = kwargs.pop('pgo_add_match_constraints', True)
        pgo_add_adj_constraints   = kwargs.pop('pgo_add_adj_constraints', False)
        # Add relative-pose constraints between middle frames of windows in each
        # reset block. Use pre-PGO poses so these edges start with zero residual.
        pgo_add_block_constraints = kwargs.pop('pgo_add_block_constraints', False)
        pgo_sigma_block           = kwargs.pop('pgo_sigma_block', 0.05)
        pgo_block_middle_count    = kwargs.pop('pgo_block_middle_count', 4)
        pgo_debug_save  = kwargs.pop('pgo_debug_save',  None) # path to save debug edges npz, e.g. '/tmp/pgo_debug.npz'
        run_pgo         = kwargs.pop('run_pgo', False)        # if True, run PGO even without LC pairs
        # Run online PGO after each batch of bridges. Corrected poses affect
        # the output, not subsequent window inference.
        online_pgo      = kwargs.pop('online_pgo',      False)
        # Reject bridges whose median translation or rotation residual exceeds
        # the threshold, then re-optimise. Zero disables each threshold; translation
        # uses model units and rotation uses degrees.
        pgo_lc_check_t     = float(kwargs.pop('pgo_lc_check_t', 1.0))
        pgo_lc_check_R     = float(kwargs.pop('pgo_lc_check_R', 0.0))
        pgo_lc_check_iters = int(kwargs.pop('pgo_lc_check_iters', 3))
        # Detect loops per window and connect them with bridge windows.
        loop_detect = kwargs.pop('loop_detect', False)
        loop_image_paths = kwargs.pop('loop_image_paths', None)  # list[str] – one path per frame
        loop_ckpt = kwargs.pop('loop_ckpt', '')               # path to SALAD checkpoint (required)
        loop_result_dir = kwargs.pop('loop_result_dir', '/tmp/loger_loop_detection')
        loop_nms_threshold = kwargs.pop('loop_nms_threshold', 25)
        loop_min_frame_gap = kwargs.pop('loop_min_frame_gap', 10)
        loop_sim_thresh = kwargs.pop('loop_sim_thresh', 0.7)
        loop_top_k = kwargs.pop('loop_top_k', 5)
        gt_traj_path = kwargs.pop('gt_traj_path', None)                       # ndarray (N, 6), for loop detection debug only
        lc_detect_use_nms = kwargs.pop('lc_detect_use_nms', False)                # Apply NMS to loop candidates.
        # Pair visualisations are off by default because saving PNGs slows inference.
        loop_save_pair_vis = kwargs.pop('loop_save_pair_vis', False)
        # Reject loop candidates with chord/arc >= lc_chord_arc_thresh.
        # Use aligned window centres over the full candidate segment.
        # None or a threshold >= 1 disables this filter.
        lc_chord_arc_thresh = kwargs.pop('lc_chord_arc_thresh', 0.80)
        # Restore TTT weights and SWA history after bridges unless keep_state is set.
        lc_bridge_keep_state = bool(kwargs.pop('lc_bridge_keep_state', False))
        # Bridge NMS: per window, keep the candidate loop window with the most frame
        # matches, drop candidates within +-lc_bridge_nms windows of it, repeat (0 = off).
        lc_bridge_nms = int(kwargs.pop('lc_bridge_nms', 2))
        n_bridges_run = n_bridges_dropped = 0
        save_bridge_pts = kwargs.pop('save_bridge_pts', False)
        # Optional snapshots; save_online_pgo_poses_dir requires online_pgo.
        save_online_pgo_poses_dir = kwargs.pop('save_online_pgo_poses_dir', None)
        # Save window poses, bridges, and call inputs for offline PGO replay.
        save_pgo_inputs = kwargs.pop('save_pgo_inputs', None)
        _pgo_rec = {"windows": None, "win_cam": {}, "bridges": {}, "calls": []}
        save_window_local_pts_dir = kwargs.pop('save_window_local_pts_dir', None)
        # Output stride. Heavy fields are sampled per window at matching global
        # indices; camera poses stay full resolution for alignment and PGO.
        inference_save_stride = max(1, int(kwargs.pop('inference_save_stride', 1)))
        # Debug scale override: use each window's Umeyama scale against GT,
        # relative to window 0. Used for Sim(3) alignment only.
        gt_poses_for_scale = kwargs.pop('gt_poses_for_scale', None)  # Optional[Tensor] (N, 4, 4)
        if sim3 and se3:
            raise ValueError("'sim3' and 'se3' alignments are mutually exclusive; enable only one.")

        # Ensure at least one decode iteration so that 'hidden' is always defined
        try:
            num_iterations = int(num_iterations)
        except Exception:
            num_iterations = 1
        if num_iterations < 1:
            num_iterations = 1
        try:
            reset_every = int(reset_every)
        except Exception:
            reset_every = 0
        if reset_every < 0:
            reset_every = 0

        # Ensure batch dimension
        if imgs.dim() == 4:
            imgs = imgs.unsqueeze(0)

        B, N, C, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        # Report loop detection and PGO time in merged["_timing"].
        # Synchronise CUDA before timing to include pending GPU work.
        _t_loop_detect = 0.0
        def _now_sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.perf_counter()

        def _rss_mb():
            """Return process RSS in MiB from /proc/self/statm."""
            try:
                with open("/proc/self/statm") as _f:
                    pages = int(_f.read().split()[1])
                return pages * _os.sysconf("SC_PAGE_SIZE") / (1024.0 ** 2)
            except Exception:
                return float("nan")

        class _SectionTimer:
            """Track time and process memory for a section, including failed calls.

            Retained RSS is the sum of per-call growth. Peak RSS covers the whole
            process; increases in its high-water mark are lower bounds on call peaks.
            """

            __slots__ = ("name", "total", "calls", "per_call", "_t0",
                         "_r0", "_h0", "rss_delta", "rss_after",
                         "hwm_delta", "rss_before_first")

            def __init__(self, name):
                self.name = name
                self.total = 0.0
                self.calls = 0
                self.per_call = []
                self.rss_delta = []        # MiB this call retained  (PGO only)
                self.hwm_delta = []        # MiB this call added to the process
                                           # high-water mark        (PGO only)
                self.rss_after = []        # MiB absolute process RSS after the
                                           # call (WHOLE process, not just PGO)
                self.rss_before_first = None

            @staticmethod
            def _hwm_mb():
                import resource
                # Linux ru_maxrss is a cumulative peak in KiB. Its increase during a call
                # is a lower bound on that call's transient memory use.
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

            def start(self):
                self._r0 = _rss_mb()
                self._h0 = self._hwm_mb()
                if self.rss_before_first is None:
                    self.rss_before_first = self._r0
                self._t0 = _now_sync()

            def stop(self):
                dt = _now_sync() - self._t0
                r1 = _rss_mb()
                self.total += dt
                self.calls += 1
                self.per_call.append(float(dt))
                self.rss_delta.append(float(r1 - self._r0))
                self.hwm_delta.append(float(self._hwm_mb() - self._h0))
                self.rss_after.append(float(r1))
                return dt

            # Use start/stop when a call already has a try/except block.
            def __enter__(self):
                self.start()
                return self

            def __exit__(self, *_exc):
                self.stop()
                return False   # never suppress exceptions

            @property
            def mean(self):
                return self.total / self.calls if self.calls else 0.0

            def summary(self):
                return {
                    "total": float(self.total),
                    "calls": int(self.calls),
                    "mean":  float(self.mean),
                    "max":   float(max(self.per_call)) if self.per_call else 0.0,
                    "per_call": list(self.per_call),
                    # Report retained RSS growth, increases in peak RSS, and total process RSS.
                    # Total RSS includes the model and images as well as PGO.
                    "rss_retained_mb": float(sum(self.rss_delta)),
                    "hwm_max_mb":      float(max(self.hwm_delta)) if self.hwm_delta else 0.0,
                    "proc_rss_peak_mb": float(max(self.rss_after)) if self.rss_after else 0.0,
                    "rss_before_mb":   float(self.rss_before_first or 0.0),
                    "rss_delta_per_call": list(self.rss_delta),
                    "hwm_delta_per_call": list(self.hwm_delta),
                    "rss_after_per_call": list(self.rss_after),
                }

        # Use one timer for all online and offline PGO calls.
        _pgo_timer = _SectionTimer("pgo")

        # Extract SALAD descriptors per window and compare with cached frames.
        streaming_loop_detector = None
        if loop_detect:
            if loop_image_paths is not None and len(loop_image_paths) >= 2:
                _os.makedirs(loop_result_dir, exist_ok=True)
                _t_ld_setup = _now_sync()
                streaming_loop_detector = LoopDetector(
                    image_list=list(loop_image_paths),
                    gt_traj_path=gt_traj_path,
                    result_dir=loop_result_dir,
                    ckpt_path=loop_ckpt,
                    nms_threshold=loop_nms_threshold,
                    min_frame_gap=loop_min_frame_gap,
                    similarity_threshold=loop_sim_thresh,
                    top_k=loop_top_k,
                    use_nms=lc_detect_use_nms,
                    save_pair_vis=loop_save_pair_vis,
                )
                streaming_loop_detector.load_model()
                _t_loop_detect += _now_sync() - _t_ld_setup
                lc_frame_pairs = []
                # Accepted frame pairs (fi, fj, sim), written to detected_loops.txt.
                lc_frame_pairs_kept: list[tuple[int, int, float]] = []
                run_pgo = True

                # ----------------------------------------------------------
                # Map window centres to window 0 using a shared overlap pose.
                # These transforms are only used by the chord/arc filter.
                # ----------------------------------------------------------
                import numpy as _np_align
                window_transforms: list[tuple[_np_align.ndarray, _np_align.ndarray]] = [
                    (_np_align.eye(3), _np_align.zeros(3))
                ]

                def _se3_kabsch(src, dst):
                    src = _np_align.asarray(src, dtype=_np_align.float64)
                    dst = _np_align.asarray(dst, dtype=_np_align.float64)
                    sm = src.mean(0); dm = dst.mean(0)
                    H = (src - sm).T @ (dst - dm)
                    U, _, Vt = _np_align.linalg.svd(H)
                    d = _np_align.sign(_np_align.linalg.det(Vt.T @ U.T))
                    D = _np_align.diag([1.0, 1.0, d])
                    R = Vt.T @ D @ U.T
                    t = dm - R @ sm
                    return R, t

                def _update_window_transform(win_idx: int):
                    """Map this window to window 0 using poses at a shared overlap frame.

                    Read rotation from the pose matrices instead of fitting camera centres,
                    which can be collinear. Call after appending the window predictions.
                    """
                    if win_idx <= 0:
                        return
                    if win_idx < len(window_transforms):
                        return
                    if win_idx >= len(all_predictions) or win_idx - 1 < 0:
                        return
                    cur_cams = all_predictions[win_idx].get("camera_poses", None)
                    prv_cams = all_predictions[win_idx - 1].get("camera_poses", None)
                    if cur_cams is None or prv_cams is None or overlap_size <= 0:
                        return
                    # Same physical frame in two local worlds:
                    #   prev window's pose at index (Nw_prev - overlap_size)
                    #   curr window's pose at index 0
                    prev_idx = max(prv_cams.shape[1] - overlap_size, 0)
                    pose_prev = prv_cams[0, prev_idx]    # (4, 4)
                    pose_curr = cur_cams[0, 0]           # (4, 4)
                    if isinstance(pose_prev, torch.Tensor):
                        pose_prev = pose_prev.detach().cpu().numpy().astype(_np_align.float64)
                    else:
                        pose_prev = _np_align.asarray(pose_prev, dtype=_np_align.float64)
                    if isinstance(pose_curr, torch.Tensor):
                        pose_curr = pose_curr.detach().cpu().numpy().astype(_np_align.float64)
                    else:
                        pose_curr = _np_align.asarray(pose_curr, dtype=_np_align.float64)
                    R_prv_loc = pose_prev[:3, :3]; t_prv_loc = pose_prev[:3, 3]
                    R_cur_loc = pose_curr[:3, :3]; t_cur_loc = pose_curr[:3, 3]
                    # SE(3) mapping (curr's local frame) -> (prev's local frame):
                    #   p_in_prev_local = R_rel @ p_in_curr_local + t_rel
                    R_rel = R_prv_loc @ R_cur_loc.T
                    t_rel = t_prv_loc - R_rel @ t_cur_loc
                    # Chain through prev's already-global transform.
                    R_prev_g, t_prev_g = window_transforms[win_idx - 1]
                    R_new = R_prev_g @ R_rel
                    t_new = R_prev_g @ t_rel + t_prev_g
                    window_transforms.append((R_new, t_new))
            else:
                import warnings
                warnings.warn(
                    "loop_detect=True but loop_image_paths was not provided; skipping loop detection.",
                    RuntimeWarning,
                )

        # --- Unified Windowed Inference ---
        if window_size <= 0 or window_size >= N:    # not use sliding window, global processing
            windows = [(0, N)]
            eff_overlap = 0
            eff_window_size = N
        else:
            windows = []
            step = max(window_size - overlap_size, 1)
            for start_idx in range(0, N, step):
                end_idx = min(start_idx + window_size, N)
                if end_idx - start_idx >= overlap_size or (end_idx == N and start_idx < N):
                    windows.append((start_idx, end_idx))
                if end_idx == N:
                    break
            eff_overlap = overlap_size
            eff_window_size = window_size

        # Cache the effective window and overlap sizes for downstream merging utilities
        self._last_window_size = eff_window_size
        self._last_overlap_size = eff_overlap

        # Select alignment for both partial and final merges.
        per_block_sim3 = sim3_on_reset and reset_every > 0 and not sim3 and not se3
        per_block_se3 = reset_every > 0 and not sim3 and not se3 and not sim3_on_reset

        def _build_merged(predictions_subset, windows_subset):
            """Merge partial or full window predictions using the selected alignment."""
            _oracle_scales = None
            if gt_poses_for_scale is not None and (sim3 or per_block_sim3):
                _oracle_scales = self._compute_oracle_window_scales(
                    predictions_subset, windows_subset, gt_poses_for_scale
                )
            elif sim3_scale_mode in (
                'w0_reference', 'max_reference', 'p75_reference',
                'median_reference', 'run_max_reference',
            ) and (sim3 or per_block_sim3):
                _mode_map = {
                    'w0_reference': 'w0',
                    'max_reference': 'max',
                    'p75_reference': 'p75',
                    'median_reference': 'median',
                    'run_max_reference': 'run_max',
                }
                _oracle_scales = self._compute_reference_scales(
                    predictions_subset, mode=_mode_map[sim3_scale_mode]
                )
            if sim3:
                return self._merge_windowed_predictions_sim3(
                    predictions_subset,
                    allow_scale=True,
                    scale_mode=sim3_scale_mode,
                    oracle_window_scales=_oracle_scales,
                )
            elif per_block_sim3:
                return self._merge_windowed_predictions_sim3(
                    predictions_subset,
                    allow_scale=True,
                    scale_mode=sim3_scale_mode,
                    reset_every=reset_every,
                    reuse_transform_within_reset_block=True,
                    oracle_window_scales=_oracle_scales,
                )
            elif se3 or per_block_se3:
                return self._merge_windowed_predictions_sim3(
                    predictions_subset,
                    allow_scale=False,
                    reset_every=reset_every,
                    reuse_transform_within_reset_block=per_block_se3,
                )
            else:
                return self._merge_windowed_predictions(
                    predictions_subset, eff_window_size, eff_overlap
                )

        def _run_pgo(merged_dict, predictions_subset, windows_subset, debug_save=None):
            """Run frame-level PGO on `merged_dict` using the configured sigmas."""
            _sR_seq = pgo_sigma_R_seq if pgo_sigma_R_seq is not None else pgo_sigma_seq
            _st_seq = pgo_sigma_t_seq if pgo_sigma_t_seq is not None else pgo_sigma_seq
            _sR_lc  = pgo_sigma_R_lc  if pgo_sigma_R_lc  is not None else pgo_sigma_lc
            _st_lc  = pgo_sigma_t_lc  if pgo_sigma_t_lc  is not None else pgo_sigma_lc
            return self._apply_frame_pgo(
                merged_dict, predictions_subset, lc_bridge_cams, windows_subset,
                dist_thresh=pgo_dist_thresh,
                sigma_R_seq=_sR_seq, sigma_t_seq=_st_seq,
                sigma_R_lc=_sR_lc,   sigma_t_lc=_st_lc,
                lc_robust=pgo_lc_robust, lc_robust_k=pgo_lc_robust_k,
                sigma_match=pgo_sigma_match,
                add_match_constraints=pgo_add_match_constraints,
                add_adj_constraints=pgo_add_adj_constraints,
                reset_every=reset_every,
                add_block_constraints=pgo_add_block_constraints,
                sigma_block=pgo_sigma_block,
                block_middle_count=pgo_block_middle_count,
                debug_save_path=debug_save,
            )

        # Streaming mode: bridges and frame matches are appended per window below.
        if streaming_loop_detector is not None:
            lc_pairs = {}
            lc_frame_matches = {}

        # Frame-index → window-index helper for the streaming path below.
        _lc_step = max(eff_window_size - eff_overlap, 1)

        def _frame_to_window_idx(f: int) -> int:
            return min(max(int(f)-eff_overlap, 0) // _lc_step, len(windows) - 1)

        # Prepare TTT states across windows
        if self.ttt_layers is not None:
            w0 = [None] * len(self.ttt_insert_after)
            w1 = [None] * len(self.ttt_insert_after)
            w2 = [None] * len(self.ttt_insert_after)
        else:
            w0 = w1 = w2 = None

        # Prepare SWA history states across windows
        swa_history = [None] * len(self.attn_insert_after) if self.swa_layers is not None else None

        def reset_adaptive_states():
            """Reset fast-weight TTT states only; SWA history is preserved across resets."""
            nonlocal w0, w1, w2
            if self.ttt_layers is not None:
                w0 = [None] * len(self.ttt_insert_after)
                w1 = [None] * len(self.ttt_insert_after)
                w2 = [None] * len(self.ttt_insert_after)

        all_predictions = []
        all_gate_scales: List[torch.Tensor] = []
        all_attn_gate_scales: List[torch.Tensor] = []

        lc_bridge_cams: dict = {}
        lc_bridge_pts:  dict = {}  # only populated when save_bridge_pts=True

        # Reuse the graph across online-PGO calls, adding new window/bridge edges
        # and warm-starting from the previous result. A final call covers any
        # windows remaining after the last loop-closure batch.
        online_pgo_state: dict = {
            'graph': None,
            'last_result': None,
            'windows_added': set(),
            'bridges_added': set(),
            'last_T_total': 0,
            'prior_added': False,
        }

        _online_pgo_call_idx = 0
        if save_online_pgo_poses_dir:
            _os.makedirs(save_online_pgo_poses_dir, exist_ok=True)
        if save_window_local_pts_dir:
            _os.makedirs(save_window_local_pts_dir, exist_ok=True)

        def _maybe_save_pgo_snapshot(_d, _n_windows, _n_loops, _windows_subset):
            """Save pre-PGO and post-PGO poses from an online-PGO call."""
            nonlocal _online_pgo_call_idx
            if not save_online_pgo_poses_dir:
                return
            try:
                _pre = _d.get("camera_poses_pre_pgo")
                _post = _d.get("camera_poses")
                if _pre is None or _post is None:
                    return
                _meta = {
                    "call_idx":  int(_online_pgo_call_idx),
                    "n_windows": int(_n_windows),
                    "n_loops":   int(_n_loops),
                    "windows":   list(_windows_subset),
                }
                _pre_path = _os.path.join(
                    save_online_pgo_poses_dir,
                    f"pgo_call_{_online_pgo_call_idx:04d}_pre.pt",
                )
                _post_path = _os.path.join(
                    save_online_pgo_poses_dir,
                    f"pgo_call_{_online_pgo_call_idx:04d}_post.pt",
                )
                torch.save({"camera_poses": _pre.detach().cpu().clone(),  **_meta}, _pre_path)
                torch.save({"camera_poses": _post.detach().cpu().clone(), **_meta}, _post_path)
            except Exception as _exc:
                import warnings as _w
                _w.warn(
                    f"save_online_pgo_poses_dir write failed @ call "
                    f"{_online_pgo_call_idx}: {_exc}",
                    RuntimeWarning,
                )
            _online_pgo_call_idx += 1

        def _run_online_pgo_inc(merged_dict, predictions_subset, windows_subset):
            """Add new windows and bridges to the online pose graph."""
            _sR_seq = pgo_sigma_R_seq if pgo_sigma_R_seq is not None else pgo_sigma_seq
            _st_seq = pgo_sigma_t_seq if pgo_sigma_t_seq is not None else pgo_sigma_seq
            _sR_lc  = pgo_sigma_R_lc  if pgo_sigma_R_lc  is not None else pgo_sigma_lc
            _st_lc  = pgo_sigma_t_lc  if pgo_sigma_t_lc  is not None else pgo_sigma_lc
            if save_pgo_inputs:
                # Everything read by _apply_frame_pgo_incremental, captured when first used.
                for _w in range(min(len(windows_subset), len(predictions_subset))):
                    _c = (predictions_subset[_w] or {}).get("camera_poses")
                    if _w not in _pgo_rec["win_cam"] and _c is not None:
                        _pgo_rec["win_cam"][_w] = _c[0].detach().float().cpu().numpy().copy()
                for _k, _b in lc_bridge_cams.items():
                    if _k not in _pgo_rec["bridges"]:
                        _pgo_rec["bridges"][_k] = {
                            "cam_poses": _b["cam_poses"][0].detach().float().cpu().numpy().copy(),
                            "selected_i_frames": list(_b["selected_i_frames"]),
                            "selected_j_frames": list(_b["selected_j_frames"]),
                            "frame_matches": list(_b["frame_matches"]),
                            "overlap_size": int(_b["overlap_size"])}
                _cp = merged_dict["camera_poses"][0].detach().float().cpu().numpy()
                _t0 = _pgo_rec["calls"][-1]["T_total"] if _pgo_rec["calls"] else 0
                _pgo_rec["calls"].append({"n_windows": len(windows_subset),
                                          "bridge_keys": list(lc_bridge_cams.keys()),
                                          "T_total": _cp.shape[0], "new_poses": _cp[_t0:].copy()})
                _pgo_rec["windows"] = [tuple(w) for w in windows_subset]
            return self._apply_frame_pgo_incremental(
                merged_dict, online_pgo_state, predictions_subset, lc_bridge_cams, windows_subset,
                dist_thresh=pgo_dist_thresh,
                sigma_R_seq=_sR_seq, sigma_t_seq=_st_seq,
                sigma_R_lc=_sR_lc,   sigma_t_lc=_st_lc,
                lc_robust=pgo_lc_robust, lc_robust_k=pgo_lc_robust_k,
                sigma_match=pgo_sigma_match,
                add_match_constraints=pgo_add_match_constraints,
                add_adj_constraints=pgo_add_adj_constraints,
                lc_check_t=pgo_lc_check_t, lc_check_R=pgo_lc_check_R,
                lc_check_iters=pgo_lc_check_iters,
            )

        def _make_ttt_dict():
            ttt_state = None
            attn_state = None
            if self.ttt_layers is not None:
                ttt_state = {
                    "ttt_op_order": self.ttt_op_order if self.ttt_op_order is not None else [],
                    "insert_after": self.ttt_insert_after,
                    "w0": w0, "w1": w1, "w2": w2,
                }
            if self.swa_layers is not None:
                attn_state = {
                    "insert_after": self.attn_insert_after,
                    "history": swa_history,
                }
            if ttt_state is None and attn_state is None:
                return None
            return {"ttt": ttt_state, "attn": attn_state}

        def _update_adaptive_state(ttt_output_info):
            nonlocal w0, w1, w2, swa_history
            if self.ttt_layers is not None and ttt_output_info is not None:
                w0, w1, w2 = ttt_output_info["w0"], ttt_output_info["w1"], ttt_output_info["w2"]
            if ttt_output_info is not None:
                swa_history = ttt_output_info.get("history", swa_history)

        # Online window queue: after each normal window j its bridge windows are
        # prepended so they run (and update TTT/SWA state) before j+1.
        win_queue: deque = deque(
            ("normal", win_idx, s, e) for win_idx, (s, e) in enumerate(windows)
        )
        windows_pbar = tqdm(total=len(windows), desc="Windows", unit="win", leave=False)

        # Count of *normal* windows processed, used for reset_every cadence.
        # Bridge windows do not count toward resets.
        normal_win_count = 0

        def maybe_detach(t, no_detach=no_detach):
            if t is None:
                return None
            return t if self.training or no_detach else t.detach().cpu()

        while win_queue:
            entry = win_queue.popleft()
            kind = entry[0]

            # ---- Prepare per-kind inputs and decode kwargs ----
            if kind == "normal":
                _, win_idx, start_idx, end_idx = entry

                if reset_every > 0 and normal_win_count > 0 and normal_win_count % reset_every == 0:
                    reset_adaptive_states()
                normal_win_count += 1

                imgs_w_raw = imgs[:, start_idx:end_idx]

                # Find loop candidates now; filter them after this window's poses are aligned.
                new_pairs_pending: list = []
                if streaming_loop_detector is not None and loop_image_paths is not None:
                    _t_ld_w = _now_sync()
                    win_frame_idx = list(range(start_idx, end_idx))
                    win_paths = [loop_image_paths[fi] for fi in win_frame_idx]
                    streaming_loop_detector.extract_window_descriptors(win_frame_idx, win_paths)
                    new_pairs_pending = streaming_loop_detector.detect_window_loops(win_frame_idx)
                    if new_pairs_pending:
                        streaming_loop_detector.visualize_new_pairs(new_pairs_pending)
                    _t_loop_detect += _now_sync() - _t_ld_w

                imgs_in = (imgs_w_raw.to(self.image_mean.device) - self.image_mean) / self.image_std
                Nw = imgs_in.shape[1]

                decode_kwargs = dict(
                    window_size=window_size, overlap_size=overlap_size,
                    is_first_window=(start_idx == 0),
                    turn_off_ttt=turn_off_ttt, turn_off_swa=turn_off_swa,
                )

            else:  # bridge: [j_end(ov) | selected_i (di) | selected_j (dj) | j_end(ov)]
                _, j_win_idx, i_win_idx, i_start, i_end, j_end_idx, frame_matches = entry
                j_start = windows[j_win_idx][0]
                j_end   = j_end_idx  # == windows[j_win_idx][1]

                # Middle slots split evenly between i and j matched frames
                _slots_mid = window_size - 2 * overlap_size
                slots_i = _slots_mid // 2
                slots_j = _slots_mid - slots_i

                if frame_matches:
                    _core_i = sorted({fi for fj, fi in frame_matches if i_start <= fi < i_end})
                    _core_j = sorted({fj for fj, fi in frame_matches if j_start <= fj < j_end})
                else:
                    # No matches: fall back to last few frames of each window
                    _core_i = list(range(max(i_end - overlap_size, i_start), i_end))
                    _core_j = list(range(max(j_end - overlap_size, j_start), j_end))

                # Adjacent fill for i side
                _selected_i = set(_core_i)
                while len(_selected_i) < slots_i:
                    _prev_i = set(_selected_i)
                    for _f in sorted(_prev_i):
                        if _f - 1 >= i_start: _selected_i.add(_f - 1)
                        if _f + 1 < i_end:    _selected_i.add(_f + 1)
                    if _selected_i == _prev_i:
                        break
                _selected_i = sorted(_selected_i)[:slots_i]

                # Adjacent fill for j side
                _selected_j = set(_core_j)
                while len(_selected_j) < slots_j:
                    _prev_j = set(_selected_j)
                    for _f in sorted(_prev_j):
                        if _f - 1 >= j_start: _selected_j.add(_f - 1)
                        if _f + 1 < j_end:    _selected_j.add(_f + 1)
                    if _selected_j == _prev_j:
                        break
                _selected_j = sorted(_selected_j)[:slots_j]

                # Build bridge: [j_end | i_frames | j_frames | j_end]
                br_j_end = imgs[:, j_end - overlap_size : j_end]
                br_i   = imgs[:, torch.tensor(_selected_i, dtype=torch.long)]
                br_j   = imgs[:, torch.tensor(_selected_j, dtype=torch.long)]
                br_raw = torch.cat([br_j_end, br_i, br_j, br_j_end], dim=1)
                imgs_in = (br_raw.to(self.image_mean.device) - self.image_mean) / self.image_std
                Nw      = imgs_in.shape[1]

                del br_j_end, br_i, br_j, br_raw

                decode_kwargs = dict(
                    window_size=Nw, overlap_size=overlap_size,
                    is_first_window=False,
                )

            # ---- Common: encode + decode + state update ----
            hidden = None  # type: ignore[assignment]
            pos    = None  # type: ignore[assignment]
            with (contextlib.nullcontext() if kind == "normal" else torch.no_grad()):
                for _ in range(num_iterations):
                    if self.ttt_layers is not None and w0 is None:
                        w0 = [None] * len(self.ttt_insert_after)
                        w1 = [None] * len(self.ttt_insert_after)
                        w2 = [None] * len(self.ttt_insert_after)
                    if self.swa_layers is not None and swa_history is None:
                        swa_history = [None] * len(self.attn_insert_after)

                    imgs_flat = imgs_in.reshape(B * Nw, C, H, W)
                    hidden_input = self.encoder(imgs_flat, is_training=True)
                    if isinstance(hidden_input, dict):
                        hidden_input = hidden_input["x_norm_patchtokens"]

                    hidden, pos, ttt_output_info, decode_avg_gate_scale, decode_avg_attn_gate_scale, _ = self.decode(
                        hidden_input, Nw, H, W, ttt_dict=_make_ttt_dict(), **decode_kwargs,
                    )
                    if kind == "normal" or not lc_bridge_keep_state:
                        _update_adaptive_state(ttt_output_info)

                    if kind == "normal":
                        if decode_avg_gate_scale is not None:
                            all_gate_scales.append(decode_avg_gate_scale.detach().cpu())
                        if decode_avg_attn_gate_scale is not None:
                            all_attn_gate_scales.append(decode_avg_attn_gate_scale.detach().cpu())

            if hidden is None:
                if kind == "normal":
                    windows_pbar.update(1)
                continue

            # ---- Kind-specific post-processing ----
            if kind == "normal":
                windows_pbar.update(1)

                point_hidden = self.point_decoder(hidden, xpos=pos)
                conf_hidden = self.conf_decoder(hidden, xpos=pos) if self.use_conf and self.conf_decoder is not None else None

                if self.pi3x and self.pi3x_metric:
                    hw = hidden.shape[1]
                    pos_hw = pos.reshape(B, Nw*hw, -1)
                    metric_hidden = self.metric_decoder(self.metric_token.repeat(B, 1, 1), hidden.reshape(B, Nw*hw, -1), xpos=pos_hw[:, 0:1], ypos=pos_hw)
                else:
                    metric_hidden = None

                camera_hidden = self.camera_decoder(hidden, xpos=pos)

                with torch.autocast(device_type='cuda', enabled=False):
                    point_hidden = point_hidden.float()
                    if self.pi3x:
                        xy, z = self.point_head(point_hidden[:, self.patch_start_idx:], patch_h=patch_h, patch_w=patch_w)
                        xy = xy.permute(0, 2, 3, 1).reshape(B, Nw, H, W, -1)
                        z = z.permute(0, 2, 3, 1).reshape(B, Nw, H, W, -1)
                        z = torch.exp(z.clamp(max=15.0))
                        local_points = torch.cat([xy * z, z], dim=-1)
                    else:
                        ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, Nw, H, W, -1)
                        xy, z = ret.split([2, 1], dim=-1)
                        z = torch.exp(z)
                        local_points = torch.cat([xy * z, z], dim=-1)

                    conf = self.conf_head([conf_hidden[:, self.patch_start_idx:].float()], (H, W)).reshape(B, Nw, H, W, -1) if conf_hidden is not None and self.conf_head is not None else None

                    camera_poses = self.camera_head(camera_hidden.float()[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, Nw, 4, 4)

                    if self.pi3x and self.pi3x_metric and metric_hidden is not None:
                        metric = self.metric_head(metric_hidden.float()).reshape(B).exp()
                        camera_poses[..., :3, 3] = camera_poses[..., :3, 3] * metric.view(B, 1, 1)
                        local_points = local_points * metric.view(B, 1, 1, 1, 1)
                    else:
                        metric = None

                # Stop immediately if this window's poses are not finite; every later
                # window would inherit the NaNs.
                if not torch.isfinite(camera_poses).all():
                    raise RuntimeError(
                        f"NaN/inf camera poses in window {win_idx + 1}/{len(windows)} "
                        f"(frames {start_idx}-{end_idx - 1}); stopping.")

                with torch.autocast(device_type='cuda', enabled=False):
                    skip_points = output_keys is not None and 'points' not in output_keys
                    points = None if skip_points else torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

                pred_dict = dict(
                    points=maybe_detach(points), local_points=maybe_detach(local_points),
                    conf=maybe_detach(conf), camera_poses=maybe_detach(camera_poses),
                    local_camera_poses=maybe_detach(None), camera_qvec=maybe_detach(None),
                    local_camera_qvec=maybe_detach(None), metric=maybe_detach(metric),
                )
                if output_keys is not None:
                    pred_dict = {k: v for k, v in pred_dict.items() if k in output_keys}
                # Sample heavy fields at global indices divisible by the output stride.
                # Keep camera poses full resolution for alignment and PGO. The merge uses
                # _strided_* metadata to remove overlaps without shifting frame indices.
                if inference_save_stride > 1:
                    _stride = int(inference_save_stride)
                    _i_start = (-int(start_idx)) % _stride
                    for _key in ("points", "local_points", "conf"):
                        _v = pred_dict.get(_key)
                        if _v is not None and hasattr(_v, "shape") and len(_v.shape) >= 2:
                            # Clone the slice so it does not retain the full window's storage.
                            pred_dict[_key] = _v[:, _i_start::_stride].clone()
                    pred_dict["_strided_stride"] = _stride
                    pred_dict["_strided_i_start"] = _i_start
                    pred_dict["_strided_w_start"] = int(start_idx)
                all_predictions.append(pred_dict)

                if save_window_local_pts_dir:
                    try:
                        # Dump live tensors before output_keys can remove local_points or conf.
                        _win_payload = {
                            "win_idx": int(win_idx),
                            "start_idx": int(start_idx),
                            "end_idx": int(end_idx),
                            "local_points": local_points.detach().cpu() if isinstance(local_points, torch.Tensor) else local_points,
                            "camera_poses": camera_poses.detach().cpu() if isinstance(camera_poses, torch.Tensor) else camera_poses,
                            "conf":         conf.detach().cpu() if isinstance(conf, torch.Tensor) else conf,
                        }
                        _win_path = _os.path.join(
                            save_window_local_pts_dir, f"window_{int(win_idx):04d}.pt"
                        )
                        torch.save(_win_payload, _win_path)
                    except Exception as _exc:
                        import warnings as _w
                        _w.warn(f"save_window_local_pts_dir write failed @ win {win_idx}: {_exc}",
                                RuntimeWarning)

                # Align the new window for the chord/arc filter.
                if streaming_loop_detector is not None:
                    _t_ld_align = _now_sync()
                    try:
                        _update_window_transform(int(win_idx))
                    except Exception as _exc:
                        import warnings as _w
                        _w.warn(f"window SE(3) align failed @ win {win_idx}: {_exc}",
                                RuntimeWarning)
                    _t_loop_detect += _now_sync() - _t_ld_align

                # Filter candidates using aligned centres, including the current window.
                if streaming_loop_detector is not None and new_pairs_pending:
                    _t_ld_filt = _now_sync()
                    _geom_pass: dict = {}

                    def _passes_chord_arc(small_w: int, big_w: int) -> bool:
                        """Compare chord length with path length using one centre per window.

                        Map each window's first camera centre to the global frame. Sampling by
                        window reduces the effect of pose noise on small per-frame movements.
                        """
                        if (lc_chord_arc_thresh is None
                                or lc_chord_arc_thresh <= 0.0
                                or lc_chord_arc_thresh >= 1.0):
                            return True
                        if not all_predictions or small_w >= len(all_predictions):
                            return True
                        cap_end = min(big_w, len(all_predictions) - 1)
                        if cap_end <= small_w:
                            return True
                        import numpy as _np_filt
                        samples = []
                        for k in range(small_w, cap_end + 1):
                            cams = all_predictions[k].get("camera_poses", None)
                            if cams is None:
                                return True
                            p = cams[0, 0, :3, 3]                # window k's first frame
                            if isinstance(p, torch.Tensor):
                                p = p.detach().cpu().numpy()
                            p = _np_filt.asarray(p, dtype=_np_filt.float64)
                            if k < len(window_transforms):
                                _R, _t = window_transforms[k]
                                p = _R @ p + _t                  # to global frame
                            samples.append(p)
                        if len(samples) < 2:
                            return True
                        samples = _np_filt.stack(samples, axis=0)   # (n_win, 3)
                        chord = float(_np_filt.linalg.norm(samples[-1] - samples[0]))
                        arc = float(_np_filt.linalg.norm(_np_filt.diff(samples, axis=0), axis=1).sum())
                        if arc <= 0.0:
                            return True
                        return (chord / arc) < lc_chord_arc_thresh

                    for fi, fj, _sim in new_pairs_pending:
                        lc_frame_pairs.append((int(fi), int(fj)))
                        wi = _frame_to_window_idx(int(fi))
                        wj = _frame_to_window_idx(int(fj))
                        if abs(wi - wj) <= 1:
                            continue
                        big_w, small_w = (wi, wj) if wi > wj else (wj, wi)
                        gkey = (small_w, big_w)
                        if gkey not in _geom_pass:
                            _geom_pass[gkey] = _passes_chord_arc(small_w, big_w)
                        if not _geom_pass[gkey]:
                            continue
                        bucket = lc_pairs.setdefault(big_w, [])
                        if small_w not in bucket:
                            bucket.append(small_w)
                        lc_frame_matches.setdefault((big_w, small_w), []).append([int(fi), int(fj)])
                        lc_frame_pairs_kept.append((int(fi), int(fj), float(_sim)))
                    _t_loop_detect += _now_sync() - _t_ld_filt

                torch.cuda.empty_cache()

                # Inject bridge windows to front of queue so they run before j+1.
                if lc_pairs is not None and win_idx in lc_pairs:
                    cand = list(lc_pairs[win_idx])
                    if lc_bridge_nms > 0 and len(cand) > 1:
                        n_edges = {i: len(lc_frame_matches.get((win_idx, i), [])) for i in cand}
                        kept_w = []
                        for i in sorted(cand, key=lambda i: (-n_edges[i], i)):
                            if all(abs(i - k) > lc_bridge_nms for k in kept_w):
                                kept_w.append(i)
                        n_bridges_dropped += len(cand) - len(kept_w)
                        cand = sorted(kept_w)
                    n_bridges_run += len(cand)
                    for i_win_idx in reversed(cand):
                        i_start, i_end = windows[i_win_idx]
                        fm = lc_frame_matches.get((win_idx, i_win_idx), []) if lc_frame_matches else []
                        win_queue.appendleft(("bridge", win_idx, i_win_idx, i_start, i_end, end_idx, fm))

            else:  # bridge: camera head only → lc_bridge_cams
                with torch.no_grad():
                    camera_hidden_br = self.camera_decoder(hidden, xpos=pos).float()
                    br_cam_poses = self.camera_head(
                        camera_hidden_br[:, self.patch_start_idx:],
                        patch_h, patch_w,
                    ).reshape(B, Nw, 4, 4)

                _j_end_frames = list(range(j_end - overlap_size, j_end))
                lc_bridge_cams[(j_win_idx, i_win_idx)] = {
                    "cam_poses": br_cam_poses.detach().cpu(),
                    "selected_i_frames": _selected_i,
                    "selected_j_frames": _selected_j,
                    "j_end_frames": _j_end_frames,
                    "frame_matches": frame_matches,
                    "overlap_size": overlap_size,
                }

                if save_bridge_pts:
                    with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
                        point_hidden_br = self.point_decoder(hidden, xpos=pos)
                        conf_hidden_br  = self.conf_decoder(hidden, xpos=pos) if self.use_conf and self.conf_decoder is not None else None
                        ret = self.point_head([point_hidden_br[:, self.patch_start_idx:].float()], (H, W)).reshape(B, Nw, H, W, -1)
                        xy_br, z_br = ret.split([2, 1], dim=-1)
                        z_br = torch.exp(z_br)
                        local_pts_br = torch.cat([xy_br * z_br, z_br], dim=-1)
                        conf_br = self.conf_head([conf_hidden_br[:, self.patch_start_idx:].float()], (H, W)).reshape(B, Nw, H, W, -1) if conf_hidden_br is not None and self.conf_head is not None else None
                        world_pts_br = torch.einsum('bnij, bnhwj -> bnhwi', br_cam_poses, homogenize_points(local_pts_br))[..., :3]
                    # j_end overlap frames come first and last; middle is [i_frames | j_frames]
                    n_j_end = overlap_size
                    br_all_frames = _j_end_frames + _selected_i + _selected_j + _j_end_frames
                    lc_bridge_pts[(j_win_idx, i_win_idx)] = {
                        "points":      world_pts_br.squeeze(0).cpu().float(),   # (Nw, H, W, 3)
                        "conf":        torch.sigmoid(conf_br).squeeze(0).cpu().float() if conf_br is not None else None,
                        "camera_poses": br_cam_poses.squeeze(0).cpu().float(),  # (Nw, 4, 4)
                        "all_frames":  br_all_frames,
                        "n_j_end":     n_j_end,
                        "selected_i_frames": _selected_i,
                        "selected_j_frames": _selected_j,
                    }

                torch.cuda.empty_cache()
                print(f"[LC] bridge window for pair j={j_win_idx} ← i={i_win_idx}, "
                      f"n_i={len(_selected_i)} n_j={len(_selected_j)} Nw={Nw}")

                # Online PGO: if this is the last bridge in the current batch
                # (the next queued item is a normal window or the queue is empty),
                # run merge + PGO on everything processed so far.
                if online_pgo and (not win_queue or win_queue[0][0] != "bridge"):
                    n_done = len(all_predictions)
                    if n_done > 0:
                        print(
                            f"[LC-PGO] starting online PGO after bridge batch "
                            f"(n_windows={n_done}, n_loops={len(lc_bridge_cams)})",
                            flush=True,
                        )
                        _pgo_timer.start()
                        try:
                            _windows_subset = windows[:n_done]
                            # Merge only camera poses for inline PGO to avoid allocating point clouds.
                            # The final merge keeps all output fields and applies the PGO delta to points.
                            _light_predictions = [
                                {"camera_poses": _p.get("camera_poses"),
                                 "local_camera_poses": _p.get("local_camera_poses"),
                                 "metric": _p.get("metric")}
                                for _p in all_predictions
                            ]
                            _partial = _build_merged(_light_predictions, _windows_subset)
                            if _partial.get("camera_poses") is not None:
                                _run_online_pgo_inc(
                                    _partial, all_predictions, _windows_subset,
                                )
                                print(
                                    f"[LC-PGO] finished online PGO "
                                    f"(n_windows={n_done}, n_loops={len(lc_bridge_cams)})",
                                    flush=True,
                                )
                                _maybe_save_pgo_snapshot(
                                    _partial, n_done, len(lc_bridge_cams), _windows_subset,
                                )
                            del _partial, _light_predictions
                            import gc as _gc; _gc.collect()
                        except Exception as _exc:
                            print(f"[LC-PGO] online PGO failed: {_exc}", flush=True)
                        _pgo_timer.stop()

        windows_pbar.close()

        # loop_vis/loop_traj.png is written after the final PGO
        # (below), so the plot can also show the loops the PGO check rejected.
        if streaming_loop_detector is not None:
            # detected_loops.txt: the loop pairs that survived the inline chord/arc
            # filter (the ones used for bridges), one "index1, index2, sim" per line.
            try:
                _det_path = _os.path.join(loop_result_dir, "detected_loops.txt")
                _orig_count, _kept_count = len(lc_frame_pairs), len(lc_frame_pairs_kept)
                with open(_det_path, "w", encoding="utf-8") as _fh:
                    _fh.write(f"# Detected loops (chord/arc < {lc_chord_arc_thresh}): {_kept_count} "
                              f"of {_orig_count} SALAD pairs kept\n")
                    _fh.write("# Format: index1, index2, sim\n")
                    for _fi, _fj, _s in lc_frame_pairs_kept:
                        _fh.write(f"{_fi}, {_fj}, {_s:.4f}\n")
                print(
                    f"[LC-filter] wrote {_kept_count}/{_orig_count} surviving "
                    f"loop pairs -> {_det_path}",
                    flush=True,
                )
                print(f"[LC-bridge] ran {n_bridges_run} bridge windows; NMS dropped "
                      f"{n_bridges_dropped} (lc_bridge_nms={lc_bridge_nms}, "
                      f"keep_state={lc_bridge_keep_state})", flush=True)
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"writing detected_loops.txt failed: {exc}",
                    RuntimeWarning,
                )

        # Merge all windows with the selected alignment.
        merged = _build_merged(all_predictions, windows)

        # Release per-window points and confidence after merging.
        # PGO only needs their camera poses.
        for _pred in all_predictions:
            if isinstance(_pred, dict):
                _pred.pop("points", None)
                _pred.pop("conf", None)
                _pred.pop("local_points", None)
                _pred.pop("_local_points_raw", None)
        import gc as _gc; _gc.collect()
        if all_gate_scales:
            merged["avg_gate_scale"] = torch.stack(all_gate_scales).mean()
        if all_attn_gate_scales:
            merged["attn_gate_scale"] = torch.stack(all_attn_gate_scales).mean()
        if lc_bridge_pts:
            merged["lc_bridge_pts"] = lc_bridge_pts

        # Save raw per-window camera poses (pre-PGO, all frames in each window
        # including overlap duplicates). Shape: (K, max_Nw, 4, 4); short windows
        # are padded with NaN. window_frame_starts/window_frame_ends give the
        # global frame index range for each window.
        try:
            cam_list = []
            valid_lens = []
            for pred in all_predictions:
                c = pred.get("camera_poses", None)
                if c is None:
                    continue
                cs = c.detach()
                if cs.dim() == 4 and cs.shape[0] == 1:
                    cs = cs.squeeze(0)               # (Nw, 4, 4)
                cam_list.append(cs)
                valid_lens.append(cs.shape[0])
            if cam_list:
                max_len = max(valid_lens)
                if all(L == max_len for L in valid_lens):
                    win_cams = torch.stack(cam_list, dim=0)   # (K, Nw, 4, 4)
                else:
                    padded = []
                    for cs in cam_list:
                        if cs.shape[0] < max_len:
                            pad = torch.full(
                                (max_len - cs.shape[0], 4, 4),
                                float('nan'), device=cs.device, dtype=cs.dtype
                            )
                            cs = torch.cat([cs, pad], dim=0)
                        padded.append(cs)
                    win_cams = torch.stack(padded, dim=0)     # (K, max_Nw, 4, 4)
                # Add leading B=1 so demo_viser's .squeeze(0) consistently strips it
                # (consistent with how the rest of the merged tensors are shaped).
                merged["window_camera_poses"] = win_cams.unsqueeze(0)   # (1, K, max_Nw, 4, 4)
                merged["window_frame_starts"] = torch.tensor(
                    [s for (s, _) in windows[: win_cams.shape[0]]], dtype=torch.int64
                ).unsqueeze(0)                                          # (1, K)
                merged["window_frame_ends"] = torch.tensor(
                    [e for (_, e) in windows[: win_cams.shape[0]]], dtype=torch.int64
                ).unsqueeze(0)                                          # (1, K)
        except Exception as _e:
            print(f"[window_camera_poses] WARNING: skipped saving per-window poses: {_e}")

        if online_pgo:
            # Add any remaining windows in a final online-PGO call. Previously added
            # windows are skipped. If no online graph exists, keep the merged poses.
            if (online_pgo_state.get('graph') is not None
                    and merged.get("camera_poses") is not None):
                print(
                    "[LC-PGO] starting final online PGO step "
                    f"(integrating windows after last loop, n_windows={len(all_predictions)})",
                    flush=True,
                )
                _pgo_timer.start()
                try:
                    merged = _run_online_pgo_inc(merged, all_predictions, windows)
                    print("[LC-PGO] finished final online PGO step", flush=True)
                    _maybe_save_pgo_snapshot(
                        merged, len(all_predictions), len(lc_bridge_cams), windows,
                    )
                except Exception as _exc:
                    print(f"[LC-PGO] final online PGO step failed: {_exc}", flush=True)
                _pgo_timer.stop()
        elif run_pgo and merged.get("camera_poses") is not None:
            with _pgo_timer:
                merged = _run_pgo(merged, all_predictions, windows, debug_save=pgo_debug_save)
        if save_pgo_inputs and _pgo_rec["calls"]:
            _pgo_rec["pgo_params"] = dict(
                dist_thresh=pgo_dist_thresh,
                sigma_R_seq=pgo_sigma_R_seq if pgo_sigma_R_seq is not None else pgo_sigma_seq,
                sigma_t_seq=pgo_sigma_t_seq if pgo_sigma_t_seq is not None else pgo_sigma_seq,
                sigma_R_lc=pgo_sigma_R_lc if pgo_sigma_R_lc is not None else pgo_sigma_lc,
                sigma_t_lc=pgo_sigma_t_lc if pgo_sigma_t_lc is not None else pgo_sigma_lc,
                lc_robust=pgo_lc_robust, lc_robust_k=pgo_lc_robust_k, sigma_match=pgo_sigma_match,
                add_match_constraints=pgo_add_match_constraints,
                add_adj_constraints=pgo_add_adj_constraints,
                lc_check_t=pgo_lc_check_t, lc_check_R=pgo_lc_check_R,
                lc_check_iters=pgo_lc_check_iters)
            _os.makedirs(_os.path.dirname(_os.path.abspath(save_pgo_inputs)), exist_ok=True)
            torch.save(_pgo_rec, save_pgo_inputs)
            print(f"[LC-PGO] saved online-PGO inputs ({len(_pgo_rec['calls'])} calls, "
                  f"{len(_pgo_rec['bridges'])} bridges) -> {save_pgo_inputs}", flush=True)

        # loop_vis/loop_traj.png: pairs that survived the chord/arc filter, with the
        # loop edges rejected by the PGO consistency check in red.
        # Pair PNGs were saved incrementally per window via visualize_new_pairs.
        if streaming_loop_detector is not None:
            _t_ld_save = _now_sync()
            _rej = sorted(online_pgo_state.get('lc_rejected', ()))
            _rej_info = online_pgo_state.get('lc_rejected_info', {})
            _lc_fac = online_pgo_state.get('lc_factors', {})
            _rej_pairs = [(int(f), int(f2)) for k in _rej for _, f, f2, _ in _lc_fac.get(k, [])]
            try:
                streaming_loop_detector.save_streaming_visualization(
                    plot_loops=lc_frame_pairs_kept, rejected_pairs=_rej_pairs,
                    n_rejected_bridges=len(_rej), n_bridges=len(_lc_fac))
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"streaming loop detector finalize failed: {exc}",
                    RuntimeWarning,
                )
            if pgo_lc_check_t > 0 or pgo_lc_check_R > 0:
                try:
                    with open(_os.path.join(loop_result_dir, "pgo_check_rejected.txt"), "w") as _fh:
                        _fh.write(f"# Bridges rejected by the online-PGO loop consistency check "
                                  f"(pgo_lc_check_t={pgo_lc_check_t}, pgo_lc_check_R={pgo_lc_check_R})\n")
                        _fh.write(f"# Rejected: {len(_rej)} / {len(_lc_fac)} bridges, "
                                  f"{len(_rej_pairs)} loop edges\n")
                        _fh.write("# Format: j_window, i_window, frame_j_start, frame_i_start, "
                                  "median t residual [units], median R residual [deg], n_edges\n")
                        for (j_w, i_w) in _rej:
                            _t_r, _R_r, _ne = _rej_info.get((j_w, i_w), (float('nan'), float('nan'), 0))
                            _fh.write(f"{j_w}, {i_w}, {windows[j_w][0]}, {windows[i_w][0]}, "
                                      f"{_t_r:.4f}, {_R_r:.4f}, {_ne}\n")
                except Exception as exc:
                    print(f"[LC-check] writing pgo_check_rejected.txt failed: {exc}", flush=True)
            _t_loop_detect += _now_sync() - _t_ld_save

        # Return full-resolution camera poses for trajectory export. Bundle saving
        # strides them separately to match the sampled points and confidence.
        _pgo_stats = _pgo_timer.summary()
        merged["_timing"] = {
            "loop_detect": float(_t_loop_detect),
            "pgo":         _pgo_stats["total"],
            "pgo_calls":   _pgo_stats["calls"],
            "pgo_mean":    _pgo_stats["mean"],
            "pgo_max":     _pgo_stats["max"],
            "pgo_per_call": _pgo_stats["per_call"],
            "pgo_rss_retained_mb": _pgo_stats["rss_retained_mb"],
            "pgo_hwm_max_mb":      _pgo_stats["hwm_max_mb"],
            "proc_rss_peak_mb":    _pgo_stats["proc_rss_peak_mb"],
            "pgo_rss_before_mb":   _pgo_stats["rss_before_mb"],
            "pgo_rss_delta_per_call": _pgo_stats["rss_delta_per_call"],
            "pgo_hwm_delta_per_call": _pgo_stats["hwm_delta_per_call"],
            "pgo_rss_after_per_call": _pgo_stats["rss_after_per_call"],
        }
        if _pgo_stats["calls"]:
            print(
                f"[PGO-timing] {_pgo_stats['calls']} call(s), "
                f"total {_pgo_stats['total']:.3f} s, "
                f"mean {_pgo_stats['mean']:.3f} s, "
                f"max {_pgo_stats['max']:.3f} s",
                flush=True,
            )
            print(
                f"[PGO-cpumem] pgo_retained {_pgo_stats['rss_retained_mb']:.1f} MiB, "
                f"pgo_transient {_pgo_stats['hwm_max_mb']:.1f} MiB, "
                f"whole_process_rss_peak {_pgo_stats['proc_rss_peak_mb']:.1f} MiB",
                flush=True,
            )
        return merged

    def _merge_windowed_predictions(self, all_predictions, window_size, overlap_size):
        """Concatenate windows along time and remove overlapping frames.

        For strided points and confidence, use _strided_* metadata to keep global
        indices before the next window starts. Full-resolution fields use the
        usual overlap removal. This keeps sampled fields on the global stride.
        """
        if not all_predictions:
            return {}
        if len(all_predictions) == 1:
            # Single window: still strip metadata keys before returning so they
            # don't leak into the caller's dict.
            single = {k: v for k, v in all_predictions[0].items() if not k.startswith("_strided_")}
            return single

        merged_predictions = {}
        keys = list(all_predictions[0].keys())
        sequence_keys = {"points", "local_points", "conf", "camera_poses", "local_camera_poses", "camera_qvec", "local_camera_qvec"}
        # Subset of sequence_keys that get the per-window stride applied upstream;
        # for these we use stride-aware overlap removal (drop frames at global idx
        # >= next window's w_start) instead of the standard overlap_size drop.
        strided_seq_keys = {"points", "local_points", "conf"}

        # Pre-compute per-window stride metadata. None entry = window has no stride.
        strided_meta = []
        for pred in all_predictions:
            if isinstance(pred, dict):
                stride = pred.get("_strided_stride")
                w_start = pred.get("_strided_w_start")
                if stride and int(stride) > 1 and w_start is not None:
                    strided_meta.append({
                        "stride":  int(stride),
                        "i_start": int(pred.get("_strided_i_start", 0)),
                        "w_start": int(w_start),
                    })
                    continue
            strided_meta.append(None)

        for key in keys:
            # Skip stride metadata — internal-use only, not part of the merged dict.
            if key.startswith("_strided_"):
                continue
            # Collect window tensors
            window_tensors = [pred.get(key, None) for pred in all_predictions]

            # Skip if all windows have None for this key
            if all(t is None for t in window_tensors):
                continue

            # Only perform overlap-aware concatenation for known sequence-shaped tensors
            if key in sequence_keys:
                result_parts = []
                n_windows = len(window_tensors)
                key_is_strided = key in strided_seq_keys

                for i, tensor in enumerate(window_tensors):
                    if tensor is None:
                        continue
                    is_last = (i == n_windows - 1)
                    n_frames = tensor.shape[1]

                    # Decide how many trailing frames to drop.
                    if is_last:
                        drop = 0
                    elif (
                        key_is_strided
                        and i < len(strided_meta)
                        and strided_meta[i] is not None
                        and i + 1 < len(strided_meta)
                        and strided_meta[i + 1] is not None
                    ):
                        # Stride-aware: keep frames whose GLOBAL idx < next w_start.
                        # Strided frame j of window i has global idx
                        #   g_j = w_start_i + i_start_i + j * stride
                        # Keep while g_j < w_start_{i+1}, i.e.
                        #   j < (w_start_{i+1} - w_start_i - i_start_i) / stride
                        m_curr = strided_meta[i]
                        m_next = strided_meta[i + 1]
                        stride = m_curr["stride"]
                        i_start = m_curr["i_start"]
                        delta = m_next["w_start"] - m_curr["w_start"] - i_start
                        keep_count = (delta + stride - 1) // stride if delta > 0 else 0
                        keep_count = max(0, min(keep_count, n_frames))
                        drop = n_frames - keep_count
                    else:
                        # Standard overlap-size drop for non-strided fields
                        # (camera_poses) or windows without stride metadata.
                        if overlap_size > 0 and n_frames > overlap_size:
                            drop = overlap_size
                        elif overlap_size > 0 and n_frames <= overlap_size:
                            # Tiny window – drop entirely.
                            drop = n_frames
                        else:
                            drop = 0

                    keep = n_frames - drop
                    if keep > 0:
                        result_parts.append(tensor[:, :keep])
                    # else drop entirely

                if result_parts:
                    merged_predictions[key] = torch.cat(result_parts, dim=1)
                else:
                    # Fallback: if everything was dropped due to tiny windows, keep last non-None
                    for t in reversed(window_tensors):
                        if t is not None:
                            merged_predictions[key] = t
                            break
            else:
                # Non-sequence keys: keep the last non-None
                for t in reversed(window_tensors):
                    if t is not None:
                        merged_predictions[key] = t
                        break

        # Instead of computing overlap losses here, export overlap prev/next tensors for trainer-side chunk losses
        if overlap_size > 0 and len(all_predictions) > 1:
            prev_cam_chunks = []
            next_cam_chunks = []
            prev_pcd_chunks = []
            next_pcd_chunks = []
            next_conf_chunks = []

            for i in range(len(all_predictions) - 1):
                pred_a = all_predictions[i]
                pred_b = all_predictions[i + 1]

                cam_a = pred_a.get("camera_poses", None)
                cam_b = pred_b.get("camera_poses", None)
                lpts_a = pred_a.get("local_points", None)
                lpts_b = pred_b.get("local_points", None)
                conf_a = pred_a.get("conf", None)
                conf_b = pred_b.get("conf", None)

                # Only collect when both sides have enough frames for a full overlap window
                if cam_a is not None and cam_b is not None and cam_a.shape[1] >= overlap_size and cam_b.shape[1] >= overlap_size:
                    S_a = cam_a.shape[1]
                    # Take last overlap_size from A and first overlap_size from B
                    prev_cam_chunks.append(cam_a[:, S_a - overlap_size: S_a])  # (B, O, 4, 4)
                    next_cam_chunks.append(cam_b[:, 0: overlap_size])         # (B, O, 4, 4)

                if lpts_a is not None and lpts_b is not None and lpts_a.shape[1] >= overlap_size and lpts_b.shape[1] >= overlap_size:
                    S_a = lpts_a.shape[1]
                    prev_pcd_chunks.append(lpts_a[:, S_a - overlap_size: S_a])  # (B, O, H, W, 3)
                    next_pcd_chunks.append(lpts_b[:, 0: overlap_size])          # (B, O, H, W, 3)
                    if conf_b is not None and conf_b.shape[1] >= overlap_size:
                        next_conf_chunks.append(conf_b[:, 0: overlap_size].squeeze(-1))  # (B, O, H, W)

            # Stack along a new chunk dimension if any collected
            if prev_cam_chunks and next_cam_chunks:
                merged_predictions["overlap_prev_cam"] = torch.stack(prev_cam_chunks, dim=1)  # (B, K, O, 4, 4)
                merged_predictions["overlap_next_cam"] = torch.stack(next_cam_chunks, dim=1)  # (B, K, O, 4, 4)
            if prev_pcd_chunks and next_pcd_chunks:
                merged_predictions["overlap_prev_pcd"] = torch.stack(prev_pcd_chunks, dim=1)  # (B, K, O, H, W, 3)
                merged_predictions["overlap_next_pcd"] = torch.stack(next_pcd_chunks, dim=1)  # (B, K, O, H, W, 3)
                if next_conf_chunks:
                    merged_predictions["overlap_next_conf"] = torch.stack(next_conf_chunks, dim=1)  # (B, K, O, H, W)

        return merged_predictions

    @staticmethod
    def _compute_reference_scales(all_predictions, mode: str = "w0"):
        """Per-window scale anchored to a reference raw-camera path length.

        mode:
          - "w0"       : reference = window 0's raw path length.
          - "max"      : reference = max raw path length across all windows.
          - "p75"      : reference = 75th percentile raw path length.
          - "median"   : reference = median raw path length.
          - "run_max"  : running (causal) max with decay 0.95 per window step.

        Returns list[float] s_k such that in the merge loop `forced_scale = s_k / s_0`
        becomes `reference / pl_k` (ref stretched to match each k). Non-compounding.
        """
        pls = []
        for pred in all_predictions:
            cam = pred.get("camera_poses", None) if isinstance(pred, dict) else None
            if cam is None or cam.shape[1] < 2:
                pls.append(None); continue
            t = cam[0, :, :3, 3].detach().cpu().float()
            steps = torch.linalg.norm(t[1:] - t[:-1], dim=-1)
            pl = float(steps.sum())
            pls.append(pl if pl > 1e-12 else None)
        finite_pls = [p for p in pls if p is not None and p == p]  # NaN-filter
        if not finite_pls:
            return [1.0] * len(pls)

        if mode == "w0":
            ref_k = [pls[0] if pls[0] is not None else finite_pls[0]] * len(pls)
        elif mode == "max":
            ref_val = max(finite_pls); ref_k = [ref_val] * len(pls)
        elif mode == "p75":
            import numpy as _np
            ref_val = float(_np.percentile(finite_pls, 75)); ref_k = [ref_val] * len(pls)
        elif mode == "median":
            import numpy as _np
            ref_val = float(_np.median(finite_pls)); ref_k = [ref_val] * len(pls)
        elif mode == "run_max":
            # causal running max with decay 0.95 per window step
            ref_k = []; running = 0.0
            for pl in pls:
                val = pl if pl is not None else running
                running = max(running * 0.95, val)
                if running < 1e-12: running = val if (val and val > 1e-12) else 1.0
                ref_k.append(running)
        else:
            raise ValueError(f"Unknown reference mode '{mode}'")

        # Output scales[k] such that scales[k]/scales[0] = ref_k[k] / pl_k
        # Choose scales[0] = ref_k[0], scales[k] = ref_k[0] * (ref_k[k] / pl_k).
        s_out = []
        scale_zero = ref_k[0] if ref_k[0] else 1.0
        for k, (pl, r) in enumerate(zip(pls, ref_k)):
            if pl is None or pl < 1e-12:
                s_out.append(scale_zero)   # → forced_scale = 1
            else:
                s_out.append(scale_zero * (r / pl))
        return s_out

    @staticmethod
    def _compute_w0_reference_scales(all_predictions):
        """Scale each window's camera path length relative to window 0.

        Return s_k so s_k / s_0 = path_length(w_0) / path_length(w_k).
        Zero or NaN path lengths use scale 1. This assumes similar motion across
        windows and may not suit stop-and-go sequences.
        """
        scales = []
        ref_pl = None
        for k, pred in enumerate(all_predictions):
            cam = pred.get("camera_poses", None) if isinstance(pred, dict) else None
            if cam is None or cam.shape[1] < 2:
                scales.append(1.0 if ref_pl is None else ref_pl); continue
            t = cam[0, :, :3, 3].detach().cpu().float()
            steps = torch.linalg.norm(t[1:] - t[:-1], dim=-1)
            pl = float(steps.sum())
            if ref_pl is None:
                ref_pl = pl if pl > 1e-12 else 1.0
                scales.append(ref_pl)
                continue
            if pl < 1e-12 or not (pl == pl):  # NaN check
                scales.append(ref_pl)
            else:
                # store 'scale to GT'-analogue: s_k s.t. s_k/s_0 = pl_0/pl_k => s_k = s_0 * (pl_0 / pl_k)
                scales.append(ref_pl * ref_pl / pl)
        return scales

    @staticmethod
    def _compute_oracle_window_scales(all_predictions, windows, gt_poses):
        """Return per-window Umeyama-Sim3 scale of estimated camera centres vs GT.

        gt_poses: (N, 4, 4) tensor/ndarray of camera-to-world poses (metric).
        windows:  list of (start, end) global frame indices.
        all_predictions[k]["camera_poses"]: (B, L, 4, 4) for window k.

        Returns list[float] of length len(windows). Windows where the fit is
        ill-posed (L<4 or non-finite) get scale 1.0.
        """
        import numpy as _np
        if isinstance(gt_poses, torch.Tensor):
            gt_np = gt_poses.detach().cpu().float().numpy()
        else:
            gt_np = _np.asarray(gt_poses, dtype=_np.float64)
        if gt_np.ndim != 3 or gt_np.shape[-2:] != (4, 4):
            raise ValueError(f"gt_poses_for_scale must be (N,4,4); got shape {gt_np.shape}")
        gt_t = gt_np[:, :3, 3].astype(_np.float64)

        def _umeyama_scale(src, dst):
            sm = src.mean(0); dm = dst.mean(0)
            sc = src - sm; dc = dst - dm
            H = sc.T @ dc / len(src)
            U, D, Vt = _np.linalg.svd(H)
            S = _np.eye(3)
            if _np.linalg.det(U) * _np.linalg.det(Vt) < 0:
                S[2, 2] = -1
            var = (sc ** 2).sum() / len(src)
            if var < 1e-20:
                return 1.0
            return float((D * _np.diag(S)).sum() / var)

        scales = []
        for k, (start, end) in enumerate(windows):
            cam = all_predictions[k].get("camera_poses", None) if k < len(all_predictions) else None
            if cam is None:
                scales.append(1.0); continue
            raw_t = cam[0, :, :3, 3].detach().cpu().float().numpy().astype(_np.float64)
            gt_seg = gt_t[start:end]
            L = min(raw_t.shape[0], gt_seg.shape[0])
            if L < 4:
                scales.append(1.0); continue
            try:
                s = _umeyama_scale(raw_t[:L], gt_seg[:L])
                if not _np.isfinite(s) or s <= 0:
                    s = 1.0
            except Exception:
                s = 1.0
            scales.append(s)
        return scales

    def _merge_windowed_predictions_sim3(
        self,
        all_predictions,
        allow_scale: bool = True,
        scale_mode: str = 'median',
        reset_every: int = 0,
        reuse_transform_within_reset_block: bool = False,
        oracle_window_scales: Optional[List[float]] = None,
    ):
        """
        Merge windowed predictions by estimating relative poses between overlaps.
        When ``allow_scale`` is True this performs Sim(3) alignment (scale+SE(3));
        when False it reduces to SE(3) alignment by keeping the scale fixed to 1.
        If ``reuse_transform_within_reset_block`` is enabled with ``reset_every > 0``,
        one transform is estimated at each reset boundary and reused for the rest of
        that reset block.
        """
        if not all_predictions:
            return {}
        if len(all_predictions) == 1:
            return all_predictions[0]

        # Locate a reference tensor to determine batch/device/dtype information
        sample_tensor = None
        for pred in all_predictions:
            for key in ("points", "camera_poses", "local_points", "conf"):
                tensor = pred.get(key, None)
                if tensor is not None:
                    sample_tensor = tensor
                    break
            if sample_tensor is not None:
                break
        if sample_tensor is None:
            raise ValueError("Sim3 merge requires at least one tensor prediction")

        device = sample_tensor.device
        dtype = sample_tensor.dtype
        batch_size = sample_tensor.shape[0]

        identity_rot = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        zero_trans = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        one_scale = torch.ones(batch_size, device=device, dtype=dtype)

        aligned_predictions: List[dict] = []
        sim3_scales: Optional[List[torch.Tensor]] = [] if allow_scale else None
        sim3_poses: List[torch.Tensor] = []

        window_size = getattr(self, "_last_window_size", -1)
        overlap_size = getattr(self, "_last_overlap_size", 0)

        def _estimate_relative_sim3(prev_aligned: dict, curr_raw: dict, overlap: int, current_allow_scale: bool, forced_scale: Optional[torch.Tensor] = None):
            if overlap <= 0:
                return torch.ones_like(one_scale), identity_rot, zero_trans

            prev_cam = prev_aligned.get("camera_poses", None)
            curr_cam = curr_raw.get("camera_poses", None)
            if prev_cam is None or curr_cam is None or prev_cam.shape[1] == 0 or curr_cam.shape[1] == 0:
                return torch.ones_like(one_scale), identity_rot, zero_trans

            prev_frames = prev_cam.shape[1]
            prev_idx = max(prev_frames - overlap, 0)

            prev_pose = prev_cam[:, prev_idx]
            curr_pose = curr_cam[:, 0]

            R_prev = prev_pose[:, :3, :3]
            t_prev = prev_pose[:, :3, 3]
            R_curr = curr_pose[:, :3, :3]
            t_curr = curr_pose[:, :3, 3]

            relative_rot = torch.matmul(R_prev, R_curr.transpose(-1, -2))

            relative_scale = torch.ones_like(one_scale)
            if forced_scale is not None:
                relative_scale = forced_scale
            elif current_allow_scale and scale_mode == 'translation_magnitude':
                # Estimate scale from motion over the same overlap frames:
                # scale = sum(||prev_t[i+1] - prev_t[i]||) / sum(||curr_t[i+1] - curr_t[i]||).
                overlap_n = min(overlap, prev_cam.shape[1] - prev_idx, curr_cam.shape[1])
                if overlap_n >= 2:
                    prev_t_overlap = prev_cam[:, prev_idx : prev_idx + overlap_n, :3, 3].to(torch.float32)
                    curr_t_overlap = curr_cam[:, :overlap_n, :3, 3].to(torch.float32)
                    prev_steps = torch.linalg.norm(prev_t_overlap[:, 1:] - prev_t_overlap[:, :-1], dim=-1)  # (B, O-1)
                    curr_steps = torch.linalg.norm(curr_t_overlap[:, 1:] - curr_t_overlap[:, :-1], dim=-1)
                    prev_total = prev_steps.sum(-1)  # (B,)
                    curr_total = curr_steps.sum(-1)
                    eps_tm = 1e-12
                    valid_tm = (curr_total > eps_tm) & torch.isfinite(prev_total) & torch.isfinite(curr_total)
                    ratio_tm = prev_total / torch.clamp(curr_total, min=eps_tm)
                    relative_scale = torch.where(valid_tm, ratio_tm.to(dtype), torch.ones_like(one_scale))
                    relative_scale = torch.clamp(relative_scale, min=1e-3, max=1e3)
                else:
                    relative_scale = torch.ones_like(one_scale)
            elif current_allow_scale and scale_mode == 'pointcloud_umeyama':
                # Fit Sim(3) between overlap point clouds in each window's local world.
                # Sample one point per 14x14 patch to limit the fitting cost.
                prev_local_raw = prev_aligned.get("local_points", None)
                if prev_local_raw is None:
                    prev_local_raw = prev_aligned.get("_local_points_raw", None)
                curr_local_raw = curr_raw.get("local_points", None)
                overlap_n_pc = min(overlap, prev_cam.shape[1] - prev_idx, curr_cam.shape[1])
                if (
                    prev_local_raw is not None and curr_local_raw is not None
                    and overlap_n_pc >= 2
                    and prev_local_raw.shape[1] > prev_idx
                    and curr_local_raw.shape[1] > 0
                ):
                    stride_pc = 14
                    prev_pts_local = prev_local_raw[:, prev_idx : prev_idx + overlap_n_pc, ::stride_pc, ::stride_pc, :]
                    curr_pts_local = curr_local_raw[:, :overlap_n_pc, ::stride_pc, ::stride_pc, :]
                    prev_cam_overlap = prev_cam[:, prev_idx : prev_idx + overlap_n_pc]
                    curr_cam_overlap = curr_cam[:, :overlap_n_pc]
                    # to world: cam @ homog(local)
                    prev_homog = homogenize_points(prev_pts_local)
                    curr_homog = homogenize_points(curr_pts_local)
                    prev_world = torch.einsum('bnij, bnhwj -> bnhwi', prev_cam_overlap.to(prev_homog.dtype), prev_homog)[..., :3]
                    curr_world = torch.einsum('bnij, bnhwj -> bnhwi', curr_cam_overlap.to(curr_homog.dtype), curr_homog)[..., :3]
                    prev_flat = prev_world.reshape(batch_size, -1, 3).to(torch.float32)
                    curr_flat = curr_world.reshape(batch_size, -1, 3).to(torch.float32)
                    scale_values = []
                    eps_pc = 1e-12
                    for b in range(batch_size):
                        src = curr_flat[b]  # scale * R * src + t = dst
                        dst = prev_flat[b]
                        valid_pc = torch.isfinite(src).all(-1) & torch.isfinite(dst).all(-1)
                        if valid_pc.sum() < 4:
                            scale_values.append(torch.tensor(1.0, device=device, dtype=torch.float32))
                            continue
                        src = src[valid_pc]; dst = dst[valid_pc]
                        sm = src.mean(0); dm = dst.mean(0)
                        sc = src - sm; dc = dst - dm
                        var = (sc ** 2).sum() / len(src)
                        if var < eps_pc:
                            scale_values.append(torch.tensor(1.0, device=device, dtype=torch.float32))
                            continue
                        Hmat = sc.T @ dc / len(src)
                        try:
                            U, D, Vt = torch.linalg.svd(Hmat)
                        except Exception:
                            scale_values.append(torch.tensor(1.0, device=device, dtype=torch.float32))
                            continue
                        S_ = torch.eye(3, device=device, dtype=torch.float32)
                        if torch.det(U) * torch.det(Vt) < 0:
                            S_[2, 2] = -1
                        s_val = (D * torch.diagonal(S_)).sum() / var
                        scale_values.append(s_val)
                    relative_scale = torch.stack(scale_values).to(dtype)
                    relative_scale = torch.clamp(relative_scale, min=1e-3, max=1e3)
                else:
                    relative_scale = torch.ones_like(one_scale)
            elif current_allow_scale:
                prev_local_raw = prev_aligned.get("local_points", None)
                if prev_local_raw is None:
                    prev_local_raw = prev_aligned.get("_local_points_raw", None)
                curr_local_raw = curr_raw.get("local_points", None)

                if (
                    prev_local_raw is not None
                    and curr_local_raw is not None
                    and prev_local_raw.shape[1] > prev_idx
                    and curr_local_raw.shape[1] > 0
                ):
                    if scale_mode in ['median_all', 'trimmed_mean_all']:
                        # Use all overlapping frames
                        actual_overlap = min(overlap, prev_local_raw.shape[1] - prev_idx, curr_local_raw.shape[1])
                        if actual_overlap > 0:
                            prev_depth = prev_local_raw[:, prev_idx : prev_idx + actual_overlap, ..., 2]
                            curr_depth = curr_local_raw[:, :actual_overlap, ..., 2]
                        else:
                            # Fallback to single frame if overlap calculation fails (should not happen given checks above)
                            prev_depth = prev_local_raw[:, prev_idx, ..., 2]
                            curr_depth = curr_local_raw[:, 0, ..., 2]
                    else:
                        # Use only the first overlapping frame (standard behavior)
                        prev_depth = prev_local_raw[:, prev_idx, ..., 2]
                        curr_depth = curr_local_raw[:, 0, ..., 2]

                    prev_depth_f32 = prev_depth.to(torch.float32)
                    curr_depth_f32 = curr_depth.to(torch.float32)
                    eps_depth = torch.finfo(torch.float32).eps
                    valid = (
                        torch.isfinite(prev_depth_f32)
                        & torch.isfinite(curr_depth_f32)
                        & (curr_depth_f32.abs() > eps_depth)
                    )

                    prev_depth_flat = prev_depth_f32.reshape(batch_size, -1)
                    curr_depth_flat = curr_depth_f32.reshape(batch_size, -1)
                    valid_flat = valid.reshape(batch_size, -1)
                    
                    if scale_mode in ['median', 'median_all']:
                        scale_values = []
                        for b in range(batch_size):
                            valid_idx = valid_flat[b]
                            if valid_idx.any():
                                ratios = prev_depth_flat[b, valid_idx] / curr_depth_flat[b, valid_idx]
                                scale_values.append(ratios.median())
                            else:
                                scale_values.append(torch.tensor(1.0, device=device, dtype=torch.float32))
                        relative_scale = torch.stack(scale_values).to(dtype)
                    elif scale_mode in ['trimmed_mean', 'trimmed_mean_all']:
                        # Compute trimmed means per batch because valid pixel counts vary.
                        
                        # To keep it simple and consistent with the median loop structure for now (which handles varying valid counts per batch):
                        scale_values = []
                        for b in range(batch_size):
                            valid_idx = valid_flat[b]
                            if valid_idx.any():
                                ratios = prev_depth_flat[b, valid_idx] / curr_depth_flat[b, valid_idx]
                                # Add the batch dimension expected by robust_scale_estimation.
                                scale_val = robust_scale_estimation(ratios.unsqueeze(0), trim_ratio=0.25).squeeze(0)
                                scale_values.append(scale_val)
                            else:
                                scale_values.append(torch.tensor(1.0, device=device, dtype=torch.float32))
                        relative_scale = torch.stack(scale_values).to(dtype)
                    elif scale_mode in ['sim3_avg1']:
                        scale_values = []
                        for b in range(batch_size):
                            valid_idx = valid_flat[b]
                            if valid_idx.any():
                                ratios = prev_depth_flat[b, valid_idx] / curr_depth_flat[b, valid_idx]
                                scale_values.append(ratios.median())
                            else:
                                scale_values.append(torch.tensor(1.0, device=device, dtype=torch.float32))
                        relative_scale = torch.stack(scale_values).to(dtype)
                        relative_scale = (relative_scale + 1.0) / 2.0
                    else:
                        raise ValueError(f"Unknown scale_mode: {scale_mode}")

                    relative_scale = torch.clamp(relative_scale, min=1e-3, max=1e3)

            rotated_curr_centers = torch.matmul(relative_rot, t_curr.unsqueeze(-1)).squeeze(-1)
            relative_trans = t_prev - relative_scale.unsqueeze(-1) * rotated_curr_centers

            return relative_scale, relative_rot.to(dtype), relative_trans.to(dtype)

        block_scale: Optional[torch.Tensor] = None
        block_rot: Optional[torch.Tensor] = None
        block_trans: Optional[torch.Tensor] = None

        # Capture overlap pcd/conf chunks before the per-iteration pop of
        # local_points (which is done below for memory). Without this, the
        # downstream _merge_windowed_predictions sees local_points=None on
        # all but the last window and cannot build overlap_*_pcd tensors.
        overlap_prev_pcd_chunks: List[torch.Tensor] = []
        overlap_next_pcd_chunks: List[torch.Tensor] = []
        overlap_prev_conf_chunks: List[torch.Tensor] = []
        overlap_next_conf_chunks: List[torch.Tensor] = []

        for window_idx, pred in enumerate(all_predictions):
            if window_idx == 0:
                current_scale = torch.ones_like(one_scale)
                current_rot = identity_rot.clone()
                current_trans = zero_trans.clone()
                if reuse_transform_within_reset_block and reset_every > 0:
                    block_scale = current_scale.clone()
                    block_rot = current_rot.clone()
                    block_trans = current_trans.clone()
            else:
                prev_aligned = aligned_predictions[-1]
                reuse_block_transform = (
                    reuse_transform_within_reset_block
                    and reset_every > 0
                    and window_idx % reset_every != 0
                    and block_rot is not None
                    and block_trans is not None
                )
                if reuse_block_transform:
                    current_rot = block_rot.clone()
                    current_trans = block_trans.clone()
                    if allow_scale and block_scale is not None:
                        current_scale = block_scale.clone()
                    else:
                        current_scale = torch.ones_like(one_scale)
                else:
                    forced = None
                    if (
                        oracle_window_scales is not None
                        and allow_scale
                        and window_idx < len(oracle_window_scales)
                        and len(oracle_window_scales) > 0
                        and oracle_window_scales[0] > 0
                    ):
                        ratio = float(oracle_window_scales[window_idx]) / float(oracle_window_scales[0])
                        import math as _math
                        if _math.isfinite(ratio) and ratio > 0:
                            forced = torch.full(
                                (batch_size,), ratio, device=device, dtype=dtype
                            )
                    current_scale, current_rot, current_trans = _estimate_relative_sim3(
                        prev_aligned, pred, overlap_size, allow_scale, forced_scale=forced,
                    )
                    if reuse_transform_within_reset_block and reset_every > 0:
                        block_scale = current_scale.clone()
                        block_rot = current_rot.clone()
                        block_trans = current_trans.clone()

            if allow_scale and sim3_scales is not None:
                sim3_scales.append(current_scale.clone())
            pose_mat = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
            pose_mat[:, :3, :3] = current_rot
            pose_mat[:, :3, 3] = current_trans
            sim3_poses.append(pose_mat)

            aligned_pred: dict = {}

            original_local_points = pred.get("local_points", None)
            aligned_pred["_local_points_raw"] = original_local_points

            if original_local_points is not None:
                if allow_scale: # Keep using global allow_scale for applying scale if we have it, or maybe we should track per-window scale application?
                    # Actually, current_scale will be 1.0 if current_allow_scale was False.
                    # So we can just always apply current_scale.
                    scale_factor = current_scale.view(batch_size, 1, 1, 1, 1)
                    aligned_local_points = original_local_points * scale_factor
                else:
                    aligned_local_points = original_local_points
            else:
                aligned_local_points = None
            aligned_pred["local_points"] = aligned_local_points

            def _transform_camera(cam_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                if cam_tensor is None:
                    return None
                frames = cam_tensor.shape[1]
                rot_local = cam_tensor[..., :3, :3]
                trans_local = cam_tensor[..., :3, 3]
                rot_global = torch.matmul(
                    current_rot.unsqueeze(1).expand(-1, frames, -1, -1),
                    rot_local
                )
                rotated_trans = torch.matmul(
                    current_rot.unsqueeze(1).expand(-1, frames, -1, -1),
                    trans_local.unsqueeze(-1)
                ).squeeze(-1)
                if allow_scale:
                    rotated_trans = rotated_trans * current_scale.view(batch_size, 1, 1)
                trans_global = rotated_trans + current_trans.unsqueeze(1)
                cam_out = cam_tensor.clone()
                cam_out[..., :3, :3] = rot_global
                cam_out[..., :3, 3] = trans_global
                return cam_out

            camera_global = _transform_camera(pred.get("camera_poses", None))
            aligned_pred["camera_poses"] = camera_global

            local_camera_global = _transform_camera(pred.get("local_camera_poses", None))
            aligned_pred["local_camera_poses"] = local_camera_global

            raw_points = pred.get("points", None)
            if raw_points is None and aligned_local_points is None:
                aligned_points = None
            elif raw_points is None and aligned_local_points is not None and camera_global is not None:
                # points were skipped during inference (output_keys); recompute only if needed
                aligned_points = None  # skip — not needed for overlap-only analysis
            elif camera_global is not None and aligned_local_points is not None:
                # Sample camera poses at the same global indices as the strided points.
                _n_lp = aligned_local_points.shape[1]
                _n_cg = camera_global.shape[1]
                if _n_lp == 0:
                    # Preserve the point tensor shape when this window has no sampled frames.
                    aligned_points = aligned_local_points.new_zeros(
                        (aligned_local_points.shape[0], 0,
                         aligned_local_points.shape[2],
                         aligned_local_points.shape[3], 3)
                    )
                else:
                    if _n_lp != _n_cg:
                        _stride_meta = int(pred.get("_strided_stride", 0) or 0)
                        _i_start_meta = int(pred.get("_strided_i_start", 0) or 0)
                        if _stride_meta > 1:
                            # Use the recorded stride and starting offset.
                            _cam_for_pts = camera_global[:, _i_start_meta::_stride_meta][:, :_n_lp]
                        else:
                            # Fallback heuristic if metadata was not propagated
                            # (shouldn't happen under inference_save_stride).
                            _stride_inferred = max(1, round(_n_cg / _n_lp))
                            _cam_for_pts = camera_global[:, ::_stride_inferred][:, :_n_lp]
                    else:
                        _cam_for_pts = camera_global
                    aligned_points = torch.einsum(
                        'bnij, bnhwj -> bnhwi',
                        _cam_for_pts,
                        homogenize_points(aligned_local_points)
                    )[..., :3]
            else:
                if raw_points is not None:
                    rotated_points = torch.einsum('bij, bnhwj -> bnhwi', current_rot, raw_points)
                    if allow_scale:
                        rotated_points = rotated_points * current_scale.view(batch_size, 1, 1, 1, 1)
                    aligned_points = rotated_points + current_trans.view(batch_size, 1, 1, 1, 3)
                else:
                    aligned_points = None
            aligned_pred["points"] = aligned_points

            aligned_pred["conf"] = pred.get("conf", None)

            for key, value in pred.items():
                if key in aligned_pred:
                    continue
                aligned_pred[key] = value

            aligned_predictions.append(aligned_pred)

            # Keep only the latest aligned local points for the next scale estimate.
            # Clear both references so older tensors can be freed.
            if len(aligned_predictions) >= 2:
                older_idx = len(aligned_predictions) - 2
                newer_idx = older_idx + 1
                # Capture the overlap pcd chunk for this boundary before popping.
                if overlap_size > 0:
                    lpts_older = aligned_predictions[older_idx].get("local_points", None)
                    lpts_newer = aligned_predictions[newer_idx].get("local_points", None)
                    conf_older = aligned_predictions[older_idx].get("conf", None)
                    conf_newer = aligned_predictions[newer_idx].get("conf", None)
                    if (
                        lpts_older is not None and lpts_newer is not None
                        and lpts_older.shape[1] >= overlap_size
                        and lpts_newer.shape[1] >= overlap_size
                    ):
                        overlap_prev_pcd_chunks.append(lpts_older[:, -overlap_size:].detach())
                        overlap_next_pcd_chunks.append(lpts_newer[:, :overlap_size].detach())
                        if conf_older is not None and conf_older.shape[1] >= overlap_size:
                            overlap_prev_conf_chunks.append(conf_older[:, -overlap_size:].squeeze(-1).detach())
                        if conf_newer is not None and conf_newer.shape[1] >= overlap_size:
                            overlap_next_conf_chunks.append(conf_newer[:, :overlap_size].squeeze(-1).detach())
                aligned_predictions[older_idx].pop("local_points", None)
                aligned_predictions[older_idx].pop("_local_points_raw", None)
                if older_idx < len(all_predictions):
                    all_predictions[older_idx].pop("local_points", None)

        aligned_predictions_clean = []
        for pred in aligned_predictions:
            cleaned = pred.copy()
            cleaned.pop("_local_points_raw", None)
            aligned_predictions_clean.append(cleaned)

        merged = self._merge_windowed_predictions(aligned_predictions_clean, window_size, overlap_size)

        # The inner merge cannot build overlap_*_pcd because local_points were
        # popped during the alignment loop; restore them from captured chunks.
        if overlap_prev_pcd_chunks and overlap_next_pcd_chunks:
            merged["overlap_prev_pcd"] = torch.stack(overlap_prev_pcd_chunks, dim=1)
            merged["overlap_next_pcd"] = torch.stack(overlap_next_pcd_chunks, dim=1)
            if overlap_prev_conf_chunks:
                merged["overlap_prev_conf"] = torch.stack(overlap_prev_conf_chunks, dim=1)
            if overlap_next_conf_chunks:
                merged["overlap_next_conf"] = torch.stack(overlap_next_conf_chunks, dim=1)

        pose_key = "chunk_sim3_poses" if allow_scale else "chunk_se3_poses"
        if allow_scale and sim3_scales:
            merged["chunk_sim3_scales"] = torch.stack(sim3_scales, dim=1)
        if sim3_poses:
            merged[pose_key] = torch.stack(sim3_poses, dim=1)
        merged["alignment_mode"] = "sim3" if allow_scale else "se3"

        return merged

    @staticmethod
    def _apply_pgo_delta_to_points(merged: dict, pre_poses, post_poses):
        """Apply T_post @ inverse(T_pre) to points in place.

        pre_poses and post_poses: (B, T_pose, 4, 4).
        points: (B, T_pts, H, W, 3).

        For strided points, infer the stride from frame counts and sample poses
        at the same indices. Point frames without a matching pose stay unchanged.
        """
        points = merged.get("points")
        if points is None or not torch.is_tensor(points):
            return
        try:
            _pre  = pre_poses.float()
            _post = post_poses.float()
            T_pose = _pre.shape[1]
            T_pts  = points.shape[1]
            # Match strided points to strided poses. Stride=1 → identity.
            if T_pts > 0 and T_pose != T_pts:
                stride_inferred = max(1, round(T_pose / T_pts))
                if stride_inferred > 1:
                    _pre  = _pre[:, ::stride_inferred][:, :T_pts]
                    _post = _post[:, ::stride_inferred][:, :T_pts]
            T_pre_inv = torch.linalg.inv(_pre)
            delta = _post @ T_pre_inv                       # (B, T', 4, 4)
        except Exception as exc:
            print(f"[LC-PGO] points re-transform skipped (inv failed): {exc}", flush=True)
            return
        R = delta[..., :3, :3]                              # (B, T', 3, 3)
        t = delta[..., :3, 3]                               # (B, T', 3)
        T_use = min(points.shape[1], R.shape[1])
        if T_use <= 0:
            return
        # Transform points in place, one frame at a time, to avoid a full copy.
        R_use = R[:, :T_use].to(points.dtype)
        t_use = t[:, :T_use].to(points.dtype)
        for f in range(T_use):
            # points[b, f, h, w, :] = R[b,f] @ pts[b, f, h, w, :] + t[b, f]
            points[:, f] = (R_use[:, f] @ points[:, f].reshape(points.shape[0], -1, 3).transpose(-1, -2)).transpose(-1, -2).reshape(points.shape[0], points.shape[2], points.shape[3], 3) + t_use[:, f, None, None, :]

    def _apply_frame_pgo_incremental(
        self,
        merged: dict,
        state: dict,
        all_predictions: list,
        lc_bridge_cams: dict,
        windows: list,
        dist_thresh: int = 5,
        sigma_R_seq: float = 0.01,
        sigma_t_seq: float = 0.01,
        sigma_R_lc:  float = 0.005,
        sigma_t_lc:  float = 0.1,
        lc_robust:   Optional[str] = 'huber',
        lc_robust_k: float = 1.345,
        sigma_match: float = 0.01,
        add_match_constraints: bool = True,
        add_adj_constraints: bool = True,
        lc_check_t: float = 0.0,
        lc_check_R: float = 0.0,
        lc_check_iters: int = 3,
    ) -> dict:
        """Extend the online pose graph with new windows and bridges.

        Warm-start from the previous result. Unlike offline PGO, skip block
        constraints because their merged reference poses change between calls.

        state is updated in place:
          graph: persistent GTSAM factor graph.
          last_result: poses from the previous optimisation.
          windows_added, bridges_added: entries already included in the graph.
          last_T_total: number of frames inserted.
          prior_added: whether frame 0 has been anchored.
        """
        try:
            import gtsam
            import numpy as np
        except ImportError:
            print("[LC-PGO] gtsam not available, skipping online PGO.")
            return merged

        cam_poses = merged.get("camera_poses")
        if cam_poses is None:
            return merged

        cam_np = cam_poses[0].float().numpy().copy()
        T_total = cam_np.shape[0]

        def _is_valid_mat(mat4):
            return np.isfinite(mat4).all()

        def mat_to_pose3(mat4):
            U, _, Vt = np.linalg.svd(mat4[:3, :3])
            R_ortho = U @ Vt
            return gtsam.Pose3(gtsam.Rot3(R_ortho), gtsam.Point3(mat4[:3, 3]))

        def rel_pose(cam, f, f2):
            return mat_to_pose3(cam[f]).between(mat_to_pose3(cam[f2]))

        def _aniso(sR, st):
            return gtsam.noiseModel.Diagonal.Sigmas(
                np.array([sR, sR, sR, st, st, st], dtype=np.float64))

        seq_noise = _aniso(sigma_R_seq, sigma_t_seq)
        _lc_base  = _aniso(sigma_R_lc,  sigma_t_lc)
        if lc_robust == 'huber':
            lc_noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(lc_robust_k), _lc_base)
        elif lc_robust == 'cauchy':
            lc_noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Cauchy.Create(lc_robust_k), _lc_base)
        else:
            lc_noise = _lc_base

        if state.get('graph') is None:
            state['graph'] = gtsam.NonlinearFactorGraph()
            state.setdefault('windows_added', set())
            state.setdefault('bridges_added', set())
            state.setdefault('last_T_total', 0)
            state.setdefault('prior_added', False)
            state.setdefault('last_result', None)
            state.setdefault('lc_factors', {})     # bridge key -> [(factor idx, f, f2, T_rel)]
            state.setdefault('lc_rejected', set())
            state.setdefault('lc_rejected_info', {})  # bridge key -> (t residual, R residual [deg], n edges)

        graph: 'gtsam.NonlinearFactorGraph' = state['graph']
        last_result = state.get('last_result')
        windows_added: set = state['windows_added']
        bridges_added: set = state['bridges_added']
        last_T_total: int = state['last_T_total']

        # Build initial values: warm-start old nodes from previous result, init
        # new nodes from the current merged poses. `initial` is local per call;
        # only `graph` and `last_result` persist across calls.
        initial = gtsam.Values()
        for f in range(T_total):
            if not _is_valid_mat(cam_np[f]):
                cam_np[f] = np.eye(4, dtype=cam_np.dtype)
            inserted = False
            if last_result is not None and f < last_T_total:
                try:
                    initial.insert(f, last_result.atPose3(f))
                    inserted = True
                except Exception:
                    pass
            if not inserted:
                initial.insert(f, mat_to_pose3(cam_np[f]))

        # Tight prior on frame 0 — added once, on first call.
        if not state['prior_added']:
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-6))
            graph.add(gtsam.PriorFactorPose3(0, mat_to_pose3(cam_np[0]), prior_noise))
            state['prior_added'] = True

        # --- Sequential edges from NEW windows only ---
        n_seq_added = 0
        for win_idx, (w_start, w_end) in enumerate(windows):
            if win_idx in windows_added:
                continue
            if win_idx >= len(all_predictions):
                continue
            pred = all_predictions[win_idx]
            local_cam = pred.get("camera_poses") if pred is not None else None
            if local_cam is None:
                continue
            local_cam_np = local_cam[0].float().numpy()
            Nw = local_cam_np.shape[0]
            for f in range(Nw):
                gf = w_start + f
                if gf >= T_total:
                    break
                if not _is_valid_mat(local_cam_np[f]):
                    continue
                for f2 in range(f + 1, min(f + dist_thresh + 1, Nw)):
                    gf2 = w_start + f2
                    if gf2 >= T_total:
                        break
                    if not _is_valid_mat(local_cam_np[f2]):
                        continue
                    T_rel = rel_pose(local_cam_np, f, f2)
                    graph.add(gtsam.BetweenFactorPose3(gf, gf2, T_rel, seq_noise))
                    n_seq_added += 1
            windows_added.add(win_idx)

        # --- LC edges from NEW bridges only ---
        n_lc_added = 0
        for key, br_data in lc_bridge_cams.items():
            if key in bridges_added:
                continue
            j_win_idx, i_win_idx = key
            br_cam = br_data["cam_poses"]
            selected_i = br_data["selected_i_frames"]
            selected_j = br_data["selected_j_frames"]
            frame_matches = br_data["frame_matches"]
            n_i = len(selected_i)
            n_j = len(selected_j)
            br_ov = br_data["overlap_size"]
            br_cam_np = br_cam[0].float().numpy()
            br_Nw = br_cam_np.shape[0]
            if j_win_idx >= len(windows):
                continue
            _j_end_br = windows[j_win_idx][1]

            def bridge_to_global(bf, _sel_i=selected_i, _sel_j=selected_j,
                                 _n_i=n_i, _n_j=n_j, _ov=br_ov, _jend=_j_end_br):
                if bf < _ov:
                    return _jend - _ov + bf
                elif bf < _ov + _n_i:
                    return _sel_i[bf - _ov]
                elif bf < _ov + _n_i + _n_j:
                    return _sel_j[bf - _ov - _n_i]
                else:
                    k = bf - _ov - _n_i - _n_j
                    return _jend - _ov + k if k < _ov else -1

            if add_match_constraints and frame_matches:
                match_noise = lc_noise
                i_to_br = {gf: br_ov + bi for bi, gf in enumerate(selected_i)}
                j_to_br = {gf: br_ov + n_i + bj for bj, gf in enumerate(selected_j)}
                pair_edges = 0
                for fj, fi in frame_matches:
                    if fi not in i_to_br or fj not in j_to_br:
                        continue
                    bi, bj = i_to_br[fi], j_to_br[fj]
                    if bi >= br_Nw or bj >= br_Nw:
                        continue
                    if not _is_valid_mat(br_cam_np[bi]) or not _is_valid_mat(br_cam_np[bj]):
                        continue
                    T_rel = rel_pose(br_cam_np, bi, bj)
                    state['lc_factors'].setdefault(key, []).append((graph.size(), fi, fj, T_rel))
                    graph.add(gtsam.BetweenFactorPose3(fi, fj, T_rel, match_noise))
                    n_lc_added += 1
                    pair_edges += 1
                if pair_edges > 0:
                    print(f"[LC-PGO] +bridge j={j_win_idx}(f{windows[j_win_idx][0]}-{_j_end_br}) "
                          f"← i={i_win_idx}(f{windows[i_win_idx][0]}-{windows[i_win_idx][1]})  "
                          f"edges={pair_edges}", flush=True)

            if add_adj_constraints:
                for side_start, side_end in [
                    (br_ov, br_ov + n_i),
                    (br_ov + n_i, br_ov + n_i + n_j),
                ]:
                    for bf in range(side_start, side_end):
                        gf = bridge_to_global(bf)
                        if gf < 0 or gf >= T_total or not _is_valid_mat(br_cam_np[bf]):
                            continue
                        for bf2 in range(bf + 1, min(bf + dist_thresh + 1, side_end)):
                            gf2 = bridge_to_global(bf2)
                            if gf2 < 0 or gf2 >= T_total or gf == gf2:
                                continue
                            if not _is_valid_mat(br_cam_np[bf2]):
                                continue
                            T_rel = rel_pose(br_cam_np, bf, bf2)
                            state['lc_factors'].setdefault(key, []).append((graph.size(), gf, gf2, T_rel))
                            graph.add(gtsam.BetweenFactorPose3(gf, gf2, T_rel, lc_noise))
                            n_lc_added += 1

            bridges_added.add(key)

        print(f"[LC-PGO] online step: +{n_seq_added} seq edges, +{n_lc_added} LC edges; "
              f"graph={graph.size()} factors, T_total={T_total}", flush=True)

        try:
            params = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
            result = optimizer.optimize()
            print(f"[LC-PGO] optimised {T_total} nodes, {graph.size()} factors, "
                  f"final error={optimizer.error():.4f}", flush=True)
        except Exception as e:
            print(f"[LC-PGO] online optimisation failed: {e}", flush=True)
            return merged

        # --- Loop consistency check: drop bridges the optimum disagrees with ---
        if lc_check_t > 0 or lc_check_R > 0:
            lc_factors, rejected = state['lc_factors'], state['lc_rejected']
            for _it in range(max(lc_check_iters, 1)):
                res = {}   # bridge key -> (median t residual, median R residual [deg])
                for key, facs in lc_factors.items():
                    if key in rejected:
                        continue
                    et, eR = [], []
                    for _, f, f2, T_rel in facs:
                        E = T_rel.between(result.atPose3(f).between(result.atPose3(f2)))
                        et.append(np.linalg.norm(E.translation()))
                        eR.append(np.degrees(np.linalg.norm(gtsam.Rot3.Logmap(E.rotation()))))
                    res[key] = (float(np.median(et)), float(np.median(eR)))
                bad = [k for k, (et, eR) in res.items()
                       if (lc_check_t > 0 and et > lc_check_t) or (lc_check_R > 0 and eR > lc_check_R)]
                if res:
                    _t = np.array([v[0] for v in res.values()]); _R = np.array([v[1] for v in res.values()])
                    print(f"[LC-check] round {_it}: {len(res)} bridges, median residual "
                          f"t p50/p90/max={np.percentile(_t, 50):.2f}/{np.percentile(_t, 90):.2f}/{_t.max():.2f} units, "
                          f"R p50/p90/max={np.percentile(_R, 50):.2f}/{np.percentile(_R, 90):.2f}/{_R.max():.2f} deg; "
                          f"rejecting {len(bad)}", flush=True)
                if not bad:
                    break
                for k in bad:
                    j_w, i_w = k
                    print(f"[LC-check]   reject bridge j={j_w} <- i={i_w}: "
                          f"t={res[k][0]:.2f}  R={res[k][1]:.2f} deg  ({len(lc_factors[k])} edges)", flush=True)
                    state['lc_rejected_info'][k] = (res[k][0], res[k][1], len(lc_factors[k]))
                    for idx, *_ in lc_factors[k]:
                        graph.remove(idx)
                    rejected.add(k)
                try:
                    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, gtsam.LevenbergMarquardtParams())
                    result = optimizer.optimize()
                    print(f"[LC-check]   re-optimised, final error={optimizer.error():.4f}", flush=True)
                except Exception as e:
                    print(f"[LC-check] re-optimisation failed: {e}", flush=True)
                    return merged
            print(f"[LC-check] total rejected bridges: {len(rejected)}/{len(lc_factors)}", flush=True)

        state['last_result'] = result
        state['last_T_total'] = T_total

        merged["camera_poses_pre_pgo"] = cam_poses.clone()
        new_cam = cam_poses.clone()
        for f in range(T_total):
            try:
                pose = result.atPose3(f)
                new_cam[0, f] = torch.tensor(pose.matrix(), dtype=new_cam.dtype)
            except Exception:
                pass
        merged["camera_poses"] = new_cam
        # Re-transform world-space points by per-frame delta so post-PGO
        # cameras and points stay consistent in the saved bundle.
        Pi3._apply_pgo_delta_to_points(merged, cam_poses, new_cam)
        return merged

    def _apply_frame_pgo(
        self,
        merged: dict,
        all_predictions: list,
        lc_bridge_cams: dict,
        windows: list,
        dist_thresh: int = 5,
        # Separate rotation and translation sigmas. Legacy isotropic arguments
        # override both axes when provided.
        sigma_R_seq: float = 0.01,
        sigma_t_seq: float = 0.01,
        sigma_R_lc:  float = 0.005,
        sigma_t_lc:  float = 0.1,
        lc_robust:   Optional[str] = 'huber',   # 'huber' | 'cauchy' | None
        lc_robust_k: float = 1.345,
        sigma_seq: Optional[float] = None,      # legacy isotropic; if set, overrides per-axis
        sigma_lc:  Optional[float] = None,
        sigma_match: float = 0.01,
        add_match_constraints: bool = True,
        add_adj_constraints: bool = True,
        reset_every: int = 0,
        add_block_constraints: bool = False,
        sigma_block: float = 0.05,
        block_middle_count: int = 4,
        debug_save_path: Optional[str] = None,
    ) -> dict:
        """
        Frame-based pose-graph optimisation using GTSAM.

        Nodes  : one Pose3 per frame in the merged trajectory.
        Edges  :
          - Sequential: relative poses between frames within the same window,
                        for all pairs with |f-f'| < dist_thresh.
          - LC        : same logic applied to bridge window output,
                        naturally giving j<->i constraints.
        Init   : merged["camera_poses"] (world-frame, from SE3 alignment).
        Prior  : tight prior on frame 0 to anchor the graph.
        """
        # Backwards-compat: if legacy isotropic sigmas were passed, fold them in.
        if sigma_seq is not None:
            sigma_R_seq = sigma_t_seq = sigma_seq
        if sigma_lc is not None:
            sigma_R_lc = sigma_t_lc = sigma_lc
        try:
            import gtsam
            import numpy as np
        except ImportError:
            print("[LC-PGO] gtsam not available, skipping PGO.")
            return merged

        cam_poses = merged.get("camera_poses")
        if cam_poses is None:
            return merged

        # Work on numpy (poses are on CPU after merge)
        cam_np = cam_poses[0].float().numpy().copy()  # (T, 4, 4) — explicit copy, avoid aliasing with the tensor
        T_total = cam_np.shape[0]

        def _is_valid_mat(mat4):
            return np.isfinite(mat4).all()

        def mat_to_pose3(mat4):
            # Project the rotation onto an orthogonal matrix before passing it to GTSAM.
            U, _, Vt = np.linalg.svd(mat4[:3, :3])
            R_ortho = U @ Vt
            R = gtsam.Rot3(R_ortho)
            t = gtsam.Point3(mat4[:3, 3])
            return gtsam.Pose3(R, t)

        def rel_pose(cam, f, f2):
            """Relative pose from frame f to frame f2 in local window."""
            p_f  = mat_to_pose3(cam[f])
            p_f2 = mat_to_pose3(cam[f2])
            return p_f.between(p_f2)

        # Accumulators for debug saving (populated regardless of debug_save_path)
        dbg_seq_gf:    list = []
        dbg_seq_gf2:   list = []
        dbg_seq_T_rel: list = []
        dbg_lc_gf:     list = []
        dbg_lc_gf2:    list = []
        dbg_lc_T_rel:  list = []
        dbg_block_gf:    list = []
        dbg_block_gf2:   list = []
        dbg_block_T_rel: list = []

        try:
            graph   = gtsam.NonlinearFactorGraph()
            initial = gtsam.Values()

            # Initialise all nodes from merged world-frame poses
            for f in range(T_total):
                if not _is_valid_mat(cam_np[f]):
                    # Fall back to identity for degenerate frames
                    cam_np[f] = np.eye(4, dtype=cam_np.dtype)
                initial.insert(f, mat_to_pose3(cam_np[f]))

            # Tight prior on frame 0 to fix gauge
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-6))
            graph.add(gtsam.PriorFactorPose3(0, mat_to_pose3(cam_np[0]), prior_noise))

            # GTSAM orders sigmas as [Rx, Ry, Rz, tx, ty, tz].
            # Use a robust kernel for loop-closure edges when enabled.
            def _aniso(sR, st):
                return gtsam.noiseModel.Diagonal.Sigmas(
                    np.array([sR, sR, sR, st, st, st], dtype=np.float64))

            print(f"[LC-PGO] σ_R_seq={sigma_R_seq} σ_t_seq={sigma_t_seq} | "
                  f"σ_R_lc={sigma_R_lc} σ_t_lc={sigma_t_lc} | "
                  f"lc_robust={lc_robust} k={lc_robust_k}", flush=True)
            seq_noise = _aniso(sigma_R_seq, sigma_t_seq)
            _lc_base  = _aniso(sigma_R_lc,  sigma_t_lc)
            if lc_robust == 'huber':
                lc_noise = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.Huber.Create(lc_robust_k), _lc_base)
            elif lc_robust == 'cauchy':
                lc_noise = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.Cauchy.Create(lc_robust_k), _lc_base)
            else:
                lc_noise = _lc_base

            # --- Sequential edges from each normal window ---
            for win_idx, (w_start, w_end) in enumerate(windows):
                pred = all_predictions[win_idx] if win_idx < len(all_predictions) else None
                if pred is None:
                    continue
                local_cam = pred.get("camera_poses")
                if local_cam is None:
                    continue
                local_cam_np = local_cam[0].float().numpy()  # (Nw, 4, 4)
                Nw = local_cam_np.shape[0]
                for f in range(Nw):
                    gf = w_start + f
                    if gf >= T_total:
                        break
                    if not _is_valid_mat(local_cam_np[f]):
                        continue
                    for f2 in range(f + 1, min(f + dist_thresh + 1, Nw)):
                        gf2 = w_start + f2
                        if gf2 >= T_total:
                            break
                        if not _is_valid_mat(local_cam_np[f2]):
                            continue
                        T_rel = rel_pose(local_cam_np, f, f2)
                        graph.add(gtsam.BetweenFactorPose3(gf, gf2, T_rel, seq_noise))
                        dbg_seq_gf.append(gf)
                        dbg_seq_gf2.append(gf2)
                        dbg_seq_T_rel.append(T_rel.matrix())

            # --- LC edges from bridge windows ---
            for (j_win_idx, i_win_idx), br_data in lc_bridge_cams.items():
                br_cam        = br_data["cam_poses"]
                selected_i    = br_data["selected_i_frames"]
                selected_j    = br_data["selected_j_frames"]
                frame_matches = br_data["frame_matches"]
                n_i = len(selected_i)
                n_j = len(selected_j)
                br_ov = br_data["overlap_size"]
                br_cam_np = br_cam[0].float().numpy()  # (br_Nw, 4, 4)
                br_Nw = br_cam_np.shape[0]
                _j_end_br = windows[j_win_idx][1]

                # Bridge layout: [j_end(br_ov) | di(n_i) | dj(n_j) | j_end(br_ov)]
                #   0 .. br_ov-1               -> j_end - br_ov + k
                #   br_ov .. br_ov+n_i-1       -> selected_i[bf - br_ov]
                #   br_ov+n_i .. br_ov+n_i+n_j-1 -> selected_j[bf - br_ov - n_i]
                #   br_ov+n_i+n_j .. br_Nw-1   -> j_end - br_ov + k
                def bridge_to_global(bf, _sel_i=selected_i, _sel_j=selected_j,
                                     _n_i=n_i, _n_j=n_j, _ov=br_ov, _jend=_j_end_br):
                    if bf < _ov:
                        return _jend - _ov + bf
                    elif bf < _ov + _n_i:
                        return _sel_i[bf - _ov]
                    elif bf < _ov + _n_i + _n_j:
                        return _sel_j[bf - _ov - _n_i]
                    else:
                        k = bf - _ov - _n_i - _n_j
                        return _jend - _ov + k if k < _ov else -1

                # Match constraints: bridge-predicted T_rel(di → dj) for each GT match pair
                _win_t_norms = []
                _win_r_degs  = []
                if add_match_constraints and frame_matches:
                    # Match and adjacency edges come from the same bridge; use the same
                    # noise model and robust kernel for both.
                    match_noise = lc_noise
                    i_to_br = {gf: br_ov + bi for bi, gf in enumerate(selected_i)}      # frame index to local window index matching
                    j_to_br = {gf: br_ov + n_i + bj for bj, gf in enumerate(selected_j)}
                    for fj, fi in frame_matches:
                        if fi not in i_to_br or fj not in j_to_br:
                            continue
                        bi, bj = i_to_br[fi], j_to_br[fj]
                        if bi >= br_Nw or bj >= br_Nw:
                            continue
                        if not _is_valid_mat(br_cam_np[bi]) or not _is_valid_mat(br_cam_np[bj]):
                            continue
                        T_rel = rel_pose(br_cam_np, bi, bj)
                        graph.add(gtsam.BetweenFactorPose3(fi, fj, T_rel, match_noise))
                        dbg_lc_gf.append(fi)
                        dbg_lc_gf2.append(fj)
                        dbg_lc_T_rel.append(T_rel.matrix())
                        _m = T_rel.matrix()
                        _win_t_norms.append(float(np.linalg.norm(_m[:3, 3])))
                        _R = _m[:3, :3]
                        _cos = np.clip((np.trace(_R) - 1.0) / 2.0, -1.0, 1.0)
                        _win_r_degs.append(float(np.degrees(np.arccos(_cos))))
                _n_edges = len(_win_t_norms)
                if _n_edges:
                    _t_mean = float(np.mean(_win_t_norms))
                    _r_mean = float(np.mean(_win_r_degs))
                    _t_max  = float(np.max(_win_t_norms))
                    _r_max  = float(np.max(_win_r_degs))
                    print(f"[LC-PGO] win j={j_win_idx}(f{windows[j_win_idx][0]}-{windows[j_win_idx][1]}) "
                          f"← i={i_win_idx}(f{windows[i_win_idx][0]}-{windows[i_win_idx][1]})  "
                          f"edges={_n_edges}  "
                          f"t: mean={_t_mean:.2f}m max={_t_max:.2f}m  "
                          f"rot: mean={_r_mean:.1f}° max={_r_max:.1f}°")
                else:
                    print(f"[LC-PGO] win j={j_win_idx} ← i={i_win_idx}  edges=0 (no valid matches)")

                # Bridge-derived adjacent constraints: within-di and within-dj only
                # (skips j_end positions and does not cross the i/j boundary)
                if add_adj_constraints:
                    for side_start, side_end in [
                        (br_ov, br_ov + n_i),           # di side
                        (br_ov + n_i, br_ov + n_i + n_j),  # dj side
                    ]:
                        for bf in range(side_start, side_end):
                            gf = bridge_to_global(bf)
                            if gf < 0 or gf >= T_total or not _is_valid_mat(br_cam_np[bf]):
                                continue
                            for bf2 in range(bf + 1, min(bf + dist_thresh + 1, side_end)):
                                gf2 = bridge_to_global(bf2)
                                if gf2 < 0 or gf2 >= T_total or gf == gf2:
                                    continue
                                if not _is_valid_mat(br_cam_np[bf2]):
                                    continue
                                T_rel = rel_pose(br_cam_np, bf, bf2)
                                graph.add(gtsam.BetweenFactorPose3(gf, gf2, T_rel, lc_noise))
                                dbg_lc_gf.append(gf)
                                dbg_lc_gf2.append(gf2)
                                dbg_lc_T_rel.append(T_rel.matrix())

            # --- Block-consistency edges (within each reset block) ---
            # For each contiguous run of up to `reset_every` windows, take the middle
            # `block_middle_count` frames of every window and add a BetweenFactorPose3
            # between every cross-window pair, with the relative pose read from the
            # merged world-frame init (cam_np). Zero-residual at init, so these edges
            # only act when LC tries to deform the block.
            if add_block_constraints and reset_every and reset_every > 0 and len(windows) > 0:
                block_noise = gtsam.noiseModel.Isotropic.Sigma(6, sigma_block)

                def _middle_global(w_start, w_end, count):
                    size = w_end - w_start
                    if size <= 0:
                        return []
                    if size <= count:
                        return list(range(w_start, w_end))
                    mid = size // 2
                    half = count // 2
                    lo = max(0, mid - half)
                    hi = lo + count
                    if hi > size:
                        hi = size
                        lo = max(0, hi - count)
                    return [w_start + k for k in range(lo, hi)]

                n_block_edges_total = 0
                n_blocks = 0
                for block_start in range(0, len(windows), reset_every):
                    block_end = min(block_start + reset_every, len(windows))
                    block_idx = list(range(block_start, block_end))
                    if len(block_idx) < 2:
                        continue
                    n_blocks += 1
                    mids = []
                    for wi in block_idx:
                        ws, we = windows[wi]
                        mids.append(_middle_global(ws, we, block_middle_count))

                    n_edges_block = 0
                    for a in range(len(block_idx)):
                        for b in range(a + 1, len(block_idx)):
                            for gf_a in mids[a]:
                                if gf_a >= T_total or not _is_valid_mat(cam_np[gf_a]):
                                    continue
                                for gf_b in mids[b]:
                                    if gf_b >= T_total or gf_a == gf_b:
                                        continue
                                    if not _is_valid_mat(cam_np[gf_b]):
                                        continue
                                    T_rel = rel_pose(cam_np, gf_a, gf_b)
                                    graph.add(gtsam.BetweenFactorPose3(gf_a, gf_b, T_rel, block_noise))
                                    dbg_block_gf.append(gf_a)
                                    dbg_block_gf2.append(gf_b)
                                    dbg_block_T_rel.append(T_rel.matrix())
                                    n_edges_block += 1
                    n_block_edges_total += n_edges_block
                print(f"[LC-PGO] block-consistency: {n_blocks} block(s), "
                      f"{n_block_edges_total} edges (reset_every={reset_every}, "
                      f"middle={block_middle_count}, sigma={sigma_block})")

            # --- Optimise ---
            params = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
            result = optimizer.optimize()
            print(f"[LC-PGO] optimised {T_total} nodes, "
                  f"{graph.size()} factors, "
                  f"final error={optimizer.error():.4f}")
        except Exception as e:
            print(f"[LC-PGO] optimisation failed: {e}")
            return merged

        # Extract result back into merged camera_poses; keep a pre-PGO snapshot
        # so demo_viser.py can save the pre-PGO trajectory alongside the optimised one.
        merged["camera_poses_pre_pgo"] = cam_poses.clone()
        new_cam = cam_poses.clone()
        for f in range(T_total):
            try:
                pose = result.atPose3(f)
                mat  = pose.matrix()
                new_cam[0, f] = torch.tensor(mat, dtype=new_cam.dtype)
            except Exception:
                pass
        merged["camera_poses"] = new_cam
        # Re-transform world-space points by per-frame delta so post-PGO
        # cameras and points stay consistent in the saved bundle.
        Pi3._apply_pgo_delta_to_points(merged, cam_poses, new_cam)

        if debug_save_path is not None:
            import numpy as _np_pgo
            save_dict = dict(
                initial_poses   = cam_np,                                          # (T,4,4) before PGO
                optimized_poses = new_cam[0].float().numpy(),                      # (T,4,4) after PGO
                seq_gf          = _np_pgo.array(dbg_seq_gf,  dtype=_np_pgo.int32),
                seq_gf2         = _np_pgo.array(dbg_seq_gf2, dtype=_np_pgo.int32),
                seq_T_rel       = _np_pgo.array(dbg_seq_T_rel,  dtype=_np_pgo.float32) if dbg_seq_T_rel  else _np_pgo.zeros((0,4,4), dtype=_np_pgo.float32),
                lc_gf           = _np_pgo.array(dbg_lc_gf,   dtype=_np_pgo.int32),
                lc_gf2          = _np_pgo.array(dbg_lc_gf2,  dtype=_np_pgo.int32),
                lc_T_rel        = _np_pgo.array(dbg_lc_T_rel,   dtype=_np_pgo.float32) if dbg_lc_T_rel   else _np_pgo.zeros((0,4,4), dtype=_np_pgo.float32),
                block_gf        = _np_pgo.array(dbg_block_gf,  dtype=_np_pgo.int32),
                block_gf2       = _np_pgo.array(dbg_block_gf2, dtype=_np_pgo.int32),
                block_T_rel     = _np_pgo.array(dbg_block_T_rel, dtype=_np_pgo.float32) if dbg_block_T_rel else _np_pgo.zeros((0,4,4), dtype=_np_pgo.float32),
            )
            for (j_idx, i_idx), br in lc_bridge_cams.items():
                j_end_frames = br["j_end_frames"]
                all_frames = (j_end_frames + list(br["selected_i_frames"])
                              + list(br["selected_j_frames"]) + j_end_frames)
                key = f"bridge_{j_idx}_{i_idx}"
                save_dict[f"{key}_cam_poses"]  = br["cam_poses"][0].float().numpy()        # (Nw,4,4)
                save_dict[f"{key}_all_frames"] = _np_pgo.array(all_frames, dtype=_np_pgo.int32)  # (Nw,)
            _np_pgo.savez(debug_save_path, **save_dict)
            print(f"[LC-PGO] debug data saved to {debug_save_path} "
                  f"({len(dbg_seq_gf)} seq edges, {len(dbg_lc_gf)} LC edges, "
                  f"{len(dbg_block_gf)} block edges, "
                  f"{len(lc_bridge_cams)} bridge windows)")

        return merged