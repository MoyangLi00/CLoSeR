#!/bin/bash
# Shared flag parsing + per-sequence dispatch for scripts/run_<dataset>.sh.
# Every dataset runner accepts exactly the same flags; only the dataset
# settings below differ. Not meant to be run directly — source it from a
# runner after setting:
#
#   MODE        demo_run_longeval.sh --mode (kitti | vbr | oxford_spires | droidw)
#   ALL_SEQS    default space-separated sequence list
#   SEQS_ENV    name of an env var that may override ALL_SEQS (e.g. KITTI_SEQS)
#   WIN_LOGER   window size for LoGeR (default model)
#   WIN_STAR    window size for LoGeR_star (--star)
#   WIN_PI3     window size for the pi3-Chunk baseline (--pi3)
#
# and optionally:
#   prepare_seq <seq>    per-sequence setup (e.g. staging); non-zero return skips it
#   seq_input_dir <seq>  image dir; a sequence whose dir is missing is skipped
#
# Flags (all forwarded to eval/demo_run_longeval.sh -> demo_viser.py):
#   model        --star | --pi3                    (default: LoGeR)
#   sequences    --seq "SEQ ..."                   (precedence: --seq > $SEQS_ENV > ALL_SEQS)
#   output       --output_dir DIR  --lc_pts_dir DIR
#                --save_viser_data (bundle in <output_dir>/<model>/viser_bundles/<seq>/)  --save_stride N (bundle only)
#                --frame_stride N (live viser display)
#   inference    --overlap N  --reset_every N  --stride N  --no_ttt  --no_swa
#                --se3  --sim3  --sim3_on_reset
#                --sim3_scale_mode MODE  --gt_scale PATH
#   PGO          --pgo  --online_pgo  --pgo_adj_constraints  --pgo_block_constraints
#                --pgo_sigma_block F  --pgo_block_middle_count N  --pgo_dist_thresh N
#                --pgo_sigma_seq F  --pgo_sigma_lc F
#                --pgo_lc_check_t U (default 1, trajectory units)  --pgo_lc_check_R DEG (default 0 = off)
#                --pgo_lc_check_iters N  (online-PGO loop consistency check; --pgo_lc_check_t 0 = off)
#   loops        --loopdetect (implies --pgo)  --loop_save_pair_vis  --lc_chord_arc_thresh F
#                --loop_min_frame_gap N  --loop_sim_thresh F  --loop_top_k N
#                --lc_bridge_nms N (default 2; 0 = a bridge per candidate)  --lc_bridge_keep_state
#   debug dumps  --save_debug_poses (<seq>_pre_pgo.txt, _chunk_*.npy, _pgo_inputs.pt next to <seq>.txt)
#                --save_online_pgo_poses_dir DIR  --save_window_local_pts_dir DIR

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

USE_STAR=0
USE_PI3=0
SEQS_ARG=""
OUTPUT_DIR=""
USE_PGO=0
USE_LOOP_DETECT=0
PASS=()       # forwarded to every model
PI3_PASS=()   # subset the pi3-Chunk baseline understands

need_value() { [[ $2 -ge 2 ]] || { echo "Error: $1 requires a value."; exit 1; }; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            sed -n '/^# Flags/,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        # --- model / sequences ---
        --star)  USE_STAR=1; shift ;;
        --pi3)   USE_PI3=1; shift ;;   # vanilla Pi3 run chunk-wise (the "pi3-Chunk" baseline)
        --seq|--seqs)
            need_value "$1" $#; SEQS_ARG="$2"; shift 2 ;;
        # --- flags that switch other behaviour on ---
        --pgo)        USE_PGO=1; shift ;;
        --loopdetect) USE_LOOP_DETECT=1; USE_PGO=1; shift ;;
        # --- value flags forwarded to every model (incl. pi3) ---
        --output_dir)
            need_value "$1" $#; OUTPUT_DIR="$2"; shift 2 ;;
        --overlap|--reset_every|--sim3_scale_mode|--stride)
            need_value "$1" $#; PASS+=("$1" "$2"); PI3_PASS+=("$1" "$2"); shift 2 ;;
        --se3|--sim3)
            PASS+=("$1"); PI3_PASS+=("$1"); shift ;;
        # --- switches forwarded as-is ---
        --no_ttt|--no_swa|--sim3_on_reset|--online_pgo|\
        --pgo_adj_constraints|--pgo_block_constraints|--save_viser_data|--loop_save_pair_vis|\
        --lc_bridge_keep_state|--save_debug_poses)
            PASS+=("$1"); shift ;;
        # --- value flags forwarded as-is ---
        --lc_pts_dir|--gt_scale|--save_stride|--frame_stride|\
        --pgo_sigma_block|--pgo_block_middle_count|--pgo_dist_thresh|--pgo_sigma_seq|--pgo_sigma_lc|\
        --lc_chord_arc_thresh|--loop_min_frame_gap|--loop_sim_thresh|--loop_top_k|--lc_bridge_nms|\
        --pgo_lc_check_t|--pgo_lc_check_R|--pgo_lc_check_iters|\
        --save_online_pgo_poses_dir|--save_window_local_pts_dir)
            need_value "$1" $#; PASS+=("$1" "$2"); shift 2 ;;
        *)
            echo "Unknown argument: $1 (see --help)"; exit 1 ;;
    esac
done

[[ "$USE_PGO" -eq 1 ]] && PASS+=("--pgo")
[[ "$USE_LOOP_DETECT" -eq 1 ]] && PASS+=("--loop_detect")
if [[ -n "$OUTPUT_DIR" ]]; then
    PASS+=("--output_dir" "$OUTPUT_DIR")
    PI3_PASS+=("--output_dir" "$OUTPUT_DIR")
fi

if [[ "$USE_PI3" -eq 1 ]]; then
    MODEL=pi3; WIN="$WIN_PI3"; MODEL_PASS=("${PI3_PASS[@]}")
elif [[ "$USE_STAR" -eq 1 ]]; then
    MODEL=LoGeR_star; WIN="$WIN_STAR"; MODEL_PASS=("${PASS[@]}")
else
    MODEL=LoGeR; WIN="$WIN_LOGER"; MODEL_PASS=("${PASS[@]}")
fi

SEQS_TO_RUN="${SEQS_ARG:-${!SEQS_ENV:-$ALL_SEQS}}"
for seq in $SEQS_TO_RUN; do
    if declare -F prepare_seq >/dev/null && ! prepare_seq "$seq"; then
        continue
    fi
    if declare -F seq_input_dir >/dev/null; then
        _dir="$(seq_input_dir "$seq")"
        if [[ ! -d "$_dir" ]]; then
            echo "  [skip] seq $seq: no images at $_dir"
            continue
        fi
    fi
    if [[ "$USE_LOOP_DETECT" -eq 1 ]]; then
        echo "  [LC] seq $seq: online loop detection enabled"
    fi
    bash "$REPO_ROOT/eval/demo_run_longeval.sh" --cuda 0 --model "$MODEL" --mode "$MODE" \
        --seq "$seq" --win "$WIN" "${MODEL_PASS[@]}"
done
