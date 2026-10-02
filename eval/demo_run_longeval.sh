#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  ./eval/demo_run_longeval.sh [--cuda ID] [--model MODEL] [--mode MODE] [--seq SEQ] [--win SIZE]

Options:
  --cuda ID        Set CUDA_VISIBLE_DEVICES (optional)
  --model MODEL    loger | loger_star | pi3 | all (default: all)
                   pi3 = vanilla Pi3 run chunk-wise (the "pi3-Chunk" baseline);
                   needs ckpts/Pi3/{latest.pt,original_config.yaml}
  --mode MODE      kitti | vbr | oxford_spires | droidw (default: kitti)
  --win SIZE       Window size for demo_viser.py (default: 32)
  --seq SEQ        Sequence id/name.
                   kitti:          data/kitti/dataset/sequences/<SEQ>/image_2
                   vbr:            data/vbr/<SEQ>_processed_aligned/rgb
                   oxford_spires:  data/oxford_spires/${OXFORD_SEQS_DIR:-processed_sequences}/<SEQ>/raw/cam0
                   droidw:         ${DROIDW_DATA_ROOT:-data/DROID-W}/<SEQ>/images_anonymized
                                   (prepared by scripts_download/download_droidw.sh)
                   If omitted, fallback to KITTI_SEQ/VBR_SEQ env vars.
  --se3            Use SE(3) transformation for TTT
  --sim3           Use Sim(3) transformation for TTT
  -h, --help       Show this help message

Examples:
  ./eval/demo_run_longeval.sh --cuda 0 --model loger --mode kitti --seq 00 --win 32
  ./eval/demo_run_longeval.sh --model loger_star --mode vbr --seq office --win 64
EOF
}

CUDA_DEVICE=""
MODEL_ARG="all"
MODE_ARG="kitti"
SEQ_ARG=""
WIN_ARG="32"
OVERLAP_ARG=""
OUTPUT_DIR=""
NO_TTT=0
NO_SWA=0
ADJ_CONSTRAINTS=0
BLOCK_CONSTRAINTS=0
PGO_SIGMA_BLOCK=""
PGO_BLOCK_MIDDLE_COUNT=""
PGO_DIST_THRESH=""
PGO_SIGMA_SEQ=""
PGO_SIGMA_LC=""
RESET_EVERY=""
USE_PGO=0
USE_SE3=0
USE_SIM3=0
USE_SIM3_ON_RESET=0
USE_LOOP_DETECT=0
USE_ONLINE_PGO=0
SAVE_ONLINE_PGO_POSES_DIR=""
SAVE_WINDOW_LOCAL_PTS_DIR=""
USE_LOOP_SAVE_PAIR_VIS=0
USE_SAVE_VISER=0
LC_CHORD_ARC_THRESH=""
USE_LC_BRIDGE_KEEP_STATE=0
SAVE_DEBUG_POSES_PY=""
LC_BRIDGE_NMS=""
PGO_LC_CHECK_ARGS=""
LOOP_MIN_FRAME_GAP=""
LOOP_SIM_THRESH=""
LOOP_TOP_K=""
FRAME_STRIDE=""
GT_SCALE=""
SIM3_SCALE_MODE=""
LC_PTS_DIR=""
STRIDE=""
SAVE_STRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)
            [[ $# -ge 2 ]] || { echo "Error: --cuda requires a value."; exit 1; }
            CUDA_DEVICE="$2"
            shift 2
            ;;
        --model)
            [[ $# -ge 2 ]] || { echo "Error: --model requires a value."; exit 1; }
            MODEL_ARG="$2"
            shift 2
            ;;
        --mode)
            [[ $# -ge 2 ]] || { echo "Error: --mode requires a value."; exit 1; }
            MODE_ARG="$2"
            shift 2
            ;;
        --seq)
            [[ $# -ge 2 ]] || { echo "Error: --seq requires a value."; exit 1; }
            SEQ_ARG="$2"
            shift 2
            ;;
        --win)
            [[ $# -ge 2 ]] || { echo "Error: --win requires a value."; exit 1; }
            WIN_ARG="$2"
            shift 2
            ;;
        --overlap)
            [[ $# -ge 2 ]] || { echo "Error: --overlap requires a value."; exit 1; }
            OVERLAP_ARG="$2"
            shift 2
            ;;
        --no_ttt)
            NO_TTT=1
            shift
            ;;
        --no_swa)
            NO_SWA=1
            shift
            ;;
        --pgo_adj_constraints)
            ADJ_CONSTRAINTS=1
            shift
            ;;
        --pgo_block_constraints)
            BLOCK_CONSTRAINTS=1
            shift
            ;;
        --pgo_sigma_block)
            [[ $# -ge 2 ]] || { echo "Error: --pgo_sigma_block requires a value."; exit 1; }
            PGO_SIGMA_BLOCK="$2"; shift 2 ;;
        --pgo_block_middle_count)
            [[ $# -ge 2 ]] || { echo "Error: --pgo_block_middle_count requires a value."; exit 1; }
            PGO_BLOCK_MIDDLE_COUNT="$2"; shift 2 ;;
        --pgo_dist_thresh)
            [[ $# -ge 2 ]] || { echo "Error: --pgo_dist_thresh requires a value."; exit 1; }
            PGO_DIST_THRESH="$2"; shift 2 ;;
        --pgo_sigma_seq)
            [[ $# -ge 2 ]] || { echo "Error: --pgo_sigma_seq requires a value."; exit 1; }
            PGO_SIGMA_SEQ="$2"; shift 2 ;;
        --pgo_sigma_lc)
            [[ $# -ge 2 ]] || { echo "Error: --pgo_sigma_lc requires a value."; exit 1; }
            PGO_SIGMA_LC="$2"; shift 2 ;;
        --reset_every)
            [[ $# -ge 2 ]] || { echo "Error: --reset_every requires a value."; exit 1; }
            RESET_EVERY="$2"
            shift 2
            ;;
        --pgo)
            USE_PGO=1
            shift
            ;;
        --output_dir)
            [[ $# -ge 2 ]] || { echo "Error: --output_dir requires a value."; exit 1; }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --lc_pts_dir)
            [[ $# -ge 2 ]] || { echo "Error: --lc_pts_dir requires a value."; exit 1; }
            LC_PTS_DIR="$2"
            shift 2
            ;;
        --se3)
            USE_SE3=1
            shift
            ;;
        --sim3)
            USE_SIM3=1
            shift
            ;;
        --sim3_on_reset)
            USE_SIM3_ON_RESET=1
            shift
            ;;
        --gt_scale)
            [[ $# -ge 2 ]] || { echo "Error: --gt_scale requires a path (file or dir)."; exit 1; }
            GT_SCALE="$2"
            shift 2
            ;;
        --sim3_scale_mode)
            [[ $# -ge 2 ]] || { echo "Error: --sim3_scale_mode requires a value."; exit 1; }
            SIM3_SCALE_MODE="$2"
            shift 2
            ;;
        --stride)
            [[ $# -ge 2 ]] || { echo "Error: --stride requires a value."; exit 1; }
            STRIDE="$2"
            shift 2
            ;;
        --save_stride)
            [[ $# -ge 2 ]] || { echo "Error: --save_stride requires a value."; exit 1; }
            SAVE_STRIDE="$2"
            shift 2
            ;;
        --loop_detect)
            USE_LOOP_DETECT=1
            shift
            ;;
        --online_pgo)
            USE_ONLINE_PGO=1
            shift
            ;;
        --save_online_pgo_poses_dir)
            [[ $# -ge 2 ]] || { echo "Error: --save_online_pgo_poses_dir requires a value."; exit 1; }
            SAVE_ONLINE_PGO_POSES_DIR="$2"
            shift 2
            ;;
        --save_window_local_pts_dir)
            [[ $# -ge 2 ]] || { echo "Error: --save_window_local_pts_dir requires a value."; exit 1; }
            SAVE_WINDOW_LOCAL_PTS_DIR="$2"
            shift 2
            ;;
        --loop_save_pair_vis)
            USE_LOOP_SAVE_PAIR_VIS=1
            shift
            ;;
        --lc_bridge_nms)
            [[ $# -ge 2 ]] || { echo "Error: --lc_bridge_nms requires a value."; exit 1; }
            LC_BRIDGE_NMS="$2"
            shift 2
            ;;
        --lc_bridge_keep_state)
            USE_LC_BRIDGE_KEEP_STATE=1
            shift
            ;;
        --save_debug_poses)
            SAVE_DEBUG_POSES_PY="--save_debug_poses"
            shift
            ;;
        --pgo_lc_check_t|--pgo_lc_check_R|--pgo_lc_check_iters)
            [[ $# -ge 2 ]] || { echo "Error: $1 requires a value."; exit 1; }
            PGO_LC_CHECK_ARGS="$PGO_LC_CHECK_ARGS $1 $2"
            shift 2
            ;;
        --lc_chord_arc_thresh)
            [[ $# -ge 2 ]] || { echo "Error: --lc_chord_arc_thresh requires a value."; exit 1; }
            LC_CHORD_ARC_THRESH="$2"
            shift 2
            ;;
        --loop_min_frame_gap)
            [[ $# -ge 2 ]] || { echo "Error: --loop_min_frame_gap requires a value."; exit 1; }
            LOOP_MIN_FRAME_GAP="$2"
            shift 2
            ;;
        --loop_sim_thresh)
            [[ $# -ge 2 ]] || { echo "Error: --loop_sim_thresh requires a value."; exit 1; }
            LOOP_SIM_THRESH="$2"
            shift 2
            ;;
        --loop_top_k)
            [[ $# -ge 2 ]] || { echo "Error: --loop_top_k requires a value."; exit 1; }
            LOOP_TOP_K="$2"
            shift 2
            ;;
        --save_viser_data)
            USE_SAVE_VISER=1
            shift
            ;;
        --frame_stride)
            [[ $# -ge 2 ]] || { echo "Error: --frame_stride requires a value."; exit 1; }
            FRAME_STRIDE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument '$1'."
            usage
            exit 1
            ;;
    esac
done

if [[ -n "$CUDA_DEVICE" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
fi

if ! [[ "$WIN_ARG" =~ ^[0-9]+$ ]] || [[ "$WIN_ARG" -le 0 ]]; then
    echo "Error: --win must be a positive integer."
    exit 1
fi
window_size="$WIN_ARG"

model_key="${MODEL_ARG,,}"
case "$model_key" in
    loger)
        ckpt_list=("LoGeR")
        ;;
    loger_star|loger-star|logerstar)
        ckpt_list=("LoGeR_star")
        ;;
    pi3|pi3_chunk|pi3-chunk)
        ckpt_list=("Pi3")
        ;;
    all)
        ckpt_list=("LoGeR" "LoGeR_star")
        ;;
    *)
        echo "Error: --model must be one of: loger, loger_star, pi3, all."
        exit 1
        ;;
esac

mode_key="${MODE_ARG,,}"
case "$mode_key" in
    kitti)
        seq="${SEQ_ARG:-${KITTI_SEQ:-}}"
        if [[ -z "$seq" ]]; then
            echo "Error: kitti mode requires --seq (or KITTI_SEQ env var)."
            exit 1
        fi
        input_path="$REPO_ROOT/data/kitti/dataset/sequences/${seq}/image_2"
        gt_traj="$REPO_ROOT/data/kitti/dataset/poses/${seq}.txt"
        end_frame=10000
        ;;
    vbr)
        seq="${SEQ_ARG:-${VBR_SEQ:-}}"
        if [[ -z "$seq" ]]; then
            echo "Error: vbr mode requires --seq (or VBR_SEQ env var)."
            exit 1
        fi
        input_path="$REPO_ROOT/data/vbr/${seq}_processed_aligned/rgb"
        gt_traj="$REPO_ROOT/data/vbr/processed_gt/${seq}_gt.txt"
        end_frame=20000
        ;;
    oxford_spires)
        seq="${SEQ_ARG:-}"
        if [[ -z "$seq" ]]; then
            echo "Error: oxford_spires mode requires --seq."
            exit 1
        fi
        # Default: rectified images from scripts_download/undistort_oxford_spires.py.
        # OXFORD_SEQS_DIR=sequences uses the raw fisheye images as released.
        _ox_root="$REPO_ROOT/data/oxford_spires/${OXFORD_SEQS_DIR:-processed_sequences}"
        input_path="$_ox_root/${seq}/raw/cam0"
        gt_traj="$_ox_root/${seq}/processed/trajectory/gt-tum.txt"
        end_frame=30000
        ;;
    droidw)
        seq="${SEQ_ARG:-}"
        if [[ -z "$seq" ]]; then
            echo "Error: droidw mode requires --seq."
            exit 1
        fi
        # Prepared by scripts_download/download_droidw.sh: images symlink + poses.txt + rgb.txt
        # (rgb.txt carries the image timestamps demo_viser.py reads).
        _droidw_root="${DROIDW_DATA_ROOT:-$REPO_ROOT/data/DROID-W}"
        input_path="$_droidw_root/${seq}/images_anonymized"
        gt_traj="$_droidw_root/${seq}/poses.txt"
        end_frame=100000
        ;;
    *)
        echo "Error: --mode must be one of: kitti, vbr, oxford_spires, droidw."
        exit 1
        ;;
esac

echo "Mode      : ${mode_key}"
echo "Sequence  : ${seq}"
echo "Input path: ${input_path}"
echo "Window    : ${window_size}"

for ckpt_name in "${ckpt_list[@]}"; do
    echo "--- Processing checkpoint: ${ckpt_name} ---"
    config_path="$REPO_ROOT/ckpts/${ckpt_name}/original_config.yaml"
    model_path="$REPO_ROOT/ckpts/${ckpt_name}/latest.pt"

    # Resolve per-checkpoint results dir, then derive per-seq paths from it
    # (mirrors scripts/run_droidw.sh layout: trajectory txt + loop_detection/
    # + logs/ all live as siblings under one RESULTS_DIR per ckpt).
    if [[ -n "$OUTPUT_DIR" ]]; then
        RESULTS_DIR="${OUTPUT_DIR}/${ckpt_name//\//_}"
    elif [[ "$mode_key" == "kitti" ]]; then
        RESULTS_DIR="$REPO_ROOT/results/viser_pi3_kitti/${ckpt_name//\//_}"
    elif [[ "$mode_key" == "oxford_spires" ]]; then
        RESULTS_DIR="$REPO_ROOT/results/viser_pi3_oxford_spires/${ckpt_name//\//_}"
    elif [[ "$mode_key" == "droidw" ]]; then
        RESULTS_DIR="$REPO_ROOT/results/viser_pi3_droidw/${ckpt_name//\//_}"
    else
        RESULTS_DIR="$REPO_ROOT/results/viser_pi3_vbr/${ckpt_name//\//_}"
    fi

    if [[ "$mode_key" == "kitti" || "$mode_key" == "droidw" ]]; then
        OUT_TXT="$RESULTS_DIR/${seq}.txt"
    else
        OUT_TXT="$RESULTS_DIR/${seq}_es.txt"
    fi
    LOOP_DIR="$RESULTS_DIR/loop_detection/${seq}"
    LOG="$RESULTS_DIR/logs/${seq}.log"
    output_txt="$OUT_TXT"   # back-compat: existing references downstream

    mkdir -p "$RESULTS_DIR"
    echo "Output txt: ${output_txt}"

    OVERLAP_ARG_PY=""
    [[ -n "$OVERLAP_ARG" ]] && OVERLAP_ARG_PY="--overlap_size $OVERLAP_ARG"
    NO_TTT_PY=""
    [[ "$NO_TTT" -eq 1 ]] && NO_TTT_PY="--no_ttt"
    NO_SWA_PY=""
    [[ "$NO_SWA" -eq 1 ]] && NO_SWA_PY="--no_swa"
    ADJ_CONSTRAINTS_PY=""
    [[ "$ADJ_CONSTRAINTS" -eq 1 ]] && ADJ_CONSTRAINTS_PY="--pgo_adj_constraints"
    BLOCK_CONSTRAINTS_PY=""
    [[ "$BLOCK_CONSTRAINTS" -eq 1 ]] && BLOCK_CONSTRAINTS_PY="--pgo_block_constraints"
    PGO_SIGMA_BLOCK_PY=""
    [[ -n "$PGO_SIGMA_BLOCK" ]] && PGO_SIGMA_BLOCK_PY="--pgo_sigma_block $PGO_SIGMA_BLOCK"
    PGO_BLOCK_MIDDLE_COUNT_PY=""
    [[ -n "$PGO_BLOCK_MIDDLE_COUNT" ]] && PGO_BLOCK_MIDDLE_COUNT_PY="--pgo_block_middle_count $PGO_BLOCK_MIDDLE_COUNT"
    PGO_DIST_THRESH_PY=""
    [[ -n "$PGO_DIST_THRESH" ]] && PGO_DIST_THRESH_PY="--pgo_dist_thresh $PGO_DIST_THRESH"
    PGO_SIGMA_SEQ_PY=""
    [[ -n "$PGO_SIGMA_SEQ" ]] && PGO_SIGMA_SEQ_PY="--pgo_sigma_seq $PGO_SIGMA_SEQ"
    PGO_SIGMA_LC_PY=""
    [[ -n "$PGO_SIGMA_LC" ]] && PGO_SIGMA_LC_PY="--pgo_sigma_lc $PGO_SIGMA_LC"
    RESET_EVERY_PY="${RESET_EVERY:-5}"
    PGO_PY=""
    [[ "$USE_PGO" -eq 1 ]] && PGO_PY="--pgo"

    if [[ -n "$LC_PTS_DIR" ]]; then
        echo "LC bridge pts output: $LC_PTS_DIR/$seq"
    else
        echo "LC bridge pts output: disabled (pass --lc_pts_dir to save bridge .pt files)"
    fi
    LC_PTS_PY=""
    [[ -n "$LC_PTS_DIR" ]] && LC_PTS_PY="--lc_pts_folder $LC_PTS_DIR/$seq"
    SE3_PY=""
    [[ "$USE_SE3" -eq 1 ]] && SE3_PY="--se3"
    SIM3_PY=""
    [[ "$USE_SIM3" -eq 1 ]] && SIM3_PY="--sim3"
    SIM3_ON_RESET_PY=""
    [[ "$USE_SIM3_ON_RESET" -eq 1 ]] && SIM3_ON_RESET_PY="--sim3_on_reset"
    SIM3_SCALE_MODE_PY=""
    [[ -n "$SIM3_SCALE_MODE" ]] && SIM3_SCALE_MODE_PY="--sim3_scale_mode $SIM3_SCALE_MODE"
    GT_SCALE_PY=""
    if [[ -n "$GT_SCALE" ]]; then
        if [[ -d "$GT_SCALE" ]]; then
            _gt_file="$GT_SCALE/${seq}.txt"
            if [[ -f "$_gt_file" ]]; then
                GT_SCALE_PY="--gt_scale $_gt_file"
                echo "[gt_scale] using $_gt_file"
            else
                echo "[gt_scale] WARNING: $_gt_file not found in dir $GT_SCALE — running without oracle scale"
            fi
        elif [[ -f "$GT_SCALE" ]]; then
            GT_SCALE_PY="--gt_scale $GT_SCALE"
            echo "[gt_scale] using $GT_SCALE"
        else
            echo "[gt_scale] WARNING: $GT_SCALE is neither a file nor a directory — ignoring"
        fi
    fi
    STRIDE_PY=""
    [[ -n "$STRIDE" ]] && STRIDE_PY="--stride $STRIDE"
    SAVE_STRIDE_PY=""
    [[ -n "$SAVE_STRIDE" ]] && SAVE_STRIDE_PY="--save_stride $SAVE_STRIDE"
    LOOP_DETECT_PY=""
    LOOP_RESULT_DIR_PY=""
    if [[ "$USE_LOOP_DETECT" -eq 1 ]]; then
        LOOP_DETECT_PY="--loop_detect"
        mkdir -p "$LOOP_DIR"
        LOOP_RESULT_DIR_PY="--loop_result_dir $LOOP_DIR"
        echo "Loop detection dir: ${LOOP_DIR}"
        # GT is only used by the loop detector to draw loop_vis/loop_traj.png.
        if [[ -f "$gt_traj" ]]; then
            LOOP_RESULT_DIR_PY="$LOOP_RESULT_DIR_PY --gt_traj_path $gt_traj"
        else
            echo "Loop vis: GT not found at ${gt_traj}; loop_traj.png will be skipped"
        fi
    fi
    ONLINE_PGO_PY=""
    [[ "$USE_ONLINE_PGO" -eq 1 ]] && ONLINE_PGO_PY="--online_pgo"
    SAVE_ONLINE_PGO_POSES_DIR_PY=""
    SAVE_WINDOW_LOCAL_PTS_DIR_PY=""
    if [[ -n "$SAVE_ONLINE_PGO_POSES_DIR" ]]; then
        _OPP_DIR="$SAVE_ONLINE_PGO_POSES_DIR/${seq}"
        mkdir -p "$_OPP_DIR"
        SAVE_ONLINE_PGO_POSES_DIR_PY="--save_online_pgo_poses_dir $_OPP_DIR"
        echo "Online PGO pose-snapshot dir: ${_OPP_DIR}"
    fi
    if [[ -n "$SAVE_WINDOW_LOCAL_PTS_DIR" ]]; then
        _WLP_DIR="$SAVE_WINDOW_LOCAL_PTS_DIR/${seq}"
        mkdir -p "$_WLP_DIR"
        SAVE_WINDOW_LOCAL_PTS_DIR_PY="--save_window_local_pts_dir $_WLP_DIR"
        echo "Per-window local-pts dir: ${_WLP_DIR}"
    fi
    LOOP_SAVE_PAIR_VIS_PY=""
    [[ "$USE_LOOP_SAVE_PAIR_VIS" -eq 1 ]] && LOOP_SAVE_PAIR_VIS_PY="--loop_save_pair_vis"
    LC_BRIDGE_NMS_PY=""
    [[ -n "$LC_BRIDGE_NMS" ]] && LC_BRIDGE_NMS_PY="--lc_bridge_nms $LC_BRIDGE_NMS"
    LC_BRIDGE_KEEP_STATE_PY=""
    [[ "$USE_LC_BRIDGE_KEEP_STATE" -eq 1 ]] && LC_BRIDGE_KEEP_STATE_PY="--lc_bridge_keep_state"
    LC_CHORD_ARC_THRESH_PY=""
    [[ -n "$LC_CHORD_ARC_THRESH" ]] && LC_CHORD_ARC_THRESH_PY="--lc_chord_arc_thresh $LC_CHORD_ARC_THRESH"
    LOOP_MIN_FRAME_GAP_PY=""
    [[ -n "$LOOP_MIN_FRAME_GAP" ]] && LOOP_MIN_FRAME_GAP_PY="--loop_min_frame_gap $LOOP_MIN_FRAME_GAP"
    LOOP_SIM_THRESH_PY=""
    [[ -n "$LOOP_SIM_THRESH" ]] && LOOP_SIM_THRESH_PY="--loop_sim_thresh $LOOP_SIM_THRESH"
    LOOP_TOP_K_PY=""
    [[ -n "$LOOP_TOP_K" ]] && LOOP_TOP_K_PY="--loop_top_k $LOOP_TOP_K"

    # Save-viser plumbing: when --save_viser_data is set, force --output_folder
    # to <RESULTS_DIR>/viser_bundles/<seq>/ so demo_viser.py writes both the
    # predictions .pt AND the sibling _viser_config.json there. --seq_name
    # cleans up the auto-generated path-based name to just <seq>.
    SAVE_VISER_PY=""
    SEQ_NAME_PY=""
    OUT_FOLDER_VAL=""
    if [[ "$USE_SAVE_VISER" -eq 1 ]]; then
        VISER_DIR="$RESULTS_DIR/viser_bundles/${seq}"
        mkdir -p "$VISER_DIR"
        OUT_FOLDER_VAL="$VISER_DIR"
        SAVE_VISER_PY="--save_viser_data"
        SEQ_NAME_PY="--seq_name $seq"
        echo "Viser bundle dir: ${VISER_DIR}"
    fi
    FRAME_STRIDE_PY=""
    [[ -n "$FRAME_STRIDE" ]] && FRAME_STRIDE_PY="--frame_stride $FRAME_STRIDE"

    mkdir -p "$(dirname "$LOG")"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python "$REPO_ROOT/demo_viser.py" \
        --input "$input_path" \
        --config "$config_path" \
        --model_name "$model_path" \
        --window_size "$window_size" \
        --end_frame "$end_frame" \
        --skip_viser \
        --output_txt "$output_txt" \
        --output_folder "$OUT_FOLDER_VAL" \
        --reset_every "$RESET_EVERY_PY" \
        $OVERLAP_ARG_PY \
        $NO_TTT_PY \
        $NO_SWA_PY \
        $ADJ_CONSTRAINTS_PY \
        $BLOCK_CONSTRAINTS_PY \
        $PGO_SIGMA_BLOCK_PY \
        $PGO_BLOCK_MIDDLE_COUNT_PY \
        $PGO_DIST_THRESH_PY \
        $PGO_SIGMA_SEQ_PY \
        $PGO_SIGMA_LC_PY \
        $PGO_PY \
        $LC_PTS_PY \
        $SE3_PY \
        $SIM3_PY \
        $SIM3_ON_RESET_PY \
        $GT_SCALE_PY \
        $SIM3_SCALE_MODE_PY \
        $STRIDE_PY \
        $SAVE_STRIDE_PY \
        $LOOP_DETECT_PY \
        $LOOP_RESULT_DIR_PY \
        $ONLINE_PGO_PY \
        $SAVE_ONLINE_PGO_POSES_DIR_PY \
        $SAVE_WINDOW_LOCAL_PTS_DIR_PY \
        $LOOP_SAVE_PAIR_VIS_PY \
        $LC_CHORD_ARC_THRESH_PY \
        $LC_BRIDGE_KEEP_STATE_PY \
        $SAVE_DEBUG_POSES_PY \
        $LC_BRIDGE_NMS_PY \
        $PGO_LC_CHECK_ARGS \
        $LOOP_MIN_FRAME_GAP_PY \
        $LOOP_SIM_THRESH_PY \
        $LOOP_TOP_K_PY \
        $SAVE_VISER_PY \
        $SEQ_NAME_PY \
        $FRAME_STRIDE_PY 2>&1 | tee "$LOG"
    echo "--- Finished processing ${ckpt_name} ---"
    echo ""
done

echo "All requested checkpoints have been processed."
