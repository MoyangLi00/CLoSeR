#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

usage() {
    cat <<'EOF_USAGE'
Usage: eval/mv_recon/run.sh [--ckpt CKPT_NAME] [options]

Options:
    --ckpt NAME            Checkpoint name under ckpts/ (default: LoGeR)
    --model-names list     Comma-separated model names (default: pi3)
    --model-update NAME    Override model_update_type passed to launch.py
    --max-frames list      Comma-separated max frame counts (default: 400)
    --frame-sampling MODE  Frame selection mode: max|uniform (default: max)
    --dataset-tag NAME     Dataset tag used in outputs (default: 7scenes)
    --window-size INT      Pi3 window size tag/override (default: 48)
    --overlap-size INT     Pi3 overlap size tag/override (default: 3)
    --num-iterations INT   Pi3 decoding iteration override
    --sim3 [true|false]    Override Pi3 Sim(3) merge flag
    --se3 [true|false]     Override Pi3 SE(3) merge flag
    --pi3-config PATH      Explicit Pi3 config file (auto-detected when absent)
    --no-pi3-config        Disable Pi3 config auto-discovery
    --weights-path PATH    Explicit checkpoint weights file
    --num-processes INT    Accelerate process count (default: 8)
    --port INT             Accelerate main process port (default: 29502)
    --output-root PATH     Base directory for outputs (default: ${REPO_ROOT}/eval_results)
    --save                 Save outputs to disk (default: false)
    -h, --help             Show this message and exit
EOF_USAGE
}

CKPT_RUN=""
DEFAULT_CKPT_NAME="LoGeR"
MODEL_NAMES=("pi3")
MODEL_UPDATE_OVERRIDE=""
MAX_FRAMES=("400")
FRAME_SAMPLING="max"
DATASET_TAG="7scenes"
WINDOW_SIZE="48"
OVERLAP_SIZE="3"
NUM_ITERATIONS=""
SIM3_OVERRIDE=""
SE3_OVERRIDE=""
NUM_PROCESSES=8
MAIN_PORT=29502
OUTPUT_ROOT="${REPO_ROOT}/eval_results"
OUTPUT_ROOT_SPECIFIED=0
WEIGHTS_PATH=""
PI3_CONFIG=""
PI3_CONFIG_SPECIFIED=0
PI3_CONFIG_ENABLED=1
SAVE_OUTPUTS=0

trim_array_values() {
    local -n arr_ref=$1
    for i in "${!arr_ref[@]}"; do
        arr_ref[$i]=$(echo "${arr_ref[$i]}" | xargs)
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)
            CKPT_RUN="$2"
            shift 2
            ;;
        --model-names)
            IFS=',' read -r -a MODEL_NAMES <<< "$2"
            shift 2
            ;;
        --model-update)
            MODEL_UPDATE_OVERRIDE="$2"
            shift 2
            ;;
        --max-frames)
            IFS=',' read -r -a MAX_FRAMES <<< "$2"
            shift 2
            ;;
        --frame-sampling)
            FRAME_SAMPLING="$2"
            shift 2
            ;;
        --dataset-tag)
            DATASET_TAG="$2"
            shift 2
            ;;
        --window-size)
            WINDOW_SIZE="$2"
            shift 2
            ;;
        --overlap-size)
            OVERLAP_SIZE="$2"
            shift 2
            ;;
        --num-iterations)
            NUM_ITERATIONS="$2"
            shift 2
            ;;
        --sim3)
            if [[ $# -ge 2 && "$2" != -* ]]; then
                SIM3_OVERRIDE="$2"
                shift 2
            else
                SIM3_OVERRIDE="true"
                shift
            fi
            ;;
        --se3)
            if [[ $# -ge 2 && "$2" != -* ]]; then
                SE3_OVERRIDE="$2"
                shift 2
            else
                SE3_OVERRIDE="true"
                shift
            fi
            ;;
        --pi3-config)
            PI3_CONFIG="$2"
            PI3_CONFIG_SPECIFIED=1
            shift 2
            ;;
        --no-pi3-config)
            PI3_CONFIG_ENABLED=0
            PI3_CONFIG=""
            PI3_CONFIG_SPECIFIED=0
            shift
            ;;
        --weights-path)
            WEIGHTS_PATH="$2"
            shift 2
            ;;
        --num-processes)
            NUM_PROCESSES="$2"
            shift 2
            ;;
        --port)
            MAIN_PORT="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            OUTPUT_ROOT_SPECIFIED=1
            shift 2
            ;;
        --save)
            SAVE_OUTPUTS=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            if [[ -z "$CKPT_RUN" ]]; then
                CKPT_RUN="$1"
            else
                echo "Unknown argument: $1" >&2
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$CKPT_RUN" ]]; then
    CKPT_RUN="$DEFAULT_CKPT_NAME"
    echo "No checkpoint specified; defaulting to ${CKPT_RUN}."
fi

if [[ ${#MODEL_NAMES[@]} -eq 0 || -z "${MODEL_NAMES[0]:-}" ]]; then
    MODEL_NAMES=("pi3")
fi

if [[ ${#MAX_FRAMES[@]} -eq 0 || -z "${MAX_FRAMES[0]:-}" ]]; then
    MAX_FRAMES=("400")
fi

if [[ -z "${FRAME_SAMPLING:-}" ]]; then
    FRAME_SAMPLING="max"
fi
FRAME_SAMPLING=$(echo "$FRAME_SAMPLING" | tr '[:upper:]' '[:lower:]')
if [[ "$FRAME_SAMPLING" != "max" && "$FRAME_SAMPLING" != "uniform" ]]; then
    echo "Error: --frame-sampling must be one of: max, uniform" >&2
    exit 1
fi

if [[ -z "${DATASET_TAG:-}" ]]; then
    DATASET_TAG="7scenes"
fi

RUN_NAME=$(basename "${CKPT_RUN%/}")
if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="$DEFAULT_CKPT_NAME"
fi

if [[ -z "${WINDOW_SIZE:-}" ]]; then
    WINDOW_SIZE="48"
fi

if [[ -z "${OVERLAP_SIZE:-}" ]]; then
    OVERLAP_SIZE="3"
fi

SIM3_SUFFIX=""
if [[ -n "$SIM3_OVERRIDE" ]]; then
    sim3_lower=$(echo "$SIM3_OVERRIDE" | tr '[:upper:]' '[:lower:]')
    if [[ "$sim3_lower" == "true" ]]; then
        SIM3_SUFFIX="_sim3"
    fi
fi

SE3_SUFFIX=""
if [[ -n "$SE3_OVERRIDE" ]]; then
    se3_lower=$(echo "$SE3_OVERRIDE" | tr '[:upper:]' '[:lower:]')
    if [[ "$se3_lower" == "true" ]]; then
        SE3_SUFFIX="_se3"
    fi
fi
WINDOW_TAG="win${WINDOW_SIZE}o${OVERLAP_SIZE}${SIM3_SUFFIX}${SE3_SUFFIX}"

if [[ $OUTPUT_ROOT_SPECIFIED -eq 0 ]]; then
    OUTPUT_ROOT="${OUTPUT_ROOT%/}/${RUN_NAME}/${WINDOW_TAG}/mv_recon"
fi

ckpt_dir="${REPO_ROOT}/ckpts/${CKPT_RUN}"
config_path="${ckpt_dir}/original_config.yaml"
weights_path="${ckpt_dir}/latest.pt"

if [[ -n "$WEIGHTS_PATH" ]]; then
    weights_path="$WEIGHTS_PATH"
fi

if [[ $PI3_CONFIG_ENABLED -eq 1 && $PI3_CONFIG_SPECIFIED -eq 0 ]]; then
    if [[ -f "$config_path" ]]; then
        PI3_CONFIG="$config_path"
    else
        PI3_CONFIG=""
    fi
fi

if [[ ! -f "$weights_path" ]]; then
    if [[ -z "$WEIGHTS_PATH" && -d "$ckpt_dir" ]]; then
        candidate=$(find "$ckpt_dir" -maxdepth 1 -type f -name '*.pt' | sort | head -n 1 || true)
        if [[ -n "${candidate:-}" ]]; then
            weights_path="$candidate"
            echo "Using checkpoint file ${weights_path}"
        fi
    fi
fi

if [[ ! -f "$weights_path" ]]; then
    echo "Error: checkpoint weights not found (${weights_path})" >&2
    echo "Expected default path: ${ckpt_dir}/latest.pt" >&2
    exit 1
fi

if [[ $PI3_CONFIG_ENABLED -eq 1 && -n "$PI3_CONFIG" && ! -f "$PI3_CONFIG" ]]; then
    echo "Warning: Pi3 config not found (${PI3_CONFIG}); continuing without it." >&2
    PI3_CONFIG=""
fi

mkdir -p "$OUTPUT_ROOT"

trim_array_values MODEL_NAMES
trim_array_values MAX_FRAMES

pi3_forward_args=()
if [[ -n "$PI3_CONFIG" ]]; then
    pi3_forward_args+=(--pi3_config "$PI3_CONFIG")
fi
if [[ -n "$WINDOW_SIZE" ]]; then pi3_forward_args+=(--pi3_window_size "$WINDOW_SIZE"); fi
if [[ -n "$OVERLAP_SIZE" ]]; then pi3_forward_args+=(--pi3_overlap_size "$OVERLAP_SIZE"); fi
if [[ -n "$NUM_ITERATIONS" ]]; then pi3_forward_args+=(--pi3_num_iterations "$NUM_ITERATIONS"); fi
if [[ -n "$SIM3_OVERRIDE" ]]; then pi3_forward_args+=(--pi3_sim3 "$SIM3_OVERRIDE"); fi
if [[ -n "$SE3_OVERRIDE" ]]; then pi3_forward_args+=(--pi3_se3 "$SE3_OVERRIDE"); fi

save_flag=()
if [[ "$SAVE_OUTPUTS" -eq 1 ]]; then
    save_flag+=(--save)
fi

for model_name in "${MODEL_NAMES[@]}"; do
    model_update="$model_name"
    if [[ -n "$MODEL_UPDATE_OVERRIDE" ]]; then
        model_update="$MODEL_UPDATE_OVERRIDE"
    fi
    normalized_model=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')
    use_pi3_args=0
    if [[ "$normalized_model" == pi3* || "$normalized_model" == *pi3* ]]; then
        use_pi3_args=1
    elif [[ $PI3_CONFIG_SPECIFIED -eq 1 && -n "$PI3_CONFIG" ]]; then
        use_pi3_args=1
    fi

    for max_frames in "${MAX_FRAMES[@]}"; do
        frames_clean=${max_frames//[^0-9]/}
        if [[ -z "$frames_clean" ]]; then
            frames_clean="$max_frames"
        fi
        run_tag="${DATASET_TAG}_${FRAME_SAMPLING}_${frames_clean}"
        output_dir="${OUTPUT_ROOT%/}/${run_tag}/${model_name}"
        mkdir -p "$output_dir"
        echo "\n>>> Evaluating ${model_name} with ${max_frames} frames (${FRAME_SAMPLING} sampling, output: ${output_dir})"

        extra_args=()
        if [[ $use_pi3_args -eq 1 && ${#pi3_forward_args[@]} -gt 0 ]]; then
            extra_args+=("${pi3_forward_args[@]}")
        fi

        ACCELERATE_TIMEOUT=360000 NCCL_TIMEOUT=360000 accelerate launch --num_processes "$NUM_PROCESSES" --multi_gpu --num_machines 1 --main_process_port "$MAIN_PORT" eval/mv_recon/launch.py \
            --weights "$weights_path" \
            --output_dir "$output_dir" \
            --model_name "$model_name" \
            --model_update_type "$model_update" \
            --max_frames "$max_frames" \
            --frame_sampling "$FRAME_SAMPLING" \
            "${save_flag[@]}" \
            --dataset "$DATASET_TAG" \
            "${extra_args[@]}"
        printf '\n'
    done
done
