#!/usr/bin/env bash
# Shared launcher, sourced by run_pems04.sh / run_pems08.sh / run_nyc_bike.sh.

run_training() {
  local dataset="$1"
  shift
  local project_root
  project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "$project_root"

  local dry_run=0
  local cli_args=()
  local arg
  for arg in "$@"; do
    if [[ "$arg" == --dry-run ]]; then
      dry_run=1
    else
      cli_args+=("$arg")
    fi
  done

  local mode_args=()
  case "$MODE" in
    tfp) ;;
    unica) mode_args=(--unica_enabled --unica_training_policy adapter_lora) ;;
    adapter_only)
      mode_args=(--unica_enabled --unica_training_policy adapter_only --tfp_moe_loss_weight 0.0)
      ;;
    *) echo "MODE must be tfp, unica or adapter_only: $MODE" >&2; return 2 ;;
  esac
  if [[ "$MODE" != tfp ]]; then
    mode_args+=(--unica_heads "$UNICA_HEADS" --unica_dropout "$UNICA_DROPOUT"
                --unica_bottleneck_dim "${UNICA_BOTTLENECK_DIM:-128}")
    if [[ "$UNICA_WITH_GATE" == 1 ]]; then
      mode_args+=(--unica_with_gate)
    fi
    if [[ "$UNICA_WITH_FUTURE" == 1 ]]; then
      mode_args+=(--unica_with_future)
    else
      mode_args+=(--no-unica_with_future)
    fi
    if [[ -n "$PRETRAINED_PATH" ]]; then
      mode_args+=(--unica_pretrained_path "$PRETRAINED_PATH")
    fi
  fi

  local shuffle_args=()
  if [[ "$SHUFFLE_TRAIN" == 1 ]]; then
    shuffle_args=(--shuffle_train)
  fi
  local norm_args=(--tfp_input_norm)
  if [[ "${TFP_INPUT_NORM:-1}" == 0 ]]; then
    norm_args=(--no-tfp_input_norm)
  fi
  local save_prefix="${SAVE_PREFIX:-./logs/$(date +%Y-%m-%d-%H-%M-%S-%N)-${MODE}-}"
  local command=(
    python -B train_tfp.py
    --data "$dataset" --data_dir "$DATA_DIR" --device "$DEVICE"
    --input_len "$INPUT_LEN" --output_len "$OUTPUT_LEN" --steps_per_day "$STEPS_PER_DAY"
    --batch_size "$BATCH_SIZE" --epochs "$EPOCHS" --lrate "$LRATE"
    --wdecay "$WDECAY" --clip "$CLIP" --seed "$SEED"
    --llm_layer "$LLM_LAYER" --U "$U" --es_patience "$ES_PATIENCE"
    --model_selection_split val --save "$save_prefix"
    "${norm_args[@]}" "${shuffle_args[@]}" "${mode_args[@]}" "${EXTRA_ARGS[@]}" "${cli_args[@]}"
  )

  printf 'Conda environment: %s; offline: %s\n' "$CONDA_ENV" "$OFFLINE"
  printf 'Command:'
  printf ' %q' "${command[@]}"
  printf '\n'
  if ((dry_run)); then
    return 0
  fi

  if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    local conda_sh="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
    if [[ ! -f "$conda_sh" ]]; then
      echo "Conda initialization script not found: $conda_sh" >&2
      echo "Set CONDA_SH, or activate $CONDA_ENV before running this script." >&2
      return 2
    fi
    source "$conda_sh"
    conda activate "$CONDA_ENV"
  fi
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  export HF_HUB_OFFLINE="$OFFLINE" TRANSFORMERS_OFFLINE="$OFFLINE"
  exec "${command[@]}"
}
