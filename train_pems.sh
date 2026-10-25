#!/usr/bin/env bash
# Usage: bash train_pems.sh pems07 [tfp|unica|adapter_only] [train_tfp.py options...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
dataset="${1:-pems07}"
mode="${2:-tfp}"
if (($# > 0)); then shift; fi
if (($# > 0)); then shift; fi
case "$dataset" in
  pems07) batch_size=4 ;;
  pems04|pems08) batch_size=32 ;;
  *) echo "Expected dataset pems04, pems07 or pems08" >&2; exit 2 ;;
esac
mode_args=()
case "$mode" in
  tfp) ;;
  unica) mode_args=(--unica_enabled --unica_training_policy adapter_lora) ;;
  adapter_only) mode_args=(--unica_enabled --unica_training_policy adapter_only --tfp_moe_loss_weight 0.0) ;;
  *) echo "Expected mode tfp, unica or adapter_only" >&2; exit 2 ;;
esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
data_args=()
if [[ "$dataset" == pems04 ]]; then data_args=(--data_dir data/pems04_12_12); fi
exec python -B train_tfp.py --data "$dataset" "${data_args[@]}" \
  --input_len 12 --output_len 12 --batch_size "$batch_size" \
  --llm_layer 6 --U 1 --epochs 200 --es_patience 50 \
  --save "./logs/$(date +%Y-%m-%d-%H-%M-%S)-${mode}-" \
  "${mode_args[@]}" "$@"
