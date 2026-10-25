#!/usr/bin/env bash
# Usage: bash train_nyc_bike.sh [tfp|unica] [train_tfp.py options...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mode="${1:-unica}"
if (($# > 0)); then shift; fi
extra=()
case "$mode" in
  tfp) ;;
  unica) extra=(--unica_enabled --unica_with_gate --unica_heads 4) ;;
  *) echo "Expected tfp or unica" >&2; exit 2 ;;
esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
exec python -B train_tfp.py --data nyc_bike --input_len 24 --output_len 24 \
  --steps_per_day 24 --batch_size 4 --llm_layer 6 --U 1 \
  --epochs 200 --es_patience 50 --model_selection_split val --shuffle_train \
  --save "./logs/$(date +%Y-%m-%d-%H-%M-%S)-nyc-${mode}-" "${extra[@]}" "$@"
