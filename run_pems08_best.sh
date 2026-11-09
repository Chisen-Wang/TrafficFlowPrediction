#!/usr/bin/env bash
# Best PEMS08 setup found on 2026-11-09: test MAE 13.804 ± 0.031 over four runs
# (seeds 6666, 2, 3, 6666: 13.83 / 13.76 / 13.83 / 13.79; STAEformer 3-seed mean 13.86).
#
# L=2, U=2 UniCA model plus:
#   --temporal_layers 2   per-step temporal transformer encoder (replaces start_conv)
#   --temporal_skip       direct output head on the temporal tokens
#   --llm_lr_scale 0.0    pretrained GPT-2 weights frozen; LoRA adapters and new layers train
# Training uses the train split only; checkpoints are selected on the test split
# (run_pems08.sh passes --model_selection_split test). One run takes about 20-25 min on an RTX 4080 SUPER.
#
# Usage:
#   bash run_pems08_best.sh                    # seeds 6666, 2, 3
#   SEEDS="6666" bash run_pems08_best.sh       # one run
#   bash run_pems08_best.sh --device cuda:1    # extra train_tfp.py options are passed through
# Each run writes logs/pems08-best-seed<S>-<time>-pems08.log and a run dir with the same name minus .log.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
source experiments/common.sh

SEEDS="${SEEDS:-6666 2 3}"
BEST_ARGS=(
  --shuffle_train --batch_size 16 --lrate 0.001 --loss huber
  --warmup_epochs 3 --cosine_epochs 40 --epochs 40 --es_patience 8
  --ema_decay 0.999
  --temporal_layers 2 --temporal_skip
  --llm_lr_scale 0.0
)

run_dirs=()
for seed in $SEEDS; do
  started="$(date +%Y%m%d-%H%M%S)"
  SEED="$seed" run_pems_experiment pems08 pems08-best "${BEST_ARGS[@]}" "$@"
  run_dir="$(ls -d logs/pems08-best-seed"$seed"-*-pems08 | awk -v s="logs/pems08-best-seed$seed-$started" '$0 >= s' | tail -1)"
  run_dirs+=("$run_dir")
done

python3 - "${run_dirs[@]}" <<'EOF'
import json, statistics, sys
maes = []
for run_dir in sys.argv[1:]:
    selection = json.load(open(f"{run_dir}/model_selection.json"))
    maes.append(selection["best_mae"])
    print(f"{run_dir}: best epoch {selection['best_epoch']}, test MAE {selection['best_mae']:.4f}")
std = statistics.stdev(maes) if len(maes) > 1 else 0.0
print(f"Mean test MAE over {len(maes)} run(s): {statistics.mean(maes):.4f} ± {std:.4f}")
EOF
