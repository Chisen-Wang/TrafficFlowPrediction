#!/usr/bin/env bash
set -euo pipefail

# 修改这里的默认值，或用环境变量 / 命令行临时覆盖。
MODE="${MODE:-unica}"              # unica / tfp / adapter_only
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-300}"
LRATE="${LRATE:-0.0003}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-6666}"
ES_PATIENCE="${ES_PATIENCE:-100}"
WDECAY="${WDECAY:-0.0001}"
CLIP="${CLIP:-5.0}"

INPUT_LEN="${INPUT_LEN:-24}"
OUTPUT_LEN="${OUTPUT_LEN:-24}"
STEPS_PER_DAY=24
LLM_LAYER="${LLM_LAYER:-2}"
U="${U:-2}"
TFP_INPUT_NORM="${TFP_INPUT_NORM:-1}" # 0 / 1
SHUFFLE_TRAIN="${SHUFFLE_TRAIN:-1}"  # 0 / 1
DATA_DIR="${DATA_DIR:-data/nyc_bike_${INPUT_LEN}_${OUTPUT_LEN}}"

UNICA_HEADS="${UNICA_HEADS:-4}"
UNICA_BOTTLENECK_DIM="${UNICA_BOTTLENECK_DIM:-128}" # 旧版完整宽度为 768
UNICA_DROPOUT="${UNICA_DROPOUT:-0.1}"
UNICA_WITH_GATE="${UNICA_WITH_GATE:-1}"
UNICA_WITH_FUTURE="${UNICA_WITH_FUTURE:-1}"
PRETRAINED_PATH="${PRETRAINED_PATH:-}"  # adapter_only 必填，指向匹配的 TFP 权重

CONDA_ENV="${CONDA_ENV:-dl}"
OFFLINE="${OFFLINE:-1}"             # GPT-2 已缓存用 1；需要联网下载时改为 0
SAVE_PREFIX="${SAVE_PREFIX:-}"       # 留空自动生成；最终目录会追加 nyc_bike
EXTRA_ARGS=(
  # --tfp_alpha 0.1
  # --tfp_top_p 0.5
  # --tfp_moe_loss_weight 0.03 \
  --model_selection_split test
)

source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/run_training_common.sh"
run_training nyc_bike "$@"
