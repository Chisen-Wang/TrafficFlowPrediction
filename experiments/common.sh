#!/usr/bin/env bash
# Shared settings, sourced by the exp*.sh scripts. Every experiment is the
# NYC-Bike L=2, U=2 model with one change, so these values stay fixed.

run_experiment() {
  local name="$1"
  shift
  cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
  local seed="${SEED:-6666}"
  local run="${name}-seed${seed}-$(date +%Y%m%d-%H%M%S)-"
  mkdir -p logs
  MODE=unica LLM_LAYER=2 U=2 BATCH_SIZE=32 EPOCHS=300 LRATE=0.0003 WDECAY=0.0001 ES_PATIENCE=20 \
    SEED="$seed" SAVE_PREFIX="./logs/$run" bash run_nyc_bike.sh "$@" 2>&1 | tee "logs/${run}nyc_bike.log"
}

# The same idea for PEMS04/PEMS08, with the PEMS launchers' own learning rate and epoch budget.
run_pems_experiment() {
  local dataset="$1" name="$2"
  shift 2
  case "$dataset" in
    pems04|pems08) ;;
    *) echo "Expected pems04 or pems08, got: $dataset" >&2; return 2 ;;
  esac
  cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
  local seed="${SEED:-6666}"
  local run="${name}-seed${seed}-$(date +%Y%m%d-%H%M%S)-"
  mkdir -p logs
  MODE=unica LLM_LAYER=2 U=2 BATCH_SIZE=32 EPOCHS=300 LRATE=0.001 WDECAY=0.0001 ES_PATIENCE=50 \
    SEED="$seed" SAVE_PREFIX="./logs/$run" bash "run_${dataset}.sh" "$@" 2>&1 | tee "logs/${run}${dataset}.log"
}
