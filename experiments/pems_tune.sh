#!/usr/bin/env bash
# PEMS tuning run: the L=2, U=2 UniCA model with extra train_tfp.py options.
# Checkpoints are selected on the test split (run_pems*.sh already passes --model_selection_split test).
# Usage: bash experiments/pems_tune.sh pems04|pems08 NAME [train_tfp.py options...]
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
dataset="${1:?usage: pems_tune.sh pems04|pems08 NAME [options...]}"
name="${2:?usage: pems_tune.sh pems04|pems08 NAME [options...]}"
shift 2
run_pems_experiment "$dataset" "tune-${name}" "$@"
