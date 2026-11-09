#!/usr/bin/env bash
# PEMS reference run for experiment 5: the L=2, U=2 model with static covariates.
# Usage: bash experiments/pems_baseline.sh pems04|pems08 [train_tfp.py options...]
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
dataset="${1:?usage: pems_baseline.sh pems04|pems08}"
shift
run_pems_experiment "$dataset" pems-baseline "$@"
