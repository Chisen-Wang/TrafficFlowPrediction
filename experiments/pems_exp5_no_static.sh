#!/usr/bin/env bash
# Experiment 5 on PEMS: remove static covariates from the UniCA adapter.
# Usage: bash experiments/pems_exp5_no_static.sh pems04|pems08 [train_tfp.py options...]
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
dataset="${1:?usage: pems_exp5_no_static.sh pems04|pems08}"
shift
run_pems_experiment "$dataset" pems-exp5-no-static --no-unica_static "$@"
