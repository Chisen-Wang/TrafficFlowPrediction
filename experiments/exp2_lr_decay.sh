#!/usr/bin/env bash
# Experiment 2: halve the learning rate after epochs 5, 10, 20 and 40.
# The constant-rate run peaked at epoch 7, so decay starts just before that.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_experiment exp2-lr-decay --lr_milestones 5 10 20 40 --lr_gamma 0.5 "$@"
