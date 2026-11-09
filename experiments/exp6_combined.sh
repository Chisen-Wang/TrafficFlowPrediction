#!/usr/bin/env bash
# Experiment 6: the four changes that each beat the baseline, together:
# no static covariates (exp5), learning-rate decay (exp2), compact TimeFilter with weight decay 5e-4 (exp4).
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_experiment exp6-combined \
  --no-unica_static --lr_milestones 5 10 20 40 --lr_gamma 0.5 --tfp_proj_dim 120 --wdecay 0.0005 "$@"
