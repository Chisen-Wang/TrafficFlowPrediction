#!/usr/bin/env bash
# Experiment 7: experiment 6 plus the level prior (exp3), which helped long horizons and RMSE most.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_experiment exp7-combined-level-prior \
  --no-unica_static --lr_milestones 5 10 20 40 --lr_gamma 0.5 --tfp_proj_dim 120 --wdecay 0.0005 \
  --level_prior residual "$@"
