#!/usr/bin/env bash
# Experiment 4: shrink TimeFilter Q/K from 768 to 120 and raise weight decay from 1e-4 to 5e-4.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_experiment exp4-compact-timefilter --tfp_proj_dim 120 --wdecay 0.0005 "$@"
