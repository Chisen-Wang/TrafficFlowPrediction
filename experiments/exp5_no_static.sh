#!/usr/bin/env bash
# Experiment 5: remove static (POI and location) covariates from the UniCA adapter.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_experiment exp5-no-static --no-unica_static "$@"
