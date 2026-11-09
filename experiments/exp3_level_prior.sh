#!/usr/bin/env bash
# Experiment 3: give the model each station's recent level through the rescaled weekly profile.
#   residual (default): predict the profile's error, with the profile also as future covariates.
#   LEVEL_PRIOR=covariate: future covariates only.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
variant="${LEVEL_PRIOR:-residual}"
run_experiment "exp3-level-prior-$variant" --level_prior "$variant" "$@"
