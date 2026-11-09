#!/usr/bin/env bash
# Experiment 1: freeze base attention so only LoRA and LayerNorm train in the graph layers.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_experiment exp1-lora-only --lora_only "$@"
