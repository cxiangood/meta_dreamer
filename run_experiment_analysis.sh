#!/bin/bash

set -euo pipefail

ROOTS=${ROOTS:-"dreamer logs result"}
OUTDIR=${OUTDIR:-result/analysis}
TOPK=${TOPK:-10}

# shellcheck disable=SC2086
python3 dreamer/tools/analyze_experiments.py \
  --roots $ROOTS \
  --outdir "$OUTDIR" \
  --topk "$TOPK"

