#!/usr/bin/env bash
set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate bme_ml

export KMP_DUPLICATE_LIB_OK=TRUE

IMAGE_ROOT="data/images"
if [[ "${1:-}" != "" && "${1:-}" != -* ]]; then
  IMAGE_ROOT="$1"
  shift
fi

python -m code.cli train \
  --csv data/data.csv \
  --image-root "${IMAGE_ROOT}" \
  --device auto \
  "$@"
