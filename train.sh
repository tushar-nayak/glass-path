#!/usr/bin/env bash
set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate bme_ml

export KMP_DUPLICATE_LIB_OK=TRUE

IMAGE_ROOT="${1:-}"
if [[ -z "${IMAGE_ROOT}" ]]; then
  echo "Usage: $0 /path/to/images [extra training args]" >&2
  echo "Example: $0 /data/LungHist700/images --ssl-dim 256 --graph-dim 128" >&2
  exit 1
fi

shift || true

python -m code.cli train \
  --csv data/data.csv \
  --image-root "${IMAGE_ROOT}" \
  --device auto \
  "$@"
