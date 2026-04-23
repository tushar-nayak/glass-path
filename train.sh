#!/usr/bin/env bash
set -euo pipefail

PYTHON="python"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
fi

export KMP_DUPLICATE_LIB_OK=TRUE

IMAGE_ROOT="data/images"
if [[ "${1:-}" != "" && "${1:-}" != -* ]]; then
  IMAGE_ROOT="$1"
  shift
fi

"${PYTHON}" -m code.cli train \
  --csv data/data.csv \
  --image-root "${IMAGE_ROOT}" \
  --device auto \
  "$@"
