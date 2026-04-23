#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="python"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
fi

echo "[preload] warming page cache for data/images"
"${PYTHON}" - <<'PY'
from pathlib import Path
import time

root = Path("data/images")
files = sorted(p for p in root.rglob("*") if p.is_file())
start = time.time()
bytes_read = 0
for p in files:
    with p.open("rb") as f:
        b = f.read()
    bytes_read += len(b)
print({"files": len(files), "bytes_gb": round(bytes_read / 1e9, 3), "sec": round(time.time() - start, 2)})
PY

echo "[train] starting"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

"${PYTHON}" -u -m code.cli train \
  --csv data/data.csv \
  --image-root data/images \
  --device cpu \
  --num-threads 16 \
  --save-dir checkpoints_cpu_16t \
  2>&1 | tee train_cpu_16t.log

