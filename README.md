# GLASS (Graph-based Learning Architecture for Spatial Structures) in Pathology

Unified federated pipeline for self-supervised image encoding, local graph reasoning, and concept-bottleneck classification.

## Setup

This repo is Python 3.11+ (we currently use Python 3.13 in this workspace).

Create a virtualenv and install dependencies:

```bash
cd glass-path
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Notes:
- `.venv/` is intentionally gitignored.
- `torchvision` is only needed for the `resnet` baseline command.

## Dataset

The pipeline expects a CSV with (at minimum) these columns:

- `superclass` (label, e.g. `aca`, `scc`, `nor`)
- `subclass` (may be empty for some rows)
- `resolution` (e.g. `20x`, `40x`)
- `image_id`
- `patient_id` (treated as the federated "client" id)

By default it resolves image paths under `data/images/` using filename templates in `code/data.py`.

## Commands

Inspect the dataset:

```bash
./first_run.sh inspect
# or:
.venv/bin/python -m code.cli inspect --csv data/data.csv
```

Verify that all CSV rows resolve to an image file under `data/images/`:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from code.data import GlassDataset

dataset = GlassDataset.from_csv("data/data.csv")
resolved = dataset.resolve_image_paths(Path("data/images"))
missing = int(resolved["image_path"].isna().sum())
print({"rows": int(len(resolved)), "missing_image_paths": missing})
PY
```

### Train (Low Resource / Smoke)

When sharing a machine, limit CPU threads and subsample patients/images:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -u -m code.cli train \
  --csv data/data.csv \
  --image-root data/images \
  --device cpu \
  --num-threads 1 \
  --max-patients 6 \
  --max-images-per-patient 4 \
  --image-size 64 \
  --quick \
  --save-dir checkpoints_smoke
```

Notes:
- `--max-patients` is deterministic and label-stratified (round-robin across labels) to make smoke runs cover multiple classes.
- With very small patient counts, validation/test splits may be empty (the splitter is patient-level and label-stratified).

### Train (Full)

This runs federated SSL pretraining and then federated supervised classification:

```bash
.venv/bin/python -m code.cli train \
  --csv data/data.csv \
  --image-root data/images \
  --device auto \
  --save-dir checkpoints
```

### ResNet Baseline

```bash
.venv/bin/python -m code.cli resnet \
  --csv data/data.csv \
  --image-root data/images \
  --device auto \
  --save-dir checkpoints_resnet
```
