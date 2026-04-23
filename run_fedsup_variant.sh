#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="python"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
fi

: "${CSV:=data/data.csv}"
: "${IMAGE_ROOT:=data/images}"
: "${DEVICE:=cuda}"
: "${NUM_THREADS:=4}"
: "${IMAGE_SIZE:=224}"
: "${BACKBONE:=resnet50}"
: "${PRETRAINED:=1}"
: "${FREEZE_BACKBONE:=0}"
: "${CLASS_WEIGHTING:=balanced}"
: "${CLIENT_FRACTION:=1.0}"
: "${PROX_MU:=0.01}"
: "${CLIENT_SAMPLING:=shuffle}"
: "${ROUNDS:=12}"
: "${LOCAL_EPOCHS:=1}"
: "${BATCH_SIZE:=16}"
: "${LR:=1e-4}"
: "${VERBOSE:=1}"
: "${SAVE_DIR:=runs/fedsup_variant}"
: "${VAL_FRACTION:=0.15}"
: "${TEST_FRACTION:=0.15}"
: "${SEED:=42}"

mkdir -p "${SAVE_DIR}"

cmd=(
  "${PYTHON}" -u -m code.cli fedsup
  --csv "${CSV}"
  --image-root "${IMAGE_ROOT}"
  --device "${DEVICE}"
  --num-threads "${NUM_THREADS}"
  --image-size "${IMAGE_SIZE}"
  --backbone "${BACKBONE}"
  --class-weighting "${CLASS_WEIGHTING}"
  --client-fraction "${CLIENT_FRACTION}"
  --prox-mu "${PROX_MU}"
  --client-sampling "${CLIENT_SAMPLING}"
  --rounds "${ROUNDS}"
  --local-epochs "${LOCAL_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --save-dir "${SAVE_DIR}"
  --val-fraction "${VAL_FRACTION}"
  --test-fraction "${TEST_FRACTION}"
  --seed "${SEED}"
)

if [[ "${PRETRAINED}" == "1" ]]; then
  cmd+=(--pretrained)
fi
if [[ "${FREEZE_BACKBONE}" == "1" ]]; then
  cmd+=(--freeze-backbone)
fi
if [[ "${VERBOSE}" == "1" ]]; then
  cmd+=(--verbose)
fi

export OMP_NUM_THREADS="${NUM_THREADS}"
export MKL_NUM_THREADS="${NUM_THREADS}"
"${cmd[@]}" 2>&1 | tee "${SAVE_DIR}/train.log"
