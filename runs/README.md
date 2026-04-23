# Runs

This folder is for local run artifacts (logs, checkpoints, metrics). Large outputs are intentionally gitignored.

## 2026-04-23_cpu_16t

Location (not committed): `runs/2026-04-23_cpu_16t/`

Command (CPU, 16 threads, page-cache warmup):
- Script: `./run_cpu_train_16t.sh`
- Outputs:
  - `runs/2026-04-23_cpu_16t/train.log`
  - `runs/2026-04-23_cpu_16t/checkpoints/{ssl_backbone.pt,classifier.pt,metrics.json}`

Results (from `metrics.json`):
- `best_round`: `0`
- Validation (n=75):
  - accuracy: `0.36`
  - balanced_accuracy: `0.3333333333333333`
  - macro_f1: `0.17647058823529413`
  - confusion_matrix:
    - `[[27,0,0],[8,0,0],[40,0,0]]`
- Test (n=69):
  - accuracy: `0.4927536231884058`
  - balanced_accuracy: `0.3333333333333333`
  - macro_f1: `0.22006472491909387`
  - confusion_matrix:
    - `[[34,0,0],[10,0,0],[25,0,0]]`

Notes:
- The confusion matrices show the model predicted class `0` for every sample; balanced accuracy staying at `1/3` reflects that.

