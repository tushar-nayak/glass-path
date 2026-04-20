from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def require_torch() -> None:
    if torch is None:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError("PyTorch is required for this baseline.") from _TORCH_IMPORT_ERROR


@dataclass
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    num_samples: int
    confusion_matrix: list[list[int]]


def _confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for truth, pred in zip(y_true, y_pred):
        matrix[truth][pred] += 1
    return matrix


def compute_classification_metrics(
    y_true: list[int], y_pred: list[int], num_classes: int
) -> ClassificationMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return ClassificationMetrics(0.0, 0.0, 0.0, 0, [[0] * num_classes for _ in range(num_classes)])
    cm = _confusion_matrix(y_true, y_pred, num_classes)
    accuracy = sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true)
    recalls = []
    f1s = []
    for cls in range(num_classes):
        tp = cm[cls][cls]
        fp = sum(cm[r][cls] for r in range(num_classes) if r != cls)
        fn = sum(cm[cls][c] for c in range(num_classes) if c != cls)
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        recalls.append(recall)
        f1s.append(f1)
    balanced_accuracy = float(sum(recalls) / num_classes)
    macro_f1 = float(sum(f1s) / num_classes)
    return ClassificationMetrics(
        accuracy=float(accuracy),
        balanced_accuracy=balanced_accuracy,
        macro_f1=macro_f1,
        num_samples=len(y_true),
        confusion_matrix=cm,
    )


def save_checkpoint(path: str | Path, model, metadata: dict | None = None) -> None:
    require_torch()
    payload = {
        "state_dict": model.cpu().state_dict(),
        "metadata": metadata or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
