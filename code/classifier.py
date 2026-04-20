from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    torch = None
    nn = None
    DataLoader = None
    Dataset = object
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from .data import SampleRecord
from .image import AugmentationConfig, augment_view, load_rgb_image
from .federated import weighted_average_state_dicts
from .ssl import PathologyBackbone, require_torch


@dataclass
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    num_samples: int
    confusion_matrix: list[list[int]]


class SupervisedImageDataset(Dataset):
    def __init__(self, records, label_to_index: dict[str, int], image_size: int = 224):
        require_torch()
        self.records = list(records)
        self.label_to_index = label_to_index
        self.image_size = image_size
        self.aug = AugmentationConfig(image_size=image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        if record.image_path is None:
            raise ValueError(
                "Encountered a record without an image_path. Resolve paths before training."
            )
        image = load_rgb_image(record.image_path, image_size=self.image_size)
        rng = np.random.default_rng()
        view = augment_view(image, rng, self.aug)
        view = torch.from_numpy(view).float()
        label = torch.tensor(self.label_to_index[record.label], dtype=torch.long)
        return view, label


class EvaluationImageDataset(Dataset):
    def __init__(self, records, label_to_index: dict[str, int], image_size: int = 224):
        require_torch()
        self.records = list(records)
        self.label_to_index = label_to_index
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        if record.image_path is None:
            raise ValueError(
                "Encountered a record without an image_path. Resolve paths before training."
            )
        image = load_rgb_image(record.image_path, image_size=self.image_size)
        view = torch.from_numpy(image).float()
        label = torch.tensor(self.label_to_index[record.label], dtype=torch.long)
        return view, label


class PathologyClassifier(nn.Module if nn is not None else object):
    def __init__(self, backbone: PathologyBackbone, num_classes: int, freeze_backbone: bool = True):
        require_torch()
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embedding_dim, num_classes)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


@dataclass
class FederatedClassifierConfig:
    image_size: int = 224
    batch_size: int = 16
    local_epochs: int = 1
    rounds: int = 2
    lr: float = 1e-3
    device: str = "cpu"
    freeze_backbone: bool = True


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
        tn = sum(cm[r][c] for r in range(num_classes) for c in range(num_classes)) - tp - fp - fn
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


def load_checkpoint(path: str | Path, model):
    require_torch()
    payload = torch.load(Path(path), map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    return model, payload.get("metadata", {})


def evaluate_classifier(
    model: PathologyClassifier,
    records,
    label_to_index: dict[str, int],
    image_size: int = 224,
    batch_size: int = 16,
    device: str = "cpu",
) -> ClassificationMetrics:
    require_torch()
    dataset = EvaluationImageDataset(records, label_to_index=label_to_index, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model = model.to(device)
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for view, label in loader:
            view = view.to(device)
            logits = model(view)
            pred = torch.argmax(logits, dim=-1).cpu().tolist()
            y_pred.extend(int(p) for p in pred)
            y_true.extend(int(t) for t in label.tolist())
    return compute_classification_metrics(y_true, y_pred, num_classes=len(label_to_index))


class FederatedClassifierTrainer:
    def __init__(self, config: FederatedClassifierConfig, label_to_index: dict[str, int]):
        require_torch()
        self.config = config
        self.label_to_index = label_to_index

    def _make_loader(self, records) -> DataLoader:
        dataset = SupervisedImageDataset(
            records,
            label_to_index=self.label_to_index,
            image_size=self.config.image_size,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def _train_local(self, model: PathologyClassifier, loader: DataLoader) -> dict:
        model.to(self.config.device)
        model.train()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=self.config.lr
        )
        criterion = nn.CrossEntropyLoss()
        for _ in range(self.config.local_epochs):
            for view, label in loader:
                view = view.to(self.config.device)
                label = label.to(self.config.device)
                logits = model(view)
                loss = criterion(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.state_dict()

    def fit(self, patient_records: Iterable[list], backbone: PathologyBackbone) -> PathologyClassifier:
        loaders = [self._make_loader(records) for records in patient_records if len(records) > 0]
        if not loaders:
            raise ValueError("No labeled images were found. Resolve paths before training.")
        global_model = PathologyClassifier(
            copy.deepcopy(backbone),
            num_classes=len(self.label_to_index),
            freeze_backbone=self.config.freeze_backbone,
        ).to(self.config.device)
        for _ in range(self.config.rounds):
            local_states = []
            weights = []
            for loader in loaders:
                local_model = PathologyClassifier(
                    copy.deepcopy(backbone),
                    num_classes=len(self.label_to_index),
                    freeze_backbone=self.config.freeze_backbone,
                ).to(self.config.device)
                local_model.load_state_dict(global_model.state_dict())
                state = self._train_local(local_model, loader)
                local_states.append(state)
                weights.append(float(len(loader.dataset)))
            global_model.load_state_dict(weighted_average_state_dicts(local_states, weights))
        return global_model.cpu()

    def train_and_evaluate(
        self,
        train_records: Iterable[list],
        val_records: Iterable[list],
        test_records: Iterable[list],
        backbone: PathologyBackbone,
    ) -> tuple[PathologyClassifier, dict]:
        model = self.fit(train_records, backbone)
        val_flat = [record for records in val_records for record in records]
        test_flat = [record for records in test_records for record in records]
        val_metrics = evaluate_classifier(
            model,
            val_flat,
            self.label_to_index,
            image_size=self.config.image_size,
            batch_size=self.config.batch_size,
            device=self.config.device,
        )
        test_metrics = evaluate_classifier(
            model,
            test_flat,
            self.label_to_index,
            image_size=self.config.image_size,
            batch_size=self.config.batch_size,
            device=self.config.device,
        )
        return model, {
            "val": val_metrics,
            "test": test_metrics,
        }

    def fit_from_frame(self, frame, backbone: PathologyBackbone) -> PathologyClassifier:
        records_by_patient = []
        for _, group in frame.groupby("patient_id", sort=True):
            records = [
                SampleRecord(
                    superclass=str(r.superclass),
                    subclass=str(r.subclass),
                    resolution=str(r.resolution),
                    image_id=str(r.image_id),
                    patient_id=str(r.patient_id),
                    image_path=str(getattr(r, "image_path", None))
                    if getattr(r, "image_path", None) is not None
                    and str(getattr(r, "image_path", None))
                    and str(getattr(r, "image_path", None)) != "nan"
                    else None,
                )
                for r in group.itertuples(index=False)
                if getattr(r, "image_path", None) is not None
                and str(getattr(r, "image_path", None))
                and str(getattr(r, "image_path", None)) != "nan"
            ]
            if records:
                records_by_patient.append(records)
        return self.fit(records_by_patient, backbone)
