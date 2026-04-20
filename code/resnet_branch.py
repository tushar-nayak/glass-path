from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    torch = None
    nn = None
    DataLoader = None
    Dataset = object
    models = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from .data import SampleRecord
from .image import AugmentationConfig, augment_view, load_rgb_image
from .classifier import ClassificationMetrics, compute_classification_metrics, save_checkpoint
from .runtime import resolve_device


def require_torch() -> None:
    if torch is None:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError(
            "PyTorch and torchvision are required for the ResNet branch."
        ) from _TORCH_IMPORT_ERROR


class ResNetImageDataset(Dataset):
    def __init__(self, records, label_to_index: dict[str, int], image_size: int = 224, augment: bool = True):
        require_torch()
        self.records = list(records)
        self.label_to_index = label_to_index
        self.image_size = image_size
        self.augment = augment
        self.aug = AugmentationConfig(image_size=image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        if record.image_path is None:
            raise ValueError("Encountered a record without an image_path.")
        image = load_rgb_image(record.image_path, image_size=self.image_size)
        if self.augment:
            rng = np.random.default_rng()
            image = augment_view(image, rng, self.aug)
        view = torch.from_numpy(image).float()
        label = torch.tensor(self.label_to_index[record.label], dtype=torch.long)
        return view, label


class ResNet50Classifier(nn.Module if nn is not None else object):
    def __init__(self, num_classes: int, pretrained: bool = False, freeze_backbone: bool = False):
        require_torch()
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def to_device(self, device: str):
        self.model = self.model.to(device)
        return self


@dataclass
class ResNetBranchConfig:
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 8
    lr: float = 1e-4
    device: str = "auto"
    pretrained: bool = False
    freeze_backbone: bool = False
    patience: int = 2
    min_delta: float = 1e-4


def _flatten_records(nested_records: Iterable[list]) -> list[SampleRecord]:
    return [record for records in nested_records for record in records]


def records_by_patient_from_frame(frame: pd.DataFrame) -> list[list[SampleRecord]]:
    grouped: list[list[SampleRecord]] = []
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
            grouped.append(records)
    return grouped


def patient_split_frames(
    frame: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    frame = frame.copy()
    if frame.empty:
        return {"train": frame.copy(), "val": frame.copy(), "test": frame.copy()}
    patient_labels = []
    for patient_id, group in frame.groupby("patient_id", sort=True):
        label = group["superclass"].mode().iloc[0]
        patient_labels.append((str(patient_id), str(label)))
    by_label: dict[str, list[str]] = {}
    for patient_id, label in patient_labels:
        by_label.setdefault(label, []).append(patient_id)
    rng = np.random.default_rng(seed)
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    for _, patient_ids in sorted(by_label.items()):
        ids = list(patient_ids)
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, int(round(n * test_fraction))) if n >= 3 else max(0, int(round(n * test_fraction)))
        n_val = max(1, int(round(n * val_fraction))) if n >= 3 else max(0, int(round(n * val_fraction)))
        if n_test + n_val >= n:
            n_test = 1 if n >= 2 else 0
            n_val = 1 if n >= 3 else 0
        test_ids.extend(ids[:n_test])
        val_ids.extend(ids[n_test : n_test + n_val])
        train_ids.extend(ids[n_test + n_val :])
    train_df = frame[frame["patient_id"].astype(str).isin(train_ids)].copy()
    val_df = frame[frame["patient_id"].astype(str).isin(val_ids)].copy()
    test_df = frame[frame["patient_id"].astype(str).isin(test_ids)].copy()
    return {"train": train_df, "val": val_df, "test": test_df}


def _make_loader(records, label_to_index: dict[str, int], image_size: int, augment: bool) -> DataLoader:
    dataset = ResNetImageDataset(
        records,
        label_to_index=label_to_index,
        image_size=image_size,
        augment=augment,
    )
    return DataLoader(dataset, batch_size=min(16, len(dataset)) or 1, shuffle=augment, drop_last=False)


def train_resnet_branch(
    train_records: Iterable[list],
    val_records: Iterable[list],
    test_records: Iterable[list],
    label_to_index: dict[str, int],
    config: ResNetBranchConfig,
    save_dir: str | Path = "checkpoints_resnet",
):
    require_torch()
    device = resolve_device(config.device)
    train_flat = _flatten_records(train_records)
    val_flat = _flatten_records(val_records)
    test_flat = _flatten_records(test_records)
    if not train_flat:
        raise ValueError("No labeled images available for the ResNet branch.")

    model = ResNet50Classifier(
        num_classes=len(label_to_index),
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
    ).to(device)
    model.to_device(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = _make_loader(train_flat, label_to_index, config.image_size, augment=True)

    best_state = copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items()})
    best_score = float("-inf")
    best_round = -1
    stale_rounds = 0

    for epoch in range(config.epochs):
        model.train()
        for view, label in train_loader:
            view = view.to(device)
            label = label.to(device)
            logits = model(view)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_resnet_branch(model, val_flat, label_to_index, config.image_size, device)
        score = val_metrics.macro_f1
        if score > best_score + config.min_delta:
            best_score = score
            best_state = copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items()})
            best_round = epoch
            stale_rounds = 0
        else:
            stale_rounds += 1
        if stale_rounds >= config.patience:
            break

    model.load_state_dict(best_state)
    model = model.to(device)
    model.to_device(device)
    val_metrics = evaluate_resnet_branch(model, val_flat, label_to_index, config.image_size, device)
    test_metrics = evaluate_resnet_branch(model, test_flat, label_to_index, config.image_size, device)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        save_dir / "resnet50_classifier.pt",
        model,
        metadata={
            "branch": "resnet50",
            "best_round": best_round,
            "label_map": label_to_index,
            "device": device,
        },
    )
    return model, {
        "best_round": best_round,
        "val": val_metrics,
        "test": test_metrics,
        "save_dir": str(save_dir),
    }


def evaluate_resnet_branch(
    model: ResNet50Classifier,
    records,
    label_to_index: dict[str, int],
    image_size: int = 224,
    device: str = "cpu",
) -> ClassificationMetrics:
    require_torch()
    if not records:
        return compute_classification_metrics([], [], num_classes=len(label_to_index))
    loader = _make_loader(records, label_to_index, image_size, augment=False)
    model = model.to(device)
    model.to_device(device)
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for view, label in loader:
            view = view.to(device)
            label = label.to(device)
            logits = model(view)
            pred = torch.argmax(logits, dim=-1).cpu().tolist()
            y_pred.extend(int(p) for p in pred)
            y_true.extend(int(t) for t in label.cpu().tolist())
    return compute_classification_metrics(y_true, y_pred, num_classes=len(label_to_index))
