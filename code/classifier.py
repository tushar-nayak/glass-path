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
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    torch = None
    nn = None
    DataLoader = None
    Dataset = object
    WeightedRandomSampler = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from .data import SampleRecord
from .image import AugmentationConfig, apply_normalization, augment_view, load_rgb_image
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
    def __init__(
        self,
        records,
        label_to_index: dict[str, int],
        image_size: int = 224,
        normalization: str = "instance",
    ):
        require_torch()
        self.records = list(records)
        self.label_to_index = label_to_index
        self.image_size = image_size
        self.aug = AugmentationConfig(image_size=image_size, normalization=normalization)

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
    def __init__(
        self,
        records,
        label_to_index: dict[str, int],
        image_size: int = 224,
        normalization: str = "instance",
    ):
        require_torch()
        self.records = list(records)
        self.label_to_index = label_to_index
        self.image_size = image_size
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        if record.image_path is None:
            raise ValueError(
                "Encountered a record without an image_path. Resolve paths before training."
            )
        image = apply_normalization(
            load_rgb_image(record.image_path, image_size=self.image_size),
            self.normalization,
        )
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

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = trainable


@dataclass
class FederatedClassifierConfig:
    image_size: int = 224
    batch_size: int = 16
    local_epochs: int = 1
    rounds: int = 6
    lr: float = 1e-3
    device: str = "cpu"
    freeze_backbone: bool = True
    head_warmup_rounds: int = 1
    patience: int = 2
    min_delta: float = 1e-4
    class_weighting: str = "none"  # "none" or "balanced"
    client_fraction: float = 1.0
    seed: int = 42
    verbose: bool = False
    normalization: str = "instance"
    prox_mu: float = 0.0
    client_sampling: str = "shuffle"  # "shuffle" or "balanced"


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
    normalization: str = "instance",
) -> ClassificationMetrics:
    require_torch()
    dataset = EvaluationImageDataset(
        records,
        label_to_index=label_to_index,
        image_size=image_size,
        normalization=normalization,
    )
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

    def _build_model(self, backbone: PathologyBackbone, train_backbone: bool) -> PathologyClassifier:
        model = PathologyClassifier(
            copy.deepcopy(backbone),
            num_classes=len(self.label_to_index),
            freeze_backbone=not train_backbone,
        )
        model.set_backbone_trainable(train_backbone)
        return model

    def _make_loader(self, records) -> DataLoader:
        dataset = SupervisedImageDataset(
            records,
            label_to_index=self.label_to_index,
            image_size=self.config.image_size,
            normalization=self.config.normalization,
        )
        sampling = self.config.client_sampling.lower().strip()
        if sampling == "balanced":
            if WeightedRandomSampler is None:  # pragma: no cover - dependency guard
                raise ModuleNotFoundError("WeightedRandomSampler requires PyTorch to be installed")
            labels = [self.label_to_index[record.label] for record in dataset.records]
            class_counts = np.bincount(labels, minlength=len(self.label_to_index)).astype(np.float32)
            sample_weights = np.asarray(
                [1.0 / max(class_counts[label], 1.0) for label in labels], dtype=np.float32
            )
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.float32),
                num_samples=len(dataset),
                replacement=True,
            )
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                sampler=sampler,
                shuffle=False,
                drop_last=False,
            )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def _clone_state_dict(self, model: PathologyClassifier) -> dict:
        # Clone tensors so we can reuse a single model instance for multiple clients.
        state = model.state_dict()
        cloned = {}
        for k, v in state.items():
            if hasattr(v, "detach"):
                cloned[k] = v.detach().to("cpu").clone()
            else:
                cloned[k] = v
        return cloned

    def _train_local(
        self,
        model: PathologyClassifier,
        loader: DataLoader,
        class_weights=None,
        global_state: dict | None = None,
    ) -> dict:
        model.to(self.config.device)
        model.train()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=self.config.lr
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        for _ in range(self.config.local_epochs):
            for view, label in loader:
                view = view.to(self.config.device)
                label = label.to(self.config.device)
                logits = model(view)
                loss = criterion(logits, label)
                if global_state is not None and self.config.prox_mu > 0:
                    prox = torch.zeros((), device=self.config.device)
                    for name, param in model.named_parameters():
                        if not param.requires_grad:
                            continue
                        ref = global_state[name].to(self.config.device)
                        prox = prox + torch.sum((param - ref) ** 2)
                    loss = loss + 0.5 * self.config.prox_mu * prox
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self._clone_state_dict(model)

    def fit_rounds(
        self,
        patient_records: Iterable[list],
        backbone: PathologyBackbone,
        rounds: int | None = None,
        train_backbone: bool = False,
    ) -> PathologyClassifier:
        loaders = [self._make_loader(records) for records in patient_records if len(records) > 0]
        if not loaders:
            raise ValueError("No labeled images were found. Resolve paths before training.")
        total_rounds = self.config.rounds if rounds is None else rounds
        global_model = self._build_model(backbone, train_backbone=train_backbone).to(self.config.device)
        rng = np.random.default_rng(self.config.seed)
        local_model = self._build_model(backbone, train_backbone=train_backbone).to(self.config.device)
        for round_idx in range(total_rounds):
            local_states = []
            weights = []
            num_clients = len(loaders)
            take = num_clients
            if self.config.client_fraction < 1.0:
                take = max(1, int(round(num_clients * float(self.config.client_fraction))))
            client_indices = list(range(num_clients))
            if take < num_clients:
                client_indices = rng.choice(client_indices, size=take, replace=False).tolist()
            global_state = global_model.state_dict()
            for idx in client_indices:
                loader = loaders[idx]
                local_model.load_state_dict(global_state)
                local_model.set_backbone_trainable(train_backbone)
                state = self._train_local(local_model, loader)
                local_states.append(state)
                weights.append(float(len(loader.dataset)))
            global_model.load_state_dict(weighted_average_state_dicts(local_states, weights))
            if self.config.verbose:
                print({"round": int(round_idx), "clients": int(len(weights))})
        return global_model.cpu()

    def _flatten_records(self, nested_records: Iterable[list]) -> list:
        return [record for records in nested_records for record in records]

    def train_with_early_stopping(
        self,
        train_records: Iterable[list],
        val_records: Iterable[list],
        backbone: PathologyBackbone,
    ) -> tuple[PathologyClassifier, dict]:
        train_loaders = [self._make_loader(records) for records in train_records if len(records) > 0]
        if not train_loaders:
            raise ValueError("No labeled images were found. Resolve paths before training.")

        val_flat = self._flatten_records(val_records)
        best_model = None
        best_metrics = None
        best_score = float("-inf")
        best_round = -1
        stale_rounds = 0

        # Class weighting is computed globally (from the full train split) and applied to every client.
        train_flat = self._flatten_records(train_records)
        class_weights = None
        if self.config.class_weighting.lower() == "balanced" and train_flat:
            counts = [0 for _ in range(len(self.label_to_index))]
            for record in train_flat:
                counts[int(self.label_to_index[record.label])] += 1
            # Inverse-frequency weights, normalized to mean 1.0
            weights = []
            for c in counts:
                weights.append((len(train_flat) / max(1, c)))
            mean_w = sum(weights) / len(weights)
            weights = [w / mean_w for w in weights]
            class_weights = torch.as_tensor(weights, dtype=torch.float32, device=self.config.device)

        global_model = self._build_model(backbone, train_backbone=False).to(self.config.device)
        max_rounds = self.config.rounds

        rng = np.random.default_rng(self.config.seed)
        local_model = self._build_model(backbone, train_backbone=False).to(self.config.device)

        for round_idx in range(max_rounds):
            train_backbone = round_idx >= self.config.head_warmup_rounds and not self.config.freeze_backbone
            local_states = []
            weights = []
            num_clients = len(train_loaders)
            take = num_clients
            if self.config.client_fraction < 1.0:
                take = max(1, int(round(num_clients * float(self.config.client_fraction))))
            client_indices = list(range(num_clients))
            if take < num_clients:
                client_indices = rng.choice(client_indices, size=take, replace=False).tolist()
            global_state = global_model.state_dict()
            local_model.set_backbone_trainable(train_backbone)
            for idx in client_indices:
                loader = train_loaders[idx]
                local_model.load_state_dict(global_state)
                state = self._train_local(
                    local_model,
                    loader,
                    class_weights=class_weights,
                    global_state=global_state,
                )
                local_states.append(state)
                weights.append(float(len(loader.dataset)))
            global_model.load_state_dict(weighted_average_state_dicts(local_states, weights))

            val_metrics = evaluate_classifier(
                global_model.cpu(),
            val_flat,
            self.label_to_index,
            image_size=self.config.image_size,
            batch_size=self.config.batch_size,
            device="cpu",
            normalization=self.config.normalization,
        )
            score = val_metrics.macro_f1
            if self.config.verbose:
                print(
                    {
                        "round": int(round_idx),
                        "clients": int(len(weights)),
                        "val_macro_f1": float(val_metrics.macro_f1),
                        "val_balanced_accuracy": float(val_metrics.balanced_accuracy),
                        "val_accuracy": float(val_metrics.accuracy),
                    }
                )
            if score > best_score + self.config.min_delta:
                best_score = score
                best_model = copy.deepcopy(global_model.cpu())
                best_metrics = val_metrics
                best_round = round_idx
                stale_rounds = 0
            else:
                stale_rounds += 1

            if stale_rounds >= self.config.patience:
                break

        if best_model is None:
            best_model = global_model.cpu()
            best_metrics = evaluate_classifier(
                best_model,
                val_flat,
                self.label_to_index,
                image_size=self.config.image_size,
                batch_size=self.config.batch_size,
                device="cpu",
                normalization=self.config.normalization,
            )
            best_round = max_rounds - 1

        return best_model, {
            "best_round": best_round,
            "val": best_metrics,
        }

    def fit(self, patient_records: Iterable[list], backbone: PathologyBackbone) -> PathologyClassifier:
        return self.fit_rounds(
            patient_records,
            backbone,
            rounds=self.config.rounds,
            train_backbone=not self.config.freeze_backbone,
        )

    def train_and_evaluate(
        self,
        train_records: Iterable[list],
        val_records: Iterable[list],
        test_records: Iterable[list],
        backbone: PathologyBackbone,
    ) -> tuple[PathologyClassifier, dict]:
        model, summary = self.train_with_early_stopping(train_records, val_records, backbone)
        test_flat = self._flatten_records(test_records)
        test_metrics = evaluate_classifier(
            model.cpu(),
            test_flat,
            self.label_to_index,
            image_size=self.config.image_size,
            batch_size=self.config.batch_size,
            device="cpu",
            normalization=self.config.normalization,
        )
        return model, {
            "best_round": summary["best_round"],
            "val": summary["val"],
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
