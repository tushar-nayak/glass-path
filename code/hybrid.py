from __future__ import annotations

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

from .backbones import make_torchvision_backbone
from .classifier import ClassificationMetrics, compute_classification_metrics, save_checkpoint
from .data import SampleRecord
from .graph import GraphClassifier
from .image import apply_normalization, load_rgb_image
from .runtime import resolve_device


def require_torch() -> None:
    if torch is None:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError("PyTorch is required for the hybrid branch.") from _TORCH_IMPORT_ERROR


def _complete_graph(num_nodes: int) -> np.ndarray:
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append((i, j))
    return np.asarray(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)


def _quadrant_patches(image: np.ndarray, patch_grid: int = 2) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected image shape [3, H, W], got {image.shape}")
    _, height, width = image.shape
    step_h = height // patch_grid
    step_w = width // patch_grid
    patches = []
    coords = []
    for row in range(patch_grid):
        for col in range(patch_grid):
            top = row * step_h
            left = col * step_w
            patch = image[:, top : top + step_h, left : left + step_w]
            patches.append(patch)
            coords.append(((col + 0.5) / patch_grid, (row + 0.5) / patch_grid))
    return np.stack(patches, axis=0), np.asarray(coords, dtype=np.float32)


class HybridImageGraphDataset(Dataset):
    def __init__(
        self,
        records,
        label_to_index: dict[str, int],
        image_size: int = 224,
        patch_grid: int = 2,
        normalization: str = "imagenet",
    ):
        require_torch()
        self.records = list(records)
        self.label_to_index = label_to_index
        self.image_size = image_size
        self.patch_grid = patch_grid
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        if record.image_path is None:
            raise ValueError("Encountered a record without an image_path.")
        image = load_rgb_image(record.image_path, image_size=self.image_size)
        image = apply_normalization(image, self.normalization)
        patches, coords = _quadrant_patches(image, patch_grid=self.patch_grid)
        label = torch.tensor(self.label_to_index[record.label], dtype=torch.long)
        return torch.from_numpy(patches).float(), torch.from_numpy(coords).float(), label


class PatchGraphClassifier(nn.Module if nn is not None else object):
    def __init__(
        self,
        patch_encoder: nn.Module,
        graph_in_dim: int,
        graph_hidden_dim: int,
        graph_embedding_dim: int,
        num_classes: int,
        heads: int = 2,
        patch_grid: int = 2,
    ):
        require_torch()
        super().__init__()
        self.patch_encoder = patch_encoder
        self.patch_grid = patch_grid
        self.num_nodes = patch_grid * patch_grid
        self.graph = GraphClassifier(
            in_dim=graph_in_dim,
            hidden_dim=graph_hidden_dim,
            embedding_dim=graph_embedding_dim,
            num_classes=num_classes,
            heads=heads,
        )

    def forward(self, patches, coords):
        if patches.ndim != 5:
            raise ValueError(f"Expected patches [B, N, C, H, W], got {patches.shape}")
        batch_size, num_nodes = patches.shape[:2]
        flat = patches.reshape(batch_size * num_nodes, *patches.shape[2:])
        feats = self.patch_encoder(flat)
        feats = feats.reshape(batch_size, num_nodes, -1)
        edge_index = _complete_graph(num_nodes)
        logits = []
        embs = []
        for i in range(batch_size):
            graph_logits, emb = self.graph(feats[i], torch.as_tensor(edge_index, device=feats.device))
            logits.append(graph_logits.squeeze(0))
            embs.append(emb.squeeze(0))
        logits = torch.stack(logits, dim=0)
        embs = torch.stack(embs, dim=0)
        return logits, embs


@dataclass
class HybridBranchConfig:
    image_size: int = 224
    patch_grid: int = 2
    batch_size: int = 8
    epochs: int = 4
    lr: float = 1e-4
    device: str = "cpu"
    pretrained: bool = True
    freeze_patch_encoder: bool = True
    patience: int = 2
    min_delta: float = 1e-4


def _flatten_records(nested_records: Iterable[list]) -> list[SampleRecord]:
    return [record for records in nested_records for record in records]


def _make_loader(
    records,
    label_to_index: dict[str, int],
    image_size: int,
    patch_grid: int,
    normalization: str,
) -> DataLoader:
    dataset = HybridImageGraphDataset(
        records,
        label_to_index=label_to_index,
        image_size=image_size,
        patch_grid=patch_grid,
        normalization=normalization,
    )
    return DataLoader(dataset, batch_size=min(8, len(dataset)) or 1, shuffle=True, drop_last=False)


def train_hybrid_branch(
    train_records: Iterable[list],
    val_records: Iterable[list],
    test_records: Iterable[list],
    label_to_index: dict[str, int],
    config: HybridBranchConfig,
    save_dir: str | Path = "runs/hybrid_branch",
):
    require_torch()
    device = resolve_device(config.device)
    train_flat = _flatten_records(train_records)
    val_flat = _flatten_records(val_records)
    test_flat = _flatten_records(test_records)
    if not train_flat:
        raise ValueError("No labeled images available for the hybrid branch.")

    patch_encoder = make_torchvision_backbone(
        name="resnet50",
        pretrained=config.pretrained,
        normalize=True,
    )
    if config.freeze_patch_encoder:
        for p in patch_encoder.parameters():
            p.requires_grad = False
    patch_encoder = patch_encoder.to(device)
    graph_in_dim = patch_encoder.embedding_dim
    model = PatchGraphClassifier(
        patch_encoder=patch_encoder,
        graph_in_dim=graph_in_dim,
        graph_hidden_dim=128,
        graph_embedding_dim=128,
        num_classes=len(label_to_index),
        heads=2,
        patch_grid=config.patch_grid,
    ).to(device)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = _make_loader(train_flat, label_to_index, config.image_size, config.patch_grid, "imagenet")

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_score = float("-inf")
    best_round = -1
    stale_rounds = 0

    for epoch in range(config.epochs):
        model.train()
        for patches, coords, label in train_loader:
            patches = patches.to(device)
            coords = coords.to(device)
            label = label.to(device)
            logits, _ = model(patches, coords)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_hybrid_branch(
            model,
            val_flat,
            label_to_index,
            config.image_size,
            device,
            patch_grid=config.patch_grid,
        )
        score = val_metrics.macro_f1
        if score > best_score + config.min_delta:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_round = epoch
            stale_rounds = 0
        else:
            stale_rounds += 1
        if stale_rounds >= config.patience:
            break

    model.load_state_dict(best_state)
    val_metrics = evaluate_hybrid_branch(
        model,
        val_flat,
        label_to_index,
        config.image_size,
        device,
        patch_grid=config.patch_grid,
    )
    test_metrics = evaluate_hybrid_branch(
        model,
        test_flat,
        label_to_index,
        config.image_size,
        device,
        patch_grid=config.patch_grid,
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        save_dir / "hybrid_classifier.pt",
        model,
        metadata={
            "branch": "hybrid_patch_graph",
            "best_round": best_round,
            "label_map": label_to_index,
            "device": device,
            "patch_grid": config.patch_grid,
            "pretrained": config.pretrained,
        },
    )
    return model, {
        "best_round": best_round,
        "val": val_metrics,
        "test": test_metrics,
        "save_dir": str(save_dir),
    }


def evaluate_hybrid_branch(
    model: PatchGraphClassifier,
    records,
    label_to_index: dict[str, int],
    image_size: int = 224,
    device: str = "cpu",
    patch_grid: int = 2,
) -> ClassificationMetrics:
    require_torch()
    if not records:
        return compute_classification_metrics([], [], num_classes=len(label_to_index))
    dataset = HybridImageGraphDataset(
        records,
        label_to_index=label_to_index,
        image_size=image_size,
        patch_grid=patch_grid,
        normalization="imagenet",
    )
    loader = DataLoader(dataset, batch_size=min(8, len(dataset)) or 1, shuffle=False, drop_last=False)
    model = model.to(device)
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for patches, coords, label in loader:
            patches = patches.to(device)
            coords = coords.to(device)
            logits, _ = model(patches, coords)
            pred = torch.argmax(logits, dim=-1).cpu().tolist()
            y_pred.extend(int(p) for p in pred)
            y_true.extend(int(t) for t in label.tolist())
    return compute_classification_metrics(y_true, y_pred, num_classes=len(label_to_index))
