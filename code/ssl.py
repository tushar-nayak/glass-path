from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import os
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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

from .image import AugmentationConfig, augment_view, load_rgb_image
from .data import SampleRecord
from .federated import weighted_average_state_dicts


def require_torch() -> None:
    if torch is None:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError(
            "PyTorch is required for training. Install project dependencies first."
        ) from _TORCH_IMPORT_ERROR


class SSLImageDataset(Dataset):
    def __init__(self, records, image_size: int = 224, return_label: bool = False):
        require_torch()
        self.records = list(records)
        self.image_size = image_size
        self.return_label = return_label
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
        view1 = augment_view(image, rng, self.aug)
        view2 = augment_view(image, rng, self.aug)
        view1 = torch.from_numpy(view1).float()
        view2 = torch.from_numpy(view2).float()
        if self.return_label:
            return view1, view2, record.label
        return view1, view2


class TransformerBlock(nn.Module if nn is not None else object):
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: int = 4, dropout: float = 0.0):
        require_torch()
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class PathologyBackbone(nn.Module if nn is not None else object):
    def __init__(
        self,
        embedding_dim: int = 256,
        patch_size: int = 16,
        hidden_dim: int = 192,
        depth: int = 2,
        num_heads: int = 4,
        max_patches: int = 196,
    ):
        require_torch()
        super().__init__()
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.max_patches = max_patches
        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches + 1, hidden_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads=num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        tokens, embedding = self.forward_features(x)
        return embedding

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        batch_size, num_patches, _ = x.shape
        if num_patches > self.max_patches:
            x = x[:, : self.max_patches, :]
            num_patches = self.max_patches
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed[:, : num_patches + 1, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        tokens = x
        cls = x[:, 0]
        embedding = self.projector(cls)
        embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return tokens, embedding


class PredictiveHead(nn.Module if nn is not None else object):
    def __init__(self, dim: int):
        require_torch()
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class FederatedSSLConfig:
    image_size: int = 224
    embedding_dim: int = 256
    batch_size: int = 16
    local_epochs: int = 2
    rounds: int = 3
    lr: float = 1e-3
    device: str = "cpu"


class FederatedSSLTrainer:
    """Federated self-supervised training using local BYOL-style updates."""

    def __init__(self, config: FederatedSSLConfig):
        require_torch()
        self.config = config

    def _make_loader(self, records) -> DataLoader:
        dataset = SSLImageDataset(records, image_size=self.config.image_size)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def _train_local(self, model: PathologyBackbone, loader: DataLoader) -> dict:
        predictor = PredictiveHead(self.config.embedding_dim).to(self.config.device)
        target = PathologyBackbone(self.config.embedding_dim).to(self.config.device)
        target.load_state_dict(model.state_dict())
        model.to(self.config.device)
        model.train()
        predictor.train()
        opt = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()), lr=self.config.lr
        )
        mse = nn.MSELoss()
        for _ in range(self.config.local_epochs):
            for view1, view2 in loader:
                view1 = view1.to(self.config.device)
                view2 = view2.to(self.config.device)
                z1 = model(view1)
                z2 = model(view2)
                with torch.no_grad():
                    t1 = target(view1)
                    t2 = target(view2)
                p1 = predictor(z1)
                p2 = predictor(z2)
                loss = mse(torch.nn.functional.normalize(p1, dim=-1), t2) + mse(
                    torch.nn.functional.normalize(p2, dim=-1), t1
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
        return model.state_dict()

    def fit(self, patient_records: Iterable[list]) -> PathologyBackbone:
        loaders = [self._make_loader(records) for records in patient_records if len(records) > 0]
        if not loaders:
            raise ValueError("No training images were found. Resolve image paths first.")
        global_model = PathologyBackbone(self.config.embedding_dim).to(self.config.device)
        for _ in range(self.config.rounds):
            local_states = []
            weights = []
            for loader in loaders:
                local_model = PathologyBackbone(self.config.embedding_dim).to(self.config.device)
                local_model.load_state_dict(global_model.state_dict())
                state = self._train_local(local_model, loader)
                local_states.append(state)
                weights.append(float(len(loader.dataset)))
            global_model.load_state_dict(weighted_average_state_dicts(local_states, weights))
        return global_model.cpu()

    def fit_from_frame(self, frame) -> PathologyBackbone:
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
        return self.fit(records_by_patient)
