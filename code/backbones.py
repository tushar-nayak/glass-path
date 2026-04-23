from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
    from torchvision import models
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    torch = None
    nn = None
    models = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def require_torchvision() -> None:
    if torch is None or models is None:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError(
            "PyTorch and torchvision are required for torchvision backbones."
        ) from _TORCH_IMPORT_ERROR


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    embedding_dim: int
    pretrained: bool = True


class TorchvisionBackbone(nn.Module if nn is not None else object):
    """Torchvision feature extractor that returns an embedding vector.

    Used to bootstrap performance on small datasets where training a backbone
    from scratch is likely to collapse.
    """

    def __init__(self, model: nn.Module, embedding_dim: int, normalize: bool = True):
        require_torchvision()
        super().__init__()
        self.model = model
        self.embedding_dim = int(embedding_dim)
        self.normalize = bool(normalize)

    def forward(self, x):
        feat = self.model(x)
        if self.normalize:
            feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat


def make_torchvision_backbone(
    name: str = "resnet50",
    pretrained: bool = True,
    normalize: bool = True,
) -> TorchvisionBackbone:
    require_torchvision()
    key = name.lower().strip()
    if key != "resnet50":
        raise ValueError(f"Unsupported torchvision backbone: {name!r} (only 'resnet50' supported)")

    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    embedding_dim = int(model.fc.in_features)
    # Remove classifier head; keep global average pooled embedding.
    model.fc = nn.Identity()
    return TorchvisionBackbone(model, embedding_dim=embedding_dim, normalize=normalize)

