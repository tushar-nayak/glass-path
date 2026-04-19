from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def require_torch() -> None:
    if torch is None:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError(
            "PyTorch is required for concept bottleneck training. Install project dependencies first."
        ) from _TORCH_IMPORT_ERROR


@dataclass
class ConceptBottleneckConfig:
    input_dim: int
    concept_dim: int
    num_classes: int
    hidden_dim: int = 128


class ConceptBottleneckNet(nn.Module if nn is not None else object):
    def __init__(self, config: ConceptBottleneckConfig):
        require_torch()
        super().__init__()
        self.config = config
        self.fusion = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.concept_head = nn.Linear(config.hidden_dim, config.concept_dim)
        self.classifier = nn.Linear(config.concept_dim, config.num_classes)

    def forward(self, x):
        fused = self.fusion(x)
        concepts = self.concept_head(fused)
        logits = self.classifier(torch.sigmoid(concepts))
        return logits, concepts


def train_concept_bottleneck(
    features,
    concept_targets,
    class_targets,
    config: ConceptBottleneckConfig,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
):
    require_torch()
    if len(features) != len(concept_targets) or len(features) != len(class_targets):
        raise ValueError("Features, concept targets, and class targets must have the same length")
    model = ConceptBottleneckNet(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    concept_loss = nn.BCEWithLogitsLoss()
    class_loss = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for feature, concept_target, class_target in zip(features, concept_targets, class_targets):
            x = torch.as_tensor(feature, dtype=torch.float32, device=device).unsqueeze(0)
            c = torch.as_tensor(concept_target, dtype=torch.float32, device=device).unsqueeze(0)
            y = torch.as_tensor([class_target], dtype=torch.long, device=device)
            logits, concept_logits = model(x)
            loss = class_loss(logits, y) + concept_loss(concept_logits, c)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.cpu()
