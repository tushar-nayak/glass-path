from __future__ import annotations

from dataclasses import dataclass

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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


@dataclass
class MultimodalConceptBottleneckConfig:
    ssl_dim: int
    graph_dim: int
    fusion_dim: int
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


class CrossAttentionFusion(nn.Module if nn is not None else object):
    def __init__(
        self,
        ssl_dim: int,
        graph_dim: int,
        fusion_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        require_torch()
        super().__init__()
        self.ssl_proj = nn.Linear(ssl_dim, fusion_dim)
        self.graph_proj = nn.Linear(graph_dim, fusion_dim)
        self.ssl_norm = nn.LayerNorm(fusion_dim)
        self.graph_norm = nn.LayerNorm(fusion_dim)
        self.ssl_to_graph = nn.MultiheadAttention(
            fusion_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.graph_to_ssl = nn.MultiheadAttention(
            fusion_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.out = nn.Sequential(
            nn.Linear(fusion_dim * 4, fusion_dim * 2),
            nn.GELU(),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )

    def forward(self, ssl_tokens, graph_tokens):
        if ssl_tokens.ndim == 2:
            ssl_tokens = ssl_tokens.unsqueeze(1)
        if graph_tokens.ndim == 2:
            graph_tokens = graph_tokens.unsqueeze(1)

        ssl_q = self.ssl_norm(self.ssl_proj(ssl_tokens))
        graph_q = self.graph_norm(self.graph_proj(graph_tokens))
        ssl_ctx, _ = self.ssl_to_graph(ssl_q, graph_q, graph_q, need_weights=False)
        graph_ctx, _ = self.graph_to_ssl(graph_q, ssl_q, ssl_q, need_weights=False)
        ssl_pool = ssl_ctx.mean(dim=1)
        graph_pool = graph_ctx.mean(dim=1)
        fused = torch.cat([ssl_pool, graph_pool, ssl_pool * graph_pool, ssl_pool - graph_pool], dim=-1)
        return self.out(fused)


class MultimodalConceptBottleneckNet(nn.Module if nn is not None else object):
    def __init__(self, config: MultimodalConceptBottleneckConfig, num_heads: int = 4):
        require_torch()
        super().__init__()
        self.config = config
        self.fusion = CrossAttentionFusion(
            config.ssl_dim,
            config.graph_dim,
            config.fusion_dim,
            num_heads=num_heads,
        )
        self.concept_head = nn.Linear(config.fusion_dim, config.concept_dim)
        self.classifier = nn.Linear(config.concept_dim, config.num_classes)

    def forward(self, ssl_tokens, graph_tokens):
        fused = self.fusion(ssl_tokens, graph_tokens)
        concepts = self.concept_head(fused)
        logits = self.classifier(torch.sigmoid(concepts))
        return logits, concepts, fused


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


def train_multimodal_concept_bottleneck(
    ssl_features,
    graph_features,
    concept_targets,
    class_targets,
    config: MultimodalConceptBottleneckConfig,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
    num_heads: int = 4,
):
    require_torch()
    if not (len(ssl_features) == len(graph_features) == len(concept_targets) == len(class_targets)):
        raise ValueError("All multimodal inputs must have the same length")
    model = MultimodalConceptBottleneckNet(config, num_heads=num_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    concept_loss = nn.BCEWithLogitsLoss()
    class_loss = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for ssl_feature, graph_feature, concept_target, class_target in zip(
            ssl_features, graph_features, concept_targets, class_targets
        ):
            ssl_x = torch.as_tensor(ssl_feature, dtype=torch.float32, device=device)
            graph_x = torch.as_tensor(graph_feature, dtype=torch.float32, device=device)
            if ssl_x.ndim == 1:
                ssl_x = ssl_x.unsqueeze(0).unsqueeze(1)
            elif ssl_x.ndim == 2:
                ssl_x = ssl_x.unsqueeze(0)
            if graph_x.ndim == 1:
                graph_x = graph_x.unsqueeze(0).unsqueeze(1)
            elif graph_x.ndim == 2:
                graph_x = graph_x.unsqueeze(0)
            c = torch.as_tensor(concept_target, dtype=torch.float32, device=device).unsqueeze(0)
            y = torch.as_tensor([class_target], dtype=torch.long, device=device)
            logits, concept_logits, _ = model(ssl_x, graph_x)
            loss = class_loss(logits, y) + concept_loss(concept_logits, c)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.cpu()
