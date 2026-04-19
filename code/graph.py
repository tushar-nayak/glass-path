from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

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
            "PyTorch is required for graph training. Install project dependencies first."
        ) from _TORCH_IMPORT_ERROR


@dataclass(frozen=True)
class GraphSample:
    x: np.ndarray
    edge_index: np.ndarray
    label: str | None = None
    image_id: str | None = None
    source: str | None = None


def build_knn_graph(coords: np.ndarray, k: int = 8) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Coordinates must have shape [num_nodes, 2]")
    num_nodes = coords.shape[0]
    if num_nodes < 2:
        return np.zeros((2, 0), dtype=np.int64)
    diffs = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt((diffs**2).sum(axis=-1))
    np.fill_diagonal(distances, np.inf)
    edges = []
    for i in range(num_nodes):
        nbrs = np.argsort(distances[i])[: min(k, num_nodes - 1)]
        for j in nbrs:
            edges.append((i, int(j)))
            edges.append((int(j), i))
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T


def graph_summary_features(coords: np.ndarray, features: np.ndarray, k: int = 8) -> np.ndarray:
    """Summarize a graph derived from image-local structure.

    This function intentionally assumes the graph comes from automatically
    extracted coordinates or other model-derived structure, not manual
    labels.
    """
    coords = np.asarray(coords, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    edge_index = build_knn_graph(coords, k=k)
    if edge_index.shape[1] == 0:
        return np.concatenate([features.mean(axis=0), features.std(axis=0)], axis=0)
    src, dst = edge_index
    degree = np.bincount(src, minlength=coords.shape[0]).astype(np.float32)
    edge_lengths = np.linalg.norm(coords[src] - coords[dst], axis=1)
    summary = np.concatenate(
        [
            features.mean(axis=0),
            features.std(axis=0),
            np.array(
                [
                    degree.mean(),
                    degree.std(),
                    edge_lengths.mean(),
                    edge_lengths.std(),
                    coords[:, 0].std(),
                    coords[:, 1].std(),
                ],
                dtype=np.float32,
            ),
        ]
    )
    return summary.astype(np.float32)


class GraphSAGELayer(nn.Module if nn is not None else object):
    def __init__(self, in_dim: int, out_dim: int):
        require_torch()
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):
        if edge_index.numel() == 0:
            neigh = torch.zeros_like(x)
        else:
            src, dst = edge_index
            neigh = torch.zeros_like(x)
            neigh.index_add_(0, dst, x[src])
            degree = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            degree = degree.clamp_min_(1.0).unsqueeze(-1)
            neigh = neigh / degree
        out = self.self_linear(x) + self.neigh_linear(neigh)
        return self.norm(torch.relu(out))


class GraphEncoder(nn.Module if nn is not None else object):
    def __init__(self, in_dim: int, hidden_dim: int = 128, embedding_dim: int = 128):
        require_torch()
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GraphSAGELayer(in_dim, hidden_dim),
                GraphSAGELayer(hidden_dim, hidden_dim),
            ]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x, edge_index, batch=None):
        for layer in self.layers:
            x = layer(x, edge_index)
        mean_pool = x.mean(dim=0, keepdim=True)
        max_pool = x.max(dim=0, keepdim=True).values
        graph_emb = torch.cat([mean_pool, max_pool], dim=-1)
        return self.readout(graph_emb)


class GraphClassifier(nn.Module if nn is not None else object):
    def __init__(self, in_dim: int, hidden_dim: int, embedding_dim: int, num_classes: int):
        require_torch()
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, edge_index, batch=None):
        emb = self.encoder(x, edge_index, batch=batch)
        return self.head(emb), emb


def collate_graphs(graphs: Iterable[GraphSample]):
    require_torch()
    xs = []
    edge_indices = []
    batches = []
    offset = 0
    for batch_idx, graph in enumerate(graphs):
        x = torch.as_tensor(graph.x, dtype=torch.float32)
        edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long)
        xs.append(x)
        if edge_index.numel() > 0:
            edge_indices.append(edge_index + offset)
        batches.append(torch.full((x.size(0),), batch_idx, dtype=torch.long))
        offset += x.size(0)
    if not xs:
        raise ValueError("No graphs to collate")
    x = torch.cat(xs, dim=0)
    edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros((2, 0), dtype=torch.long)
    batch = torch.cat(batches, dim=0)
    return x, edge_index, batch


def train_graph_classifier(
    graphs: list[GraphSample],
    labels: list[int],
    in_dim: int,
    num_classes: int,
    hidden_dim: int = 128,
    embedding_dim: int = 128,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """Train a graph classifier on derived graphs.

    The pipeline treats graphs as optional, locally constructed inputs
    generated from images or model outputs. It does not require any
    ground-truth graph supervision.
    """
    require_torch()
    if len(graphs) != len(labels):
        raise ValueError("Graphs and labels must have the same length")
    model = GraphClassifier(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for graph, label in zip(graphs, labels):
            x = torch.as_tensor(graph.x, dtype=torch.float32, device=device)
            edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)
            y = torch.as_tensor([label], dtype=torch.long, device=device)
            logits, _ = model(x, edge_index)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.cpu()
