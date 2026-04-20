from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import os
import numpy as np

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


class GraphAttentionLayer(nn.Module if nn is not None else object):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 1, dropout: float = 0.0):
        require_torch()
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim * heads))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim * heads)
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x, edge_index):
        if edge_index.numel() == 0:
            out = self.proj(x)
            return self.norm(torch.relu(out + self.bias))

        num_nodes = x.size(0)
        h = self.proj(x).view(num_nodes, self.heads, self.out_dim)
        src, dst = edge_index
        out = torch.zeros_like(h)
        for node in range(num_nodes):
            mask = dst == node
            if not torch.any(mask):
                out[node] = h[node]
                continue
            node_src = src[mask]
            node_dst = dst[mask]
            h_src = h[node_src]
            h_dst = h[node_dst]
            scores = (h_src * self.attn_src).sum(dim=-1) + (h_dst * self.attn_dst).sum(dim=-1)
            scores = self.leaky_relu(scores)
            weights = torch.softmax(scores, dim=0)
            weights = self.dropout(weights)
            neigh = (weights.unsqueeze(-1) * h_src).sum(dim=0)
            out[node] = neigh
        out = out.reshape(num_nodes, self.heads * self.out_dim) + self.bias
        return self.norm(torch.relu(out))


class GraphEncoder(nn.Module if nn is not None else object):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        heads: int = 2,
    ):
        require_torch()
        attn_dim = max(1, hidden_dim // heads)
        message_dim = attn_dim * heads
        super().__init__()
        self.hidden_dim = message_dim
        self.layers = nn.ModuleList(
            [
                GraphAttentionLayer(in_dim, attn_dim, heads=heads),
                GraphAttentionLayer(message_dim, attn_dim, heads=heads),
            ]
        )
        self.readout = nn.Sequential(
            nn.Linear(message_dim * 2, message_dim),
            nn.ReLU(inplace=True),
            nn.Linear(message_dim, embedding_dim),
        )

    def forward(self, x, edge_index, batch=None):
        for layer in self.layers:
            x = layer(x, edge_index)
        mean_pool = x.mean(dim=0, keepdim=True)
        max_pool = x.max(dim=0, keepdim=True).values
        graph_emb = torch.cat([mean_pool, max_pool], dim=-1)
        return self.readout(graph_emb)


class GraphClassifier(nn.Module if nn is not None else object):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_classes: int,
        heads: int = 2,
    ):
        require_torch()
        super().__init__()
        self.encoder = GraphEncoder(
            in_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim, heads=heads
        )
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
    heads: int = 2,
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
        heads=heads,
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
