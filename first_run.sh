#!/usr/bin/env bash
set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate bme_ml

export KMP_DUPLICATE_LIB_OK=TRUE

MODE="${1:-inspect}"

case "$MODE" in
  inspect)
    python -m code.cli inspect --csv data/data.csv
    ;;
  smoke)
    python - <<'PY'
import torch
from code.ssl import PathologyBackbone
from code.graph import GraphClassifier
from code.concept import MultimodalConceptBottleneckConfig, MultimodalConceptBottleneckNet

ssl_model = PathologyBackbone(embedding_dim=64, hidden_dim=64, depth=1, num_heads=4, max_patches=64)
img = torch.randn(2, 3, 128, 128)
tokens, emb = ssl_model.forward_features(img)
print("ssl", tokens.shape, emb.shape)

x = torch.randn(5, 16)
edge_index = torch.tensor([[0, 1, 2, 3, 4, 0, 2], [1, 2, 3, 4, 0, 2, 4]], dtype=torch.long)
graph_model = GraphClassifier(in_dim=16, hidden_dim=32, embedding_dim=32, num_classes=3, heads=2)
logits, gemb = graph_model(x, edge_index)
print("graph", logits.shape, gemb.shape)

config = MultimodalConceptBottleneckConfig(
    ssl_dim=64,
    graph_dim=32,
    fusion_dim=48,
    concept_dim=8,
    num_classes=3,
)
mm_model = MultimodalConceptBottleneckNet(config)
ssl_tokens = torch.randn(2, 9, 64)
graph_tokens = torch.randn(2, 5, 32)
logits, concepts, fused = mm_model(ssl_tokens, graph_tokens)
print("concept", logits.shape, concepts.shape, fused.shape)
PY
    ;;
  train)
    python -m code.cli train --csv data/data.csv --device auto
    ;;
  *)
    echo "Usage: $0 [inspect|smoke|train]" >&2
    exit 1
    ;;
esac
