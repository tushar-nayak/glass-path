# Hybrid Patch-Graph Sketch

This note sketches the hybrid architecture we want for LungHist700:

## Goal

Use a pretrained image backbone for local appearance, then let a graph encoder reason over patch-level structure.

## Proposed flow

1. Load the whole histology image and normalize it with ImageNet stats.
2. Split it into a small grid of patches, starting with `2x2` quadrants.
3. Encode each patch with a pretrained CNN backbone (currently `ResNet50`).
4. Treat the patch embeddings as graph nodes.
5. Connect the nodes with a simple complete graph or kNN graph.
6. Run a graph encoder over the patch embeddings.
7. Pool the graph embedding and classify the image.

## Why this is reasonable

- It is much cheaper than full slide tiling.
- It uses a real pretrained image representation instead of training graph features from scratch.
- It still lets the graph branch express relationships across regions, which a plain pooled CNN cannot.

## What this is not yet

- Not a nuclei-level graph.
- Not a full MIL pipeline.
- Not a production pathology graph model.

## What we will test first

- `patch_grid=2`
- pretrained `ResNet50` patch encoder
- frozen patch encoder for the first smoke test
- patient-disjoint split
- balanced accuracy and macro F1 on the tiny run

## Success criteria

- The code runs end-to-end in tmux.
- Metrics are non-collapsed and comparable to the supervised baseline.
- If it underperforms the pretrained ResNet baseline, we keep it as a novelty/interpretability branch, not the main accuracy claim.

