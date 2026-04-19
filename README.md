# G.L.A.S.S. Pathology

Unified federated pipeline for self-supervised image encoding, local graph reasoning, and concept-bottleneck classification.

## What is included

- Patient-level federated partitioning from `data/data.csv`
- Image loading and augmentation utilities
- Federated self-supervised encoder
- Local graph encoder for derived image structure
- Concept bottleneck classifier
- CLI for inspecting and wiring the pipeline together

## Commands

Install the package in editable mode first:

```bash
pip install -e .
```

Inspect the dataset:

```bash
python -m glass_path.cli inspect --csv data/data.csv
```

Prepare the training pipeline:

```bash
python -m glass_path.cli train --csv data/data.csv --image-root /path/to/images
```

The repository currently contains metadata and the paper PDF, but no raw image files. The code is written to resolve image paths from an external image root when you add the image assets later.

The dataset in this repo provides image-level labels only. It does not include manual cell masks, nucleus coordinates, or segmentation annotations. Any graph inputs must therefore be derived automatically from images or image embeddings.
