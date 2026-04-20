# GLASS (Graph-based Learning Architecture for Spatial Structures) in Pathology

Unified federated pipeline for self-supervised image encoding, local graph reasoning, and concept-bottleneck classification.

## What is included

- Patient-level federated partitioning from `data/data.csv`
- Image loading and augmentation utilities
- Transformer-based federated self-supervised encoder
- GAT-style local graph encoder for derived image structure
- Cross-attention concept bottleneck classifier
- CLI for inspecting and wiring the pipeline together

## Commands

Activate the conda environment first:

```bash
conda activate bme_ml
```

Inspect the dataset:

```bash
python -m code.cli inspect --csv data/data.csv
```

Prepare the training pipeline:

```bash
python -m code.cli train --csv data/data.csv --image-root /path/to/images
```

On macOS, the training command defaults to `--device auto`, which picks MPS when available.

For this repository, the image root is `data/images` by default, and the label is encoded by the subfolder name, for example `data/images/aca_bd/` or `data/images/scc_pd/`.

The `train` command now does two stages by default:
- federated SSL pretraining
- supervised classification on `aca`, `nor`, and `scc`

There is also a separate baseline branch:

```bash
python -m code.cli resnet --csv data/data.csv --image-root data/images --epochs 2
```

That branch trains a plain ResNet-50 classifier and writes its checkpoint to `checkpoints_resnet/`.

The repository currently contains metadata and the paper PDF, but no raw image files. The code is written to resolve image paths from an external image root when you add the image assets later.

If you need to install the package into the environment anyway, run `pip install -e .` after activating `bme_ml`.

The dataset in this repo provides image-level labels only. It does not include manual cell masks, nucleus coordinates, or segmentation annotations. Any graph inputs must therefore be derived automatically from images or image embeddings.
