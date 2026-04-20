# G.L.A.S.S. Pathology

This branch contains a simple ResNet-50 classification baseline for the dataset in `data/`.

## Commands

Activate the conda environment first:

```bash
conda activate bme_ml
```

Inspect the dataset:

```bash
python -m code.cli inspect --csv data/data.csv
```

Train the ResNet baseline:

```bash
./resnet
```

Or run the CLI directly:

```bash
python -m code.cli resnet --csv data/data.csv --image-root data/images
```

The image root defaults to `data/images`, and the labels are encoded by the subfolder names such as `aca_bd`, `aca_md`, `aca_pd`, `nor`, `scc_bd`, `scc_md`, and `scc_pd`.

The baseline writes its checkpoint to `checkpoints_resnet/resnet50_classifier.pt` and saves validation/test metrics alongside it.
