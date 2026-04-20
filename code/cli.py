from __future__ import annotations

import argparse
from pathlib import Path

from .data import GlassDataset
from .resnet_branch import (
    ResNetBranchConfig,
    patient_split_frames,
    records_by_patient_from_frame,
    train_resnet_branch,
)
from .runtime import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glass-path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="Summarize the dataset")
    inspect.add_argument("--csv", required=True, help="Path to data.csv")

    resnet = subparsers.add_parser("resnet", help="Train a simple ResNet-50 classifier")
    resnet.add_argument("--csv", required=True, help="Path to data.csv")
    resnet.add_argument("--image-root", default="data/images", help="Directory containing images")
    resnet.add_argument("--image-size", type=int, default=224)
    resnet.add_argument("--batch-size", type=int, default=16)
    resnet.add_argument("--epochs", type=int, default=8)
    resnet.add_argument("--lr", type=float, default=1e-4)
    resnet.add_argument("--device", default="auto", help="auto, mps, cuda, or cpu")
    resnet.add_argument("--pretrained", action="store_true", help="Use torchvision pretrained weights")
    resnet.add_argument("--freeze-backbone", action="store_true", help="Freeze all layers except fc")
    resnet.add_argument("--patience", type=int, default=2)
    resnet.add_argument("--save-dir", default="checkpoints_resnet", help="Directory for model checkpoints")
    resnet.add_argument("--val-fraction", type=float, default=0.15)
    resnet.add_argument("--test-fraction", type=float, default=0.15)
    resnet.add_argument("--seed", type=int, default=42)

    return parser


def cmd_inspect(args: argparse.Namespace) -> int:
    dataset = GlassDataset.from_csv(Path(args.csv))
    print(dataset.summary())
    return 0


def cmd_resnet(args: argparse.Namespace) -> int:
    dataset = GlassDataset.from_csv(Path(args.csv))
    image_root = Path(args.image_root)
    resolved = dataset.resolve_image_paths(image_root)
    splits = patient_split_frames(
        resolved,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    label_map = {label: idx for idx, label in enumerate(sorted(str(v) for v in dataset.frame["superclass"].unique()))}
    config = ResNetBranchConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=resolve_device(args.device),
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        patience=args.patience,
    )
    train_records = records_by_patient_from_frame(splits["train"])
    val_records = records_by_patient_from_frame(splits["val"])
    test_records = records_by_patient_from_frame(splits["test"])
    model, metrics = train_resnet_branch(
        train_records,
        val_records,
        test_records,
        label_map,
        config,
        save_dir=args.save_dir,
    )
    print(
        {
            "model": type(model).__name__,
            "device": config.device,
            "label_map": label_map,
            "metrics": {
                "best_round": metrics["best_round"],
                "val": metrics["val"].__dict__,
                "test": metrics["test"].__dict__,
            },
            "save_dir": metrics["save_dir"],
        }
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "inspect":
        return cmd_inspect(args)
    if args.command == "resnet":
        return cmd_resnet(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
