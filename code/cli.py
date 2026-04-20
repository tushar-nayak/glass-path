from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import PipelineConfig, UnifiedGlassPipeline
from .runtime import resolve_device
from .classifier import save_checkpoint
from .resnet_branch import ResNetBranchConfig, train_resnet_branch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glass-path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="Summarize the dataset")
    inspect.add_argument("--csv", required=True, help="Path to data.csv")

    train = subparsers.add_parser("train", help="Run the unified pipeline")
    train.add_argument("--csv", required=True, help="Path to data.csv")
    train.add_argument("--image-root", default=None, help="Directory containing images")
    train.add_argument("--image-size", type=int, default=224)
    train.add_argument("--ssl-dim", type=int, default=256)
    train.add_argument("--graph-dim", type=int, default=128)
    train.add_argument("--concept-dim", type=int, default=8)
    train.add_argument("--num-classes", type=int, default=3)
    train.add_argument("--device", default="auto", help="auto, mps, cuda, or cpu")
    train.add_argument("--quick", action="store_true", help="Use fewer rounds for a faster run")
    train.add_argument("--save-dir", default="checkpoints", help="Directory for model checkpoints")
    train.add_argument("--val-fraction", type=float, default=0.15)
    train.add_argument("--test-fraction", type=float, default=0.15)
    train.add_argument("--seed", type=int, default=42)

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

    return parser


def cmd_inspect(args: argparse.Namespace) -> int:
    pipeline = UnifiedGlassPipeline(PipelineConfig(csv_path=Path(args.csv)))
    report = pipeline.inspect()
    print(report)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    pipeline = UnifiedGlassPipeline(
        PipelineConfig(
            csv_path=Path(args.csv),
            image_root=Path(args.image_root) if args.image_root else Path("data/images"),
            image_size=args.image_size,
            ssl_embedding_dim=args.ssl_dim,
            graph_embedding_dim=args.graph_dim,
            concept_dim=args.concept_dim,
            num_classes=args.num_classes,
            device=resolve_device(args.device),
        )
    )
    resolved = pipeline.resolved_frame()
    with_paths = resolved["image_path"].notna().sum() if "image_path" in resolved.columns else 0
    image_root = Path(args.image_root) if args.image_root else Path("data/images")
    splits = pipeline.patient_split_frames(
        resolved,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    train_frame = splits["train"]
    val_frame = splits["val"]
    test_frame = splits["test"]
    ssl_cfg = pipeline.quick_ssl_config() if args.quick else pipeline.ssl_config()
    classifier_cfg = pipeline.quick_classifier_config() if args.quick else pipeline.classifier_config()
    print(
        {
            "rows": len(resolved),
            "image_paths_resolved": int(with_paths),
            "clients": len(resolved.groupby("patient_id", sort=True)),
            "split_rows": {
                "train": int(len(train_frame)),
                "val": int(len(val_frame)),
                "test": int(len(test_frame)),
            },
            "device": pipeline.config.device,
            "image_root": str(image_root),
            "labels": pipeline.label_map(),
            "ssl_config": ssl_cfg,
            "classifier_config": classifier_cfg,
            "concept_config": pipeline.concept_config(),
        }
    )
    if with_paths == 0:
        print("No image files were resolved. Provide --image-root to run training.")
        return 1
    try:
        ssl_model = pipeline.train_ssl_from_frame(train_frame, quick=args.quick)
        classifier_model, classifier_summary = pipeline.train_classifier_with_splits(
            train_frame,
            val_frame,
            test_frame,
            ssl_model,
            quick=args.quick,
        )
    except ModuleNotFoundError as exc:
        print(str(exc))
        return 1
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        save_dir / "ssl_backbone.pt",
        ssl_model,
        metadata={
            "stage": "ssl_pretraining",
            "device": pipeline.config.device,
            "image_root": str(image_root),
            "label_map": pipeline.label_map(),
        },
    )
    save_checkpoint(
        save_dir / "classifier.pt",
        classifier_model,
        metadata={
            "stage": "supervised_classification",
            "device": pipeline.config.device,
            "image_root": str(image_root),
            "label_map": pipeline.label_map(),
        },
    )
    metrics_payload = {
        "best_round": classifier_summary["best_round"],
        "val": classifier_summary["val"].__dict__,
        "test": classifier_summary["test"].__dict__,
    }
    (save_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    print({"ssl_model": type(ssl_model).__name__, "classifier_model": type(classifier_model).__name__})
    print(metrics_payload)
    print(
        "Federated SSL pretraining completed, then the supervised classifier was trained on "
        "image-level labels for aca, scc, and nor with backbone fine-tuning and early stopping."
    )
    return 0


def cmd_resnet(args: argparse.Namespace) -> int:
    pipeline = UnifiedGlassPipeline(
        PipelineConfig(
            csv_path=Path(args.csv),
            image_root=Path(args.image_root),
            image_size=args.image_size,
            device=resolve_device(args.device),
        )
    )
    resolved = pipeline.resolved_frame()
    splits = pipeline.patient_split_frames(
        resolved,
        val_fraction=0.15,
        test_fraction=0.15,
        seed=42,
    )
    label_map = pipeline.label_map()
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
    model, metrics = train_resnet_branch(
        pipeline.client_records_from_frame(splits["train"]),
        pipeline.client_records_from_frame(splits["val"]),
        pipeline.client_records_from_frame(splits["test"]),
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
    if args.command == "train":
        return cmd_train(args)
    if args.command == "resnet":
        return cmd_resnet(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
