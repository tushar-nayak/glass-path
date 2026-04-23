from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import PipelineConfig, UnifiedGlassPipeline
from .runtime import resolve_device
from .classifier import save_checkpoint
from .backbones import make_torchvision_backbone
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
    train.add_argument(
        "--max-patients",
        type=int,
        default=0,
        help="If set, only use the first N patients (sorted by patient_id) to reduce runtime.",
    )
    train.add_argument(
        "--max-images-per-patient",
        type=int,
        default=0,
        help="If set, cap the number of images used per patient to reduce runtime.",
    )
    train.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="If set, limit PyTorch CPU threads (recommended 1 when sharing a machine).",
    )

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

    fedsup = subparsers.add_parser(
        "fedsup",
        help="Federated supervised training using a pretrained torchvision backbone (recommended baseline)",
    )
    fedsup.add_argument("--csv", required=True, help="Path to data.csv")
    fedsup.add_argument("--image-root", default="data/images", help="Directory containing images")
    fedsup.add_argument("--image-size", type=int, default=224)
    fedsup.add_argument("--device", default="auto", help="auto, mps, cuda, or cpu")
    fedsup.add_argument("--backbone", default="resnet50", help="torchvision backbone (resnet50)")
    fedsup.add_argument("--pretrained", action="store_true", help="Use torchvision pretrained weights")
    fedsup.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone and train head only")
    fedsup.add_argument("--class-weighting", default="balanced", help="none or balanced")
    fedsup.add_argument("--rounds", type=int, default=8)
    fedsup.add_argument("--local-epochs", type=int, default=1)
    fedsup.add_argument("--batch-size", type=int, default=16)
    fedsup.add_argument("--lr", type=float, default=1e-4)
    fedsup.add_argument("--client-fraction", type=float, default=0.3)
    fedsup.add_argument("--prox-mu", type=float, default=0.01)
    fedsup.add_argument("--verbose", action="store_true")
    fedsup.add_argument("--num-threads", type=int, default=0)
    fedsup.add_argument("--save-dir", default=None, help="Output directory (defaults under runs/)")
    fedsup.add_argument("--val-fraction", type=float, default=0.15)
    fedsup.add_argument("--test-fraction", type=float, default=0.15)
    fedsup.add_argument("--seed", type=int, default=42)

    return parser


def cmd_inspect(args: argparse.Namespace) -> int:
    pipeline = UnifiedGlassPipeline(PipelineConfig(csv_path=Path(args.csv)))
    report = pipeline.inspect()
    print(report)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    if args.num_threads and args.num_threads > 0:
        try:
            import torch
        except ModuleNotFoundError:
            torch = None
        if torch is not None:
            torch.set_num_threads(int(args.num_threads))
            try:
                torch.set_num_interop_threads(1)
            except (AttributeError, RuntimeError):
                pass

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

    # Optional deterministic subsampling to keep resource usage low.
    frame = resolved
    if args.max_patients and args.max_patients > 0:
        # Prefer a label-stratified selection so small smoke runs cover multiple classes.
        # Deterministic: within each label, patients are sorted by patient_id and then
        # sampled round-robin across labels.
        grouped = frame.groupby("patient_id", sort=True)["superclass"]
        patient_to_label = grouped.agg(lambda s: str(s.mode().iloc[0]) if len(s.mode()) else str(s.iloc[0]))
        by_label: dict[str, list[str]] = {}
        for patient_id, label in patient_to_label.items():
            by_label.setdefault(str(label), []).append(str(patient_id))
        for label in by_label:
            by_label[label] = sorted(by_label[label])

        selected: list[str] = []
        labels = sorted(by_label)
        while len(selected) < int(args.max_patients):
            progressed = False
            for label in labels:
                ids = by_label.get(label) or []
                if not ids:
                    continue
                selected.append(ids.pop(0))
                progressed = True
                if len(selected) >= int(args.max_patients):
                    break
            if not progressed:
                break

        keep = set(selected)
        frame = frame[frame["patient_id"].astype(str).isin(keep)].copy()
    if args.max_images_per_patient and args.max_images_per_patient > 0:
        k = int(args.max_images_per_patient)
        frame = frame.groupby("patient_id", sort=True, group_keys=False).head(k).copy()

    with_paths = frame["image_path"].notna().sum() if "image_path" in frame.columns else 0
    image_root = Path(args.image_root) if args.image_root else Path("data/images")
    splits = pipeline.patient_split_frames(
        frame,
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
            "rows": int(len(frame)),
            "image_paths_resolved": int(with_paths),
            "clients": int(len(frame.groupby("patient_id", sort=True))),
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


def cmd_fedsup(args: argparse.Namespace) -> int:
    if args.num_threads and args.num_threads > 0:
        try:
            import torch
        except ModuleNotFoundError:
            torch = None
        if torch is not None:
            torch.set_num_threads(int(args.num_threads))
            try:
                torch.set_num_interop_threads(1)
            except (AttributeError, RuntimeError):
                pass

    pipeline = UnifiedGlassPipeline(
        PipelineConfig(
            csv_path=Path(args.csv),
            image_root=Path(args.image_root),
            image_size=args.image_size,
            device=resolve_device(args.device),
        )
    )
    resolved = pipeline.resolved_frame()
    with_paths = resolved["image_path"].notna().sum() if "image_path" in resolved.columns else 0
    if with_paths == 0:
        print("No image files were resolved. Provide --image-root to run training.")
        return 1

    splits = pipeline.patient_split_frames(
        resolved,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    train_records = pipeline.client_records_from_frame(splits["train"])
    val_records = pipeline.client_records_from_frame(splits["val"])
    test_records = pipeline.client_records_from_frame(splits["test"])
    label_map = pipeline.label_map()

    from .classifier import FederatedClassifierConfig, FederatedClassifierTrainer

    backbone = make_torchvision_backbone(
        name=args.backbone,
        pretrained=bool(args.pretrained),
        normalize=True,
    )
    config = FederatedClassifierConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        rounds=args.rounds,
        lr=args.lr,
        device=resolve_device(args.device),
        freeze_backbone=bool(args.freeze_backbone),
        class_weighting=str(args.class_weighting),
        client_fraction=float(args.client_fraction),
        seed=int(args.seed),
        verbose=bool(args.verbose),
        normalization="imagenet" if bool(args.pretrained) else "instance",
        prox_mu=float(args.prox_mu),
    )
    trainer = FederatedClassifierTrainer(config, label_map)
    model, metrics = trainer.train_and_evaluate(train_records, val_records, test_records, backbone)

    # Default output location under runs/.
    save_dir = Path(args.save_dir) if args.save_dir else Path("runs") / "fedsup_resnet50"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        save_dir / "classifier.pt",
        model,
        metadata={
            "branch": "fedsup",
            "backbone": args.backbone,
            "pretrained": bool(args.pretrained),
            "freeze_backbone": bool(args.freeze_backbone),
            "label_map": label_map,
            "device": config.device,
        },
    )
    metrics_payload = {
        "best_round": int(metrics["best_round"]),
        "val": metrics["val"].__dict__,
        "test": metrics["test"].__dict__,
    }
    (save_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    print({"save_dir": str(save_dir), "metrics": metrics_payload})
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
    if args.command == "fedsup":
        return cmd_fedsup(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
