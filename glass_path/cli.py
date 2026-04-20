from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PipelineConfig, UnifiedGlassPipeline


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
    train.add_argument("--device", default="cpu")

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
            image_root=Path(args.image_root) if args.image_root else None,
            image_size=args.image_size,
            ssl_embedding_dim=args.ssl_dim,
            graph_embedding_dim=args.graph_dim,
            concept_dim=args.concept_dim,
            num_classes=args.num_classes,
            device=args.device,
        )
    )
    resolved = pipeline.resolved_frame()
    with_paths = resolved["image_path"].notna().sum() if "image_path" in resolved.columns else 0
    print(
        {
            "rows": len(resolved),
            "image_paths_resolved": int(with_paths),
            "clients": len(resolved.groupby("patient_id", sort=True)),
            "ssl_config": pipeline.ssl_config(),
            "concept_config": pipeline.concept_config(),
        }
    )
    if with_paths == 0:
        print("No image files were resolved. Provide --image-root to run training.")
        return 1
    try:
        model = pipeline.train_ssl()
    except ModuleNotFoundError as exc:
        print(str(exc))
        return 1
    print({"ssl_model": type(model).__name__})
    print(
        "Federated SSL training completed for the image encoder. "
        "Graph and concept stages are ready for derived graph features and concept labels."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "inspect":
        return cmd_inspect(args)
    if args.command == "train":
        return cmd_train(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
