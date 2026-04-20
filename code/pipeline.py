from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import SampleRecord
from .concept import ConceptBottleneckConfig, MultimodalConceptBottleneckConfig
from .classifier import FederatedClassifierConfig
from .data import GlassDataset
from .federated import split_by_patient
from .runtime import resolve_device


@dataclass
class PipelineConfig:
    csv_path: str | Path
    image_root: str | Path | None = Path("data/images")
    image_patterns: list[str] = field(default_factory=list)
    image_size: int = 224
    ssl_embedding_dim: int = 256
    graph_embedding_dim: int = 128
    concept_dim: int = 8
    hidden_dim: int = 128
    num_classes: int = 3
    device: str = "auto"


class UnifiedGlassPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.dataset = GlassDataset.from_csv(config.csv_path)

    def inspect(self) -> dict:
        return {
            "dataset": self.dataset.summary(),
            "clients": len(split_by_patient(self.dataset.frame)),
            "config": self.config,
        }

    def resolved_frame(self) -> pd.DataFrame:
        if self.config.image_root is None:
            return self.dataset.frame.copy()
        image_root = Path(self.config.image_root)
        if not image_root.exists():
            return self.dataset.frame.copy()
        patterns = self.config.image_patterns or None
        return self.dataset.resolve_image_paths(image_root, patterns=patterns)

    def client_records(self) -> list[list[SampleRecord]]:
        frame = self.resolved_frame()
        return self.client_records_from_frame(frame)

    def client_records_from_frame(self, frame: pd.DataFrame) -> list[list[SampleRecord]]:
        client_records: list[list[SampleRecord]] = []
        for _, group in frame.groupby("patient_id", sort=True):
            records = [
                SampleRecord(
                    superclass=str(row.superclass),
                    subclass=str(row.subclass),
                    resolution=str(row.resolution),
                    image_id=str(row.image_id),
                    patient_id=str(row.patient_id),
                    image_path=str(row.image_path)
                    if hasattr(row, "image_path") and pd.notna(row.image_path) and str(row.image_path)
                    else None,
                )
                for row in group.itertuples(index=False)
                if hasattr(row, "image_path") and pd.notna(row.image_path) and str(row.image_path)
            ]
            if records:
                client_records.append(records)
        return client_records

    def patient_split_frames(
        self,
        frame: pd.DataFrame | None = None,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        seed: int = 42,
    ) -> dict[str, pd.DataFrame]:
        frame = frame.copy() if frame is not None else self.resolved_frame()
        if frame.empty:
            return {
                "train": frame.copy(),
                "val": frame.copy(),
                "test": frame.copy(),
            }
        patient_labels = []
        for patient_id, group in frame.groupby("patient_id", sort=True):
            label = group["superclass"].mode().iloc[0]
            patient_labels.append((str(patient_id), str(label)))
        by_label: dict[str, list[str]] = {}
        for patient_id, label in patient_labels:
            by_label.setdefault(label, []).append(patient_id)
        rng = np.random.default_rng(seed)
        train_ids: list[str] = []
        val_ids: list[str] = []
        test_ids: list[str] = []
        for label, patient_ids in sorted(by_label.items()):
            ids = list(patient_ids)
            rng.shuffle(ids)
            n = len(ids)
            n_test = max(1, int(round(n * test_fraction))) if n >= 3 else max(0, int(round(n * test_fraction)))
            n_val = max(1, int(round(n * val_fraction))) if n >= 3 else max(0, int(round(n * val_fraction)))
            if n_test + n_val >= n:
                n_test = 1 if n >= 2 else 0
                n_val = 1 if n >= 3 else 0
            test_ids.extend(ids[:n_test])
            val_ids.extend(ids[n_test : n_test + n_val])
            train_ids.extend(ids[n_test + n_val :])
        train_df = frame[frame["patient_id"].astype(str).isin(train_ids)].copy()
        val_df = frame[frame["patient_id"].astype(str).isin(val_ids)].copy()
        test_df = frame[frame["patient_id"].astype(str).isin(test_ids)].copy()
        return {"train": train_df, "val": val_df, "test": test_df}

    def ssl_config(self):
        from .ssl import FederatedSSLConfig

        return FederatedSSLConfig(
            image_size=self.config.image_size,
            embedding_dim=self.config.ssl_embedding_dim,
            device=resolve_device(self.config.device),
        )

    def quick_ssl_config(self):
        config = self.ssl_config()
        config.local_epochs = 1
        config.rounds = 1
        config.batch_size = min(config.batch_size, 8)
        return config

    def concept_config(self) -> ConceptBottleneckConfig:
        return ConceptBottleneckConfig(
            input_dim=self.config.ssl_embedding_dim + self.config.graph_embedding_dim,
            concept_dim=self.config.concept_dim,
            num_classes=self.config.num_classes,
            hidden_dim=self.config.hidden_dim,
        )

    def multimodal_concept_config(self) -> MultimodalConceptBottleneckConfig:
        return MultimodalConceptBottleneckConfig(
            ssl_dim=self.config.ssl_embedding_dim,
            graph_dim=self.config.graph_embedding_dim,
            fusion_dim=self.config.hidden_dim,
            concept_dim=self.config.concept_dim,
            num_classes=self.config.num_classes,
            hidden_dim=self.config.hidden_dim,
        )

    def classifier_config(self) -> FederatedClassifierConfig:
        return FederatedClassifierConfig(
            image_size=self.config.image_size,
            device=resolve_device(self.config.device),
        )

    def quick_classifier_config(self) -> FederatedClassifierConfig:
        config = self.classifier_config()
        config.local_epochs = 1
        config.rounds = 1
        config.batch_size = min(config.batch_size, 8)
        return config

    def label_map(self) -> dict[str, int]:
        labels = sorted(str(label) for label in self.dataset.frame["superclass"].unique())
        return {label: idx for idx, label in enumerate(labels)}

    def train_ssl(self):
        from .ssl import FederatedSSLTrainer

        trainer = FederatedSSLTrainer(self.ssl_config())
        client_records = self.client_records()
        return trainer.fit(client_records)

    def train_ssl_from_frame(self, frame: pd.DataFrame, quick: bool = False):
        from .ssl import FederatedSSLTrainer

        trainer = FederatedSSLTrainer(self.quick_ssl_config() if quick else self.ssl_config())
        client_records = self.client_records_from_frame(frame)
        return trainer.fit(client_records)

    def train_ssl_quick(self):
        from .ssl import FederatedSSLTrainer

        trainer = FederatedSSLTrainer(self.quick_ssl_config())
        client_records = self.client_records()
        return trainer.fit(client_records)

    def train_classifier(self, backbone):
        from .classifier import FederatedClassifierTrainer

        trainer = FederatedClassifierTrainer(self.classifier_config(), self.label_map())
        client_records = self.client_records()
        return trainer.fit(client_records, backbone)

    def train_classifier_from_frame(self, frame: pd.DataFrame, backbone, quick: bool = False):
        from .classifier import FederatedClassifierTrainer

        trainer = FederatedClassifierTrainer(
            self.quick_classifier_config() if quick else self.classifier_config(),
            self.label_map(),
        )
        client_records = self.client_records_from_frame(frame)
        return trainer.fit(client_records, backbone)

    def train_graph(self, graphs, labels, input_dim: int):
        from .graph import train_graph_classifier

        return train_graph_classifier(
            graphs=graphs,
            labels=labels,
            in_dim=input_dim,
            num_classes=self.config.num_classes,
            hidden_dim=self.config.hidden_dim,
            embedding_dim=self.config.graph_embedding_dim,
            device=resolve_device(self.config.device),
        )

    def graph_inputs_from_images(self, graph_builder, image_feature_extractor):
        """Build graph inputs from images without relying on manual labels.

        `graph_builder` should convert an image or image-derived representation
        into a `GraphSample`. `image_feature_extractor` should produce node
        features from the same source.
        """
        frame = self.resolved_frame()
        graphs = []
        for row in frame.itertuples(index=False):
            image_path = getattr(row, "image_path", None)
            if not image_path or pd.isna(image_path):
                continue
            graph = graph_builder(
                image_path=image_path,
                image_id=str(row.image_id),
                patient_id=str(row.patient_id),
                label=str(row.superclass),
                subclass=str(row.subclass),
                resolution=str(row.resolution),
            )
            graphs.append(graph)
        return graphs

    def train_concept(self, features, concept_targets, class_targets):
        from .concept import train_concept_bottleneck

        return train_concept_bottleneck(
            features=features,
            concept_targets=concept_targets,
            class_targets=class_targets,
            config=self.concept_config(),
            device=resolve_device(self.config.device),
        )

    def train_multimodal_concept(self, ssl_features, graph_features, concept_targets, class_targets):
        from .concept import train_multimodal_concept_bottleneck

        return train_multimodal_concept_bottleneck(
            ssl_features=ssl_features,
            graph_features=graph_features,
            concept_targets=concept_targets,
            class_targets=class_targets,
            config=self.multimodal_concept_config(),
            device=resolve_device(self.config.device),
        )
