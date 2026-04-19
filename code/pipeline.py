from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .data import SampleRecord
from .concept import ConceptBottleneckConfig
from .data import GlassDataset
from .federated import split_by_patient


@dataclass
class PipelineConfig:
    csv_path: str | Path
    image_root: str | Path | None = None
    image_patterns: list[str] = field(default_factory=list)
    image_size: int = 224
    ssl_embedding_dim: int = 256
    graph_embedding_dim: int = 128
    concept_dim: int = 8
    hidden_dim: int = 128
    num_classes: int = 3
    device: str = "cpu"


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
        patterns = self.config.image_patterns or None
        return self.dataset.resolve_image_paths(self.config.image_root, patterns=patterns)

    def client_records(self) -> list[list[SampleRecord]]:
        frame = self.resolved_frame()
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

    def ssl_config(self):
        from .ssl import FederatedSSLConfig

        return FederatedSSLConfig(
            image_size=self.config.image_size,
            embedding_dim=self.config.ssl_embedding_dim,
            device=self.config.device,
        )

    def concept_config(self) -> ConceptBottleneckConfig:
        return ConceptBottleneckConfig(
            input_dim=self.config.ssl_embedding_dim + self.config.graph_embedding_dim,
            concept_dim=self.config.concept_dim,
            num_classes=self.config.num_classes,
            hidden_dim=self.config.hidden_dim,
        )

    def train_ssl(self):
        from .ssl import FederatedSSLTrainer

        trainer = FederatedSSLTrainer(self.ssl_config())
        client_records = self.client_records()
        return trainer.fit(client_records)

    def train_graph(self, graphs, labels, input_dim: int):
        from .graph import train_graph_classifier

        return train_graph_classifier(
            graphs=graphs,
            labels=labels,
            in_dim=input_dim,
            num_classes=self.config.num_classes,
            hidden_dim=self.config.hidden_dim,
            embedding_dim=self.config.graph_embedding_dim,
            device=self.config.device,
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
            device=self.config.device,
        )
