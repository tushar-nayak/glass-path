from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = ("superclass", "subclass", "resolution", "image_id", "patient_id")


@dataclass(frozen=True)
class SampleRecord:
    superclass: str
    subclass: str
    resolution: str
    image_id: str
    patient_id: str
    image_path: str | None = None

    @property
    def label(self) -> str:
        return self.superclass


class GlassDataset:
    """CSV-backed dataset with patient-level federated partitioning."""

    def __init__(self, frame: pd.DataFrame, csv_path: str | Path | None = None):
        self.frame = frame.copy()
        self.csv_path = Path(csv_path) if csv_path else None
        self._validate()

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "GlassDataset":
        frame = pd.read_csv(csv_path, dtype=str).fillna("")
        return cls(frame, csv_path=csv_path)

    def _validate(self) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in self.frame.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        for column in REQUIRED_COLUMNS:
            if self.frame[column].astype(str).eq("").any() and column != "subclass":
                raise ValueError(f"Column '{column}' contains empty values")

    def __len__(self) -> int:
        return len(self.frame)

    def __iter__(self) -> Iterable[SampleRecord]:
        for _, row in self.frame.iterrows():
            yield SampleRecord(
                superclass=str(row["superclass"]),
                subclass=str(row["subclass"]),
                resolution=str(row["resolution"]),
                image_id=str(row["image_id"]),
                patient_id=str(row["patient_id"]),
            )

    def summary(self) -> dict:
        frame = self.frame
        return {
            "rows": len(frame),
            "patients": frame["patient_id"].nunique(),
            "superclasses": frame["superclass"].value_counts().sort_index().to_dict(),
            "subclasses": frame["subclass"].value_counts().sort_index().to_dict(),
            "resolutions": frame["resolution"].value_counts().sort_index().to_dict(),
        }

    def partition_by_patient(self) -> list[pd.DataFrame]:
        return [group.copy() for _, group in self.frame.groupby("patient_id", sort=True)]

    def records_for_patient(self, patient_id: str) -> list[SampleRecord]:
        subset = self.frame[self.frame["patient_id"].astype(str) == str(patient_id)]
        return [
            SampleRecord(
                superclass=str(row.superclass),
                subclass=str(row.subclass),
                resolution=str(row.resolution),
                image_id=str(row.image_id),
                patient_id=str(row.patient_id),
            )
            for row in subset.itertuples(index=False)
        ]

    def resolve_image_paths(
        self,
        image_root: str | Path,
        patterns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Attach image paths using a list of filename templates."""
        root = Path(image_root)
        patterns = patterns or [
            "{patient_id}/{image_id}_{resolution}.jpg",
            "{patient_id}/{image_id}_{resolution}.jpeg",
            "{patient_id}/{image_id}_{resolution}.png",
            "{patient_id}/{image_id}.jpg",
            "{patient_id}/{image_id}.jpeg",
            "{patient_id}/{image_id}.png",
            "{resolution}/{patient_id}_{image_id}.jpg",
            "{image_id}_{resolution}.jpg",
            "{image_id}.jpg",
        ]
        resolved = self.frame.copy()
        paths: list[str | None] = []
        for row in resolved.itertuples(index=False):
            candidate = None
            for pattern in patterns:
                rel = pattern.format(
                    superclass=row.superclass,
                    subclass=row.subclass,
                    resolution=row.resolution,
                    image_id=row.image_id,
                    patient_id=row.patient_id,
                )
                path = root / rel
                if path.exists():
                    candidate = str(path)
                    break
            paths.append(candidate)
        resolved["image_path"] = paths
        return resolved

    def to_records(self, include_image_path: bool = False) -> list[SampleRecord]:
        records: list[SampleRecord] = []
        for row in self.frame.itertuples(index=False):
            image_path = getattr(row, "image_path", None) if include_image_path else None
            records.append(
                SampleRecord(
                    superclass=str(row.superclass),
                    subclass=str(row.subclass),
                    resolution=str(row.resolution),
                    image_id=str(row.image_id),
                    patient_id=str(row.patient_id),
                    image_path=str(image_path) if image_path else None,
                )
            )
        return records
