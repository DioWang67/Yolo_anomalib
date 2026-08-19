"""Presentation-only dataset preparation summary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tools.processing_reports import ProcessingReport


@dataclass(frozen=True)
class DatasetPreparationViewModel:
    visible: bool = False
    dry_run: bool = True
    dataset_id: str = ""
    accepted_count: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    dataset_hash: str = ""
    artifact_path: str = ""
    preparation_report_path: str = ""
    message: str = ""

    @classmethod
    def from_report(cls, report: ProcessingReport, root: str | Path) -> DatasetPreparationViewModel:
        result = next(
            (
                item
                for item in report.sample_results
                if item.step_id == "dataset_preparation"
            ),
            None,
        )
        if result is None:
            return cls()
        metadata = result.metadata
        splits = metadata.get("split_counts", {})
        report_path = ""
        if result.artifacts:
            report_path = str((Path(root) / result.artifacts[0].relative_path).resolve())
        return cls(
            visible=True,
            dry_run=bool(metadata.get("dry_run", report.dry_run)),
            dataset_id=str(metadata.get("dataset_id") or ""),
            accepted_count=int(metadata.get("accepted_count") or 0),
            train_count=int(splits.get("train") or 0),
            val_count=int(splits.get("val") or 0),
            test_count=int(splits.get("test") or 0),
            dataset_hash=str(metadata.get("dataset_hash") or ""),
            artifact_path=str(metadata.get("dataset_path") or ""),
            preparation_report_path=report_path,
            message=result.message,
        )
