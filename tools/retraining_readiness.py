"""Estimate whether a retraining submission can reach YOLO training.

The training project only discovers a data shortage after the operator has
finished reviewing, finished annotating, and the pipeline has already run
augmentation and lint. This module reproduces the *necessary* split conditions
so the same shortage can be reported before any of that work is spent.

The thresholds mirror ``Yolo11_auto_train``:

- ``picture_tool.gui.operator_handoff.apply_handoff_to_config`` pins
  ``split.minimum_source_groups`` to ``{"val": 5, "test": 10}``.
- ``picture_tool.split.dataset_splitter.split_dataset`` requires at least three
  independent source groups overall, and requires the groups that are *not*
  forced into train to cover ``minimum_val_groups + dynamic_minimum_test_groups``.

Passing this check is necessary but not sufficient: per-class instance minimums
are only verifiable once the labels exist, so the operator text says so.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

# Mirrors Yolo11_auto_train split.minimum_source_groups.
MINIMUM_VAL_SOURCE_GROUPS = 5
MINIMUM_TEST_SOURCE_GROUPS = 10
# Mirrors dataset_splitter's "at least three independent source groups".
MINIMUM_TOTAL_SOURCE_GROUPS = 3


@dataclass(frozen=True)
class RetrainingReadiness:
    """Projected split feasibility for one product/station submission."""

    submitted_count: int
    historical_count: int
    required_historical_count: int

    @property
    def total_count(self) -> int:
        return self.submitted_count + self.historical_count

    @property
    def historical_shortfall(self) -> int:
        return max(0, self.required_historical_count - self.historical_count)

    @property
    def total_shortfall(self) -> int:
        return max(0, MINIMUM_TOTAL_SOURCE_GROUPS - self.total_count)

    @property
    def shortfall(self) -> int:
        """Images still needed before training can start."""
        return max(self.historical_shortfall, self.total_shortfall)

    @property
    def is_ready(self) -> bool:
        return self.shortfall == 0

    def to_operator_text(self, *, language: str = "zh_TW") -> str:
        """Return a concise, actionable summary for the submission dialog."""
        if language != "zh_TW":
            if self.is_ready:
                return (
                    f"Accumulated {self.total_count} trainable image(s) "
                    f"({self.submitted_count} from this submission). "
                    "Class-level minimums are still checked during training."
                )
            return (
                f"Accumulated {self.total_count} trainable image(s), of which "
                f"{self.historical_count} are earlier submissions. Safe "
                f"validation needs at least {self.required_historical_count} "
                f"earlier images, so about {self.shortfall} more are required "
                "before retraining can start."
            )
        if self.is_ready:
            return (
                f"目前累計可訓練影像 {self.total_count} 張"
                f"（本次 {self.submitted_count} 張）。"
                "數量已達安全切分下限；各類別的樣本數仍會在訓練時檢查。"
            )
        return (
            f"目前累計可訓練影像 {self.total_count} 張，"
            f"其中先前累積的有 {self.historical_count} 張。\n"
            f"系統需要至少 {self.required_historical_count} 張先前累積的影像"
            "才能安全切分驗證集與測試集，"
            f"因此還需要約 {self.shortfall} 張才會開始補訓。"
        )


def required_historical_count(position_golden_count: int = 0) -> int:
    """Return how many earlier images the splitter needs outside this batch.

    Position golden samples are forced into test, which reduces how many
    historical groups the dynamic test split must supply.
    """
    dynamic_test_groups = max(
        0, MINIMUM_TEST_SOURCE_GROUPS - max(0, int(position_golden_count))
    )
    return MINIMUM_VAL_SOURCE_GROUPS + dynamic_test_groups


def count_accumulated_training_images(
    training_data_dir: str | Path, *, product: str, area: str
) -> int:
    """Count distinct samples already promoted into a station's raw dataset."""
    manifest = (
        Path(training_data_dir)
        / _safe_name(product)
        / _safe_name(area)
        / "metadata"
        / "review_dataset_manifest.csv"
    )
    if not manifest.is_file():
        return 0
    try:
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            return len(
                {
                    str(row.get("sample_id") or "").strip()
                    for row in csv.DictReader(handle)
                    if str(row.get("sample_id") or "").strip()
                }
            )
    except (OSError, UnicodeDecodeError, csv.Error):
        # An unreadable manifest must not block a submission; the training
        # project re-validates the dataset before it trains.
        return 0


def evaluate_retraining_readiness(
    training_data_dir: str | Path,
    *,
    product: str,
    area: str,
    submitted_count: int,
    position_golden_count: int = 0,
) -> RetrainingReadiness:
    """Project whether this submission can reach training.

    Args:
        training_data_dir: Root of the shared training data directory.
        product: Product identifier for the submission.
        area: Station identifier for the submission.
        submitted_count: Images this submission contributes to the raw dataset,
            including cases that still need annotation.
        position_golden_count: Position golden samples forced into the test
            split by this submission.

    Returns:
        A readiness projection with the remaining image shortfall.
    """
    accumulated = count_accumulated_training_images(
        training_data_dir, product=product, area=area
    )
    submitted = max(0, int(submitted_count))
    # Re-reviewed images already appear in the accumulated manifest. Treating
    # the overlap as historical keeps the estimate optimistic, so the check
    # never blocks a submission that would actually have trained.
    historical = max(0, accumulated - submitted)
    return RetrainingReadiness(
        submitted_count=submitted,
        historical_count=historical,
        required_historical_count=required_historical_count(position_golden_count),
    )


def _safe_name(value: str) -> str:
    """Match the dataset folder naming used by ``export_review_dataset``."""
    text = str(value or "unknown").strip() or "unknown"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
