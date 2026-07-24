"""Button-based review workflow for inference cases used in retraining."""

from __future__ import annotations

import csv
import json
import logging
import os
import socket
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from PyQt5.QtCore import QDateTime, QEvent, QProcess, Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDateTimeEdit,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.gui.async_image_service import AsyncImageService
from app.gui.dialog_geometry import configure_responsive_dialog
from app.gui.historical_cleanup_dialog import HistoricalCleanupDialog
from app.gui.historical_cleanup_view_model import HistoricalCleanupViewModel
from app.gui.processing_batch_dialog import (
    ProcessingBatchDialog,
    processing_annotation_step_enabled,
    processing_color_step_enabled,
    processing_dataset_dry_run_enabled,
    processing_dataset_step_enabled,
    processing_execution_framework_enabled,
    processing_pipeline_enabled,
)
from app.gui.processing_execution_view_model import ProcessingExecutionViewModel
from app.gui.processing_summary_view_model import ProcessingSummaryViewModel
from app.gui.retraining_settings_dialog import RetrainingSettingsDialog
from app.gui.review_selection_gallery import ReviewSelectionGallery, record_is_reviewed
from app.gui.review_workspace import (
    ReviewImageViewer,
    ReviewThumbnailPanel,
    build_review_details,
    build_review_list_item,
    is_text_input_focus,
)
from app.gui.training_batch_dialog import (
    ANNOTATION_LABELS,
    COLOR_REVIEW_LABELS,
    DIRECT_TRAIN_LABELS,
    TrainingBatchDialog,
)
from core.retraining_options import RetrainingOptions
from core.services.inspection_repository import InspectionRepository
from core.workspace import load_workspace_paths
from tools.annotation_packages import load_annotation_package
from tools.annotation_resume import AnnotationResumeService
from tools.annotation_revisions import AnnotationRevisionStore
from tools.annotation_tool_launcher import ManualAnnotationToolLauncher
from tools.collect_review_cases import (
    collect_review_cases,
    timestamp_in_range,
    write_manifest,
)
from tools.color_calibration_packages import load_color_calibration_package
from tools.color_calibration_resume import (
    ColorCalibrationApprovalService,
    ColorCalibrationResumeService,
)
from tools.color_calibration_service import CalibrationPolicy
from tools.color_configuration_resolver import ColorConfigurationResolver
from tools.color_configuration_revisions import ColorConfigurationRevisionStore
from tools.color_feedback import read_color_feedback_progress
from tools.export_review_dataset import (
    export_operator_handoff,
    update_operator_job_status,
)
from tools.historical_cleanup import HistoricalCleanupAnalyzer, HistoricalCleanupError
from tools.portable_training_package import (
    PortableTrainingPackageError,
    export_portable_training_package,
)
from tools.processing_engine_factory import build_processing_execution_engine
from tools.processing_pipeline import ProcessingPlanner, sha256_file
from tools.processing_plan_validation import (
    ProcessingPlanValidator,
    build_validation_context_from_manifest,
)
from tools.processing_run_store import ProcessingRunStore
from tools.review_action_planner import (
    DETECTION_PRESENT,
    DETECTION_UNKNOWN,
    SKIP_CONFIRMED_FAILURE,
    SKIP_EQUIPMENT_LIGHTING,
    SKIP_IMAGE_QUALITY,
    SKIP_OTHER,
    SKIP_UNJUDGEABLE,
    ReviewActionPlan,
    detection_state,
    plan_fail,
    plan_pass,
    plan_skip,
    skip_custom_note_from_record,
    skip_ui_reason_from_record,
)
from tools.review_classification import (
    COLOR_ISSUE,
    LIGHTING_ISSUE,
    MISCLASSIFICATION,
    MISSED_DETECTION,
    NEW_DEFECT_TYPE,
    OTHER_FAILURE,
    THRESHOLD_NOT_MET,
    WRONG_BOX,
    WRONG_CLASS,
    ReviewFailureClassification,
    threshold_source_keys,
)
from tools.review_routing import (
    ReviewDecision,
    action_route,
    color_summary,
    has_color_failure,
    has_non_color_failure,
    has_threshold_color_failure,
)
from tools.review_workflow import (
    REVIEW_CORE_FIELDS,
    ReviewWorkflowValidationError,
    blocking_violations,
    ensure_transition_valid,
    record_identity,
    validate_record_consistency,
)
from tools.submission_history import record_submission_history

logger = logging.getLogger(__name__)
LEGACY_SELECTED_PAGE_ENV = "YOLO_REVIEW_LEGACY_SELECTED_PAGE"
LEGACY_REVIEW_LAYOUT_ENV = "YOLO_REVIEW_LEGACY_LAYOUT"
REVIEW_FILTER_STATE_SCHEMA_VERSION = 2


def _has_human_review_result(row: dict[str, str]) -> bool:
    """Return whether Phase 1B fields contain an explicit operator result."""
    return bool(
        str(row.get("review_outcome") or "").strip()
        or str(row.get("review_label") or "").strip()
    )

REVIEW_ACTIONS = (
    (
        "confirmed_ng",
        "確認 NG（AI 判定正確）",
        "Confirm NG (AI verdict correct)",
    ),
    (
        "confirmed_ok",
        "確認 OK（AI 判定正確）",
        "Confirm OK (AI verdict correct)",
    ),
    (
        "verified_empty",
        "實際無目標（負樣本）",
        "No target present (negative sample)",
    ),
    ("false_positive", "實際 OK（AI 過殺）", "Actually OK (AI overkill)"),
    ("false_negative", "實際 NG（AI 漏檢）", "Actually NG (AI missed it)"),
    (
        "needs_annotation",
        "確認 NG，但標註需修正",
        "Confirm NG, but fix annotation",
    ),
    (
        "color_confirmed_ng",
        "顏色確實 NG（顏色判定正確）",
        "Color is truly NG (color verdict correct)",
    ),
    (
        "color_false_reject",
        "顏色其實 OK（門檻過嚴）",
        "Color is actually OK (threshold too strict)",
    ),
    (
        "color_needs_annotation",
        "顏色覆核＋框需修正",
        "Review color and correct the box",
    ),
    (
        "image_quality_issue",
        "圖片無法判定（不採用）",
        "Image cannot be judged (exclude)",
    ),
)

STORED_REVIEW_LABELS = frozenset(
    {
        "confirmed_ng",
        "confirmed_ok",
        "verified_empty",
        "false_positive",
        "false_negative",
        "wrong_box",
        "wrong_class",
        "image_quality_issue",
        "color_confirmed_ng",
        "color_false_reject",
    }
)
NON_TRAINING_LABELS = frozenset({"confirmed_ok", "uncertain", "image_quality_issue"})
ANNOTATION_CORRECTIONS = (
    ("wrong_box", "框的位置／數量錯誤", "Box position or count is wrong"),
    ("wrong_class", "框的類別錯誤", "Box class is wrong"),
)

ACTION_COLORS = {
    "confirmed_ng": "#b42318",
    "confirmed_ok": "#237a3b",
    "verified_empty": "#326b85",
    "false_positive": "#d97706",
    "false_negative": "#a13b32",
    "needs_annotation": "#8754a1",
    "image_quality_issue": "#5f6368",
    "color_confirmed_ng": "#8f2d56",
    "color_false_reject": "#00796b",
    "color_needs_annotation": "#6a4c93",
}

PASS_SAMPLE_INTERVAL = 100

OPERATOR_ACTION_LABELS = {
    "confirmed_ng": ("確認 NG（AI 判定正確）", "Confirmed NG (AI verdict correct)"),
    "confirmed_ok": ("確認 OK（AI 判定正確）", "Confirmed OK (AI verdict correct)"),
    "verified_empty": ("實際無目標（負樣本）", "No target present (negative sample)"),
    "false_positive": ("實際 OK（AI 過殺）", "Actually OK (AI overkill)"),
    "false_negative": ("實際 NG（AI 漏檢）", "Actually NG (AI missed it)"),
    "needs_annotation": ("確認 NG，但標註需修正", "Confirmed NG, annotation needs correction"),
    "wrong_box": ("NG：框的位置／數量需修正", "NG: box position or count needs correction"),
    "wrong_class": ("NG：框的類別需修正", "NG: box class needs correction"),
    "uncertain": ("舊版未判定（不送訓）", "Legacy undecided (do not train)"),
    "image_quality_issue": (
        "圖片無法判定（不採用）",
        "Image cannot be judged (exclude)",
    ),
    "color_confirmed_ng": (
        "顏色確實 NG（顏色判定正確）",
        "Color is truly NG (color verdict correct)",
    ),
    "color_false_reject": (
        "顏色其實 OK（門檻過嚴）",
        "Color is actually OK (threshold too strict)",
    ),
    "color_needs_annotation": (
        "顏色覆核＋框需修正",
        "Review color and correct the box",
    ),
}

FAILURE_CATEGORY_ACTIONS = (
    (
        THRESHOLD_NOT_MET,
        "閾值未達標",
        "Threshold not met",
    ),
)
FAILURE_REASON_OPTIONS = (
    (MISCLASSIFICATION, "誤判／過殺", "Misclassification / false reject"),
    (MISSED_DETECTION, "漏判／漏檢", "Missed detection"),
    (NEW_DEFECT_TYPE, "新缺陷類型", "New defect type"),
    (WRONG_BOX, "框的位置或數量錯誤", "Wrong box position or count"),
    (WRONG_CLASS, "類別判定錯誤", "Wrong class"),
    (THRESHOLD_NOT_MET, "閾值未達標", "Threshold not met"),
    (COLOR_ISSUE, "顏色判定問題", "Color issue"),
    (LIGHTING_ISSUE, "光源／曝光問題", "Lighting / exposure issue"),
    (OTHER_FAILURE, "其他（必須填寫原因）", "Other (note required)"),
)
SKIP_REASON_OPTIONS = (
    ("image_quality_issue", "影像品質有問題", "Image quality issue"),
    (
        "confirmed_failure",
        "AI 確實判定為 Fail，不需回訓",
        "AI correctly found a failure; no retraining",
    ),
)
SKIP_DISPLAY_OPTIONS = (
    (SKIP_UNJUDGEABLE, "無法判定", "Unable to judge"),
    (SKIP_IMAGE_QUALITY, "圖片品質不良", "Poor image quality"),
    (SKIP_EQUIPMENT_LIGHTING, "設備／光源問題", "Equipment / lighting issue"),
    (
        SKIP_CONFIRMED_FAILURE,
        "確認為既有失敗但不適合重訓",
        "Known failure not suitable for retraining",
    ),
    (SKIP_OTHER, "其他（請填寫原因）", "Other (note required)"),
)
FAILURE_SOURCE_LABELS = {
    "yolo": ("YOLO 偵測", "YOLO detection"),
    "color": ("顏色檢查", "Color inspection"),
}
TRAINABLE_REVIEW_LABELS = frozenset(
    DIRECT_TRAIN_LABELS | ANNOTATION_LABELS | COLOR_REVIEW_LABELS
)


class ReviewManifestStore:
    """Persist review decisions immediately after every button click."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        database_path: str | Path | None = None,
        synchronize_repository: bool = True,
        initial_rows: list[dict[str, str]] | None = None,
        diagnose_rows: bool = True,
    ) -> None:
        self.path = Path(manifest_path)
        self.rows = (
            [dict(row) for row in initial_rows]
            if initial_rows is not None
            else self._load_rows()
        )
        self.repository = (
            InspectionRepository(database_path) if database_path is not None else None
        )
        self._submitted_identities: set[str] = set()
        self._ensure_contract_columns()
        self.workflow_diagnostics: dict[str, tuple[Any, ...]] = {}
        if diagnose_rows:
            self._diagnose_existing_rows()
        if synchronize_repository:
            self._sync_existing_rows()

    def _load_rows(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    def set_review(self, index: int, review_label: str) -> None:
        """Set one standardized review label and save it atomically."""
        if review_label not in STORED_REVIEW_LABELS:
            raise ValueError(f"Unsupported review label: {review_label}")
        if not 0 <= index < len(self.rows):
            raise IndexError(f"Review row index out of range: {index}")
        self.set_decision(index, ReviewDecision.from_legacy_label(review_label))

    def set_decision(self, index: int, decision: ReviewDecision) -> None:
        """Persist one structured decision without losing legacy compatibility."""
        if decision.review_label not in STORED_REVIEW_LABELS:
            raise ValueError(f"Unsupported review label: {decision.review_label}")
        if not 0 <= index < len(self.rows):
            raise IndexError(f"Review row index out of range: {index}")
        self._persist_updates(
            {
                index: {
                    **decision.to_columns(),
                    "review_selected": "1",
                    "training_selected": (
                        "0" if decision.review_label in NON_TRAINING_LABELS else "1"
                    ),
                }
            }
        )

    def set_triage(
        self,
        index: int,
        *,
        outcome: str,
        decision: ReviewDecision,
        classification: ReviewFailureClassification | None = None,
        skip_reason: str = "",
        add_to_training_set: bool | None = None,
    ) -> None:
        """Persist the simplified Pass/Fail/Skip operator contract."""
        if outcome not in {"pass", "fail", "skip"}:
            raise ValueError(f"Unsupported review outcome: {outcome}")
        if outcome == "fail" and classification is None:
            raise ValueError("Fail outcome requires a failure classification")
        valid_skip_reasons = {value for value, _zh, _en in SKIP_REASON_OPTIONS}
        if outcome == "skip" and skip_reason not in valid_skip_reasons:
            raise ValueError(f"Unsupported skip reason: {skip_reason}")
        if outcome != "skip" and skip_reason:
            raise ValueError("skip_reason is only valid for skipped reviews")
        selected_for_training = (
            outcome == "fail"
            if add_to_training_set is None
            else bool(add_to_training_set)
        )
        if outcome == "skip" and selected_for_training:
            raise ValueError("Skipped reviews cannot be added to the Training Set")
        if selected_for_training and (
            decision.review_label not in TRAINABLE_REVIEW_LABELS
            or decision.action_route == "none"
        ):
            raise ValueError("This review decision is not eligible for the Training Set")
        values = {
            **decision.to_columns(),
            "review_selected": "1",
            "review_outcome": outcome,
            "skip_reason": skip_reason,
            "training_selected": "1" if selected_for_training else "0",
        }
        values.update(
            classification.to_columns()
            if classification is not None
            else {
                "failure_category": "",
                "failure_source": "",
                "failure_note": "",
            }
        )
        self._persist_updates({index: values})

    def validate_proposed_update(
        self,
        index: int,
        updates: dict[str, str],
        *,
        revision_reason: str = "",
    ) -> dict[str, str]:
        """Validate one in-memory proposal before entering the atomic save path."""
        if not 0 <= index < len(self.rows):
            raise IndexError(f"Review row index out of range: {index}")
        before = dict(self.rows[index])
        after = {**before, **updates}
        identity = self._row_identity(before, index)
        before_for_validation = dict(before)
        after_for_validation = dict(after)
        if identity in self._submitted_identities:
            before_for_validation["submission_status"] = "submitted"
            after_for_validation["submission_status"] = "submitted"
        normalized_revision_reason = revision_reason.strip()
        if normalized_revision_reason:
            after_for_validation["revision_reason"] = normalized_revision_reason
        violations = blocking_violations(
            validate_record_consistency(after_for_validation)
        )
        if violations:
            raise ReviewWorkflowValidationError(after_for_validation, violations)
        ensure_transition_valid(before_for_validation, after_for_validation)
        return after

    def apply_review_updates(
        self,
        index: int,
        updates: dict[str, str],
        *,
        revision_reason: str = "",
    ) -> None:
        """Validate then atomically persist one Phase 2B action plan."""
        normalized_revision_reason = revision_reason.strip()
        self.validate_proposed_update(
            index,
            updates,
            revision_reason=normalized_revision_reason,
        )
        self._persist_updates(
            {index: updates},
            revision_reason=normalized_revision_reason,
        )

    def set_color_review(
        self,
        index: int,
        color_verdict: str,
        *,
        detection_verdict: str = "correct",
    ) -> None:
        """Persist a color verdict and route box corrections independently."""
        if not 0 <= index < len(self.rows):
            raise IndexError(f"Review row index out of range: {index}")
        decision = ReviewDecision.for_color(
            color_verdict,
            detection_verdict=detection_verdict,
            has_non_color_failure=has_non_color_failure(self.rows[index]),
        )
        self.set_decision(index, decision)

    def _ensure_contract_columns(self) -> None:
        for row in self.rows:
            row.setdefault("product_verdict", "")
            row.setdefault("detection_verdict", "")
            row.setdefault("color_verdict", "")
            row.setdefault("action_route", "")
            row.setdefault("failure_category", "")
            row.setdefault("failure_source", "")
            row.setdefault("failure_note", "")
            row.setdefault("review_outcome", "")
            row.setdefault("skip_reason", "")
            row.setdefault("review_selected", "0")
            row.setdefault("training_selected", "1")

    def set_review_selection(
        self,
        candidate_indices: set[int],
        selected_indices: set[int],
    ) -> None:
        """Persist failure-overview checkboxes independently of training routing."""
        if not selected_indices.issubset(candidate_indices):
            raise ValueError("Selected review rows must belong to the candidate set")
        if any(index < 0 or index >= len(self.rows) for index in candidate_indices):
            raise IndexError("Review candidate row index out of range")
        self._persist_updates(
            {
                index: {"review_selected": "1" if index in selected_indices else "0"}
                for index in candidate_indices
            }
        )

    def set_failure_classification(
        self,
        index: int,
        classification: ReviewFailureClassification | None,
    ) -> None:
        """Persist or clear a failure cause without changing its training route."""
        if not 0 <= index < len(self.rows):
            raise IndexError(f"Review row index out of range: {index}")
        values = (
            classification.to_columns()
            if classification is not None
            else {
                "failure_category": "",
                "failure_source": "",
                "failure_note": "",
            }
        )
        self._persist_updates({index: values})

    def set_training_selection(
        self,
        candidate_indices: set[int],
        selected_indices: set[int],
    ) -> None:
        """Persist which reviewed cases belong to the next training batch."""
        if not selected_indices.issubset(candidate_indices):
            raise ValueError("Selected training rows must belong to the candidate set")
        if any(index < 0 or index >= len(self.rows) for index in candidate_indices):
            raise IndexError("Training candidate row index out of range")
        updates: dict[int, dict[str, str]] = {}
        for index in candidate_indices:
            selected = index in selected_indices
            values = {"training_selected": "1" if selected else "0"}
            if selected and str(self.rows[index].get("review_label") or "").strip():
                values["review_selected"] = "1"
            updates[index] = values
        self._persist_updates(updates)

    def mark_submitted_indices(self, indices: set[int]) -> None:
        """Attach non-persisted submission context used by transition validation."""
        if any(index < 0 or index >= len(self.rows) for index in indices):
            raise IndexError("Submitted review row index out of range")
        self._submitted_identities.update(
            self._row_identity(self.rows[index], index) for index in indices
        )

    def first_pending_index(self) -> int:
        """Return the first unreviewed row, or zero when all rows are reviewed."""
        for index, row in enumerate(self.rows):
            if not str(row.get("review_label") or "").strip():
                return index
        return 0

    def reviewed_count(self) -> int:
        """Return how many rows already contain an operator decision."""
        return sum(bool(str(row.get("review_label") or "").strip()) for row in self.rows)

    def _persist_updates(
        self,
        updates: dict[int, dict[str, str]],
        *,
        revision_reason: str = "",
    ) -> None:
        """Merge row-scoped updates under a lock so parallel windows do not clobber."""
        if not updates:
            return
        identities = {
            index: self._row_identity(self.rows[index], index) for index in updates
        }
        review_fields = {
            "review_label",
            "review_outcome",
            "failure_category",
            "failure_note",
            "skip_reason",
        }
        append_event_by_identity = {
            identities[index]: bool(review_fields & set(values))
            for index, values in updates.items()
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        normalized_revision_reason = revision_reason.strip()
        rows_to_sync: list[tuple[dict[str, str], bool]] = []
        submitted_revisions: list[tuple[str, tuple[str, ...]]] = []
        with _review_manifest_lock(self.path):
            latest_rows = self._load_rows()
            if not latest_rows:
                latest_rows = [dict(row) for row in self.rows]
            latest_by_identity = {
                self._row_identity(row, index): index
                for index, row in enumerate(latest_rows)
            }
            for source_index, values in updates.items():
                target_index = latest_by_identity.get(identities[source_index])
                if target_index is None:
                    raise ValueError("Review manifest changed; reopen the review window")
                before = dict(latest_rows[target_index])
                after = {**before, **values}
                identity = identities[source_index]
                before_for_validation = dict(before)
                after_for_validation = dict(after)
                if identity in self._submitted_identities:
                    before_for_validation["submission_status"] = "submitted"
                    after_for_validation["submission_status"] = "submitted"
                if normalized_revision_reason:
                    after_for_validation["revision_reason"] = (
                        normalized_revision_reason
                    )
                try:
                    before_codes = {
                        violation.code
                        for violation in blocking_violations(
                            validate_record_consistency(before_for_validation)
                        )
                    }
                    after_violations = blocking_violations(
                        validate_record_consistency(after_for_validation)
                    )
                    changed_review_contract = bool(
                        REVIEW_CORE_FIELDS & set(values)
                    )
                    rejected_violations = (
                        after_violations
                        if changed_review_contract
                        else tuple(
                            violation
                            for violation in after_violations
                            if violation.code not in before_codes
                        )
                    )
                    if rejected_violations:
                        raise ReviewWorkflowValidationError(
                            after_for_validation,
                            rejected_violations,
                        )
                    ensure_transition_valid(
                        before_for_validation,
                        after_for_validation,
                    )
                except ReviewWorkflowValidationError as exc:
                    logger.error(
                        "Review workflow validation rejected sample=%s fields=%s "
                        "violations=%s",
                        record_identity(after_for_validation),
                        {
                            field: after_for_validation.get(field, "")
                            for field in (
                                "review_selected",
                                "review_outcome",
                                "review_label",
                                "failure_category",
                                "skip_reason",
                                "product_verdict",
                                "detection_verdict",
                                "color_verdict",
                                "action_route",
                                "training_selected",
                            )
                        },
                        [violation.code for violation in exc.violations],
                    )
                    raise
                changed_review_fields = tuple(
                    sorted(
                        field
                        for field in REVIEW_CORE_FIELDS
                        if str(before.get(field) or "").strip()
                        != str(after.get(field) or "").strip()
                    )
                )
                if (
                    identity in self._submitted_identities
                    and changed_review_fields
                    and normalized_revision_reason
                ):
                    submitted_revisions.append((identity, changed_review_fields))
                latest_rows[target_index] = after
            self._ensure_contract_columns_for(latest_rows)
            self._write_rows_atomic(latest_rows)
            self.rows = latest_rows
            rows_to_sync = [
                (
                    dict(self.rows[latest_by_identity[identity]]),
                    append_event_by_identity[identity],
                )
                for identity in identities.values()
            ]
        if self.repository is not None:
            for row, append_event in rows_to_sync:
                self.repository.sync_review_row(row, append_event=append_event)
        for identity, changed_fields in submitted_revisions:
            logger.info(
                "Submitted review revision saved sample_id=%s "
                "changed_fields=%s revision_reason=%s",
                identity,
                list(changed_fields),
                normalized_revision_reason,
            )

    def _ensure_contract_columns_for(self, rows: list[dict[str, str]]) -> None:
        for row in rows:
            row.setdefault("product_verdict", "")
            row.setdefault("detection_verdict", "")
            row.setdefault("color_verdict", "")
            row.setdefault("action_route", "")
            row.setdefault("failure_category", "")
            row.setdefault("failure_source", "")
            row.setdefault("failure_note", "")
            row.setdefault("review_outcome", "")
            row.setdefault("skip_reason", "")
            row.setdefault("review_selected", "0")
            row.setdefault("training_selected", "1")

    def _sync_existing_rows(self) -> None:
        """Incrementally backfill snapshots and changed reviews into SQLite."""
        if self.repository is None:
            return
        indexed = self.repository.load_manifest_sync_state()
        for row in self.rows:
            snapshot_path = Path(str(row.get("config_snapshot_path") or ""))
            if not snapshot_path.is_file():
                continue
            resolved_path = str(snapshot_path.resolve())
            stored_review = indexed.get(resolved_path)
            if stored_review is None:
                self.repository.upsert_snapshot_file(snapshot_path)
            if (
                str(row.get("review_label") or "").strip()
                and stored_review != self._database_review_state(row)
            ):
                self.repository.sync_review_row(row, append_event=False)

    @staticmethod
    def _database_review_state(row: dict[str, str]) -> dict[str, str]:
        """Normalize one manifest row to the persisted SQLite review contract."""
        return {
            "review_outcome": str(row.get("review_outcome") or ""),
            "review_label": str(row.get("review_label") or ""),
            "failure_category": str(row.get("failure_category") or ""),
            "failure_source": str(row.get("failure_source") or ""),
            "failure_note": str(
                row.get("failure_note") or row.get("review_note") or ""
            ),
            "skip_reason": str(row.get("skip_reason") or ""),
            "action_route": str(row.get("action_route") or ""),
            "training_selected": (
                "1" if str(row.get("training_selected") or "0") == "1" else "0"
            ),
        }

    def _diagnose_existing_rows(self) -> None:
        """Warn about legacy contradictions without rewriting or rejecting reads."""
        blocking_identities: list[str] = []
        for index, row in enumerate(self.rows):
            violations = validate_record_consistency(row)
            if not violations:
                continue
            identity = self._row_identity(row, index)
            self.workflow_diagnostics[identity] = violations
            if blocking_violations(violations):
                blocking_identities.append(identity)
        if self.workflow_diagnostics:
            logger.warning(
                "Legacy review workflow diagnostics records=%s blocking=%s "
                "blocking_samples=%s; source data was not modified",
                len(self.workflow_diagnostics),
                len(blocking_identities),
                blocking_identities[:10],
            )

    @staticmethod
    def _row_identity(row: dict[str, str], index: int) -> str:
        return str(
            row.get("config_snapshot_path")
            or row.get("original_path")
            or f"manifest-row:{index}"
        )

    def _write_rows_atomic(self, rows: list[dict[str, str]]) -> None:
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        fieldnames = list(rows[0])
        for row in rows[1:]:
            fieldnames.extend(field for field in row if field not in fieldnames)
        try:
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def _review_manifest_lock(path: Path, timeout_seconds: float = 5.0):
    """Serialize row-scoped manifest updates across review windows."""
    lock_path = path.with_name(f".{path.name}.lock")
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Review selection is being updated by another window"
                ) from None
            time.sleep(0.05)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def prepare_review_manifest(
    *,
    result_root: str | Path,
    manifest_path: str | Path,
    product: str | None,
    area: str | None,
) -> tuple[Path, list[dict[str, str]]]:
    """Build and index one target manifest without touching any QWidget."""
    result_path = Path(result_root)
    target_manifest = _target_manifest_path(
        Path(manifest_path), product=product, area=area
    )
    cases = _with_pass_sampling(
        collect_review_cases(
            result_path,
            include_pass=True,
            product=product,
            area=area,
        )
    )
    write_manifest(cases, target_manifest)
    store = ReviewManifestStore(
        target_manifest,
        database_path=result_path / "inspection_records.sqlite3",
    )
    return target_manifest, [dict(row) for row in store.rows]


class ReviewCasesDialog(QDialog):
    """Show saved inference images and guide an operator through each decision."""

    back_to_inspection_requested = pyqtSignal()

    def __init__(
        self,
        *,
        result_root: str | Path,
        manifest_path: str | Path,
        training_data_dir: str | Path,
        language: str = "zh_TW",
        product: str | None = None,
        area: str | None = None,
        start_in_overview: bool = False,
        use_legacy_selected_page: bool = False,
        use_legacy_review_layout: bool = False,
        embedded: bool = False,
        manifest_prepared: bool = False,
        prepared_rows: list[dict[str, str]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.result_root = Path(result_root)
        self.manifest_path = _target_manifest_path(Path(manifest_path), product=product, area=area)
        self.training_data_dir = Path(training_data_dir)
        self.language = language
        self.product = product
        self.area = area
        self.current_index = 0
        self.visible_indices: list[int] = []
        self._filtered_indices: list[int] = []
        self._review_scope_indices: set[int] = set()
        self._dismissed_review_identities: set[str] = set()
        self._handed_off_indices: set[int] = set()
        self._start_in_overview = start_in_overview
        self._use_legacy_selected_page = use_legacy_selected_page
        self._use_legacy_review_layout = use_legacy_review_layout
        self._embedded = embedded
        self._submission_active = False
        self._retry_review_action: Any | None = None
        self._review_thumbnail_visible_indices: tuple[int, ...] = ()
        self._filter_state_path = self.manifest_path.with_name(f".{self.manifest_path.stem}_filter.json")

        if not manifest_prepared:
            _target_path, prepared_rows = prepare_review_manifest(
                result_root=self.result_root,
                manifest_path=manifest_path,
                product=product,
                area=area,
            )
        self.store = ReviewManifestStore(
            self.manifest_path,
            database_path=self.result_root / "inspection_records.sqlite3",
            synchronize_repository=False,
            initial_rows=prepared_rows,
            diagnose_rows=prepared_rows is None,
        )
        self._reconcile_handed_off_selection()
        self._restore_review_scope_from_persistence()
        self.current_index = self.store.first_pending_index()
        self.visible_indices = list(range(len(self.store.rows)))
        self.image_service = AsyncImageService.from_environment(parent=self)

        self.setWindowTitle(self._text("產線模型補訓｜資料複核", "Production Retraining | Data Review"))
        if embedded:
            self.setWindowFlags(Qt.Widget)
        else:
            configure_responsive_dialog(
                self,
                preferred=(1250, 820),
                minimum=(760, 520),
                parent=parent,
            )
        self._build_ui()
        self._restore_time_filter()
        self._apply_time_filter()

    def done(self, result: int) -> None:
        """Invalidate image requests before QDialog hides or releases child widgets."""
        if self._embedded:
            self.back_to_inspection_requested.emit()
            return
        self.image_service.shutdown()
        super().done(result)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._embedded:
            event.ignore()
            self.back_to_inspection_requested.emit()
            return
        self.image_service.shutdown()
        super().closeEvent(event)

    def shutdown_workspace(self) -> None:
        """Release asynchronous resources when the host replaces the workspace."""
        self.image_service.shutdown()

    def refresh_workspace(self) -> None:
        """Refresh visible state when returning from another application page."""
        if self.workflow_stack.currentWidget() in {
            self.review_selection_page,
            self.selected_review_page,
            self.classification_page,
        }:
            self._apply_time_filter()
        current = self.workflow_stack.currentWidget()
        if hasattr(current, "refresh_jobs"):
            current.refresh_jobs()

    def _text(self, zh: str, en: str) -> str:
        return zh if str(self.language).lower().startswith("zh") else en

    def _reconcile_handed_off_selection(self) -> None:
        """Exclude legacy rows that already exist in ready or pending data."""
        handed_off_artifacts = _load_handed_off_artifacts(
            self.training_data_dir,
            self.store.rows,
        )
        if not handed_off_artifacts:
            self._handed_off_indices = set()
            return
        handed_off_indices = {
            index
            for index, row in enumerate(self.store.rows)
            if any(
                _artifact_identity(row.get(field, "")) in handed_off_artifacts
                for field in (
                    "config_snapshot_path",
                    "original_path",
                    "preprocessed_path",
                    "annotated_path",
                )
            )
        }
        self._handed_off_indices = handed_off_indices
        if not handed_off_indices:
            return
        self.store.mark_submitted_indices(handed_off_indices)
        changed_indices = {
            index
            for index in handed_off_indices
            if str(self.store.rows[index].get("training_selected") or "0") != "0"
        }
        if not changed_indices:
            return
        try:
            self.store.set_training_selection(changed_indices, set())
        except (OSError, sqlite3.Error, ValueError, IndexError):
            return

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        if self._embedded:
            navigation = QHBoxLayout()
            back_button = QPushButton(
                self._text("← 返回檢測主畫面", "← Back to inspection")
            )
            back_button.setObjectName("secondaryAction")
            back_button.clicked.connect(self.back_to_inspection_requested.emit)
            navigation.addWidget(back_button)
            navigation.addWidget(
                QLabel(
                    self._text(
                        "補訓在背景執行；離開此頁不會中斷工作。",
                        "Retraining continues in the background when you leave this page.",
                    )
                )
            )
            navigation.addStretch()
            root_layout.addLayout(navigation)
        self.workflow_stack = QStackedWidget()
        root_layout.addWidget(self.workflow_stack)
        self._build_review_selection_page()
        self._build_selected_review_page()
        self._build_classification_page()
        self.workflow_stack.setCurrentWidget(
            (
                self.selected_review_page
                if self._use_legacy_selected_page and self._review_scope_indices
                else self.review_selection_page
            )
            if self._start_in_overview
            else self.classification_page
        )

    def _build_review_selection_page(self) -> None:
        """Build stage one: choose failure images before assigning verdicts."""
        self.review_selection_page = QWidget()
        layout = QVBoxLayout(self.review_selection_page)
        instruction = QLabel(
            self._text(
                "模型補訓第 1 階段：先一次瀏覽失敗圖，勾選真正需要人工審核的照片；雙擊可放大。",
                "Retraining stage 1: scan failures in bulk and select only images that need human review; double-click to enlarge.",
            )
        )
        instruction.setWordWrap(True)
        instruction.setStyleSheet(
            "QLabel { background:#243447;color:white;padding:12px;"
            "font-size:13pt;font-weight:bold; }"
        )
        layout.addWidget(instruction)
        saved_hint = QLabel(
            self._text(
                "這一階段只累積失敗圖片；勾選會立即保存。你可以先關閉，等樣本足夠後再進入分類與補訓。",
                "This stage only collects failed images. Selections are saved immediately; close now and classify later when enough samples have accumulated.",
            )
        )
        saved_hint.setWordWrap(True)
        saved_hint.setStyleSheet(
            "QLabel { background:#eef6ff;color:#174a7e;padding:9px;"
            "border:1px solid #a9c8e8;border-radius:5px; }"
        )
        layout.addWidget(saved_hint)
        layout.addLayout(self._build_time_filter())
        self.review_gallery = ReviewSelectionGallery(
            language=self.language,
            image_service=self.image_service,
        )
        self.review_gallery.selection_changed.connect(self._on_review_selection_changed)
        layout.addWidget(self.review_gallery, 1)

        footer = QGridLayout()
        self.review_selection_summary = QLabel()
        self.review_selection_summary.setStyleSheet(
            "font-size:11pt;font-weight:bold;color:#344054;"
        )
        footer.addWidget(self.review_selection_summary, 0, 0, 1, 2)
        self.review_selection_saved_label = QLabel()
        self.review_selection_saved_label.setStyleSheet("color:#237a3b;")
        footer.addWidget(self.review_selection_saved_label, 0, 2, 1, 2)
        close_button = QPushButton(self._text("關閉", "Close"))
        close_button.clicked.connect(self.reject)
        self.save_review_selection_button = QPushButton(
            self._text(
                "儲存並關閉（稍後補訓）",
                "Save and close (retrain later)",
            )
        )
        self.save_review_selection_button.setMinimumHeight(44)
        self.save_review_selection_button.clicked.connect(self.accept)
        self.start_review_button = QPushButton(
            self._text("開始複核", "Start review")
        )
        self.start_review_button.setMinimumHeight(44)
        self.start_review_button.setStyleSheet(
            "QPushButton { background:#237a3b;color:white;font-weight:bold;padding:8px 18px; }"
            "QPushButton:disabled { background:#9aa0a6; }"
        )
        self.start_review_button.clicked.connect(
            self._show_selected_review_page
            if self._use_legacy_selected_page
            else self._start_selected_review
        )
        footer.addWidget(self.save_review_selection_button, 1, 0, 1, 2)
        footer.addWidget(close_button, 1, 2)
        footer.addWidget(self.start_review_button, 1, 3)
        footer.setColumnStretch(1, 1)
        layout.addLayout(footer)
        self.workflow_stack.addWidget(self.review_selection_page)

    def _build_selected_review_page(self) -> None:
        """Build a target-wide gallery containing selected images only."""
        self.selected_review_page = QWidget()
        layout = QVBoxLayout(self.selected_review_page)
        instruction = QLabel(
            self._text(
                "模型補訓第 2 階段：此畫面只顯示已選擇並保存的圖片，不混入未選圖片。取消勾選會立即移出此清單。",
                "Retraining stage 2: this page contains saved selections only. Unselected images are never mixed in; unchecking removes an image immediately.",
            )
        )
        instruction.setWordWrap(True)
        instruction.setStyleSheet(
            "QLabel { background:#174a7e;color:white;padding:12px;"
            "font-size:13pt;font-weight:bold; }"
        )
        layout.addWidget(instruction)
        self.selected_review_gallery = ReviewSelectionGallery(
            language=self.language,
            selected_only=True,
            image_service=self.image_service,
        )
        self.selected_review_gallery.selection_changed.connect(
            self._on_selected_review_selection_changed
        )
        layout.addWidget(self.selected_review_gallery, 1)

        footer = QGridLayout()
        self.selected_review_summary = QLabel()
        self.selected_review_summary.setStyleSheet(
            "font-size:11pt;font-weight:bold;color:#344054;"
        )
        footer.addWidget(self.selected_review_summary, 0, 0, 1, 3)
        choose_more_button = QPushButton(
            self._text("返回候選圖片，繼續選擇", "Back to candidates")
        )
        choose_more_button.clicked.connect(self._show_candidate_review_page)
        save_and_close_button = QPushButton(
            self._text("保存並關閉（稍後決定）", "Save and close (decide later)")
        )
        save_and_close_button.clicked.connect(self.accept)
        self.classify_selected_button = QPushButton(
            self._text("開始逐張分類", "Classify selected images")
        )
        self.classify_selected_button.setMinimumHeight(44)
        self.classify_selected_button.setStyleSheet(
            "QPushButton { background:#237a3b;color:white;font-weight:bold;padding:8px 18px; }"
            "QPushButton:disabled { background:#9aa0a6; }"
        )
        self.classify_selected_button.clicked.connect(self._start_selected_review)
        footer.addWidget(choose_more_button, 1, 0)
        footer.addWidget(save_and_close_button, 1, 1)
        footer.addWidget(self.classify_selected_button, 1, 2)
        layout.addLayout(footer)
        self.workflow_stack.addWidget(self.selected_review_page)

    def _build_classification_page(self) -> None:
        """Build the active per-image layout behind an independent rollback flag."""
        if self._use_legacy_review_layout:
            self._build_legacy_classification_page()
        else:
            self._build_review_workspace_page()
        self._install_review_shortcuts()

    def _build_review_workspace_page(self) -> None:
        """Build one three-column workspace without duplicating save behavior."""
        self.classification_page = QWidget()
        layout = QVBoxLayout(self.classification_page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("font-size:11pt;font-weight:bold;color:#243447;")
        layout.addWidget(self.progress_label)

        splitter = QSplitter(Qt.Horizontal)
        self.review_thumbnail_panel = ReviewThumbnailPanel(
            language=self.language,
            image_service=self.image_service,
        )
        self.review_thumbnail_panel.setMinimumWidth(140)
        self.review_thumbnail_panel.current_changed.connect(
            self._select_review_thumbnail
        )
        splitter.addWidget(self.review_thumbnail_panel)

        self.image_viewer = ReviewImageViewer(
            language=self.language,
            image_service=self.image_service,
        )
        splitter.addWidget(self.image_viewer)

        right_scroll = QScrollArea()
        self.review_detail_scroll = right_scroll
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_panel = QWidget()
        right_panel.setMinimumWidth(0)
        right_panel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 4, 8, 4)
        self.question_label = QLabel()
        self.question_label.setWordWrap(True)
        self.question_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.question_label.setStyleSheet("font-size:12pt;font-weight:bold;")
        self.details_label = QLabel()
        self.details_label.setWordWrap(True)
        self.details_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        right_layout.addWidget(self.question_label)

        self.review_details_toggle = QPushButton(
            self._text("詳細資訊 ▼", "Details ▼")
        )
        self.review_details_toggle.setObjectName("ReviewDetailsToggle")
        self.review_details_toggle.setCheckable(True)
        self.review_details_toggle.setStyleSheet(
            "QPushButton { text-align:left;background:#eef2f6;color:#344054;"
            "border:1px solid #c8d1dc;border-radius:5px;padding:7px 9px; }"
        )
        self.review_details_toggle.toggled.connect(self._toggle_review_details)
        self.review_details_panel = QFrame()
        self.review_details_panel.setObjectName("ReviewDetailsPanel")
        self.review_details_panel.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )
        details_layout = QVBoxLayout(self.review_details_panel)
        details_layout.setContentsMargins(4, 2, 4, 4)
        details_layout.addWidget(self.details_label)

        self.ai_summary_label = QLabel()
        self.metadata_summary_label = QLabel()
        self.domain_summary_label = QLabel()
        for title, target in (
            (self._text("AI 與標註摘要", "AI and annotation"), self.ai_summary_label),
            (self._text("設備資訊", "Equipment"), self.metadata_summary_label),
            (self._text("目前覆核語意", "Current review semantics"), self.domain_summary_label),
        ):
            title_label = QLabel(title)
            title_label.setStyleSheet("font-weight:bold;color:#344054;margin-top:6px;")
            target.setWordWrap(True)
            target.setTextInteractionFlags(Qt.TextSelectableByMouse)
            target.setMinimumWidth(0)
            target.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            details_layout.addWidget(title_label)
            details_layout.addWidget(target)
        self.review_details_panel.setVisible(False)

        self.triage_panel = QFrame()
        triage_layout = QVBoxLayout(self.triage_panel)
        triage_layout.setContentsMargins(0, 6, 0, 0)
        outcome_row = QGridLayout()
        outcome_row.setHorizontalSpacing(7)
        outcome_row.setVerticalSpacing(7)
        self.pass_review_button = QPushButton(self._text("PASS", "PASS"))
        self.fail_review_button = QPushButton(self._text("FAIL", "FAIL"))
        self.skip_review_button = QPushButton(self._text("略過", "SKIP"))
        self.pass_review_button.setToolTip(
            self._text("產品可接受", "Product is acceptable")
        )
        self.fail_review_button.setToolTip(
            self._text(
                "產品或 AI 結果需修正",
                "Product or AI result requires correction",
            )
        )
        self.skip_review_button.setToolTip(
            self._text("不納入本次處理", "Exclude from this review run")
        )
        for position, (button, color) in enumerate((
            (self.pass_review_button, "#237a3b"),
            (self.fail_review_button, "#b42318"),
            (self.skip_review_button, "#5f6368"),
        )):
            button.setMinimumHeight(44)
            button.setStyleSheet(
                f"QPushButton {{ background:{color};color:white;font-weight:bold;"
                "border:0;border-radius:5px;padding:8px; }"
            )
            if position < 2:
                outcome_row.addWidget(button, 0, position)
            else:
                outcome_row.addWidget(button, 1, 0, 1, 2)
        self.pass_review_button.clicked.connect(self._save_pass_triage)
        self.fail_review_button.clicked.connect(
            lambda: self._show_triage_details("fail")
        )
        self.skip_review_button.clicked.connect(
            lambda: self._show_triage_details("skip")
        )
        triage_layout.addLayout(outcome_row)

        self.fail_details_panel = QFrame()
        fail_layout = QVBoxLayout(self.fail_details_panel)
        fail_layout.setContentsMargins(0, 4, 0, 0)
        self.fail_reason_combo = QComboBox()
        for value, zh_label, en_label in FAILURE_REASON_OPTIONS:
            self.fail_reason_combo.addItem(self._text(zh_label, en_label), value)
        self.custom_failure_note = QLineEdit()
        self.custom_failure_note.setMaxLength(500)
        self.custom_failure_note.setPlaceholderText(
            self._text(
                "可輸入現場原因；選『其他』時必填",
                "On-site reason; required for Other",
            )
        )
        self.proposed_semantics_label = QLabel()
        self.proposed_semantics_label.setWordWrap(True)
        self.proposed_semantics_label.setStyleSheet(
            "background:#eef6ff;color:#174a7e;padding:6px;"
        )
        self.save_fail_button = QPushButton(self._text("儲存 FAIL", "Save FAIL"))
        self.save_fail_button.setMinimumHeight(38)
        self.save_fail_button.clicked.connect(self._save_fail_triage)
        self.fail_reason_combo.currentIndexChanged.connect(
            self._update_fail_action_label
        )
        self.custom_failure_note.textChanged.connect(self._update_fail_action_label)
        fail_layout.addWidget(self.fail_reason_combo)
        fail_layout.addWidget(self.custom_failure_note)
        fail_layout.addWidget(self.proposed_semantics_label)
        fail_layout.addWidget(self.save_fail_button)
        triage_layout.addWidget(self.fail_details_panel)

        self.skip_details_panel = QFrame()
        skip_layout = QVBoxLayout(self.skip_details_panel)
        skip_layout.setContentsMargins(0, 4, 0, 0)
        self.skip_reason_combo = QComboBox()
        for value, zh_label, en_label in SKIP_DISPLAY_OPTIONS:
            self.skip_reason_combo.addItem(self._text(zh_label, en_label), value)
        self.custom_skip_note = QLineEdit()
        self.custom_skip_note.setMaxLength(500)
        self.custom_skip_note.setPlaceholderText(
            self._text("其他原因請在此填寫", "Enter the other reason here")
        )
        self.save_skip_button = QPushButton(
            self._text("確認略過", "Confirm skip")
        )
        self.save_skip_button.setMinimumHeight(38)
        self.save_skip_button.clicked.connect(self._save_skip_triage)
        skip_layout.addWidget(self.skip_reason_combo)
        skip_layout.addWidget(self.custom_skip_note)
        skip_layout.addWidget(self.save_skip_button)
        triage_layout.addWidget(self.skip_details_panel)
        right_layout.addWidget(self.triage_panel)

        self.feedback_label = QLabel()
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setMinimumHeight(32)
        right_layout.addWidget(self.feedback_label)
        self.retry_save_button = QPushButton(self._text("重試保存", "Retry save"))
        self.retry_save_button.clicked.connect(self._retry_last_review_save)
        self.retry_save_button.setVisible(False)
        right_layout.addWidget(self.retry_save_button)

        self.reprocess_button = QPushButton()
        self.reprocess_button.clicked.connect(self._requeue_handed_off_scope)
        self.reprocess_button.setVisible(False)
        right_layout.addWidget(self.reprocess_button)

        self.batch_preview_button = QPushButton(self._text("待送清單", "Pending queue"))
        self.submission_history_button = QPushButton(
            self._text("已送出紀錄", "Submitted history")
        )
        self.progress_button = QPushButton(self._text("補訓進度", "Training progress"))
        self.batch_preview_button.clicked.connect(self._open_selected_training_queue)
        self.submission_history_button.clicked.connect(self._open_submission_history)
        self.progress_button.clicked.connect(self._open_update_progress)
        for button in (
            self.batch_preview_button,
            self.submission_history_button,
            self.progress_button,
        ):
            right_layout.addWidget(button)
        right_layout.addWidget(self.review_details_toggle)
        right_layout.addWidget(self.review_details_panel)
        right_layout.addStretch()
        right_scroll.setWidget(right_panel)
        right_scroll.setMinimumWidth(240)
        splitter.addWidget(right_scroll)
        splitter.setSizes([220, 650, 330])
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        navigation_layout = QGridLayout()
        self.overview_button = QPushButton(
            self._text("返回失敗圖總覽", "Back to failure overview")
        )
        self.previous_button = QPushButton(self._text("上一張", "Previous"))
        self.next_button = QPushButton(self._text("下一張", "Next"))
        self.navigation_progress_label = QLabel()
        self.export_button = QPushButton(
            self._text("完成並檢查待送清單", "Finish and review queue")
        )
        close_button = QPushButton(self._text("關閉", "Close"))
        self.overview_button.clicked.connect(self._show_review_overview)
        self.previous_button.clicked.connect(lambda: self._move(-1))
        self.next_button.clicked.connect(lambda: self._move(1))
        self.export_button.clicked.connect(self._open_selected_training_queue)
        close_button.clicked.connect(self.accept)
        navigation_layout.addWidget(self.overview_button, 0, 0)
        navigation_layout.addWidget(self.previous_button, 0, 1)
        navigation_layout.addWidget(self.navigation_progress_label, 0, 2)
        navigation_layout.addWidget(self.next_button, 0, 3)
        navigation_layout.addWidget(self.export_button, 1, 0, 1, 3)
        navigation_layout.addWidget(close_button, 1, 3)
        navigation_layout.setColumnStretch(2, 1)
        layout.addLayout(navigation_layout)

        self.failure_category_buttons = {}
        self.clear_failure_category_button = QPushButton()
        self.clear_failure_category_button.setVisible(False)
        self.failure_classification_label = QLabel()
        self.legacy_failure_category_panel = QWidget()
        self.legacy_failure_category_panel.setVisible(False)
        self.review_buttons = {}
        self.review_layout = QGridLayout()
        self.legacy_review_panel = QWidget()
        self.legacy_review_panel.setVisible(False)
        self.original_label = QLabel()
        self.annotated_label = QLabel()
        self.original_label.setVisible(False)
        self.annotated_label.setVisible(False)
        self._show_triage_details("")
        self._update_fail_action_label()
        self.workflow_stack.addWidget(self.classification_page)

    def _toggle_review_details(self, expanded: bool) -> None:
        """Show verbose evidence without pushing primary review actions away."""
        self.review_details_panel.setVisible(expanded)
        self.review_details_toggle.setText(
            self._text("詳細資訊 ▲", "Details ▲")
            if expanded
            else self._text("詳細資訊 ▼", "Details ▼")
        )

    def _build_legacy_classification_page(self) -> None:
        """Build stage three: classify only the cases selected in the overview."""
        self.classification_page = QWidget()
        layout = QVBoxLayout(self.classification_page)
        instruction = QLabel(
            self._text(
                "模型補訓第 3 階段：人工只需選 Pass、Fail 或略過；Fail 再選原因並可輸入現場說明。",
                "Retraining stage 3: choose Pass, Fail, or Skip; failed reviews also require a reason and may include an on-site note.",
            )
        )
        instruction.setWordWrap(True)
        instruction.setStyleSheet(
            "QLabel { background: #243447; color: white; padding: 10px; font-size: 11pt; font-weight: bold; }"
        )
        layout.addWidget(instruction)
        self.progress_label = QLabel()
        self.question_label = QLabel()
        self.question_label.setStyleSheet("font-size: 13pt; font-weight: bold;")
        self.details_label = QLabel()
        self.details_label.setWordWrap(True)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.question_label)
        layout.addWidget(self.details_label)

        failure_category_layout = QHBoxLayout()
        failure_category_title = QLabel(
            self._text("失敗原因分類（不影響送訓路由）：", "Failure cause (does not change routing):")
        )
        failure_category_title.setStyleSheet("font-weight:bold;color:#344054;")
        failure_category_layout.addWidget(failure_category_title)
        self.failure_category_buttons: dict[str, QPushButton] = {}
        for category, zh_label, en_label in FAILURE_CATEGORY_ACTIONS:
            button = QPushButton(self._text(zh_label, en_label))
            button.setCheckable(True)
            button.clicked.connect(
                lambda _checked=False, selected=category: self._set_failure_category(
                    selected
                )
            )
            self.failure_category_buttons[category] = button
            failure_category_layout.addWidget(button)
        self.failure_classification_label = QLabel()
        self.failure_classification_label.setStyleSheet("color:#174a7e;font-weight:bold;")
        failure_category_layout.addWidget(self.failure_classification_label)
        self.clear_failure_category_button = QPushButton(
            self._text("清除失敗原因", "Clear failure cause")
        )
        self.clear_failure_category_button.clicked.connect(
            self._clear_failure_category
        )
        failure_category_layout.addWidget(self.clear_failure_category_button)
        failure_category_layout.addStretch()
        self.legacy_failure_category_panel = QWidget()
        self.legacy_failure_category_panel.setLayout(failure_category_layout)
        self.legacy_failure_category_panel.setVisible(False)
        layout.addWidget(self.legacy_failure_category_panel)

        image_layout = QHBoxLayout()
        self.original_label = self._image_panel(self._text("原圖", "Original"))
        self.annotated_label = self._image_panel(self._text("推理結果", "Inference"))
        image_layout.addWidget(self.original_label)
        image_layout.addWidget(self.annotated_label)
        layout.addLayout(image_layout, 1)

        self.triage_panel = QFrame()
        self.triage_panel.setObjectName("TriagePanel")
        self.triage_panel.setStyleSheet(
            "QFrame#TriagePanel { background:#f8fafc; border:1px solid #d8dee6; "
            "border-radius:8px; }"
        )
        triage_layout = QVBoxLayout(self.triage_panel)
        triage_layout.setContentsMargins(12, 10, 12, 10)
        triage_title = QLabel(
            self._text(
                "人工檢核：只選 Pass、Fail 或略過",
                "Manual review: choose Pass, Fail, or Skip",
            )
        )
        triage_title.setStyleSheet("font-size:11pt;font-weight:bold;color:#344054;")
        triage_layout.addWidget(triage_title)

        outcome_row = QHBoxLayout()
        self.pass_review_button = QPushButton(self._text("PASS", "PASS"))
        self.fail_review_button = QPushButton(self._text("FAIL", "FAIL"))
        self.skip_review_button = QPushButton(self._text("略過", "SKIP"))
        for button, color in (
            (self.pass_review_button, "#237a3b"),
            (self.fail_review_button, "#b42318"),
            (self.skip_review_button, "#5f6368"),
        ):
            button.setMinimumHeight(44)
            button.setStyleSheet(
                f"QPushButton {{ background:{color};color:white;font-weight:bold;"
                "border:0;border-radius:5px;padding:8px 18px; }"
            )
            outcome_row.addWidget(button, 1)
        self.pass_review_button.clicked.connect(self._save_pass_triage)
        self.fail_review_button.clicked.connect(
            lambda: self._show_triage_details("fail")
        )
        self.skip_review_button.clicked.connect(
            lambda: self._show_triage_details("skip")
        )
        triage_layout.addLayout(outcome_row)

        self.fail_details_panel = QFrame()
        fail_details = QHBoxLayout(self.fail_details_panel)
        fail_details.setContentsMargins(0, 6, 0, 0)
        fail_details.addWidget(QLabel(self._text("失敗原因", "Failure reason")))
        self.fail_reason_combo = QComboBox()
        for value, zh_label, en_label in FAILURE_REASON_OPTIONS:
            self.fail_reason_combo.addItem(self._text(zh_label, en_label), value)
        self.fail_reason_combo.setMinimumWidth(210)
        fail_details.addWidget(self.fail_reason_combo)
        self.custom_failure_note = QLineEdit()
        self.custom_failure_note.setMaxLength(500)
        self.custom_failure_note.setPlaceholderText(
            self._text(
                "可直接輸入現場原因；選『其他』時必填",
                "Add an on-site note; required for Other",
            )
        )
        fail_details.addWidget(self.custom_failure_note, 1)
        self.save_fail_button = QPushButton(
            self._text("儲存並加入 Training Set", "Save and add to Training Set")
        )
        self.save_fail_button.setMinimumHeight(38)
        self.save_fail_button.clicked.connect(self._save_fail_triage)
        self.fail_reason_combo.currentIndexChanged.connect(
            self._update_fail_action_label
        )
        self.custom_failure_note.textChanged.connect(self._update_fail_action_label)
        fail_details.addWidget(self.save_fail_button)
        triage_layout.addWidget(self.fail_details_panel)

        self.skip_details_panel = QFrame()
        skip_details = QHBoxLayout(self.skip_details_panel)
        skip_details.setContentsMargins(0, 6, 0, 0)
        skip_details.addWidget(QLabel(self._text("略過原因", "Skip reason")))
        self.skip_reason_combo = QComboBox()
        for value, zh_label, en_label in SKIP_DISPLAY_OPTIONS:
            self.skip_reason_combo.addItem(self._text(zh_label, en_label), value)
        skip_details.addWidget(self.skip_reason_combo, 1)
        self.custom_skip_note = QLineEdit()
        self.custom_skip_note.setMaxLength(500)
        self.custom_skip_note.setPlaceholderText(
            self._text("其他原因請在此填寫", "Enter the other reason here")
        )
        skip_details.addWidget(self.custom_skip_note, 1)
        self.save_skip_button = QPushButton(
            self._text("確認略過（不加入回訓）", "Skip without retraining")
        )
        self.save_skip_button.setMinimumHeight(38)
        self.save_skip_button.clicked.connect(self._save_skip_triage)
        skip_details.addWidget(self.save_skip_button)
        triage_layout.addWidget(self.skip_details_panel)
        self._show_triage_details("")
        self._update_fail_action_label()
        layout.addWidget(self.triage_panel)

        self.review_layout = QGridLayout()
        self.review_buttons: dict[str, QPushButton] = {}
        for action_index, (value, zh_label, en_label) in enumerate(REVIEW_ACTIONS):
            zh_label, en_label = OPERATOR_ACTION_LABELS[value]
            button = QPushButton(self._text(zh_label, en_label))
            button.setMinimumHeight(46)
            button.setStyleSheet(
                f"QPushButton {{ background: {ACTION_COLORS[value]}; color: white; "
                "font-weight: bold; padding: 8px; } "
                "QPushButton:checked { border: 4px solid #ffd54f; }"
            )
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, selected=value: self._handle_review_action(selected))
            self.review_buttons[value] = button
            self.review_layout.addWidget(button, action_index // 3, action_index % 3)
        self.legacy_review_panel = QWidget()
        self.legacy_review_panel.setLayout(self.review_layout)
        self.legacy_review_panel.setVisible(False)
        layout.addWidget(self.legacy_review_panel)

        self.feedback_label = QLabel()
        self.feedback_label.setMinimumHeight(32)
        self.feedback_label.setAlignment(Qt.AlignCenter)
        self.feedback_label.setStyleSheet(
            "QLabel { color: #1b5e20; background: #e8f5e9; font-size: 11pt; font-weight: bold; padding: 5px; }"
        )
        layout.addWidget(self.feedback_label)
        self.retry_save_button = QPushButton(self._text("重試保存", "Retry save"))
        self.retry_save_button.clicked.connect(self._retry_last_review_save)
        self.retry_save_button.setVisible(False)
        layout.addWidget(self.retry_save_button)

        self.reprocess_button = QPushButton()
        self.reprocess_button.setMinimumHeight(40)
        self.reprocess_button.setStyleSheet(
            "QPushButton { background:#6f42c1;color:white;font-weight:bold;"
            "border:0;border-radius:5px;padding:8px 14px; }"
            "QPushButton:disabled { background:#9aa0a6; }"
        )
        self.reprocess_button.clicked.connect(self._requeue_handed_off_scope)
        self.reprocess_button.setVisible(False)
        layout.addWidget(self.reprocess_button)

        workspace_layout = QHBoxLayout()
        workspace_title = QLabel(self._text("資料工作台", "Data workspace"))
        workspace_title.setStyleSheet("font-weight:bold;color:#344054;padding-right:8px;")
        workspace_layout.addWidget(workspace_title)
        self.previous_button = QPushButton(self._text("上一張", "Previous"))
        self.previous_button.clicked.connect(lambda: self._move(-1))
        self.next_button = QPushButton(self._text("下一張", "Next"))
        self.next_button.clicked.connect(lambda: self._move(1))
        self.export_button = QPushButton(
            self._text(
                "完成本範圍並檢查待送清單",
                "Finish this range and review queue",
            )
        )
        self.export_button.setMinimumHeight(42)
        self.export_button.setStyleSheet(
            "QPushButton { background: #237a3b; color: white; font-weight: bold; "
            "padding: 8px 14px; } QPushButton:disabled { background: #9aa0a6; }"
        )
        self.export_button.clicked.connect(self._open_selected_training_queue)
        report_miss_button = QPushButton(
            self._text(
                "補登漏檢照片",
                "Add a missed detection",
            )
        )
        report_miss_button.setToolTip(
            self._text(
                "從已保存的推理結果補登漏檢案例，不接受來源不明的外部圖片。",
                "Add a missed case from a saved inference result; arbitrary external images are not accepted.",
            )
        )
        report_miss_button.clicked.connect(self._report_missed_image)
        self.batch_preview_button = QPushButton(self._text("待送清單", "Pending queue"))
        self.batch_preview_button.setMinimumHeight(42)
        self.batch_preview_button.setStyleSheet(
            "QPushButton { background: #2563a6; color: white; font-weight: bold; "
            "padding: 8px 14px; } QPushButton:disabled { background: #9aa0a6; }"
        )
        self.batch_preview_button.clicked.connect(self._open_selected_training_queue)
        self.submission_history_button = QPushButton(
            self._text("已送出紀錄", "Submitted history")
        )
        self.submission_history_button.setMinimumHeight(42)
        self.submission_history_button.setStyleSheet(
            "QPushButton { background: #495867; color: white; font-weight: bold; "
            "padding: 8px 14px; }"
        )
        self.submission_history_button.clicked.connect(self._open_submission_history)
        self.progress_button = QPushButton(self._text("補訓進度", "Training progress"))
        self.progress_button.setMinimumHeight(42)
        self.progress_button.setStyleSheet(
            "QPushButton { background: #00796b; color: white; font-weight: bold; "
            "padding: 8px 14px; }"
        )
        self.progress_button.clicked.connect(self._open_update_progress)
        close_button = QPushButton(self._text("關閉", "Close"))
        close_button.clicked.connect(self.accept)
        workspace_layout.addWidget(report_miss_button)
        workspace_layout.addWidget(self.batch_preview_button)
        workspace_layout.addWidget(self.submission_history_button)
        workspace_layout.addWidget(self.progress_button)
        workspace_layout.addStretch()
        layout.addLayout(workspace_layout)

        navigation_layout = QHBoxLayout()
        overview_button = QPushButton(self._text("返回失敗圖總覽", "Back to failure overview"))
        overview_button.clicked.connect(self._show_review_overview)
        navigation_layout.addWidget(overview_button)
        navigation_layout.addWidget(self.previous_button)
        navigation_layout.addWidget(self.next_button)
        navigation_layout.addStretch()
        navigation_layout.addWidget(self.export_button)
        navigation_layout.addWidget(close_button)
        layout.addLayout(navigation_layout)
        self.workflow_stack.addWidget(self.classification_page)

    def _build_time_filter(self) -> QHBoxLayout:
        """Build preset and custom inclusive timestamp controls."""
        layout = QHBoxLayout()
        layout.addWidget(QLabel(self._text("時間範圍", "Time range")))
        self.time_range_combo = QComboBox()
        self.time_range_combo.addItem(self._text("全部", "All"), "all")
        self.time_range_combo.addItem(self._text("本日", "Today"), "today")
        self.time_range_combo.addItem(self._text("最近 7 日", "Last 7 days"), "last_7_days")
        self.time_range_combo.addItem(self._text("最近 30 日", "Last 30 days"), "last_30_days")
        self.time_range_combo.addItem(self._text("自訂", "Custom"), "custom")
        self.start_time_edit = QDateTimeEdit(QDateTime.currentDateTime().addDays(-7))
        self.end_time_edit = QDateTimeEdit(QDateTime.currentDateTime())
        for editor in (self.start_time_edit, self.end_time_edit):
            editor.setCalendarPopup(True)
            editor.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.apply_time_button = QPushButton(self._text("套用", "Apply"))
        self.apply_time_button.clicked.connect(self._apply_time_filter)
        self.time_range_combo.currentIndexChanged.connect(self._on_time_preset_changed)
        layout.addWidget(self.time_range_combo)
        layout.addWidget(self.start_time_edit)
        self.time_separator_label = QLabel(self._text("至", "to"))
        layout.addWidget(self.time_separator_label)
        layout.addWidget(self.end_time_edit)
        layout.addWidget(self.apply_time_button)
        layout.addStretch()
        self._on_time_preset_changed()
        return layout

    def _on_time_preset_changed(self) -> None:
        """Update displayed bounds for the selected time preset."""
        mode = str(self.time_range_combo.currentData())
        now = datetime.now()
        starts = {
            "today": now.replace(hour=0, minute=0, second=0, microsecond=0),
            "last_7_days": now - timedelta(days=7),
            "last_30_days": now - timedelta(days=30),
        }
        if mode in starts:
            self.start_time_edit.setDateTime(QDateTime(starts[mode]))
            self.end_time_edit.setDateTime(QDateTime(now))
        custom = mode == "custom"
        self.start_time_edit.setEnabled(custom)
        self.end_time_edit.setEnabled(custom)
        self.start_time_edit.setVisible(custom)
        self.end_time_edit.setVisible(custom)
        self.time_separator_label.setVisible(custom)
        self.apply_time_button.setVisible(custom)
        if hasattr(self, "progress_label") and not custom:
            self._apply_time_filter()

    def _selected_time_bounds(self) -> tuple[datetime | None, datetime | None]:
        """Return inclusive local bounds selected by the operator."""
        if self.time_range_combo.currentData() == "all":
            return None, None
        return (
            self.start_time_edit.dateTime().toPyDateTime(),
            self.end_time_edit.dateTime().toPyDateTime(),
        )

    def _apply_time_filter(self) -> None:
        """Restrict overview candidates to the selected inclusive time range."""
        if hasattr(self, "feedback_label") and not self._submission_active:
            self.feedback_label.clear()
        start_time, end_time = self._selected_time_bounds()
        if start_time and end_time and start_time > end_time:
            QMessageBox.warning(
                self,
                self.windowTitle(),
                self._text(
                    "起始時間不得晚於結束時間。",
                    "The start time must not be later than the end time.",
                ),
            )
            return
        self._filtered_indices = [
            index
            for index, row in enumerate(self.store.rows)
            if timestamp_in_range(
                str(row.get("timestamp") or ""),
                start_time=start_time,
                end_time=end_time,
            )
        ]
        failure_indices = [
            index
            for index in self._filtered_indices
            if _is_failure_review_candidate(self.store.rows[index])
        ]
        self._refresh_review_gallery(failure_indices)
        if self._start_in_overview:
            self.visible_indices = failure_indices
            self._save_time_filter()
            return
        self.visible_indices = list(self._filtered_indices)
        self.current_index = next(
            (
                index
                for index in self.visible_indices
                if not str(self.store.rows[index].get("review_label") or "").strip()
            ),
            self.visible_indices[0] if self.visible_indices else 0,
        )
        self._save_time_filter()
        self._show_current()

    def _refresh_review_gallery(self, failure_indices: list[int]) -> None:
        """Render filtered failures without discarding selections from other ranges."""
        entries = [(index, self.store.rows[index]) for index in failure_indices]
        self.review_gallery.set_entries(entries, self._review_scope_indices)
        self._update_review_selection_summary()
        self._refresh_selected_review_gallery()

    def _refresh_selected_review_gallery(self) -> None:
        """Render every persisted selection without mixing in candidate rows."""
        selected_indices = sorted(
            index
            for index in self._review_scope_indices
            if 0 <= index < len(self.store.rows)
            and _is_failure_review_candidate(self.store.rows[index])
        )
        entries = [(index, self.store.rows[index]) for index in selected_indices]
        self.selected_review_gallery.set_entries(entries, set(selected_indices))
        self.selected_review_summary.setText(
            self._text(
                f"已保存 {len(selected_indices)} 張；這裡只顯示已選圖片",
                f"{len(selected_indices)} saved; only selected images are shown here",
            )
        )
        self.classify_selected_button.setEnabled(bool(selected_indices))
        self.classify_selected_button.setText(
            self._text(
                f"開始逐張分類（{len(selected_indices)}）",
                f"Classify selected images ({len(selected_indices)})",
            )
        )

    def _on_review_selection_changed(self, selected_indices: set[int]) -> None:
        """Persist visible checkbox changes and retain selections in other ranges."""
        visible_gallery_indices = self.review_gallery.entry_indices()
        try:
            self._persist_review_scope_selection(
                visible_gallery_indices,
                selected_indices,
            )
        except (OSError, sqlite3.Error, TimeoutError, ValueError, IndexError) as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            self._restore_review_scope_from_persistence()
            self._refresh_review_gallery(
                sorted(self.review_gallery.source_entry_indices())
            )
            return
        self._update_review_selection_summary()
        self._refresh_selected_review_gallery()

    def _on_selected_review_selection_changed(
        self, selected_indices: set[int]
    ) -> None:
        """Remove unchecked rows from the persistent selected-only gallery."""
        displayed_indices = self.selected_review_gallery.entry_indices()
        try:
            self._persist_review_scope_selection(displayed_indices, selected_indices)
        except (OSError, sqlite3.Error, TimeoutError, ValueError, IndexError) as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            self._restore_review_scope_from_persistence()
        self._refresh_selected_review_gallery()
        self.review_gallery.set_entries(
            [
                (index, self.store.rows[index])
                for index in sorted(self.review_gallery.source_entry_indices())
            ],
            self._review_scope_indices,
        )
        self._update_review_selection_summary()

    def _persist_review_scope_selection(
        self,
        displayed_indices: set[int],
        selected_indices: set[int],
    ) -> None:
        """Save queue membership without erasing completed human-review history."""
        if not selected_indices.issubset(displayed_indices):
            raise ValueError("Selected review rows must belong to the displayed set")
        reviewed_indices = {
            index
            for index in displayed_indices
            if _has_human_review_result(self.store.rows[index])
        }
        manifest_candidates = displayed_indices - reviewed_indices
        manifest_candidates.update(
            index
            for index in selected_indices & reviewed_indices
            if str(self.store.rows[index].get("review_selected") or "0") != "1"
        )
        self.store.set_review_selection(
            manifest_candidates,
            selected_indices & manifest_candidates,
        )

        removed_reviewed = (
            self._review_scope_indices & displayed_indices & reviewed_indices
        ) - selected_indices
        selected_reviewed = selected_indices & reviewed_indices
        self._dismissed_review_identities.update(
            self._review_queue_identity(index) for index in removed_reviewed
        )
        self._dismissed_review_identities.difference_update(
            self._review_queue_identity(index) for index in selected_reviewed
        )
        self._review_scope_indices.difference_update(displayed_indices)
        self._review_scope_indices.update(selected_indices)
        try:
            self._save_time_filter(strict=True)
        except OSError:
            self._restore_review_scope_from_persistence()
            raise

    def _review_queue_identity(self, index: int) -> str:
        identity = record_identity(self.store.rows[index])
        return identity if identity != "unknown" else f"manifest-row:{index}"

    def _restore_review_scope_from_persistence(self) -> None:
        state = self._read_filter_state()
        dismissed = state.get("dismissed_review_identities", [])
        self._dismissed_review_identities = {
            str(identity) for identity in dismissed if str(identity).strip()
        } if isinstance(dismissed, list) else set()
        self._review_scope_indices = {
            index
            for index, row in enumerate(self.store.rows)
            if str(row.get("review_selected") or "0") == "1"
            and _is_failure_review_candidate(row)
            and self._review_queue_identity(index)
            not in self._dismissed_review_identities
        }

    def _update_review_selection_summary(self) -> None:
        range_indices = self.review_gallery.source_entry_indices()
        visible_selected = self._review_scope_indices & range_indices
        hidden_selected_count = len(self._review_scope_indices - range_indices)
        hidden_suffix = (
            self._text(
                f"｜其他時間範圍另保留 {hidden_selected_count} 張勾選",
                f" | {hidden_selected_count} selection(s) retained in other ranges",
            )
            if hidden_selected_count
            else ""
        )
        self.review_selection_summary.setText(
            self._text(
                f"本範圍已勾選 {len(visible_selected)} 張",
                f"{len(visible_selected)} selected in this range",
            )
            + hidden_suffix
        )
        self.review_selection_saved_label.setText(
            self._text(
                f"已保存 {len(self._review_scope_indices)} 張",
                f"{len(self._review_scope_indices)} saved",
            )
        )
        self.start_review_button.setEnabled(bool(self._review_scope_indices))
        self.start_review_button.setText(
            self._text(
                (
                    f"查看已選圖片（{len(self._review_scope_indices)}）"
                    if self._use_legacy_selected_page
                    else f"開始複核（{len(self._review_scope_indices)}）"
                ),
                (
                    f"View selected images ({len(self._review_scope_indices)})"
                    if self._use_legacy_selected_page
                    else f"Start review ({len(self._review_scope_indices)})"
                ),
            )
        )

    def _show_selected_review_page(self) -> None:
        """Open the selected-only gallery across every time range."""
        self._refresh_selected_review_gallery()
        self.workflow_stack.setCurrentWidget(self.selected_review_page)

    def _show_candidate_review_page(self) -> None:
        """Return to the current time range to add more candidates."""
        self.workflow_stack.setCurrentWidget(self.review_selection_page)
        failure_indices = [
            index
            for index in self._filtered_indices
            if _is_failure_review_candidate(self.store.rows[index])
        ]
        self.visible_indices = failure_indices
        self._refresh_review_gallery(failure_indices)

    def _start_selected_review(self) -> None:
        """Lock the visible checked scope and continue to per-image classification."""
        selected_indices = set(self._review_scope_indices)
        if not selected_indices:
            QMessageBox.warning(
                self,
                self.windowTitle(),
                self._text(
                    "請先勾選至少一張需要審核的失敗圖。",
                    "Select at least one failed image that needs review.",
                ),
            )
            return
        if not self._validate_review_scope(selected_indices):
            return
        self.visible_indices = sorted(selected_indices)
        self.current_index = next(
            (
                index
                for index in self.visible_indices
                if not record_is_reviewed(self.store.rows[index])
            ),
            self.visible_indices[0],
        )
        self.workflow_stack.setCurrentWidget(self.classification_page)
        self._show_current()

    def _show_review_overview(self) -> None:
        """Return from classification without losing persisted review decisions."""
        if self._use_legacy_selected_page:
            self._show_selected_review_page()
        else:
            self._show_candidate_review_page()

    def _validate_review_scope(self, selected_indices: set[int]) -> bool:
        """Block inconsistent selections through the Phase 1B validator."""
        invalid_records: list[tuple[int, tuple[Any, ...]]] = []
        for index in sorted(selected_indices):
            if not 0 <= index < len(self.store.rows):
                logger.error(
                    "Review start blocked invalid manifest index=%s row_count=%s",
                    index,
                    len(self.store.rows),
                )
                QMessageBox.critical(
                    self,
                    self.windowTitle(),
                    self._text(
                        f"選取資料索引 {index} 已失效，請重新開啟複核畫面。",
                        f"Selected row index {index} is stale. Reopen the review dialog.",
                    ),
                )
                return False
            row = self.store.rows[index]
            violations = blocking_violations(validate_record_consistency(row))
            if violations:
                invalid_records.append((index, violations))
                logger.error(
                    "Review start blocked sample_id=%s fields=%r violations=%s",
                    record_identity(row),
                    {
                        field: row.get(field, "")
                        for field in sorted(
                            set(REVIEW_CORE_FIELDS)
                            | {"review_selected", "training_selected", "job_status"}
                        )
                    },
                    [violation.code for violation in violations],
                )
        if not invalid_records:
            return True
        details = []
        for index, violations in invalid_records[:5]:
            sample_id = record_identity(self.store.rows[index])
            codes = ", ".join(violation.code for violation in violations)
            details.append(f"• {sample_id}: {codes}")
        if len(invalid_records) > 5:
            details.append(
                self._text(
                    f"• 另有 {len(invalid_records) - 5} 筆不一致資料",
                    f"• {len(invalid_records) - 5} more inconsistent record(s)",
                )
            )
        QMessageBox.critical(
            self,
            self.windowTitle(),
            self._text(
                "選取資料不一致，已阻止開始複核：\n",
                "Selected data is inconsistent; review start was blocked:\n",
            )
            + "\n".join(details),
        )
        return False

    def _restore_time_filter(self) -> None:
        """Restore the last product/station review range when available."""
        state = self._read_filter_state()

        mode = str(state.get("mode") or "last_7_days")
        if mode not in {"all", "today", "last_7_days", "last_30_days", "custom"}:
            mode = "last_7_days"
        self.time_range_combo.blockSignals(True)
        try:
            index = self.time_range_combo.findData(mode)
            self.time_range_combo.setCurrentIndex(index if index >= 0 else 0)
            if mode == "custom":
                for editor, key in (
                    (self.start_time_edit, "start"),
                    (self.end_time_edit, "end"),
                ):
                    try:
                        restored = datetime.fromisoformat(str(state.get(key) or ""))
                    except ValueError:
                        continue
                    editor.setDateTime(QDateTime(restored))
        finally:
            self.time_range_combo.blockSignals(False)
        self._on_time_preset_changed()

    def _read_filter_state(self) -> dict[str, Any]:
        """Read optional UI state without treating it as review-domain data."""
        try:
            payload = json.loads(self._filter_state_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except FileNotFoundError:
            return {}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning("Review UI state could not be read path=%s error=%s", self._filter_state_path, exc)
        return {}

    def _save_time_filter(self, *, strict: bool = False) -> None:
        """Persist the current review range atomically for this target."""
        start_time, end_time = self._selected_time_bounds()
        payload = {
            "schema_version": REVIEW_FILTER_STATE_SCHEMA_VERSION,
            "mode": str(self.time_range_combo.currentData() or "all"),
            "start": start_time.isoformat() if start_time else "",
            "end": end_time.isoformat() if end_time else "",
            "dismissed_review_identities": sorted(
                self._dismissed_review_identities
            ),
        }
        temporary = self._filter_state_path.with_name(f".{self._filter_state_path.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(self._filter_state_path)
        except OSError as exc:
            logger.warning("Review UI state could not be saved path=%s error=%s", self._filter_state_path, exc)
            if strict:
                raise
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _image_panel(self, title: str) -> QLabel:
        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(280, 240)
        label.setStyleSheet("QLabel { background: #202124; color: white; }")
        return label

    def _show_current(self) -> None:
        total = len(self.visible_indices)
        reviewed = sum(
            record_is_reviewed(self.store.rows[index]) for index in self.visible_indices
        )
        if total == 0:
            self.progress_label.setText(self._text("所選時間範圍內沒有案例", "No cases in selected range"))
            self.details_label.setText(
                self._text(
                    "請調整時間範圍後重新套用篩選。",
                    "Adjust the time range and apply the filter again.",
                )
            )
            self.question_label.clear()
            self.original_label.clear()
            self.annotated_label.clear()
            if hasattr(self, "image_viewer"):
                self.image_viewer.set_images(original_path="", overlay_path="")
            if hasattr(self, "review_thumbnail_panel"):
                self.review_thumbnail_panel.set_items([], current_index=None)
                self._review_thumbnail_visible_indices = ()
            for button in self.review_buttons.values():
                button.setEnabled(False)
                button.setVisible(False)
            for button in self.failure_category_buttons.values():
                button.setEnabled(False)
                button.setChecked(False)
            self.clear_failure_category_button.setEnabled(False)
            self.failure_classification_label.clear()
            self._show_triage_details("")
            self._update_reprocess_button()
            self._update_submit_button()
            return

        if self.current_index not in self.visible_indices:
            self.current_index = self.visible_indices[0]
        visible_position = self.visible_indices.index(self.current_index)
        row = self.store.rows[self.current_index]
        failure_index = self.fail_reason_combo.findData(
            str(row.get("failure_category") or "")
        )
        if failure_index >= 0:
            self.fail_reason_combo.setCurrentIndex(failure_index)
        self.custom_failure_note.setText(str(row.get("failure_note") or ""))
        skip_index = self.skip_reason_combo.findData(skip_ui_reason_from_record(row))
        if skip_index >= 0:
            self.skip_reason_combo.setCurrentIndex(skip_index)
        self.custom_skip_note.setText(skip_custom_note_from_record(row))
        self._show_triage_details("")
        self._update_failure_classification_controls(row)
        current_value = str(row.get("review_label") or "")
        current_label = (
            self._text(*OPERATOR_ACTION_LABELS[current_value])
            if current_value in OPERATOR_ACTION_LABELS
            else self._text("尚未確認", "Pending")
        )
        self.progress_label.setText(
            self._text(
                f"第 {visible_position + 1}/{total} 張｜已複核 {reviewed} 張",
                f"Case {visible_position + 1}/{total} | Reviewed {reviewed}",
            )
        )
        if hasattr(self, "navigation_progress_label"):
            self.navigation_progress_label.setText(
                f"{visible_position + 1} / {total}"
            )
        detection_state = _row_detection_state(row)
        has_detection = detection_state == DETECTION_PRESENT
        detection_unknown = detection_state == DETECTION_UNKNOWN
        color_failure = has_color_failure(row)
        status = str(row.get("status") or "").strip().upper()
        self.question_label.setText(
            self._text(
                "顏色檢查未過。實物顏色究竟是 OK 還是 NG？"
                if color_failure
                else (
                    "AI 判定 PASS，人工覆核結果是？"
                    if status == "PASS"
                    else (
                        "AI 判定 NG，人工覆核結果是？"
                        if has_detection
                        else (
                            "舊資料未保存框明細，請依原圖與推理結果判斷。"
                            if detection_unknown
                            else "系統沒有畫框，影像中是否有應檢目標？"
                        )
                    )
                ),
                "The color check failed. Is the physical color actually OK or NG?"
                if color_failure
                else (
                    "AI reported PASS. What is the operator verdict?"
                    if status == "PASS"
                    else (
                        "AI reported NG. What is the operator verdict?"
                        if has_detection
                        else (
                            "Legacy data did not preserve box details; judge from both images."
                            if detection_unknown
                            else "No box was detected. Is there a target in the image?"
                        )
                    )
                ),
            )
        )
        detection_notice = ""
        if detection_unknown:
            detection_notice = self._text(
                "此舊紀錄缺少結構化框資料，系統不會將它誤判為確定漏檢；請依保存影像選擇實際結果。\n",
                "This legacy record has no structured box data. It is not treated as a confirmed miss; choose from the saved images.\n",
            )
        elif not has_detection:
            detection_notice = self._text(
                "未偵測到任何框；影像中應有目標請選擇「漏檢」，確認沒有應檢目標才可選擇負樣本。\n",
                "No boxes were detected. Select missed detection when a target exists; use background only when no target should be present.\n",
            )
        color_notice = ""
        if color_failure:
            summary = color_summary(row) or self._text(
                "舊資料未保存顏色分數，請勿送出校正",
                "Legacy snapshot has no color score; do not submit calibration.",
            )
            color_notice = self._text("顏色失敗：", "Color failure: ") + summary + "\n"
        self.details_label.setText(
            detection_notice
            + color_notice
            + f"{row.get('product', '')}/{row.get('area', '')}  "
            f"{row.get('timestamp', '')}  [{current_label}]"
        )
        if self._use_legacy_review_layout:
            visible_actions = _visible_action_values(
                status,
                has_detection,
                color_failure=color_failure,
                detection_unknown=detection_unknown,
            )
            color_false_reject_button = self.review_buttons.get("color_false_reject")
            if color_false_reject_button is not None:
                color_false_reject_button.setText(
                    self._text(
                        "顏色其實 OK（門檻過嚴）"
                        if has_threshold_color_failure(row)
                        else "顏色其實 OK（顏色規則過嚴）",
                        "Color is actually OK (threshold too strict)"
                        if has_threshold_color_failure(row)
                        else "Color is actually OK (color rule too strict)",
                    )
                )
            for button in self.review_buttons.values():
                self.review_layout.removeWidget(button)
            visible_index = 0
            for value, button in self.review_buttons.items():
                button.setVisible(value in visible_actions)
                button.setEnabled(value in visible_actions)
                button.setChecked(
                    value == current_value
                    or value == "needs_annotation"
                    and current_value
                    in {correction[0] for correction in ANNOTATION_CORRECTIONS}
                )
                if value in visible_actions:
                    self.review_layout.addWidget(button, 0, visible_index)
                    visible_index += 1
            self._set_pixmap(self.original_label, row.get("original_path", ""))
            self._set_pixmap(self.annotated_label, _best_inference_image(row))
        else:
            details = build_review_details(row, language=self.language)
            self.ai_summary_label.setText(details.ai_summary)
            self.metadata_summary_label.setText(details.metadata_summary)
            self.domain_summary_label.setText(details.semantics_summary)
            self.image_viewer.set_images(
                original_path=row.get("original_path", ""),
                overlay_path=_best_inference_image(row),
                sample_id=record_identity(row),
            )
            self._refresh_review_thumbnail_panel()
            self._update_fail_action_label()
        self._update_reprocess_button()
        self._update_submit_button()

    def _update_failure_classification_controls(
        self, row: dict[str, str]
    ) -> None:
        """Render a cause tag independently from the operator verdict buttons."""
        category = str(row.get("failure_category") or "")
        source = str(row.get("failure_source") or "")
        for value, button in self.failure_category_buttons.items():
            button.setEnabled(True)
            button.setChecked(value == category)
        self.clear_failure_category_button.setEnabled(bool(category or source))
        reason_labels = {
            value: self._text(zh_label, en_label)
            for value, zh_label, en_label in FAILURE_REASON_OPTIONS
        }
        if category:
            note = str(row.get("failure_note") or "").strip()
            note_suffix = f"｜{note}" if note else ""
            self.failure_classification_label.setText(
                self._text(
                    f"已記錄：{reason_labels.get(category, category)}｜來源："
                    f"{self._failure_source_label(source)}{note_suffix}",
                    f"Saved: {reason_labels.get(category, category)} | Source: "
                    f"{self._failure_source_label(source)}{note_suffix}",
                )
            )
        else:
            self.failure_classification_label.setText(
                self._text("尚未設定失敗原因", "No failure cause selected")
            )

    def _set_failure_category(self, category: str) -> None:
        """Ask for an extensible subsystem source, then persist the cause tag."""
        if not self.store.rows or self.current_index >= len(self.store.rows):
            return
        row = self.store.rows[self.current_index]
        source_keys = threshold_source_keys(row)
        source_labels = [self._failure_source_label(source) for source in source_keys]
        selected_label, accepted = QInputDialog.getItem(
            self,
            self._text("選擇閾值來源", "Select threshold source"),
            self._text(
                "是哪一個檢查模組的閾值未達標？",
                "Which inspection subsystem did not meet its threshold?",
            ),
            source_labels,
            0,
            False,
        )
        if not accepted:
            self._update_failure_classification_controls(row)
            return
        source = source_keys[source_labels.index(selected_label)]
        try:
            self.store.set_failure_classification(
                self.current_index,
                ReviewFailureClassification(category=category, source=source),
            )
        except (OSError, sqlite3.Error, TimeoutError, ValueError, IndexError) as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            self._update_failure_classification_controls(row)
            return
        self._update_failure_classification_controls(
            self.store.rows[self.current_index]
        )

    def _clear_failure_category(self) -> None:
        """Clear only the cause tag while preserving selection and review verdict."""
        if not self.store.rows or self.current_index >= len(self.store.rows):
            return
        try:
            self.store.set_failure_classification(self.current_index, None)
        except (OSError, sqlite3.Error, TimeoutError, ValueError, IndexError) as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return
        self._update_failure_classification_controls(
            self.store.rows[self.current_index]
        )

    def _failure_source_label(self, source: str) -> str:
        labels = FAILURE_SOURCE_LABELS.get(source)
        return self._text(*labels) if labels else source

    def _update_submit_button(self) -> None:
        """Expose the next step only after the selected queue is complete."""
        eligible_labels = DIRECT_TRAIN_LABELS | ANNOTATION_LABELS | COLOR_REVIEW_LABELS
        queue_count = sum(
            str(row.get("review_label") or "") in eligible_labels and str(row.get("training_selected") or "1") != "0"
            for row in self.store.rows
        )
        self.batch_preview_button.setEnabled(queue_count > 0)
        self.batch_preview_button.setText(
            self._text(
                f"待送清單（{queue_count}）",
                f"Pending queue ({queue_count})",
            )
        )
        if self._submission_active:
            self.export_button.setEnabled(False)
            self.export_button.setText(self._text("本批次已送出", "Batch submitted"))
            return
        reviewed_indices = {
            index for index in self.visible_indices if str(self.store.rows[index].get("review_label") or "").strip()
        }
        selected_count = sum(
            str(self.store.rows[index].get("training_selected") or "1") != "0" for index in reviewed_indices
        )
        remaining = sum(
            not str(self.store.rows[index].get("review_label") or "").strip() for index in self.visible_indices
        )
        self.export_button.setEnabled(bool(self.visible_indices) and remaining == 0 and selected_count > 0)
        self.export_button.setText(
            self._text(
                ("完成本範圍並檢查待送清單" if selected_count > 0 else "本範圍沒有可送訓影像")
                if remaining == 0
                else f"尚有 {remaining} 張未確認",
                ("Finish this range and review queue" if selected_count > 0 else "No trainable image in this range")
                if remaining == 0
                else f"{remaining} case(s) remaining",
            )
        )

    def _set_pixmap(self, target: QLabel, path_value: Any) -> None:
        path = Path(str(path_value or ""))
        pixmap = QPixmap(str(path)) if path.exists() else QPixmap()
        if pixmap.isNull():
            target.setText(self._text("圖片不存在", "Image unavailable"))
            return
        target.setPixmap(pixmap.scaled(target.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _set_review(self, value: str) -> None:
        if not self.store.rows:
            return
        try:
            self.store.set_review(self.current_index, value)
        except (OSError, sqlite3.Error, ValueError, IndexError) as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return
        self._after_review_recorded(value)

    def _show_triage_details(self, outcome: str) -> None:
        """Show only the second-level controls required by Fail or Skip."""
        self.fail_details_panel.setVisible(outcome == "fail")
        self.skip_details_panel.setVisible(outcome == "skip")

    def _save_pass_triage(self) -> None:
        """Plan and persist PASS through the shared Phase 1B save boundary."""
        if not self.store.rows:
            return
        try:
            plan = plan_pass(self.store.rows[self.current_index])
        except (TypeError, ValueError) as exc:
            self._show_review_save_error(exc, self._save_pass_triage)
            return
        self._save_review_plan(plan, self._save_pass_triage)

    def _reprocessable_handed_off_indices(self) -> set[int]:
        """Return selected historical cases that can be queued again."""
        return {
            index
            for index in self.visible_indices
            if index in self._handed_off_indices
            and str(self.store.rows[index].get("training_selected") or "0") == "0"
            and _is_trainable_review(self.store.rows[index])
        }

    def _update_reprocess_button(self) -> None:
        """Expose historical reprocessing without silently duplicating a handoff."""
        if not hasattr(self, "reprocess_button"):
            return
        count = len(self._reprocessable_handed_off_indices())
        self.reprocess_button.setVisible(count > 0)
        self.reprocess_button.setEnabled(count > 0)
        self.reprocess_button.setText(
            self._text(
                f"重新加入待送清單（{count} 張已送過）",
                f"Re-add submitted cases ({count})",
            )
        )

    def _requeue_handed_off_scope(self) -> None:
        """Queue previously submitted evidence after one explicit operator action."""
        indices = self._reprocessable_handed_off_indices()
        if not indices:
            return
        try:
            self.store.set_training_selection(indices, indices)
        except (OSError, sqlite3.Error, TimeoutError, ValueError, IndexError) as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return
        self.feedback_label.setText(
            self._text(
                f"已將 {len(indices)} 張已送過影像重新加入待送清單。",
                f"Re-added {len(indices)} submitted case(s) to the pending queue.",
            )
        )
        self._update_reprocess_button()
        self._update_submit_button()

    def _save_fail_triage(self) -> None:
        """Plan and persist FAIL without selecting a route in the QWidget."""
        if not self.store.rows:
            return
        category = str(self.fail_reason_combo.currentData() or "")
        note = self.custom_failure_note.text().strip()
        try:
            plan = plan_fail(
                self.store.rows[self.current_index],
                category=category,
                note=note,
            )
        except (TypeError, ValueError) as exc:
            self._show_review_save_error(exc, self._save_fail_triage)
            return
        self._save_review_plan(plan, self._save_fail_triage)

    def _update_fail_action_label(self) -> None:
        """Preview typed semantics produced by the shared action planner."""
        self.save_fail_button.setText(self._text("儲存 FAIL", "Save FAIL"))
        if not hasattr(self, "proposed_semantics_label") or not self.store.rows:
            return
        try:
            plan = plan_fail(
                self.store.rows[self.current_index],
                category=str(self.fail_reason_combo.currentData() or ""),
                note=self.custom_failure_note.text().strip(),
            )
        except (TypeError, ValueError) as exc:
            self.proposed_semantics_label.setText(str(exc))
            return
        semantics = plan.semantics
        self.proposed_semantics_label.setText(
            "\n".join(
                (
                    f"product verdict: {semantics.product_verdict.value}",
                    f"AI correctness: {semantics.ai_correctness.value}",
                    f"annotation validity: {semantics.annotation_validity.value}",
                    f"required action: {semantics.required_action.value}",
                )
            )
        )

    def _save_skip_triage(self) -> None:
        """Map a richer skip reason onto the unchanged compatibility fields."""
        if not self.store.rows:
            return
        try:
            plan = plan_skip(
                self.store.rows[self.current_index],
                ui_reason=str(self.skip_reason_combo.currentData() or ""),
                note=self.custom_skip_note.text().strip(),
            )
        except (TypeError, ValueError) as exc:
            self._show_review_save_error(exc, self._save_skip_triage)
            return
        self._save_review_plan(plan, self._save_skip_triage)

    def _save_review_plan(
        self,
        plan: ReviewActionPlan,
        retry_action: Any,
    ) -> None:
        """Validate an in-memory proposal, atomically save it, then navigate."""
        updates = dict(plan.updates)
        revision_reason = ""
        try:
            self.store.validate_proposed_update(self.current_index, updates)
        except ReviewWorkflowValidationError as exc:
            if not self._requires_submitted_revision(exc):
                self._show_review_save_error(exc, retry_action)
                return
            prompted_reason = self._prompt_submitted_revision_reason()
            if prompted_reason is None:
                self._show_review_save_error(exc, retry_action)
                return
            revision_reason = prompted_reason
        except (OSError, sqlite3.Error, TimeoutError, ValueError, IndexError) as exc:
            self._show_review_save_error(exc, retry_action)
            return
        try:
            self.store.apply_review_updates(
                self.current_index,
                updates,
                revision_reason=revision_reason,
            )
        except (
            OSError,
            sqlite3.Error,
            TimeoutError,
            ReviewWorkflowValidationError,
            ValueError,
            IndexError,
        ) as exc:
            self._show_review_save_error(exc, retry_action)
            return
        self._retry_review_action = None
        self.retry_save_button.setVisible(False)
        self._set_feedback_error_state(False)
        self._after_review_recorded(plan.review_label)

    @staticmethod
    def _requires_submitted_revision(exc: ReviewWorkflowValidationError) -> bool:
        """Use the shared workflow violation code instead of duplicating its rule."""
        return any(
            violation.code == "submitted_review_modified_without_revision"
            for violation in exc.violations
        )

    def _prompt_submitted_revision_reason(self) -> str | None:
        """Require an explicit operator reason before revising submitted evidence."""
        reason, accepted = QInputDialog.getText(
            self,
            self._text("建立覆核修訂", "Create review revision"),
            self._text(
                "這張圖片曾經送出。若要修改覆核結果，請輸入重新處理原因：",
                "This image was already submitted. Enter a reason for revising it:",
            ),
        )
        normalized_reason = str(reason).strip()
        if not accepted:
            return None
        if not normalized_reason:
            QMessageBox.warning(
                self,
                self.windowTitle(),
                self._text(
                    "修訂原因不可空白，資料尚未保存。",
                    "A revision reason is required. Nothing was saved.",
                ),
            )
            return None
        return normalized_reason

    def _show_review_save_error(self, exc: Exception, retry_action: Any) -> None:
        row = self.store.rows[self.current_index] if self.store.rows else {}
        sample_id = record_identity(row)
        codes = (
            [violation.code for violation in exc.violations]
            if isinstance(exc, ReviewWorkflowValidationError)
            else [type(exc).__name__]
        )
        message = self._text(
            f"保存失敗｜樣本：{sample_id}｜錯誤：{', '.join(codes)}\n{exc}",
            f"Save failed | sample: {sample_id} | error: {', '.join(codes)}\n{exc}",
        )
        logger.error(
            "Review save failed sample_id=%s violations=%s error=%s",
            sample_id,
            codes,
            exc,
            exc_info=not isinstance(exc, ReviewWorkflowValidationError),
        )
        self.feedback_label.setText(message)
        self._set_feedback_error_state(True)
        self._retry_review_action = retry_action
        self.retry_save_button.setVisible(True)

    def _retry_last_review_save(self) -> None:
        if self._retry_review_action is not None:
            self._retry_review_action()

    def _set_feedback_error_state(self, is_error: bool) -> None:
        self.feedback_label.setStyleSheet(
            "QLabel { color:%s;background:%s;padding:5px;font-weight:bold; }"
            % (("#8b1e1e", "#fdecec") if is_error else ("#1b5e20", "#e8f5e9"))
        )

    def _set_color_review(
        self, color_verdict: str, *, detection_verdict: str = "correct"
    ) -> None:
        """Persist color truth and optional independent detection correction."""
        if not self.store.rows:
            return
        try:
            self.store.set_color_review(
                self.current_index,
                color_verdict,
                detection_verdict=detection_verdict,
            )
        except (OSError, sqlite3.Error, ValueError, IndexError) as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return
        label = (
            "color_confirmed_ng"
            if color_verdict == "confirmed_ng"
            else "color_false_reject"
        )
        self._after_review_recorded(label)

    def _after_review_recorded(self, value: str) -> None:
        """Update navigation and feedback after any persisted decision."""
        self._refresh_review_thumbnail_item(self.current_index)
        selected_label = self._text(*OPERATOR_ACTION_LABELS[value])
        remaining = sum(
            not str(self.store.rows[index].get("review_label") or "").strip() for index in self.visible_indices
        )
        if remaining:
            self.feedback_label.setText(
                self._text(
                    f"已記錄「{selected_label}」，已切換至下一張。尚有 {remaining} 張。",
                    f'Recorded "{selected_label}". {remaining} case(s) remaining.',
                )
            )
            self._move_to_next_pending()
            return
        self._show_current()
        if self.export_button.isEnabled():
            self.feedback_label.setText(
                self._text(
                    f"已記錄「{selected_label}」。本範圍已全部確認，請進行下一步。",
                    f'Recorded "{selected_label}". Review complete; continue below.',
                )
            )
            self.export_button.setFocus()
        else:
            self.feedback_label.setText(
                self._text(
                    f"已記錄「{selected_label}」。目前沒有可送訓影像，資料已暫存。",
                    f'Recorded "{selected_label}". No image is eligible for training.',
                )
            )

    def _handle_review_action(self, value: str) -> None:
        """Persist a verdict or ask for the second-level annotation correction."""
        if value == "color_confirmed_ng":
            self._set_color_review("confirmed_ng")
            return
        if value == "color_false_reject":
            self._set_color_review("actually_ok")
            return
        if value == "color_needs_annotation":
            self._handle_color_and_annotation_review()
            return
        if value != "needs_annotation":
            self._set_review(value)
            return
        option_labels = [self._text(zh, en) for _stored, zh, en in ANNOTATION_CORRECTIONS]
        selected, accepted = QInputDialog.getItem(
            self,
            self._text("選擇標註修正類型", "Select annotation correction"),
            self._text("需要修正哪一項？", "What needs correction?"),
            option_labels,
            0,
            False,
        )
        if not accepted:
            self._show_current()
            return
        selected_index = option_labels.index(selected)
        self._set_review(ANNOTATION_CORRECTIONS[selected_index][0])

    def _handle_color_and_annotation_review(self) -> None:
        """Collect the independent color truth and box correction type."""
        color_options = [
            self._text("顏色確實 NG", "Color is truly NG"),
            self._text("顏色其實 OK（門檻過嚴）", "Color is actually OK"),
        ]
        selected_color, accepted = QInputDialog.getItem(
            self,
            self._text("顏色覆核", "Color review"),
            self._text("先確認實物顏色：", "Confirm the physical color:"),
            color_options,
            0,
            False,
        )
        if not accepted:
            self._show_current()
            return
        correction_options = [
            self._text(zh, en) for _stored, zh, en in ANNOTATION_CORRECTIONS
        ]
        selected_correction, accepted = QInputDialog.getItem(
            self,
            self._text("框選修正", "Annotation correction"),
            self._text("框的哪一部分需要修正？", "What needs correction?"),
            correction_options,
            0,
            False,
        )
        if not accepted:
            self._show_current()
            return
        color_verdict = (
            "confirmed_ng"
            if color_options.index(selected_color) == 0
            else "actually_ok"
        )
        detection_verdict = ANNOTATION_CORRECTIONS[
            correction_options.index(selected_correction)
        ][0]
        self._set_color_review(
            color_verdict, detection_verdict=detection_verdict
        )

    def _move_to_next_pending(self) -> None:
        total = len(self.visible_indices)
        if total == 0:
            self._show_current()
            return
        current_position = self.visible_indices.index(self.current_index)
        for offset in range(1, total + 1):
            index = self.visible_indices[(current_position + offset) % total]
            if not record_is_reviewed(self.store.rows[index]):
                self.current_index = index
                self._show_current()
                return
        self._show_current()

    def _refresh_review_thumbnail_panel(self) -> None:
        if not hasattr(self, "review_thumbnail_panel"):
            return
        visible_key = tuple(self.visible_indices)
        if visible_key == self._review_thumbnail_visible_indices:
            self._refresh_review_thumbnail_item(self.current_index)
            self.review_thumbnail_panel.set_current_index(
                self.current_index if self.visible_indices else None
            )
            return
        items = [
            build_review_list_item(
                index,
                self.store.rows[index],
                language=self.language,
            )
            for index in self.visible_indices
        ]
        self.review_thumbnail_panel.set_items(
            items,
            current_index=self.current_index if self.visible_indices else None,
        )
        self._review_thumbnail_visible_indices = visible_key

    def _refresh_review_thumbnail_item(self, row_index: int) -> None:
        if (
            not hasattr(self, "review_thumbnail_panel")
            or row_index not in self.visible_indices
        ):
            return
        self.review_thumbnail_panel.update_item(
            build_review_list_item(
                row_index,
                self.store.rows[row_index],
                language=self.language,
            )
        )

    def _select_review_thumbnail(self, row_index: int) -> None:
        if row_index not in self.visible_indices:
            return
        self.feedback_label.clear()
        self._retry_review_action = None
        self.retry_save_button.setVisible(False)
        self.current_index = row_index
        self._show_current()

    def _install_review_shortcuts(self) -> None:
        self.classification_page.installEventFilter(self)
        for widget in self.classification_page.findChildren(QWidget):
            widget.installEventFilter(self)

    def eventFilter(self, watched: Any, event: Any) -> bool:  # noqa: N802 - Qt API
        if (
            hasattr(self, "classification_page")
            and event.type() == QEvent.KeyPress
            and (
                watched is self.classification_page
                or self.classification_page.isAncestorOf(watched)
            )
        ):
            key_actions = {
                Qt.Key_P: "pass",
                Qt.Key_F: "fail",
                Qt.Key_S: "skip",
                Qt.Key_Left: "previous",
                Qt.Key_Right: "next",
                Qt.Key_Space: "toggle_image",
            }
            if event.key() in {Qt.Key_Return, Qt.Key_Enter} and (
                self.fail_details_panel.isVisible()
                or self.skip_details_panel.isVisible()
            ):
                self._dispatch_review_shortcut("confirm")
                return True
            if event.key() == Qt.Key_Escape and (
                self.fail_details_panel.isVisible()
                or self.skip_details_panel.isVisible()
            ):
                self._dispatch_review_shortcut("escape")
                return True
            action = key_actions.get(event.key())
            if action is not None and event.modifiers() in {
                Qt.NoModifier,
                Qt.KeypadModifier,
            }:
                if is_text_input_focus(watched):
                    return super().eventFilter(watched, event)
                self._dispatch_review_shortcut(action)
                return True
        return super().eventFilter(watched, event)

    def _dispatch_review_shortcut(self, action: str) -> None:
        focus = QApplication.focusWidget()
        if is_text_input_focus(focus) and action in {
            "pass",
            "fail",
            "skip",
            "previous",
            "next",
            "toggle_image",
        }:
            return
        if action == "pass":
            self.pass_review_button.click()
        elif action == "fail":
            self.fail_review_button.click()
        elif action == "skip":
            self.skip_review_button.click()
        elif action == "previous":
            self._move(-1)
        elif action == "next":
            self._move(1)
        elif action == "toggle_image" and hasattr(self, "image_viewer"):
            self.image_viewer.toggle_mode()
        elif action == "confirm":
            if self.fail_details_panel.isVisible():
                self.save_fail_button.click()
            elif self.skip_details_panel.isVisible():
                self.save_skip_button.click()
        elif action == "escape":
            self._show_triage_details("")

    def _move(self, offset: int) -> None:
        self.feedback_label.clear()
        self._retry_review_action = None
        self.retry_save_button.setVisible(False)
        if self.visible_indices:
            current_position = self.visible_indices.index(self.current_index)
            self.current_index = self.visible_indices[(current_position + offset) % len(self.visible_indices)]
            self._show_current()

    def _open_selected_training_queue(self) -> None:
        """Show the target-wide persistent queue and run one explicit action."""
        if processing_pipeline_enabled():
            self._open_processing_pipeline()
            return
        eligible_labels = DIRECT_TRAIN_LABELS | ANNOTATION_LABELS | COLOR_REVIEW_LABELS
        entries = [
            (index, row)
            for index, row in enumerate(self.store.rows)
            if str(row.get("review_label") or "") in eligible_labels and str(row.get("training_selected") or "1") != "0"
        ]
        if not entries:
            QMessageBox.information(
                self,
                self.windowTitle(),
                self._text(
                    "目前沒有已選擇的補訓照片。請先在複核頁確認照片。",
                    "No image is selected for retraining. Review images first.",
                ),
            )
            return
        dialog = TrainingBatchDialog(
            entries,
            language=self.language,
            queue_mode=True,
            parent=self,
        )
        if self._embedded:
            self._open_embedded_dialog(
                dialog,
                on_accepted=lambda: self._apply_training_queue(dialog, entries),
            )
            return
        if dialog.exec_() != QDialog.Accepted:
            return
        self._apply_training_queue(dialog, entries)

    def _apply_training_queue(
        self,
        dialog: TrainingBatchDialog,
        entries: list[tuple[int, dict[str, str]]],
    ) -> None:
        """Persist one queue decision and dispatch its explicit next action."""
        candidate_indices = {index for index, _row in entries}
        selected_indices = dialog.selected_indices()
        try:
            self.store.set_training_selection(candidate_indices, selected_indices)
        except (OSError, sqlite3.Error, ValueError, IndexError) as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return
        action_indices = dialog.action_selected_indices()
        self._show_current()
        if dialog.selected_action == "portable" and action_indices:
            self._export_portable_selected_indices(action_indices)
        elif action_indices:
            self._submit_selected_indices(action_indices)

    def _open_embedded_dialog(
        self,
        dialog: QDialog,
        *,
        on_accepted: Callable[[], None] | None = None,
    ) -> None:
        """Render a workflow dialog as a page and return without a nested event loop."""
        return_page = self.workflow_stack.currentWidget()
        dialog.setWindowFlags(Qt.Widget)
        self.workflow_stack.addWidget(dialog)

        def finish(result: int) -> None:
            self.workflow_stack.setCurrentWidget(return_page)
            if result == QDialog.Accepted and on_accepted is not None:
                on_accepted()
            self.workflow_stack.removeWidget(dialog)
            dialog.deleteLater()

        dialog.finished.connect(finish)
        dialog.show()
        self.workflow_stack.setCurrentWidget(dialog)

    def _open_processing_pipeline(self) -> None:
        """Build one batch plan and show the Phase 3A summary-only UI."""
        entries = [
            (index, row)
            for index, row in enumerate(self.store.rows)
            if _has_human_review_result(row)
        ]
        if not entries:
            QMessageBox.information(
                self,
                self.windowTitle(),
                self._text(
                    "目前沒有已完成人工複核的資料可建立處理計畫。",
                    "No reviewed sample is available for a processing plan.",
                ),
            )
            return
        try:
            manifest_sha = sha256_file(self.manifest_path)
        except OSError as exc:
            manifest_sha = ""
            logger.warning(
                "Could not hash review manifest before processing plan; using snapshot hash: path=%s error=%s",
                self.manifest_path,
                exc,
            )
        operator = str(
            os.environ.get("USERNAME")
            or os.environ.get("USER")
            or socket.gethostname()
        )
        plan = ProcessingPlanner().create_plan(
            entries,
            operator=operator,
            source_manifest_sha=manifest_sha,
        )
        if processing_execution_framework_enabled():
            run_store = ProcessingRunStore(
                self.manifest_path.parent / ".processing_runs"
            )
            validator = ProcessingPlanValidator()

            def current_context(current_plan):
                return build_validation_context_from_manifest(
                    self.manifest_path,
                    current_plan,
                    artifact_root=run_store.artifacts_dir,
                )

            dataset_step_enabled = processing_dataset_step_enabled()
            annotation_step_enabled = processing_annotation_step_enabled()
            color_step_enabled = processing_color_step_enabled()
            engine = build_processing_execution_engine(
                plan=plan,
                manifest_path=self.manifest_path,
                validator=validator,
                context_provider=current_context,
                store=run_store,
                dataset_step_enabled=dataset_step_enabled,
                dataset_dry_run=processing_dataset_dry_run_enabled(),
                annotation_step_enabled=annotation_step_enabled,
                color_step_enabled=color_step_enabled,
                models_root=self.manifest_path.parent / "models",
                color_revisions_root=self.manifest_path.parent / ".color_revisions",
            )
            annotation_resume = None
            annotation_launcher = None
            color_decider = None
            color_resume = None
            color_rollback = None
            if annotation_step_enabled:
                revision_store = AnnotationRevisionStore(
                    root=self.manifest_path.parent / ".annotation_revisions",
                    source_manifest=self.manifest_path,
                )
                resume_service = AnnotationResumeService(
                    revision_store=revision_store,
                    planner=ProcessingPlanner(),
                    validator=validator,
                )
                annotation_launcher = ManualAnnotationToolLauncher()

                def annotation_resume(package_path, reason):
                    package = load_annotation_package(package_path)
                    current_entries = tuple(
                        (index, dict(self.store.rows[index]))
                        for index, _row in entries
                        if index < len(self.store.rows)
                    )
                    return resume_service.resume(
                        package_path,
                        plan,
                        current_entries=current_entries,
                        current_manifest_sha=sha256_file(self.manifest_path),
                        revision_reasons={
                            item.sample_id: reason for item in package.items
                        },
                        operator=operator,
                    )
            if color_step_enabled:
                color_resolver = ColorConfigurationResolver(
                    models_root=self.manifest_path.parent / "models",
                    revisions_root=self.manifest_path.parent / ".color_revisions",
                )
                color_revision_store = ColorConfigurationRevisionStore(
                    root=self.manifest_path.parent / ".color_revisions"
                )
                color_approval_service = ColorCalibrationApprovalService()

                def current_color_config(scope):
                    resolved = color_resolver.resolve(scope)
                    return resolved.source_path, resolved.config_sha256

                color_resume_service = ColorCalibrationResumeService(
                    revision_store=color_revision_store,
                    planner=ProcessingPlanner(),
                    current_config_resolver=current_color_config,
                )

                def color_decider(package_path, scope_hash, approved, reviewer, reason):
                    return color_approval_service.decide(
                        package_path,
                        scope_hash,
                        approved=approved,
                        reviewer=reviewer,
                        reason=reason,
                    )

                def color_resume(package_path):
                    current_entries = tuple(
                        (index, dict(self.store.rows[index]))
                        for index, _row in entries
                        if index < len(self.store.rows)
                    )
                    completion = color_resume_service.resume(
                        package_path,
                        plan,
                        current_entries=current_entries,
                        current_manifest_sha=sha256_file(self.manifest_path),
                        operator=operator,
                    )
                    if completion.follow_up_plan is not None:
                        run_store.persist_plan(completion.follow_up_plan)
                    return completion

                def color_rollback(
                    package_path, scope_hash, target_revision_id, rollback_operator, reason
                ):
                    package = load_color_calibration_package(package_path)
                    scope = next(
                        (item for item in package.scopes if item.scope_hash == scope_hash),
                        None,
                    )
                    if scope is None:
                        raise ValueError(f"Unknown color scope: {scope_hash}")
                    return color_revision_store.rollback(
                        scope,
                        target_revision_id,
                        operator=rollback_operator,
                        reason=reason,
                    )
            view_model = ProcessingExecutionViewModel(
                plan,
                engine=engine,
                validator=validator,
                context_provider=current_context,
                store=run_store,
                language=self.language,
                annotation_launcher=annotation_launcher,
                annotation_resume=annotation_resume,
                color_decider=color_decider,
                color_resume=color_resume,
                color_rollback=color_rollback,
            )
        else:
            view_model = ProcessingSummaryViewModel(plan, language=self.language)
        dialog = ProcessingBatchDialog(
            view_model,
            language=self.language,
            cleanup_launcher=self._open_historical_cleanup,
            parent=self,
        )
        if self._embedded:
            self._open_embedded_dialog(dialog)
        else:
            dialog.exec_()

    def _open_historical_cleanup(self) -> None:
        """Open the read-only-first RC-1 assistant for the complete manifest."""
        operator = str(
            os.environ.get("USERNAME")
            or os.environ.get("USER")
            or socket.gethostname()
        )
        try:
            analysis = HistoricalCleanupAnalyzer().analyze(
                self.manifest_path,
                operator=operator,
            )
            view_model = HistoricalCleanupViewModel(
                analysis,
                page_size=50,
            )
        except (HistoricalCleanupError, OSError, ValueError) as exc:
            logger.exception(
                "Historical cleanup audit failed: manifest=%s",
                self.manifest_path,
            )
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return
        HistoricalCleanupDialog(
            view_model,
            language=self.language,
            parent=self,
        ).exec_()

    def _export_portable_selected_indices(
        self,
        selected_indices: set[int],
        *,
        training_options: RetrainingOptions | None = None,
        destination: str | None = None,
    ) -> None:
        """Create a verified ZIP without starting training on the inference PC."""
        if not selected_indices:
            return
        default_name = (
            f"portable_training_{self.product or 'product'}_"
            f"{self.area or 'area'}_{datetime.now():%Y%m%d_%H%M%S}.zip"
        )
        if destination is None:
            destination, _selected_filter = QFileDialog.getSaveFileName(
                self,
                self._text("匯出離線補訓包", "Export offline training package"),
                str(Path.home() / default_name),
                self._text("ZIP 壓縮檔 (*.zip)", "ZIP archive (*.zip)"),
            )
        if not destination:
            return
        if training_options is None:
            settings_dialog = RetrainingSettingsDialog(
                len(selected_indices),
                parent=self,
            )
            if self._embedded:
                self._open_embedded_dialog(
                    settings_dialog,
                    on_accepted=lambda: self._export_portable_selected_indices(
                        selected_indices,
                        training_options=settings_dialog.options(),
                        destination=destination,
                    ),
                )
                return
            if settings_dialog.exec_() != QDialog.Accepted:
                return
            training_options = settings_dialog.options()
        try:
            selected_manifest = self._write_selected_manifest(
                selected_indices,
                scope_indices=sorted(selected_indices),
            )
            handoff_report = export_operator_handoff(
                selected_manifest,
                self.training_data_dir,
                inference_models_dir=self.result_root.parent / "models",
                training_options=training_options.to_dict(),
            )
            if len(handoff_report.targets) != 1:
                raise ValueError(
                    "An offline training package must contain exactly one product/area."
                )
            package_report = export_portable_training_package(
                handoff_report.handoff_path,
                destination,
            )
            if not self._record_submission_audit(
                selected_manifest,
                selected_indices,
                handoff_report,
                action="portable",
            ):
                return
            self._remove_submitted_rows_from_queue(selected_indices)
        except (
            OSError,
            ValueError,
            IndexError,
            csv.Error,
            PortableTrainingPackageError,
        ) as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return
        QMessageBox.information(
            self,
            self.windowTitle(),
            self._text(
                f"離線補訓包已建立：\n{package_report.package_path}\n\n"
                f"檔案數：{package_report.file_count}\n"
                f"大小：{package_report.total_bytes / (1024 * 1024):.1f} MB\n\n"
                "請在訓練電腦執行 import_operator_training.bat 並選擇此 ZIP。",
                f"Offline package created:\n{package_report.package_path}\n\n"
                f"Files: {package_report.file_count}\n"
                f"Size: {package_report.total_bytes / (1024 * 1024):.1f} MB\n\n"
                "Run import_operator_training.bat with this ZIP on the training PC.",
            ),
        )

    def _submit_selected_indices(
        self,
        selected_indices: set[int],
        *,
        training_options: RetrainingOptions | None = None,
        settings_confirmed: bool = False,
    ) -> None:
        """Export an immutable snapshot and open the shared training center."""
        if not selected_indices:
            QMessageBox.warning(
                self,
                self.windowTitle(),
                self._text(
                    "目前沒有選擇任何補訓照片。",
                    "No retraining image is selected.",
                ),
            )
            return
        feedback_only = _is_confirmation_only_submission(
            self.store.rows,
            selected_indices,
        )
        color_only = _is_color_only_submission(
            self.store.rows,
            selected_indices,
        )
        if not feedback_only and not color_only and not settings_confirmed:
            settings_dialog = RetrainingSettingsDialog(
                len(selected_indices),
                parent=self,
            )
            if self._embedded:
                self._open_embedded_dialog(
                    settings_dialog,
                    on_accepted=lambda: self._submit_selected_indices(
                        selected_indices,
                        training_options=settings_dialog.options(),
                        settings_confirmed=True,
                    ),
                )
                return
            if settings_dialog.exec_() != QDialog.Accepted:
                return
            training_options = settings_dialog.options()
            settings_confirmed = True
        output_dir = self.training_data_dir
        if not output_dir.parent.exists():
            QMessageBox.critical(
                self,
                self.windowTitle(),
                self._text(
                    f"訓練中心目錄不存在：\n{output_dir.parent}\n\n請通知系統維護人員。",
                    f"Training center not found; contact engineering:\n{output_dir.parent}",
                ),
            )
            return
        try:
            selected_manifest = self._write_selected_manifest(
                selected_indices,
                scope_indices=sorted(selected_indices),
            )
            report = export_operator_handoff(
                selected_manifest,
                output_dir,
                inference_models_dir=self.result_root.parent / "models",
                training_options=(
                    training_options.to_dict() if training_options is not None else None
                ),
            )
        except (OSError, ValueError, IndexError, csv.Error) as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return
        message = self._text(
            f"本次可訓練：{report.ready_count} 張\n"
            f"累計訓練集：{report.total_ready_count} 張\n"
            f"待人工標註總數：{report.total_pending_count} 張\n"
            f"顏色校正回饋：{report.color_feedback_count} 項\n"
            f"未複核：{report.skipped_count} 張",
            f"Trainable this submission: {report.ready_count}\n"
            f"Total training set: {report.total_ready_count}\n"
            f"Total needing annotation: {report.total_pending_count}\n"
            f"Color calibration feedback: {report.color_feedback_count} item(s)\n"
            f"Unreviewed: {report.skipped_count}",
        )
        if color_only and report.color_feedback_count > 0:
            try:
                color_progress = self._color_feedback_progress_message(report)
            except (OSError, ValueError, csv.Error) as exc:
                QMessageBox.critical(
                    self,
                    self.windowTitle(),
                    self._text(
                        f"顏色資料已匯出，但回讀驗證失敗；待送清單暫不清除：\n{exc}",
                        "Color data was exported, but read-back verification failed; "
                        f"the pending queue was kept:\n{exc}",
                    ),
                )
                return
            message = f"{message}\n{color_progress}"
            if len(report.targets) != 1:
                QMessageBox.information(
                    self,
                    self.windowTitle(),
                    message
                    + self._text(
                        "\n\n所選資料包含多個產品，請依產品分別提交。",
                        "\n\nMultiple targets found; send one selected target at a time.",
                    ),
                )
                return
            if not self._record_submission_audit(
                selected_manifest,
                selected_indices,
                report,
                action="color",
            ):
                return
            self._remove_submitted_rows_from_queue(selected_indices)
            self._show_submission_active(
                message,
                reused_existing=False,
                color_feedback=True,
            )
            self._show_color_submission_result(message)
            return
        if report.ready_count <= 0 and report.total_pending_count <= 0:
            QMessageBox.warning(self, self.windowTitle(), message)
            return
        if len(report.targets) != 1:
            QMessageBox.information(
                self,
                self.windowTitle(),
                message
                + self._text(
                    "\n\n所選資料包含多個產品，請依產品分別提交。",
                    "\n\nMultiple targets found; send one selected target at a time.",
                ),
            )
            return
        submission_action = _submission_action(self.store.rows, selected_indices)
        if not self._record_submission_audit(
            selected_manifest,
            selected_indices,
            report,
            action=submission_action,
        ):
            return
        if feedback_only:
            try:
                update_operator_job_status(
                    report.status_path,
                    state="waiting_feedback",
                    message=("正確案例已加入樣本庫；累積足夠的補框、錯框、錯類別或確認無目標照片後才會開始補訓。"),
                    progress=10,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                QMessageBox.critical(self, self.windowTitle(), str(exc))
                return
            self._remove_submitted_rows_from_queue(selected_indices)
            self._show_submission_active(
                message,
                reused_existing=False,
                feedback_only=True,
            )
            return
        if report.reused_existing:
            self._remove_submitted_rows_from_queue(selected_indices)
            self._show_submission_active(
                message,
                reused_existing=True,
            )
            return
        if self._start_training_center(
            report.handoff_path,
            report.status_path,
            initial_state=("waiting_annotation" if report.pending_count else "queued"),
        ):
            self._remove_submitted_rows_from_queue(selected_indices)
            self._show_submission_active(message, reused_existing=False)

    def _record_submission_audit(
        self,
        selected_manifest: Path,
        selected_indices: set[int],
        report: Any,
        *,
        action: str,
    ) -> bool:
        """Persist a read-only audit record before removing submitted rows."""
        if len(report.targets) != 1:
            return False
        product, area = report.targets[0]
        try:
            record_submission_history(
                self.training_data_dir,
                selected_manifest,
                action=action,
                product=product,
                area=area,
                case_count=len(selected_indices),
                ready_count=report.ready_count,
                pending_count=report.pending_count,
                color_feedback_count=report.color_feedback_count,
                job_id=report.job_id,
                handoff_path=report.handoff_path,
            )
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.critical(
                self,
                self.windowTitle(),
                self._text(
                    f"資料已匯出，但送訓歷史保存失敗；清單暫不清除：\n{exc}",
                    f"Data was exported, but submission history could not be saved; "
                    f"the queue was kept:\n{exc}",
                ),
            )
            return False
        return True

    def _remove_submitted_rows_from_queue(
        self,
        submitted_indices: set[int],
    ) -> None:
        """Remove a successfully handed-off snapshot from the pending queue."""
        try:
            self.store.set_training_selection(submitted_indices, set())
        except (OSError, sqlite3.Error, ValueError, IndexError) as exc:
            QMessageBox.warning(
                self,
                self.windowTitle(),
                self._text(
                    f"補訓已送出，但無法更新待送清單：\n{exc}",
                    f"Retraining was submitted, but the pending queue could not be updated:\n{exc}",
                ),
            )
            return
        self._handed_off_indices.update(submitted_indices)
        self._show_current()

    def _show_submission_active(
        self,
        summary: str,
        *,
        reused_existing: bool,
        feedback_only: bool = False,
        color_feedback: bool = False,
    ) -> None:
        """Keep review context visible after handing an immutable batch off."""
        self._submission_active = True
        self.progress_button.setVisible(True)
        self.export_button.setEnabled(False)
        self.export_button.setText(
            self._text(
                "顏色資料已送出"
                if color_feedback
                else ("已加入樣本庫" if feedback_only else "本批次已送出"),
                "Color data submitted"
                if color_feedback
                else ("Added to sample library" if feedback_only else "Batch submitted"),
            )
        )
        if color_feedback:
            suffix = self._text(
                "顏色覆核資料已加入校正樣本庫；這條流程不會啟動 YOLO 訓練。",
                "Color review data was added to the calibration sample library; "
                "this route does not start YOLO training.",
            )
        elif feedback_only:
            suffix = self._text(
                "這批都是原本辨識正確的照片，已加入樣本庫但不會單獨啟動補訓。"
                "累積足夠的補框、錯框、錯類別或確認無目標照片後再一起訓練。",
                "These already-correct cases were added to the sample library without "
                "starting training. Add a corrected or verified-empty case first.",
            )
        else:
            suffix = self._text(
                "相同批次已在補訓流程中；已送出的照片已移出清單，其他照片仍保留。"
                if reused_existing
                else "已送入補訓流程；已送出的照片已移出清單，其他照片仍保留。",
                "This batch is already in retraining; submitted images were removed "
                "from the queue and the remaining images are preserved."
                if reused_existing
                else "Sent to retraining; submitted images were removed from the queue "
                "and the remaining images are preserved.",
            )
        self.feedback_label.setText(f"{summary.replace(chr(10), '｜')}｜{suffix}")

    def _show_color_submission_result(self, summary: str) -> None:
        """Make a successful color submission visible in the active workflow."""
        title = self._text(
            "顏色校正資料已送出",
            "Color calibration data submitted",
        )
        explanation = self._text(
            "所選照片已從待送清單移除，並加入顏色校正樣本庫。\n"
            "這次不會啟動 YOLO 模型訓練，也不會立即改變門檻。\n"
            "各顏色 Scope 達到樣本門檻後，仍需建立提案、核准並啟用；"
            "後續可在送出紀錄確認這批資料。",
            "The selected images were removed from the pending queue and added to "
            "the color calibration sample library.\nThis does not start YOLO model "
            "training or immediately change a threshold. Each color scope must meet "
            "the evidence policy, then be proposed, approved, and activated. Use "
            "submission history to verify this batch.",
        )
        if not self._embedded:
            QMessageBox.information(self, title, f"{summary}\n\n{explanation}")
            return

        return_page = self.classification_page
        page = QWidget()
        page.setObjectName("ColorSubmissionResultPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(36, 36, 36, 36)
        layout.addStretch()
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(
            "font-size:22pt;font-weight:bold;color:#00796b;padding:12px;"
        )
        layout.addWidget(title_label)
        summary_label = QLabel(f"{summary}\n\n{explanation}")
        summary_label.setObjectName("ColorSubmissionSummary")
        summary_label.setAlignment(Qt.AlignCenter)
        summary_label.setWordWrap(True)
        summary_label.setStyleSheet(
            "font-size:12pt;color:#243447;background:#e0f2f1;"
            "border:1px solid #80cbc4;border-radius:8px;padding:18px;"
        )
        layout.addWidget(summary_label)
        actions = QHBoxLayout()
        history_button = QPushButton(
            self._text("查看已送出紀錄", "View submission history")
        )
        back_button = QPushButton(
            self._text("返回資料複核", "Back to data review")
        )
        back_button.setObjectName("primaryAction")

        def close_result_page() -> None:
            self.workflow_stack.setCurrentWidget(return_page)
            self.workflow_stack.removeWidget(page)
            page.deleteLater()

        history_button.clicked.connect(self._open_submission_history)
        back_button.clicked.connect(close_result_page)
        actions.addStretch()
        actions.addWidget(history_button)
        actions.addWidget(back_button)
        actions.addStretch()
        layout.addLayout(actions)
        layout.addStretch()
        self.workflow_stack.addWidget(page)
        self.workflow_stack.setCurrentWidget(page)

    def _color_feedback_progress_message(self, report: Any) -> str:
        """Verify the exported manifests and explain when they become actionable."""
        manifest_paths = tuple(
            getattr(report, "color_manifest_paths", ()) or ()
        )
        if not manifest_paths:
            handoff_path = getattr(report, "handoff_path", None)
            manifest_paths = (handoff_path,) if handoff_path else ()
        if not manifest_paths:
            raise ValueError("Color feedback export returned no manifest path")
        progress_items = read_color_feedback_progress(manifest_paths)
        if not progress_items:
            raise ValueError("Color feedback manifest contains no persisted evidence")

        policy = CalibrationPolicy()
        lines = [
            self._text(
                "校正證據已回讀確認。每個顏色 Scope 的門檻為：總數 30、OK 5、NG 5。",
                "Calibration evidence was verified. Each color scope requires "
                "30 total, 5 OK, and 5 NG samples.",
            )
        ]
        for item in progress_items[:8]:
            ready = (
                item.total_count >= policy.minimum_total
                and item.ok_count >= policy.minimum_ok
                and item.ng_count >= policy.minimum_ng
            )
            state = self._text("可建立校正提案", "ready for a proposal") if ready else self._text(
                "繼續累積", "keep collecting"
            )
            lines.append(
                f"{item.threshold_key}: {item.total_count}/{policy.minimum_total}｜"
                f"OK {item.ok_count}/{policy.minimum_ok}｜"
                f"NG {item.ng_count}/{policy.minimum_ng}｜{state}"
            )
        if len(progress_items) > 8:
            lines.append(
                self._text(
                    f"另有 {len(progress_items) - 8} 個 Scope，請至處理報告查看。",
                    f"{len(progress_items) - 8} additional scopes are available in the processing report.",
                )
            )
        return "\n".join(lines)

    def _open_update_progress(self) -> None:
        """Show progress in the current workspace without blocking navigation."""
        from app.gui.model_update_status_dialog import ModelUpdateStatusDialog

        dialog = ModelUpdateStatusDialog(
            data_root=self.training_data_dir,
            language=self.language,
            selected_product=self.product,
            selected_area=self.area,
            background_refresh=True,
            parent=self,
        )
        if self._embedded:
            self._open_embedded_dialog(dialog)
        else:
            dialog.exec_()

    def show_progress_page(self) -> None:
        """Public navigation entry used by the main-window progress action."""
        self._open_update_progress()

    def _open_submission_history(self) -> None:
        """Open immutable submitted batches without changing the pending queue."""
        from app.gui.submission_history_dialog import SubmissionHistoryDialog

        dialog = SubmissionHistoryDialog(
            data_root=self.training_data_dir,
            language=self.language,
            selected_product=self.product,
            selected_area=self.area,
            parent=self,
        )
        if self._embedded:
            self._open_embedded_dialog(dialog)
        else:
            dialog.exec_()

    def _report_missed_image(self) -> None:
        """Mark one traceable saved inference result as a missed detection.

        A line operator must never manufacture a training record from an
        arbitrary image: that would lose the exact preprocessing, active
        weights and ordered class contract.  Only images already referenced
        by a persisted result snapshot are accepted here.
        """
        if not self.product or not self.area:
            QMessageBox.warning(
                self,
                self.windowTitle(),
                self._text(
                    "請先選擇產品與站別。",
                    "Select a product and station first.",
                ),
            )
            return
        file_path, _filter = QFileDialog.getOpenFileName(
            self,
            self._text(
                "選擇系統已保存的檢測結果",
                "Select a saved inspection result",
            ),
            str(self.result_root),
            "Images (*.bmp *.jpg *.jpeg *.png *.tif *.tiff *.webp)",
        )
        if not file_path:
            return
        source = Path(file_path)
        if not source.is_file():
            return

        try:
            case_index = _find_saved_case_index(self.store.rows, source)
        except OSError as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return

        if case_index is None:
            QMessageBox.warning(
                self,
                self.windowTitle(),
                self._text(
                    "這張圖片不是系統保存的檢測結果，因此沒有模型版本與前處理紀錄，"
                    "不能安全送訓。\n\n請先回到檢測畫面，用目前模型檢測並保存這張圖片，"
                    "再從結果中選擇「有目標，但系統沒有框」。",
                    "This image is not linked to a saved inspection, so its model and "
                    "preprocessing are unknown. Run it through the current inspection "
                    "model first, then mark the saved result as a missed detection.",
                ),
            )
            return

        row = self.store.rows[case_index]
        if not _row_has_ordered_class_contract(row):
            QMessageBox.warning(
                self,
                self.windowTitle(),
                self._text(
                    "這是缺少類別順序的舊版結果，不能安全補標。\n\n請用目前模型重新檢測這張圖片後再回報。",
                    "This legacy result has no ordered class contract. Re-run the "
                    "image with the current model before reporting it.",
                ),
            )
            return

        answer = QMessageBox.question(
            self,
            self.windowTitle(),
            self._text(
                "請確認：影像中確實有應檢目標，但系統少畫了一個或多個框。\n\n確認後，標註工具會在下一步引導你補框。",
                "Confirm that one or more required targets are present but were not "
                "boxed. The annotation tool will guide you in the next step.",
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        try:
            self.store.set_review(case_index, "false_negative")
        except (OSError, sqlite3.Error, ValueError, IndexError) as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return

        all_index = self.time_range_combo.findData("all")
        if all_index >= 0:
            self.time_range_combo.setCurrentIndex(all_index)
        self.visible_indices = list(range(len(self.store.rows)))
        self.current_index = case_index
        self._show_current()
        QMessageBox.information(
            self,
            self.windowTitle(),
            self._text(
                "已記錄為漏檢。下一步只需在標註工具補上缺少的框。",
                "Missed detection recorded. Next, add the missing box in the annotation tool.",
            ),
        )

    def _write_selected_manifest(
        self,
        selected_indices: set[int] | None = None,
        *,
        include_excluded: bool = False,
        scope_indices: list[int] | None = None,
    ) -> Path:
        """Write the requested rows as one immutable submission snapshot."""
        path = self.manifest_path.with_name(f"{self.manifest_path.stem}_selected{self.manifest_path.suffix}")
        active_indices = self.visible_indices if scope_indices is None else scope_indices
        if any(index < 0 or index >= len(self.store.rows) for index in active_indices):
            raise IndexError("Submission row index out of range")
        if selected_indices is None:
            selected_indices = {
                index for index in active_indices if str(self.store.rows[index].get("training_selected") or "1") != "0"
            }
        rows = [self.store.rows[index] for index in active_indices if include_excluded or index in selected_indices]
        if not rows:
            raise ValueError("No training candidate was selected")
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return path

    def _start_training_center(
        self,
        handoff_path: Path,
        status_path: Path | None = None,
        *,
        initial_state: str = "queued",
    ) -> bool:
        """Start the operator training window and report whether it launched."""
        training_root = self.training_data_dir.parent
        launcher = training_root / "open_operator_training.bat"
        if not launcher.exists():
            QMessageBox.critical(
                self,
                self.windowTitle(),
                self._text(
                    f"找不到訓練啟動器：\n{launcher}",
                    f"Training launcher not found:\n{launcher}",
                ),
            )
            if status_path:
                update_operator_job_status(
                    status_path,
                    state="failed",
                    message="找不到模型更新啟動器",
                    error=str(launcher),
                )
            return False
        result = QProcess.startDetached(
            "cmd.exe",
            ["/c", str(launcher), str(handoff_path), "--background"],
            str(training_root),
        )
        started = result[0] if isinstance(result, tuple) else bool(result)
        if not started:
            QMessageBox.critical(
                self,
                self.windowTitle(),
                self._text("訓練中心啟動失敗", "Failed to start training center"),
            )
            if status_path:
                update_operator_job_status(
                    status_path,
                    state="failed",
                    message="模型更新中心啟動失敗",
                )
            return False
        if status_path:
            process_id = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            update_operator_job_status(
                status_path,
                state=initial_state,
                message="模型更新中心已啟動",
                training_process_id=process_id,
                training_process_host=socket.gethostname(),
            )
        return True


def _is_failure_review_candidate(row: dict[str, Any]) -> bool:
    """Return whether a saved inference row belongs in the failure overview."""
    status = str(row.get("status") or "").strip().upper()
    return status not in {"PASS", "OK", "SUCCESS"}


def run_review_dialog(
    *,
    result_root: str | Path = "Result",
    manifest_path: str | Path = "review_manifest.csv",
    training_data_dir: str | Path | None = None,
    language: str = "zh_TW",
    product: str | None = None,
    area: str | None = None,
    use_legacy_selected_page: bool | None = None,
    use_legacy_review_layout: bool | None = None,
    parent: QWidget | None = None,
) -> int:
    """Open the review dialog, creating a QApplication when run standalone."""
    if training_data_dir is None:
        training_data_dir = load_workspace_paths().training_data
    if use_legacy_selected_page is None:
        use_legacy_selected_page = _environment_flag_enabled(
            LEGACY_SELECTED_PAGE_ENV
        )
    if use_legacy_review_layout is None:
        use_legacy_review_layout = _environment_flag_enabled(
            LEGACY_REVIEW_LAYOUT_ENV
        )
    app = QApplication.instance()
    owns_application = app is None
    if app is None:
        app = QApplication([])
    product, area, accepted = _select_target(result_root, product=product, area=area, language=language, parent=parent)
    if not accepted:
        if owns_application:
            app.quit()
        return QDialog.Rejected
    dialog = ReviewCasesDialog(
        result_root=result_root,
        manifest_path=manifest_path,
        training_data_dir=training_data_dir,
        language=language,
        product=product,
        area=area,
        start_in_overview=True,
        use_legacy_selected_page=use_legacy_selected_page,
        use_legacy_review_layout=use_legacy_review_layout,
        parent=parent,
    )
    result = dialog.exec_()
    if owns_application:
        app.quit()
    return result


def _environment_flag_enabled(name: str) -> bool:
    """Read an opt-in UI rollback flag without changing persisted configuration."""
    return str(os.environ.get(name) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _select_target(
    result_root: str | Path,
    *,
    product: str | None,
    area: str | None,
    language: str,
    parent: QWidget | None,
) -> tuple[str | None, str | None, bool]:
    """Let an OP choose a product/area from a list without typing paths."""
    if product and area:
        return product, area, True
    cases = _with_pass_sampling(
        collect_review_cases(
            result_root,
            include_pass=True,
            product=product,
            area=area,
        )
    )
    targets = sorted(
        {
            (case.product, case.area)
            for case in cases
            if (not product or case.product == product) and (not area or case.area == area)
        }
    )
    if not targets:
        return product, area, True
    if len(targets) == 1:
        return targets[0][0], targets[0][1], True
    labels = [f"{target_product} / {target_area}" for target_product, target_area in targets]
    selected, accepted = QInputDialog.getItem(
        parent,
        "選擇產品／站別" if language.lower().startswith("zh") else "Select Target",
        "複核目標：" if language.lower().startswith("zh") else "Review target:",
        labels,
        0,
        False,
    )
    if not accepted:
        return product, area, False
    index = labels.index(selected)
    return targets[index][0], targets[index][1], True


def _target_manifest_path(path: Path, *, product: str | None, area: str | None) -> Path:
    """Use one decision manifest per target to avoid cross-target overwrites."""
    if not product or not area:
        return path
    safe_product = "".join(character if character.isalnum() or character in "._-" else "_" for character in product)
    safe_area = "".join(character if character.isalnum() or character in "._-" else "_" for character in area)
    return path.with_name(f"{path.stem}_{safe_product}_{safe_area}{path.suffix}")


def _load_handed_off_artifacts(
    training_data_dir: Path,
    review_rows: list[dict[str, str]],
) -> set[str]:
    """Return source evidence already handed to the shared training center."""
    targets = {
        (
            _safe_target_name(str(row.get("product") or "unknown")),
            _safe_target_name(str(row.get("area") or "unknown")),
        )
        for row in review_rows
    }
    artifacts: set[str] = set()
    for product, area in targets:
        target_root = training_data_dir / product / area
        for manifest_path in (
            target_root / "metadata" / "review_dataset_manifest.csv",
            target_root / "review_pending" / "manifest.csv",
            target_root / "color_review" / "feedback.csv",
        ):
            if not manifest_path.is_file():
                continue
            try:
                with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
                    for row in csv.DictReader(handle):
                        for field in ("config_snapshot_path", "source_image"):
                            identity = _artifact_identity(row.get(field, ""))
                            if identity:
                                artifacts.add(identity)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
    return artifacts


def _is_trainable_review(row: dict[str, Any]) -> bool:
    """Return whether a saved verdict represents actionable model feedback."""
    if str(row.get("review_outcome") or "") == "skip":
        return False
    return (
        str(row.get("review_label") or "") in TRAINABLE_REVIEW_LABELS
        and action_route(row) != "none"
    )


def _artifact_identity(value: Any) -> str:
    """Normalize a persisted evidence path for cross-manifest matching."""
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        # Identity comparison does not need to touch the filesystem. On
        # Windows, Path.resolve() performs one or more native stat calls per
        # path and made a 400-row page spend ~400 ms on the UI thread.
        normalized = os.path.abspath(os.path.expanduser(text))
    except (OSError, RuntimeError, ValueError):
        normalized = text
    return os.path.normcase(normalized).replace("\\", "/").casefold()


def _safe_target_name(value: str) -> str:
    """Mirror the training export's target directory normalization."""
    text = str(value or "unknown").strip() or "unknown"
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in text)


def _with_pass_sampling(cases: list[Any]) -> list[Any]:
    """Keep every failure plus a deterministic one-percent PASS audit sample."""
    pass_counts: dict[tuple[str, str], int] = {}
    selected: list[Any] = []
    for case in cases:
        if str(case.status).upper() != "PASS":
            selected.append(case)
            continue
        key = (str(case.product), str(case.area))
        index = pass_counts.get(key, 0)
        pass_counts[key] = index + 1
        if index % PASS_SAMPLE_INTERVAL == 0:
            selected.append(case)
    return selected


def _visible_action_values(
    status: str,
    has_detection: bool,
    *,
    color_failure: bool = False,
    detection_unknown: bool = False,
) -> set[str]:
    """Return only decisions that make sense for the displayed inference result."""
    if color_failure:
        actions = {
            "color_confirmed_ng",
            "color_false_reject",
            "image_quality_issue",
        }
        if has_detection:
            actions.add("color_needs_annotation")
        return actions
    if str(status or "").strip().upper() == "PASS":
        return {
            "confirmed_ok",
            "false_negative",
            "image_quality_issue",
        }
    if detection_unknown:
        return {
            "confirmed_ng",
            "false_positive",
            "needs_annotation",
            "verified_empty",
            "false_negative",
            "image_quality_issue",
        }
    if has_detection:
        return {
            "confirmed_ng",
            "false_positive",
            "needs_annotation",
            "image_quality_issue",
        }
    return {
        "verified_empty",
        "false_negative",
        "image_quality_issue",
    }


def _is_confirmation_only_submission(
    rows: list[dict[str, str]],
    selected_indices: set[int],
) -> bool:
    """Return whether a submission only adds already-correct replay cases."""
    if not selected_indices:
        return False
    if any(index < 0 or index >= len(rows) for index in selected_indices):
        raise IndexError("Submission row index out of range")
    return {str(rows[index].get("review_label") or "").strip().lower() for index in selected_indices} == {
        "confirmed_ng"
    }


def _submission_action(
    rows: list[dict[str, str]], selected_indices: set[int]
) -> str:
    """Return the single queue route used for an operator submission."""
    categories: set[str] = set()
    for index in selected_indices:
        if index < 0 or index >= len(rows):
            raise IndexError("Submission row index out of range")
        row = rows[index]
        route = action_route(row)
        review_label = str(row.get("review_label") or "").strip().lower()
        if route == "color":
            categories.add("color")
        elif route == "both" or review_label in ANNOTATION_LABELS:
            categories.add("annotation")
        else:
            categories.add("direct")
    if len(categories) != 1:
        raise ValueError("A submission must contain exactly one queue category")
    return categories.pop()


def _is_color_only_submission(
    rows: list[dict[str, str]], selected_indices: set[int]
) -> bool:
    """Return whether every selected row is routed only to color calibration."""
    if not selected_indices:
        return False
    if any(index < 0 or index >= len(rows) for index in selected_indices):
        raise IndexError("Submission row index out of range")
    return all(action_route(rows[index]) == "color" for index in selected_indices)


def _row_has_detection(row: dict[str, str]) -> bool:
    """Return whether a review row contains at least one usable detection box."""
    return _row_detection_state(row) == DETECTION_PRESENT


def _row_detection_state(row: dict[str, str]) -> str:
    """Compatibility wrapper around the non-UI evidence classifier."""
    return detection_state(row)


def _row_has_ordered_class_contract(row: dict[str, str]) -> bool:
    """Return whether a review row carries a usable ordered class contract."""
    try:
        names = json.loads(str(row.get("class_names_json") or "[]"))
    except (TypeError, json.JSONDecodeError):
        return False
    return (
        isinstance(names, list)
        and bool(names)
        and all(isinstance(name, str) and name.strip() for name in names)
        and len(set(names)) == len(names)
    )


def _find_saved_case_index(rows: list[dict[str, str]], selected_image: Path) -> int | None:
    """Find the result snapshot that owns a selected persisted image.

    Matching is path-based rather than filename-based so files copied from
    outside the result tree can never inherit another inspection's metadata.
    """
    selected = selected_image.expanduser().resolve(strict=True)
    for index, row in enumerate(rows):
        for field in ("original_path", "preprocessed_path", "annotated_path"):
            raw_path = str(row.get(field) or "").strip()
            if not raw_path:
                continue
            candidate = Path(raw_path).expanduser()
            try:
                if candidate.resolve(strict=True) == selected:
                    return index
            except OSError:
                continue
    return None


def _best_inference_image(row: dict[str, str]) -> str:
    """Return annotated evidence, falling back to the preprocessed image."""
    for field in ("annotated_path", "preprocessed_path", "original_path"):
        value = str(row.get(field) or "")
        if value and Path(value).is_file():
            return value
    return ""
