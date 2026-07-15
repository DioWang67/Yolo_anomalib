"""Button-based review workflow for inference cases used in retraining."""

from __future__ import annotations

import csv
import json
import socket
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QDateTime, QProcess, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDateTimeEdit,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.gui.training_batch_dialog import (
    ANNOTATION_LABELS,
    DIRECT_TRAIN_LABELS,
    TrainingBatchDialog,
)
from tools.collect_review_cases import (
    collect_review_cases,
    timestamp_in_range,
    write_manifest,
)
from tools.export_review_dataset import (
    export_operator_handoff,
    update_operator_job_status,
)

REVIEW_ACTIONS = (
    (
        "confirmed_ng",
        "框數量、位置與類別皆正確（可訓練）",
        "Count, boxes, and classes correct (trainable)",
    ),
    (
        "verified_empty",
        "確認影像中無應檢目標（負樣本）",
        "Confirmed background image (trainable)",
    ),
    ("false_positive", "誤檢／多餘框（需標註）", "False/extra box (annotation)"),
    ("false_negative", "漏檢（需補框）", "Missed detection (annotation)"),
    ("wrong_class", "類別錯誤（需修正）", "Wrong class (annotation)"),
    (
        "image_quality_issue",
        "影像品質異常（不送訓）",
        "Image quality issue (do not train)",
    ),
)

ACTION_COLORS = {
    "confirmed_ng": "#237a3b",
    "verified_empty": "#326b85",
    "false_positive": "#a13b32",
    "false_negative": "#ad641f",
    "wrong_class": "#8754a1",
    "image_quality_issue": "#8b1e1e",
}

PASS_SAMPLE_INTERVAL = 100

OPERATOR_ACTION_LABELS = {
    "confirmed_ng": ("辨識結果正確", "Detection is correct"),
    "verified_empty": ("影像中沒有目標", "No target in image"),
    "false_positive": ("框的位置或數量錯誤", "Box position or count is wrong"),
    "false_negative": ("有目標，但系統沒有框", "Target exists but no box was detected"),
    "wrong_class": ("框的類別錯誤", "Box class is wrong"),
    "uncertain": ("舊版未判定（不送訓）", "Legacy undecided (do not train)"),
    "image_quality_issue": (
        "影像過曝／模糊／遮擋",
        "Overexposed, blurred, or obstructed image",
    ),
}


class ReviewManifestStore:
    """Persist review decisions immediately after every button click."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.path = Path(manifest_path)
        self.rows = self._load_rows()

    def _load_rows(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    def set_review(self, index: int, review_label: str) -> None:
        """Set one standardized review label and save it atomically."""
        if review_label not in {action[0] for action in REVIEW_ACTIONS}:
            raise ValueError(f"Unsupported review label: {review_label}")
        if not 0 <= index < len(self.rows):
            raise IndexError(f"Review row index out of range: {index}")
        for row in self.rows:
            row.setdefault("training_selected", "1")
        self.rows[index]["review_label"] = review_label
        self.rows[index]["training_selected"] = "0" if review_label in {"uncertain", "image_quality_issue"} else "1"
        self._save()

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
        for row in self.rows:
            row.setdefault("training_selected", "1")
        for index in candidate_indices:
            self.rows[index]["training_selected"] = "1" if index in selected_indices else "0"
        self._save()

    def first_pending_index(self) -> int:
        """Return the first unreviewed row, or zero when all rows are reviewed."""
        for index, row in enumerate(self.rows):
            if not str(row.get("review_label") or "").strip():
                return index
        return 0

    def reviewed_count(self) -> int:
        """Return how many rows already contain an operator decision."""
        return sum(bool(str(row.get("review_label") or "").strip()) for row in self.rows)

    def _save(self) -> None:
        if not self.rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        fieldnames = list(self.rows[0])
        try:
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.rows)
            temporary.replace(self.path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


class ReviewCasesDialog(QDialog):
    """Show saved inference images and guide an operator through each decision."""

    def __init__(
        self,
        *,
        result_root: str | Path,
        manifest_path: str | Path,
        training_data_dir: str | Path,
        language: str = "zh_TW",
        product: str | None = None,
        area: str | None = None,
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
        self._submission_active = False
        self._filter_state_path = self.manifest_path.with_name(f".{self.manifest_path.stem}_filter.json")

        cases = _with_pass_sampling(collect_review_cases(self.result_root, include_pass=True))
        if product:
            cases = [case for case in cases if case.product == product]
        if area:
            cases = [case for case in cases if case.area == area]
        write_manifest(cases, self.manifest_path)
        self.store = ReviewManifestStore(self.manifest_path)
        self._reconcile_handed_off_selection()
        self.current_index = self.store.first_pending_index()
        self.visible_indices = list(range(len(self.store.rows)))

        self.setWindowTitle(self._text("產線模型補訓｜資料複核", "Production Retraining | Data Review"))
        self.resize(1250, 820)
        self._build_ui()
        self._restore_time_filter()
        self._apply_time_filter()

    def _text(self, zh: str, en: str) -> str:
        return zh if str(self.language).lower().startswith("zh") else en

    def _reconcile_handed_off_selection(self) -> None:
        """Exclude legacy rows that already exist in ready or pending data."""
        handed_off_artifacts = _load_handed_off_artifacts(
            self.training_data_dir,
            self.store.rows,
        )
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
        if not handed_off_indices:
            return
        try:
            self.store.set_training_selection(handed_off_indices, set())
        except (OSError, ValueError, IndexError):
            return

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        instruction = QLabel(
            self._text(
                "模型補訓第 1 階段：逐張確認辨識結果；選擇答案後會自動前往下一張。",
                "Retraining stage 1: review each result; the next case opens automatically.",
            )
        )
        instruction.setWordWrap(True)
        instruction.setStyleSheet(
            "QLabel { background: #243447; color: white; padding: 10px; font-size: 11pt; font-weight: bold; }"
        )
        layout.addWidget(instruction)
        layout.addLayout(self._build_time_filter())
        self.progress_label = QLabel()
        self.question_label = QLabel()
        self.question_label.setStyleSheet("font-size: 13pt; font-weight: bold;")
        self.details_label = QLabel()
        self.details_label.setWordWrap(True)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.question_label)
        layout.addWidget(self.details_label)

        image_layout = QHBoxLayout()
        self.original_label = self._image_panel(self._text("原圖", "Original"))
        self.annotated_label = self._image_panel(self._text("推理結果", "Inference"))
        image_layout.addWidget(self.original_label)
        image_layout.addWidget(self.annotated_label)
        layout.addLayout(image_layout, 1)

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
            button.clicked.connect(lambda _checked=False, selected=value: self._set_review(selected))
            self.review_buttons[value] = button
            self.review_layout.addWidget(button, action_index // 3, action_index % 3)
        layout.addLayout(self.review_layout)

        self.feedback_label = QLabel()
        self.feedback_label.setMinimumHeight(32)
        self.feedback_label.setAlignment(Qt.AlignCenter)
        self.feedback_label.setStyleSheet(
            "QLabel { color: #1b5e20; background: #e8f5e9; font-size: 11pt; font-weight: bold; padding: 5px; }"
        )
        layout.addWidget(self.feedback_label)

        navigation_layout = QHBoxLayout()
        previous_button = QPushButton(self._text("上一張", "Previous"))
        previous_button.clicked.connect(lambda: self._move(-1))
        next_button = QPushButton(self._text("下一張", "Next"))
        next_button.clicked.connect(lambda: self._move(1))
        self.export_button = QPushButton(
            self._text(
                "複核完成：開啟已選擇清單",
                "Review complete: open selected queue",
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
                "從已保存結果回報漏檢",
                "Report miss from saved result",
            )
        )
        report_miss_button.clicked.connect(self._report_missed_image)
        self.batch_preview_button = QPushButton(self._text("已選擇清單", "Selected queue"))
        self.batch_preview_button.setMinimumHeight(42)
        self.batch_preview_button.setStyleSheet(
            "QPushButton { background: #2563a6; color: white; font-weight: bold; "
            "padding: 8px 14px; } QPushButton:disabled { background: #9aa0a6; }"
        )
        self.batch_preview_button.clicked.connect(self._open_selected_training_queue)
        self.progress_button = QPushButton(self._text("查看補訓進度", "View retraining progress"))
        self.progress_button.setVisible(False)
        self.progress_button.clicked.connect(self._open_update_progress)
        close_button = QPushButton(self._text("關閉", "Close"))
        close_button.clicked.connect(self.accept)
        navigation_layout.addWidget(previous_button)
        navigation_layout.addWidget(next_button)
        navigation_layout.addStretch()
        navigation_layout.addWidget(report_miss_button)
        navigation_layout.addWidget(self.batch_preview_button)
        navigation_layout.addWidget(self.progress_button)
        navigation_layout.addWidget(self.export_button)
        navigation_layout.addWidget(close_button)
        layout.addLayout(navigation_layout)

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
        """Restrict navigation and export to the selected time range."""
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
        self.visible_indices = [
            index
            for index, row in enumerate(self.store.rows)
            if timestamp_in_range(
                str(row.get("timestamp") or ""),
                start_time=start_time,
                end_time=end_time,
            )
        ]
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

    def _restore_time_filter(self) -> None:
        """Restore the last product/station review range when available."""
        state: dict[str, Any] = {}
        try:
            payload = json.loads(self._filter_state_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                state = payload
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
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

    def _save_time_filter(self) -> None:
        """Persist the current review range atomically for this target."""
        start_time, end_time = self._selected_time_bounds()
        payload = {
            "schema_version": 1,
            "mode": str(self.time_range_combo.currentData() or "all"),
            "start": start_time.isoformat() if start_time else "",
            "end": end_time.isoformat() if end_time else "",
        }
        temporary = self._filter_state_path.with_name(f".{self._filter_state_path.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(self._filter_state_path)
        except OSError:
            pass
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _image_panel(self, title: str) -> QLabel:
        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(480, 420)
        label.setStyleSheet("QLabel { background: #202124; color: white; }")
        return label

    def _show_current(self) -> None:
        total = len(self.visible_indices)
        reviewed = sum(
            bool(str(self.store.rows[index].get("review_label") or "").strip()) for index in self.visible_indices
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
            for button in self.review_buttons.values():
                button.setEnabled(False)
                button.setVisible(False)
            self._update_submit_button()
            return

        if self.current_index not in self.visible_indices:
            self.current_index = self.visible_indices[0]
        visible_position = self.visible_indices.index(self.current_index)
        row = self.store.rows[self.current_index]
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
        has_detection = _row_has_detection(row)
        self.question_label.setText(
            self._text(
                "框的位置、數量與類別是否正確？" if has_detection else "系統沒有畫框，影像中是否有應檢目標？",
                "Are all boxes and classes correct?"
                if has_detection
                else "No box was detected. Is there a target in the image?",
            )
        )
        detection_notice = ""
        if not has_detection:
            detection_notice = self._text(
                "未偵測到任何框；影像中應有目標請選擇「漏檢」，確認沒有應檢目標才可選擇負樣本。\n",
                "No boxes were detected. Select missed detection when a target exists; use background only when no target should be present.\n",
            )
        self.details_label.setText(
            detection_notice + f"{row.get('product', '')}/{row.get('area', '')}  "
            f"{row.get('timestamp', '')}  [{current_label}]"
        )
        visible_actions = _visible_action_values(has_detection)
        for button in self.review_buttons.values():
            self.review_layout.removeWidget(button)
        visible_index = 0
        for value, button in self.review_buttons.items():
            button.setVisible(value in visible_actions)
            button.setEnabled(value in visible_actions)
            button.setChecked(value == current_value)
            if value in visible_actions:
                self.review_layout.addWidget(button, 0, visible_index)
                visible_index += 1
        self._set_pixmap(self.original_label, row.get("original_path", ""))
        self._set_pixmap(self.annotated_label, _best_inference_image(row))
        self._update_submit_button()

    def _update_submit_button(self) -> None:
        """Expose the next step only after the selected queue is complete."""
        eligible_labels = DIRECT_TRAIN_LABELS | ANNOTATION_LABELS
        queue_count = sum(
            str(row.get("review_label") or "") in eligible_labels and str(row.get("training_selected") or "1") != "0"
            for row in self.store.rows
        )
        self.batch_preview_button.setEnabled(queue_count > 0)
        self.batch_preview_button.setText(
            self._text(
                f"已選擇清單（{queue_count}）",
                f"Selected queue ({queue_count})",
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
                ("複核完成：開啟已選擇清單" if selected_count > 0 else "本範圍沒有可送訓影像")
                if remaining == 0
                else f"尚有 {remaining} 張未確認",
                ("Review complete: open selected queue" if selected_count > 0 else "No trainable image in this range")
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
        except (OSError, ValueError, IndexError) as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return
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

    def _move_to_next_pending(self) -> None:
        total = len(self.visible_indices)
        if total == 0:
            self._show_current()
            return
        current_position = self.visible_indices.index(self.current_index)
        for offset in range(1, total + 1):
            index = self.visible_indices[(current_position + offset) % total]
            if not str(self.store.rows[index].get("review_label") or "").strip():
                self.current_index = index
                self._show_current()
                return
        self._show_current()

    def _move(self, offset: int) -> None:
        self.feedback_label.clear()
        if self.visible_indices:
            current_position = self.visible_indices.index(self.current_index)
            self.current_index = self.visible_indices[(current_position + offset) % len(self.visible_indices)]
            self._show_current()

    def _open_selected_training_queue(self) -> None:
        """Show the target-wide persistent queue and run one explicit action."""
        eligible_labels = DIRECT_TRAIN_LABELS | ANNOTATION_LABELS
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
        if dialog.exec_() != QDialog.Accepted:
            return
        candidate_indices = {index for index, _row in entries}
        selected_indices = dialog.selected_indices()
        try:
            self.store.set_training_selection(candidate_indices, selected_indices)
        except (OSError, ValueError, IndexError) as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return
        action_indices = dialog.action_selected_indices()
        self._show_current()
        if action_indices:
            self._submit_selected_indices(action_indices)

    def _submit_selected_indices(
        self,
        selected_indices: set[int],
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
            )
        except (OSError, ValueError, IndexError, csv.Error) as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return
        message = self._text(
            f"本次可訓練：{report.ready_count} 張\n"
            f"累計訓練集：{report.total_ready_count} 張\n"
            f"待人工標註總數：{report.total_pending_count} 張\n"
            f"未複核：{report.skipped_count} 張",
            f"Trainable this submission: {report.ready_count}\n"
            f"Total training set: {report.total_ready_count}\n"
            f"Total needing annotation: {report.total_pending_count}\n"
            f"Unreviewed: {report.skipped_count}",
        )
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

    def _remove_submitted_rows_from_queue(
        self,
        submitted_indices: set[int],
    ) -> None:
        """Remove a successfully handed-off snapshot from the pending queue."""
        try:
            self.store.set_training_selection(submitted_indices, set())
        except (OSError, ValueError, IndexError) as exc:
            QMessageBox.warning(
                self,
                self.windowTitle(),
                self._text(
                    f"補訓已送出，但無法更新已選擇清單：\n{exc}",
                    f"Retraining was submitted, but the selected queue could not be updated:\n{exc}",
                ),
            )
            return
        self._show_current()

    def _show_submission_active(
        self,
        summary: str,
        *,
        reused_existing: bool,
        feedback_only: bool = False,
    ) -> None:
        """Keep review context visible after handing an immutable batch off."""
        self._submission_active = True
        self.progress_button.setVisible(True)
        self.export_button.setEnabled(False)
        self.export_button.setText(
            self._text(
                "已加入樣本庫" if feedback_only else "本批次已送出",
                "Added to sample library" if feedback_only else "Batch submitted",
            )
        )
        if feedback_only:
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

    def _open_update_progress(self) -> None:
        """Open the shared progress view without discarding review context."""
        from app.gui.model_update_status_dialog import ModelUpdateStatusDialog

        ModelUpdateStatusDialog(
            data_root=self.training_data_dir,
            language=self.language,
            selected_product=self.product,
            selected_area=self.area,
            parent=self,
        ).exec_()

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
        except (OSError, ValueError, IndexError) as exc:
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
            ["/c", str(launcher), str(handoff_path)],
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


def run_review_dialog(
    *,
    result_root: str | Path = "Result",
    manifest_path: str | Path = "review_manifest.csv",
    training_data_dir: str | Path = "../Yolo11_auto_train/data",
    language: str = "zh_TW",
    product: str | None = None,
    area: str | None = None,
    parent: QWidget | None = None,
) -> int:
    """Open the review dialog, creating a QApplication when run standalone."""
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
        parent=parent,
    )
    result = dialog.exec_()
    if owns_application:
        app.quit()
    return result


def _select_target(
    result_root: str | Path,
    *,
    product: str | None,
    area: str | None,
    language: str,
    parent: QWidget | None,
) -> tuple[str | None, str | None, bool]:
    """Let an OP choose a product/area from a list without typing paths."""
    cases = _with_pass_sampling(collect_review_cases(result_root, include_pass=True))
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


def _artifact_identity(value: Any) -> str:
    """Normalize a persisted evidence path for cross-manifest matching."""
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        normalized = Path(text).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        normalized = Path(text)
    return str(normalized).replace("\\", "/").casefold()


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


def _visible_action_values(has_detection: bool) -> set[str]:
    """Return only decisions that make sense for the displayed inference result."""
    if has_detection:
        return {
            "confirmed_ng",
            "false_positive",
            "wrong_class",
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


def _row_has_detection(row: dict[str, str]) -> bool:
    """Return whether a review row contains at least one usable detection box."""
    try:
        detections = json.loads(str(row.get("detections_json") or "[]"))
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(detections, list):
        return False
    return any(
        isinstance(detection, dict) and isinstance(detection.get("bbox"), list) and len(detection["bbox"]) == 4
        for detection in detections
    )


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
