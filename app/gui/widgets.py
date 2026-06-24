from __future__ import annotations

import os
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.gui.i18n import normalize_language, tr
from core.services.results.customer_message import build_customer_message
from core.services.results.position_summary import (
    format_fixture_shift_hint,
    summarize_position_records,
)

if TYPE_CHECKING:
    from core.types import DetectionResult


class BigStatusLabel(QLabel):
    """Large, colorful status indicator for PASS/FAIL results."""

    def __init__(self) -> None:
        super().__init__("READY")
        self.setAlignment(Qt.AlignCenter)
        self.setFont(QFont("Arial Black", 22, QFont.Bold))
        self.setStyleSheet(
            """
            QLabel {
                background-color: #e5e7eb;
                color: #374151;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 14px;
                min-height: 54px;
            }
            """
        )

    def set_status(self, status: str) -> None:
        """Update the label appearance based on status."""
        status = status.upper()
        if status == "PASS":
            self.setText("PASS")
            self.setStyleSheet(
                """
                QLabel {
                    background-color: #16794c;
                    color: white;
                    border: 1px solid #12643f;
                    border-radius: 8px;
                    padding: 14px;
                }
                """
            )
        elif status in {"FAIL", "DETECTION_FAIL"}:
            self.setText("DETECTION FAIL" if status == "DETECTION_FAIL" else "FAIL")
            self.setStyleSheet(
                """
                QLabel {
                    background-color: #b42318;
                    color: white;
                    border: 1px solid #971d14;
                    border-radius: 8px;
                    padding: 14px;
                }
                """
            )
        elif status in {"ERROR", "INFERENCE_ERROR"}:
            self.setText("INFERENCE ERROR" if status == "INFERENCE_ERROR" else "ERROR")
            self.setStyleSheet(
                """
                QLabel {
                    background-color: #f59e0b;
                    color: #111827;
                    border: 1px solid #b45309;
                    border-radius: 8px;
                    padding: 14px;
                }
                """
            )
        else:
            self.setText(status if status else "READY")
            self.setStyleSheet(
                """
                QLabel {
                    background-color: #e5e7eb;
                    color: #374151;
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                    padding: 14px;
                }
                """
            )


class FailReasonLabel(QLabel):
    """One-line summary of why the last result failed."""

    _STYLE_FAIL = (
        "background-color: #fff3cd; color: #856404;"
        "border: 1px solid #ffc107; border-radius: 4px; padding: 4px 8px;"
        "font-size: 10pt;"
    )
    _STYLE_CLEAR = "color: transparent; border: none; padding: 4px 8px;"

    def __init__(self) -> None:
        super().__init__("")
        self._language = "en"
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setStyleSheet(self._STYLE_CLEAR)

    def set_language(self, language: str) -> None:
        """Set display language for future updates."""
        self._language = normalize_language(language)

    def update_from_result(self, result: DetectionResult) -> None:
        """Extract top-level FAIL reasons and display them."""
        if result.status not in {"FAIL", "DETECTION_FAIL", "INFERENCE_ERROR"}:
            self.clear_reason()
            return

        if result.status == "INFERENCE_ERROR":
            self.setText(result.error or "Inference/backend runtime error")
            self.setStyleSheet(self._STYLE_FAIL)
            return

        reasons: list[str] = []
        metadata = result.metadata or {}

        if result.error:
            reasons.append(str(result.error))

        decision = metadata.get("decision")
        decision_reasons: list[str] = []
        if isinstance(decision, dict):
            decision_reasons = [str(reason) for reason in (decision.get("reasons") or [])]
        if "BOARD_ALIGNMENT" in decision_reasons:
            detail = tr(self._language, "position_offset")
            alignment_quality = metadata.get("alignment_quality")
            if isinstance(alignment_quality, dict):
                issues = alignment_quality.get("issues") or []
                if issues:
                    issue_text = ", ".join(_alignment_issue_label(str(i), self._language) for i in issues[:2])
                    detail = f"{detail}: {issue_text}"
            reasons.append(detail)

        missing = result.missing_items or []
        if missing:
            suffix = "..." if len(missing) > 3 else ""
            reasons.append(
                f"{tr(self._language, 'missing')}: {', '.join(str(i) for i in missing[:3])}{suffix}"
            )

        color_check = result.color_check or {}
        if color_check and not color_check.get("is_ok", True):
            bad = [
                c.get("class_name", "?")
                for c in (color_check.get("items") or [])
                if not c.get("is_ok", True)
            ]
            reasons.append(
                f"{tr(self._language, 'color_error')}: {', '.join(str(i) for i in bad[:3])}"
                if bad
                else tr(self._language, "color_error")
            )

        seq = result.sequence_check or {}
        if seq and not seq.get("is_ok", True):
            reason_key = seq.get("reason", "")
            reasons.append(
                tr(self._language, "sequence_order_error")
                if reason_key == "order_mismatch"
                else tr(self._language, "sequence_length_error")
            )

        position_summary = summarize_position_records(
            [{"label": item.label, **item.metadata} for item in (result.items or [])]
        )
        if position_summary.fail_count > 0:
            labels = [issue.label for issue in position_summary.issues]
            suffix = "..." if len(labels) > 3 else ""
            reasons.append(
                f"{tr(self._language, 'position_offset')}: {', '.join(labels[:3])}{suffix}"
            )
            fixture_hint = format_fixture_shift_hint(position_summary)
            if fixture_hint:
                reasons.append(fixture_hint)

        self.setText(" | ".join(reasons) if reasons else tr(self._language, "unknown_reason"))
        self.setStyleSheet(self._STYLE_FAIL)

    def clear_reason(self) -> None:
        """Hide the fail reason row."""
        self.setText("")
        self.setStyleSheet(self._STYLE_CLEAR)


class OperatorGuidanceCard(QFrame):
    """Customer-facing guidance card with next action."""

    _STYLES = {
        "success": (
            "QFrame { background-color: #edf7ed; border: 1px solid #7bc47f; "
            "border-radius: 8px; padding: 8px; }"
        ),
        "warning": (
            "QFrame { background-color: #fff7e6; border: 1px solid #f0b429; "
            "border-radius: 8px; padding: 8px; }"
        ),
        "danger": (
            "QFrame { background-color: #fff1f0; border: 1px solid #d9534f; "
            "border-radius: 8px; padding: 8px; }"
        ),
    }

    def __init__(self) -> None:
        super().__init__()
        self._language = "en"
        self._last_result: DetectionResult | None = None
        self._headline = QLabel()
        self._action = QLabel()
        self._details = QLabel()
        self._setup_ui()
        self.show_waiting()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self._headline.setFont(QFont("Microsoft JhengHei", 14, QFont.Bold))
        self._headline.setWordWrap(True)

        self._action.setFont(QFont("Microsoft JhengHei", 10, QFont.Bold))
        self._action.setWordWrap(True)

        self._details.setFont(QFont("Microsoft JhengHei", 9))
        self._details.setWordWrap(True)

        layout.addWidget(self._headline)
        layout.addWidget(self._action)
        layout.addWidget(self._details)
        self.setLayout(layout)

    def set_language(self, language: str) -> None:
        """Update card language and re-render the current result."""
        self._language = normalize_language(language)
        if self._last_result is None:
            self.show_waiting()
        else:
            self.update_from_result(self._last_result)

    def show_waiting(self) -> None:
        """Render the idle state."""
        self.show_message(
            tr(self._language, "waiting_detection"),
            tr(self._language, "place_part_start"),
            "warning",
            [],
        )

    def update_from_result(self, result: DetectionResult) -> None:
        """Render guidance for the latest result."""
        self._last_result = result
        if self._language == "zh":
            message = build_customer_message(result)
            self.show_message(
                message.headline,
                message.action,
                message.severity,
                message.details,
            )
            return

        missing = result.missing_items or []
        over = result.over_items or []
        unexpected = result.unexpected_items or []
        if result.status == "PASS":
            self.show_message("PASS", "Continue production.", "success", [])
            return
        if result.status == "INFERENCE_ERROR":
            self.show_message(
                "INFERENCE ERROR",
                "Check model/backend status and run inspection again.",
                "danger",
                [result.error] if result.error else [],
            )
            return

        details: list[str] = []
        if missing:
            details.append(f"{tr(self._language, 'missing')}: {', '.join(str(i) for i in missing)}")
        if unexpected:
            details.append(
                f"{tr(self._language, 'unexpected_items')}: {', '.join(str(i) for i in unexpected)}"
            )
        self.show_message(
            "DETECTION FAIL",
            "Review listed defects, reposition the product, then inspect again.",
            "danger",
            details,
        )

    def show_message(
        self,
        headline: str,
        action: str,
        severity: str,
        details: list[str],
    ) -> None:
        """Display a guidance message."""
        self.setStyleSheet(self._STYLES.get(severity, self._STYLES["warning"]))
        self._headline.setText(headline)
        self._action.setText(action)
        self._details.setText("\n".join(f"- {detail}" for detail in details) if details else "")


class SessionStatsWidget(QGroupBox):
    """Current-shift PASS/FAIL counters and consecutive-fail alert."""

    CONSECUTIVE_FAIL_ALERT: int = 3
    consecutive_fail_reached = pyqtSignal(int)

    _STYLE_NORMAL = "color: #155724; background: #d4edda; border-radius: 3px; padding: 2px 6px;"
    _STYLE_ALERT = "color: #721c24; background: #f8d7da; border-radius: 3px; padding: 2px 6px;"

    def __init__(self) -> None:
        super().__init__("Shift Stats")
        self._language = "en"
        self._pass = 0
        self._fail = 0
        self._consecutive_fails = 0
        self._build_ui()
        self.set_language(self._language)

    def _build_ui(self) -> None:
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)

        lbl_font = QFont("Microsoft JhengHei", 9)
        num_font = QFont("Consolas", 11, QFont.Bold)

        def _header(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setFont(lbl_font)
            lbl.setStyleSheet("color: #6c757d;")
            lbl.setAlignment(Qt.AlignCenter)
            return lbl

        def _value() -> QLabel:
            lbl = QLabel("0")
            lbl.setFont(num_font)
            lbl.setAlignment(Qt.AlignCenter)
            return lbl

        self._pass_header = _header("PASS")
        self._fail_header = _header("FAIL")
        self._yield_header = _header("Yield")
        self._consec_header = _header("Consecutive NG")

        grid.addWidget(self._pass_header, 0, 0)
        grid.addWidget(self._fail_header, 0, 1)
        grid.addWidget(self._yield_header, 0, 2)
        grid.addWidget(self._consec_header, 0, 3)

        self._pass_lbl = _value()
        self._fail_lbl = _value()
        self._yield_lbl = _value()
        self._consec_lbl = _value()

        self._pass_lbl.setStyleSheet("color: #28a745;")
        self._fail_lbl.setStyleSheet("color: #dc3545;")

        grid.addWidget(self._pass_lbl, 1, 0)
        grid.addWidget(self._fail_lbl, 1, 1)
        grid.addWidget(self._yield_lbl, 1, 2)
        grid.addWidget(self._consec_lbl, 1, 3)

        self.setLayout(grid)

    @property
    def consecutive_fails(self) -> int:
        """Read-only view of the current consecutive-FAIL counter."""
        return self._consecutive_fails

    def set_language(self, language: str) -> None:
        """Update labels and tooltip language."""
        self._language = normalize_language(language)
        self.setTitle(tr(self._language, "shift_stats"))
        self._yield_header.setText(tr(self._language, "yield"))
        self._consec_header.setText(tr(self._language, "consecutive_ng"))
        self._refresh()

    def record_result(self, status: str) -> None:
        """Update counters based on detection status."""
        if status == "PASS":
            self._pass += 1
            self._consecutive_fails = 0
        elif status in {"FAIL", "DETECTION_FAIL"}:
            self._fail += 1
            self._consecutive_fails += 1
        self._refresh()

    def reset_session(self) -> None:
        """Reset all counters to zero."""
        self._pass = 0
        self._fail = 0
        self._consecutive_fails = 0
        self._refresh()

    def _refresh(self) -> None:
        total = self._pass + self._fail
        yield_pct = f"{self._pass / total * 100:.1f}%" if total else "0.0%"

        self._pass_lbl.setText(str(self._pass))
        self._fail_lbl.setText(str(self._fail))
        self._yield_lbl.setText(yield_pct)
        self._consec_lbl.setText(str(self._consecutive_fails))

        if self._consecutive_fails >= self.CONSECUTIVE_FAIL_ALERT:
            self._consec_lbl.setStyleSheet(self._STYLE_ALERT)
            self._consec_lbl.setToolTip(
                tr(self._language, "consecutive_fail").format(count=self._consecutive_fails)
            )
            self.consecutive_fail_reached.emit(self._consecutive_fails)
        else:
            self._consec_lbl.setStyleSheet(self._STYLE_NORMAL)
            self._consec_lbl.setToolTip("")


class ImageViewer(QLabel):
    """Simple image container with placeholder text."""

    def __init__(self, title: str = "image") -> None:
        super().__init__()
        self._title = title
        self._language = "en"
        self._load_token = 0
        self._last_image_path: str | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setMinimumSize(300, 300)
        self.setStyleSheet(
            """
            QLabel {
                border: 1px solid #111827;
                border-radius: 6px;
                background-color: #0b1120;
                color: #9ca3af;
                font-size: 11pt;
            }
            """
        )
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)
        self._set_waiting_text()

    def set_language(self, language: str) -> None:
        """Update placeholder language."""
        self._language = normalize_language(language)
        if self.pixmap() is None:
            self._set_waiting_text()

    def clear(self) -> None:
        """Cancel in-flight deferred loads before clearing."""
        self._load_token += 1
        self._last_image_path = None
        super().clear()
        self._set_waiting_text()

    def set_title(self, title: str) -> None:
        """Update placeholder title without disturbing a loaded pixmap."""
        self._title = title
        if self.pixmap() is None:
            self._set_waiting_text()

    def set_image(self, image_path: str) -> None:
        """Load and display an image from disk."""
        self._load_token += 1
        token = self._load_token

        if os.path.exists(image_path):
            self._last_image_path = image_path
            self.setText("Loading image...")
            QTimer.singleShot(0, lambda p=image_path, t=token: self._load_and_display(p, t))
        else:
            self.setText(f"Unable to load {self._title}")

    def _load_and_display(self, image_path: str, token: int) -> None:
        """Read file from disk and scale for display."""
        if token != self._load_token:
            return
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled)
        else:
            self.setText(f"Unable to load {self._title}")

    def display_image(self, image: QImage | np.ndarray) -> None:
        """Display a QImage or BGR numpy array."""
        try:
            if isinstance(image, QImage):
                self.setPixmap(
                    QPixmap.fromImage(image).scaled(
                        self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                )
            elif isinstance(image, np.ndarray) and image.size > 0:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(
                    rgb_image.data,
                    w,
                    h,
                    bytes_per_line,
                    QImage.Format_RGB888,
                )
                self.setPixmap(
                    QPixmap.fromImage(qt_image).scaled(
                        self.size(), Qt.KeepAspectRatio, Qt.FastTransformation
                    )
                )
            else:
                self.clear()
                self.setText("Unable to display image")
        except Exception:
            self.clear()
            self.setText("Unable to display image")

    def _set_waiting_text(self) -> None:
        if self._language == "zh":
            self.setText(f"{tr(self._language, 'waiting_detection')}: {self._title}")
        else:
            self.setText(f"Waiting for {self._title}")


class ResultDisplayWidget(QWidget):
    """Scrollable textual representation of detection outcomes."""

    def __init__(self) -> None:
        super().__init__()
        self._language = "en"
        self._last_result: DetectionResult | None = None
        self._title = QLabel()
        self._result_text = QTextEdit()
        self._setup_ui()
        self.set_language(self._language)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()

        self._title.setFont(QFont("Microsoft JhengHei", 12, QFont.Bold))
        self._title.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")

        self._result_text.setFont(QFont("Consolas", 9))
        self._result_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
            }
            """
        )
        self._result_text.setReadOnly(True)

        layout.addWidget(self._title)
        layout.addWidget(self._result_text)
        self.setLayout(layout)

    def set_language(self, language: str) -> None:
        """Update detail panel language."""
        self._language = normalize_language(language)
        self._title.setText(tr(self._language, "result_details"))
        if self._last_result is not None:
            self.update_result(self._last_result)

    def update_result(self, result: DetectionResult) -> None:
        """Render detection result details into the text panel."""
        self._last_result = result

        product = result.product or "N/A"
        area = result.area or "N/A"
        inference_type = result.inference_type or "N/A"
        ckpt_name = os.path.basename(result.ckpt_path) if result.ckpt_path else "N/A"
        missing = result.missing_items or []
        over = result.over_items or []
        unexpected = result.unexpected_items or []

        lines: list[str] = []
        lines.extend(self._field_decision_lines(result))

        lines.append(f"\n=== {tr(self._language, 'basic_info')} ===")
        lines.append(f"{tr(self._language, 'status')}: {result.status}")
        lines.append(f"{tr(self._language, 'product_area')}: {product} / {area}")
        lines.append(f"{tr(self._language, 'type')}: {inference_type}")
        lines.append(f"{tr(self._language, 'model_file')}: {ckpt_name}")
        lines.append(f"{tr(self._language, 'latency')}: {result.latency * 1000:.1f} ms")

        lines.append(f"\n=== {tr(self._language, 'key_counts')} ===")
        lines.append(f"{tr(self._language, 'detections')}: {len(result.items)}")
        if result.anomaly_score is not None:
            lines.append(f"{tr(self._language, 'anomaly_score')}: {result.anomaly_score}")
        lines.append(f"{tr(self._language, 'missing_count')}: {len(missing)}")
        lines.append(f"{tr(self._language, 'extra_count')}: {len(over)}")
        if isinstance(unexpected, (list, tuple)):
            lines.append(f"{tr(self._language, 'unexpected_count')}: {len(unexpected)}")

        self._append_sequence_lines(lines, result)
        self._append_color_lines(lines, result)
        self._append_list_section(lines, "missing_list", missing)
        self._append_list_section(lines, "extra_items", over)
        self._append_list_section(lines, "unexpected_items", unexpected)
        self._append_position_lines(lines, result)
        self._append_decision_lines(lines, result)
        self._append_alignment_lines(lines, result)
        self._append_detection_lines(lines, result)

        lines.append("===================")
        self._result_text.setPlainText("\n".join(lines))

    def _field_decision_lines(self, result: DetectionResult) -> list[str]:
        if self._language == "zh":
            message = build_customer_message(result)
            lines = [
                f"=== {tr(self._language, 'field_decision')} ===",
                f"{tr(self._language, 'conclusion')}: {message.headline}",
                f"{tr(self._language, 'next_action')}: {message.action}",
            ]
            lines.extend(f"- {detail}" for detail in message.details)
            return lines

        if result.status == "PASS":
            conclusion = "PASS"
            action = "Continue production."
        elif result.status == "INFERENCE_ERROR":
            conclusion = "INFERENCE ERROR"
            action = "Check model/backend status and run inspection again."
        else:
            conclusion = "DETECTION FAIL"
            action = "Review defects and run inspection again."
        return [
            f"=== {tr(self._language, 'field_decision')} ===",
            f"{tr(self._language, 'conclusion')}: {conclusion}",
            f"{tr(self._language, 'next_action')}: {action}",
        ]

    def _append_sequence_lines(self, lines: list[str], result: DetectionResult) -> None:
        seq_check = result.sequence_check
        if not seq_check:
            return
        is_ok = seq_check.get("is_ok", True)
        status_str = "PASS" if is_ok else "FAIL"
        lines.append(f"{tr(self._language, 'sequence_check')}: {status_str}")
        if is_ok:
            return
        reason = seq_check.get("reason", "")
        if reason == "length_mismatch":
            reason_text = tr(self._language, "sequence_length_error")
        elif reason == "order_mismatch":
            reason_text = tr(self._language, "sequence_order_error")
        else:
            reason_text = str(reason or tr(self._language, "unknown_reason"))
        lines.append(f"  - {tr(self._language, 'error_reason')}: {reason_text}")
        lines.append(f"  - {tr(self._language, 'expected_order')}: {seq_check.get('expected')}")
        lines.append(f"  - {tr(self._language, 'observed_order')}: {seq_check.get('observed')}")

    def _append_color_lines(self, lines: list[str], result: DetectionResult) -> None:
        color_info = result.color_check
        if not color_info:
            lines.append(f"\n=== {tr(self._language, 'color_check')}: {tr(self._language, 'not_run')} ===")
            return
        color_ok = color_info.get("is_ok", True)
        color_status = "PASS" if color_ok else "FAIL"
        lines.append(f"\n=== {tr(self._language, 'color_check')}: {color_status} ===")
        color_items = color_info.get("items") or []
        for color_item in color_items:
            item_status = "OK" if color_item.get("is_ok", True) else "NG"
            cls_name = color_item.get("class_name", "?")
            pred = color_item.get("best_color", "?")
            diff = color_item.get("diff", 0)
            threshold = color_item.get("threshold", 0)
            lines.append(
                f"  {item_status} {cls_name} -> {pred} "
                f"(diff={diff:.2f}, thr={threshold:.2f})"
            )
        if not color_items:
            lines.append(f"  ({tr(self._language, 'none')})")

    def _append_list_section(self, lines: list[str], title_key: str, items: object) -> None:
        lines.append(f"\n=== {tr(self._language, title_key)} ===")
        if items:
            for item in items:
                lines.append(f"- {item}")
        else:
            lines.append(tr(self._language, "none"))

    def _append_position_lines(self, lines: list[str], result: DetectionResult) -> None:
        position_summary = summarize_position_records(
            [{"label": item.label, **item.metadata} for item in (result.items or [])]
        )
        if position_summary.total_with_position <= 0 and position_summary.skipped_count <= 0:
            return

        pos_status = "PASS" if position_summary.fail_count == 0 else "FAIL"
        lines.append(f"\n=== {tr(self._language, 'position_check')}: {pos_status} ===")
        lines.append(
            "  "
            f"{tr(self._language, 'correct')} {position_summary.correct_count} / "
            f"{tr(self._language, 'abnormal')} {position_summary.fail_count} / "
            f"{tr(self._language, 'skipped')} {position_summary.skipped_count}"
        )
        fixture_hint = format_fixture_shift_hint(position_summary)
        if fixture_hint:
            lines.append(f"  {fixture_hint}")
        if position_summary.issues:
            for issue in position_summary.issues[:5]:
                detail = f"  - {issue.label} [{issue.status}]"
                if issue.error is not None:
                    detail += f" d={issue.error:.1f}"
                if issue.dx is not None and issue.dy is not None:
                    detail += f" dx={issue.dx:+.1f}, dy={issue.dy:+.1f}"
                lines.append(detail)
        else:
            lines.append(
                "  "
                + tr(self._language, "all_positions_ok").format(
                    count=position_summary.correct_count
                )
            )

    def _append_decision_lines(self, lines: list[str], result: DetectionResult) -> None:
        decision = result.metadata.get("decision") if result.metadata else None
        if not isinstance(decision, dict):
            return
        decision_reasons = decision.get("reasons") or []
        if decision_reasons:
            lines.append("\n=== Decision Reasons ===")
            lines.append(", ".join(str(reason) for reason in decision_reasons))

    def _append_alignment_lines(self, lines: list[str], result: DetectionResult) -> None:
        alignment_quality = result.metadata.get("alignment_quality") if result.metadata else None
        if not isinstance(alignment_quality, dict) or not alignment_quality.get("enabled"):
            return
        lines.append("\n=== Alignment Quality ===")
        lines.append("PASS" if alignment_quality.get("is_ok", True) else "FAIL")
        issues = alignment_quality.get("issues") or []
        if issues:
            lines.append(
                "issues: "
                + ", ".join(_alignment_issue_label(str(issue), self._language) for issue in issues)
            )
        try:
            dx = float(alignment_quality.get("dx", 0.0))
            dy = float(alignment_quality.get("dy", 0.0))
            lines.append(f"shift: dx={dx:+.1f}, dy={dy:+.1f}")
        except (TypeError, ValueError):
            pass
        lines.append(
            "sources: "
            f"{alignment_quality.get('observed_source_count', 0)}/"
            f"{alignment_quality.get('required_source_count', 0)}"
        )

    def _append_detection_lines(self, lines: list[str], result: DetectionResult) -> None:
        if not result.items:
            return
        lines.append(f"\n=== {tr(self._language, 'detection_details')} ===")
        for idx, item in enumerate(result.items[:5], start=1):
            parts = [f"{idx}. {item.label}", f"conf={item.confidence:.3f}"]
            pos_status = item.metadata.get("position_status")
            if pos_status:
                parts.append(f"pos={pos_status}")
            lines.append(" | ".join(parts))


def _alignment_issue_label(issue: str, language: str = "en") -> str:
    """Return localized text for alignment quality issue codes."""
    labels = {
        "en": {
            "insufficient_alignment_sources": "Insufficient alignment sources",
            "insufficient_alignment_inliers": "Insufficient alignment inliers",
            "alignment_dx_out_of_range": "Horizontal shift out of range",
            "alignment_dy_out_of_range": "Vertical shift out of range",
            "alignment_shift_out_of_range": "Board shift out of range",
        },
        "zh": {
            "insufficient_alignment_sources": "可用對位特徵不足",
            "insufficient_alignment_inliers": "對位特徵一致性不足",
            "alignment_dx_out_of_range": "水平偏移超出範圍",
            "alignment_dy_out_of_range": "垂直偏移超出範圍",
            "alignment_shift_out_of_range": "整板偏移超出範圍",
        },
    }
    lang = normalize_language(language)
    return labels.get(lang, labels["en"]).get(issue, issue)
