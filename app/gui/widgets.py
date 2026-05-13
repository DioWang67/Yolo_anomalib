from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

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
        self.setFont(QFont("Arial Black", 24, QFont.Bold))
        self.setStyleSheet(
            """
            QLabel {
                background-color: #e9ecef;
                color: #495057;
                border: 2px solid #ced4da;
                border-radius: 8px;
                padding: 15px;
                min-height: 60px;
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
                    background-color: #28a745;
                    color: white;
                    border: 2px solid #1e7e34;
                    border-radius: 8px;
                    padding: 15px;
                }
                """
            )
        elif status in {"FAIL", "DETECTION_FAIL"}:
            self.setText("DETECTION FAIL" if status == "DETECTION_FAIL" else "FAIL")
            self.setStyleSheet(
                """
                QLabel {
                    background-color: #dc3545;
                    color: white;
                    border: 2px solid #bd2130;
                    border-radius: 8px;
                    padding: 15px;
                }
                """
            )
        elif status in {"ERROR", "INFERENCE_ERROR"}:
            self.setText("INFERENCE ERROR" if status == "INFERENCE_ERROR" else "ERROR")
            self.setStyleSheet(
                """
                QLabel {
                    background-color: #ffc107;
                    color: black;
                    border: 2px solid #d39e00;
                    border-radius: 8px;
                    padding: 15px;
                }
                """
            )
        else:
            self.setText(status if status else "READY")
            self.setStyleSheet(
                """
                QLabel {
                    background-color: #e9ecef;
                    color: #495057;
                    border: 2px solid #ced4da;
                    border-radius: 8px;
                    padding: 15px;
                }
                """
            )




class FailReasonLabel(QLabel):
    """One-line summary of why the last result was FAIL.

    Appears below BigStatusLabel.  Hidden when status is PASS or READY.
    """

    _STYLE_FAIL = (
        "background-color: #fff3cd; color: #856404;"
        "border: 1px solid #ffc107; border-radius: 4px; padding: 4px 8px;"
        "font-size: 10pt;"
    )
    _STYLE_CLEAR = "color: transparent; border: none; padding: 4px 8px;"

    def __init__(self) -> None:
        super().__init__("")
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setStyleSheet(self._STYLE_CLEAR)

    def update_from_result(self, result: "DetectionResult") -> None:
        """Extract top-level FAIL reasons and display them."""
        if result.status not in {"FAIL", "DETECTION_FAIL", "INFERENCE_ERROR"}:
            self.setText("")
            self.setStyleSheet(self._STYLE_CLEAR)
            return

        if result.status == "INFERENCE_ERROR":
            self.setText(result.error or "Inference/backend runtime error")
            self.setStyleSheet(self._STYLE_FAIL)
            return

        reasons: list[str] = []

        if result.error:
            reasons.append(str(result.error))

        missing = result.missing_items or []
        if missing:
            reasons.append(f"缺件：{', '.join(str(i) for i in missing[:3])}"
                           + ("…" if len(missing) > 3 else ""))

        color_check = result.color_check or {}
        if color_check and not color_check.get("is_ok", True):
            bad = [c.get("class_name", "?") for c in (color_check.get("items") or [])
                   if not c.get("is_ok", True)]
            reasons.append(f"顏色錯誤：{', '.join(bad[:3])}" if bad else "顏色錯誤")

        seq = result.sequence_check or {}
        if seq and not seq.get("is_ok", True):
            reason_key = seq.get("reason", "")
            reasons.append("排列順序錯誤" if reason_key == "order_mismatch" else "排列長度不符")

        all_items = result.items or []
        position_summary = summarize_position_records(
            [
                {"label": item.label, **item.metadata}
                for item in all_items
            ]
        )
        if position_summary.fail_count > 0:
            labels = [issue.label for issue in position_summary.issues]
            reasons.append(f"位置偏移：{', '.join(labels[:3])}"
                           + ("…" if len(labels) > 3 else ""))
            fixture_hint = format_fixture_shift_hint(position_summary)
            if fixture_hint:
                reasons.append(f"⚙ {fixture_hint}")

        if reasons:
            self.setText("⚠ " + "　|　".join(reasons))
        else:
            self.setText("⚠ 原因不明，請查看詳細結果")
        self.setStyleSheet(self._STYLE_FAIL)

    def clear_reason(self) -> None:
        self.setText("")
        self.setStyleSheet(self._STYLE_CLEAR)


# ---------------------------------------------------------------------------


class OperatorGuidanceCard(QFrame):
    """Customer-facing guidance card with next action."""

    _STYLES = {
        "success": (
            "QFrame { background-color: #e8f6ec; border: 2px solid #7bc47f; "
            "border-radius: 10px; padding: 8px; }"
        ),
        "warning": (
            "QFrame { background-color: #fff4db; border: 2px solid #f0b429; "
            "border-radius: 10px; padding: 8px; }"
        ),
        "danger": (
            "QFrame { background-color: #fdecea; border: 2px solid #d9534f; "
            "border-radius: 10px; padding: 8px; }"
        ),
    }

    def __init__(self) -> None:
        super().__init__()
        self._headline = QLabel()
        self._action = QLabel()
        self._details = QLabel()
        self._setup_ui()
        self.show_message("等待檢測", "請放置工件後開始檢測", "warning", [])

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self._headline.setFont(QFont("Microsoft JhengHei", 15, QFont.Bold))
        self._headline.setWordWrap(True)

        self._action.setFont(QFont("Microsoft JhengHei", 11, QFont.Bold))
        self._action.setWordWrap(True)

        self._details.setFont(QFont("Microsoft JhengHei", 10))
        self._details.setWordWrap(True)

        layout.addWidget(self._headline)
        layout.addWidget(self._action)
        layout.addWidget(self._details)
        self.setLayout(layout)

    def update_from_result(self, result: "DetectionResult") -> None:
        message = build_customer_message(result)
        self.show_message(
            message.headline,
            message.action,
            message.severity,
            message.details,
        )

    def show_message(
        self,
        headline: str,
        action: str,
        severity: str,
        details: list[str],
    ) -> None:
        self.setStyleSheet(self._STYLES.get(severity, self._STYLES["warning"]))
        self._headline.setText(headline)
        self._action.setText(action)
        self._details.setText("\n".join(f"- {detail}" for detail in details) if details else "")


class SessionStatsWidget(QGroupBox):
    """Current-shift statistics: PASS/FAIL counts, yield rate, consecutive FAIL alert.

    Call :meth:`record_result` after every detection.
    Call :meth:`reset_session` at shift start or manually.
    """

    CONSECUTIVE_FAIL_ALERT: int = 3  # alert threshold

    # Emitted exactly once when consecutive_fails reaches CONSECUTIVE_FAIL_ALERT.
    # Argument is the count that triggered the alert.
    consecutive_fail_reached = pyqtSignal(int)

    _STYLE_NORMAL = "color: #155724; background: #d4edda; border-radius: 3px; padding: 2px 6px;"
    _STYLE_ALERT  = "color: #721c24; background: #f8d7da; border-radius: 3px; padding: 2px 6px;"

    def __init__(self) -> None:
        super().__init__("當班統計")
        self._pass = 0
        self._fail = 0
        self._consecutive_fails = 0
        self._build_ui()

    # ------------------------------------------------------------------
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

        grid.addWidget(_header("PASS"), 0, 0)
        grid.addWidget(_header("FAIL"), 0, 1)
        grid.addWidget(_header("良率"), 0, 2)
        grid.addWidget(_header("連續NG"), 0, 3)

        self._pass_lbl = _value()
        self._fail_lbl = _value()
        self._yield_lbl = _value()
        self._consec_lbl = _value()

        self._pass_lbl.setStyleSheet("color: #28a745;")
        self._fail_lbl.setStyleSheet("color: #dc3545;")

        grid.addWidget(self._pass_lbl,   1, 0)
        grid.addWidget(self._fail_lbl,   1, 1)
        grid.addWidget(self._yield_lbl,  1, 2)
        grid.addWidget(self._consec_lbl, 1, 3)

        self.setLayout(grid)

    # ------------------------------------------------------------------
    @property
    def consecutive_fails(self) -> int:
        """Read-only view of the current consecutive-FAIL counter."""
        return self._consecutive_fails

    # ------------------------------------------------------------------
    def record_result(self, status: str) -> None:
        """Update counters based on detection status ('PASS' or 'FAIL')."""
        if status == "PASS":
            self._pass += 1
            self._consecutive_fails = 0
        elif status in {"FAIL", "DETECTION_FAIL"}:
            self._fail += 1
            self._consecutive_fails += 1
        self._refresh()

    def reset_session(self) -> None:
        """Reset all counters to zero (new shift / manual reset)."""
        self._pass = 0
        self._fail = 0
        self._consecutive_fails = 0
        self._refresh()

    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        total = self._pass + self._fail
        yield_pct = f"{self._pass / total * 100:.1f}%" if total else "—"

        self._pass_lbl.setText(str(self._pass))
        self._fail_lbl.setText(str(self._fail))
        self._yield_lbl.setText(yield_pct)
        self._consec_lbl.setText(str(self._consecutive_fails))

        if self._consecutive_fails >= self.CONSECUTIVE_FAIL_ALERT:
            self._consec_lbl.setStyleSheet(self._STYLE_ALERT)
            self._consec_lbl.setToolTip(
                f"連續 {self._consecutive_fails} 次 NG！請確認產線狀況。"
            )
            # Emit on every FAIL at or above the threshold so the banner
            # re-appears if the operator dismissed it and NG continues.
            self.consecutive_fail_reached.emit(self._consecutive_fails)
        else:
            self._consec_lbl.setStyleSheet(self._STYLE_NORMAL)
            self._consec_lbl.setToolTip("")


# ---------------------------------------------------------------------------

class ImageViewer(QLabel):
    """Simple image container with dashed placeholder."""

    def __init__(self, title: str = "影像") -> None:
        super().__init__()
        self._title = title
        self._load_token: int = 0   # incremented on every clear/new-load to cancel stale timers
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setMinimumSize(300, 300)
        self.setStyleSheet(
            """
            QLabel {
                border: 2px dashed #adb5bd;
                border-radius: 8px;
                background-color: #f8f9fa;
            }
        """
        )
        self.setAlignment(Qt.AlignCenter)
        self.setText(f"等待{self._title}")
        self.setScaledContents(True)

    def clear(self) -> None:
        """Override to cancel any in-flight deferred loads before clearing."""
        self._load_token += 1
        self._last_image_path = None
        super().clear()
        self.setText(f"等待{self._title}")

    def set_image(self, image_path: str) -> None:
        # Cancel stale deferred loads from previous run
        self._load_token += 1
        token = self._load_token

        if os.path.exists(image_path):
            self._last_image_path = image_path
            self.setText("載入中...")
            # Defer the heavy disk-read + scale to the next event-loop tick
            # so the UI stays responsive (e.g. during pick_image flow).
            QTimer.singleShot(
                0, lambda p=image_path, t=token: self._load_and_display(p, t)
            )
        else:
            self.setText(f"無法載入{self._title}")

    def _load_and_display(self, image_path: str, token: int) -> None:
        """Heavy lifting: read file from disk and scale for display.

        Guards against stale callbacks: if a newer set_image/clear has been
        called since this timer was scheduled, ``token`` will not match
        ``_load_token`` and the call is silently dropped.
        """
        if token != self._load_token:
            return  # superseded by a newer load or clear()
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled)
        else:
            self.setText(f"無法載入{self._title}")

    def display_image(self, image) -> None:
        """Display an image directly.

        Accepts either a QImage (fast path from worker) or a BGR ndarray (legacy).
        """
        try:
            if isinstance(image, QImage):
                # Fast path: already converted in worker thread
                self.setPixmap(QPixmap.fromImage(image).scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
            elif isinstance(image, np.ndarray) and image.size > 0:
                # Legacy path: convert BGR→RGB on main thread
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.setPixmap(QPixmap.fromImage(qt_image).scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
            else:
                self.clear()
                self.setText("無法顯示影像")
        except Exception:
            self.clear()
            self.setText("無法顯示影像")


class ResultDisplayWidget(QWidget):
    """Scrollable textual representation of detection outcomes."""

    def __init__(self) -> None:
        super().__init__()
        self._result_text = QTextEdit()
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()

        title = QLabel("檢測結果")
        title.setFont(QFont("Microsoft JhengHei", 12, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")

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

        layout.addWidget(title)
        layout.addWidget(self._result_text)
        self.setLayout(layout)

    def update_result(self, result: DetectionResult) -> None:
        """Render detection result details into the text panel."""
        import os

        product = result.product or "N/A"
        area = result.area or "N/A"
        inference_type = result.inference_type or "N/A"
        ckpt_name = os.path.basename(result.ckpt_path) if result.ckpt_path else "N/A"

        lines: list[str] = []
        customer_message = build_customer_message(result)
        lines.append("=== 現場判定 ===")
        lines.append(f"結論: {customer_message.headline}")
        lines.append(f"建議: {customer_message.action}")
        for detail in customer_message.details:
            lines.append(f"- {detail}")

        lines.append("\n=== 基本資訊 ===")
        lines.append(f"狀態: {result.status}")
        lines.append(f"產品 / 區域: {product} / {area}")
        lines.append(f"類型: {inference_type}")
        lines.append(f"模型: {ckpt_name}")
        lines.append(f"延遲: {result.latency * 1000:.1f} ms")

        lines.append("\n=== 關鍵數量 ===")
        lines.append(f"偵測數量: {len(result.items)}")

        if result.anomaly_score is not None:
            lines.append(f"異常分數: {result.anomaly_score}")

        missing = result.missing_items or []
        lines.append(f"缺件數量: {len(missing)}")

        unexpected = result.unexpected_items or []
        if isinstance(unexpected, (list, tuple)):
            lines.append(f"多餘件數: {len(unexpected)}")

        # --- 排列檢查 ---
        seq_check = result.sequence_check
        if seq_check:
            is_ok = seq_check.get("is_ok", True)
            status_str = "PASS" if is_ok else "FAIL"
            lines.append(f"排列檢查: {status_str}")
            if not is_ok:
                reason = seq_check.get("reason", "不明錯誤")
                if reason == "length_mismatch":
                    lines.append("  ↳ 錯誤原因: 排列長度不符")
                elif reason == "order_mismatch":
                    lines.append("  ↳ 錯誤原因: 排列順序錯誤")
                lines.append(f"  ↳ 預期順序: {seq_check.get('expected')}")
                lines.append(f"  ↳ 實際順序: {seq_check.get('observed')}")

        # --- 顏色檢測 ---
        color_info = result.color_check
        if color_info:
            color_ok = color_info.get("is_ok", True)
            color_status = "PASS" if color_ok else "FAIL"
            lines.append(f"\n=== 顏色檢測: {color_status} ===")
            color_items = color_info.get("items") or []
            for ci in color_items:
                ci_ok = "✔" if ci.get("is_ok", True) else "✘"
                ci_class = ci.get("class_name", "?")
                ci_pred = ci.get("best_color", "?")
                ci_diff = ci.get("diff", 0)
                ci_thr = ci.get("threshold", 0)
                lines.append(
                    f"  {ci_ok} {ci_class} → {ci_pred}"
                    f" (diff={ci_diff:.2f}, thr={ci_thr:.2f})"
                )
            if not color_items:
                lines.append("  (無細節)")
        else:
            lines.append("\n=== 顏色檢測: 未執行 ===")

        # --- 缺失列表 ---
        lines.append("\n=== 缺失列表 ===")
        if missing:
            for item in missing:
                lines.append(f"- {item}")
        else:
            lines.append("無")

        # --- 未預期項目 ---
        lines.append("\n=== 未預期項目 ===")
        if unexpected:
            for item in unexpected:
                lines.append(f"- {item}")
        else:
            lines.append("無")

        # --- 位置檢查匯總 ---
        position_summary = summarize_position_records(
            [
                {"label": item.label, **item.metadata}
                for item in (result.items or [])
            ]
        )
        if position_summary.total_with_position > 0 or position_summary.skipped_count > 0:
            pos_summary = "PASS" if position_summary.fail_count == 0 else "FAIL"
            lines.append(f"\n=== 位置檢查：{pos_summary} ===")
            lines.append(
                "  "
                f"正確 {position_summary.correct_count} 件 / "
                f"異常 {position_summary.fail_count} 件 / "
                f"略過 {position_summary.skipped_count} 件"
            )
            fixture_hint = format_fixture_shift_hint(position_summary)
            if fixture_hint:
                lines.append(f"  {fixture_hint}")
            if position_summary.issues:
                for issue in position_summary.issues[:5]:
                    detail = f"  ✘ {issue.label} [{issue.status}]"
                    if issue.error is not None:
                        detail += f" d={issue.error:.1f}"
                    if issue.dx is not None and issue.dy is not None:
                        detail += f" dx={issue.dx:+.1f}, dy={issue.dy:+.1f}"
                    lines.append(detail)
            else:
                lines.append(f"  全部 {position_summary.correct_count} 件位置正確")

        # --- 偵測細節 ---
        if result.items:
            lines.append("\n=== 偵測細節 (前 5 筆) ===")
            for idx, item in enumerate(result.items[:5], start=1):
                cls = item.label
                conf = item.confidence
                pos_status = item.metadata.get("position_status")

                parts = [f"{idx}. {cls}"]
                parts.append(f"conf={conf:.3f}")
                if pos_status:
                    parts.append(f"pos={pos_status}")
                lines.append(" | ".join(parts))

        lines.append("===================")
        self._result_text.setPlainText("\n".join(lines))
