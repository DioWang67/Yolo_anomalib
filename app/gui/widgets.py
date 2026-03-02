from __future__ import annotations

import os
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget

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
        elif status == "FAIL":
            self.setText("FAIL")
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
        elif status == "ERROR":
            self.setText("ERROR")
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




class ImageViewer(QLabel):
    """Simple image container with dashed placeholder."""

    def __init__(self, title: str = "影像") -> None:
        super().__init__()
        self._title = title
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

    def set_image(self, image_path: str) -> None:
        # Skip redundant disk reads for the same path (retry scenario)
        if getattr(self, "_last_image_path", None) == image_path:
            return
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
            self._last_image_path = image_path
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
        lines.append("=== 檢測摘要 ===")
        lines.append(f"狀態: {result.status}")
        lines.append(f"產品 / 區域: {product} / {area}")
        lines.append(f"類型: {inference_type}")
        lines.append(f"模型: {ckpt_name}")
        lines.append(f"延遲: {result.latency * 1000:.1f} ms")

        lines.append("\n=== 數據 ===")
        lines.append(f"偵測數量: {len(result.items)}")

        if result.anomaly_score is not None:
            lines.append(f"異常分數: {result.anomaly_score}")

        missing = result.missing_items or []
        lines.append(f"缺失項目: {len(missing)}")

        unexpected = result.unexpected_items or []
        if isinstance(unexpected, (list, tuple)):
            lines.append(f"未預期項目: {len(unexpected)}")

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
