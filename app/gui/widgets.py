from __future__ import annotations

import os
from typing import Dict

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget


class StatusWidget(QWidget):
    """Small status dashboard showing current system state."""

    _STATUS_COLORS: Dict[str, tuple[str, str]] = {
        "idle": ("#6c757d", "系統待命中"),
        "running": ("#ffc107", "檢測執行中..."),
        "success": ("#28a745", "檢測完成"),
        "error": ("#dc3545", "檢測錯誤"),
        "warning": ("#fd7e14", "警告"),
    }

    def __init__(self) -> None:
        super().__init__()
        self._status_indicator = QLabel("")
        self._status_text = QLabel("系統就緒")
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()

        title_label = QLabel("系統狀態")
        title_label.setFont(QFont("Microsoft JhengHei", 12, QFont.Bold))
        title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")

        self._status_indicator.setFont(QFont("Arial", 20))
        self._status_indicator.setAlignment(Qt.AlignCenter)

        self._status_text.setAlignment(Qt.AlignCenter)
        self._status_text.setFont(QFont("Microsoft JhengHei", 10))

        layout.addWidget(title_label)
        layout.addWidget(self._status_indicator)
        layout.addWidget(self._status_text)
        layout.addStretch()

        self.setLayout(layout)
        self.setFixedWidth(150)
        self.setStyleSheet(
            """
            QWidget {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 10px;
            }
        """
        )

        self.set_status("idle")

    def set_status(self, status: str) -> None:
        color, text = self._STATUS_COLORS.get(status, ("#6c757d", "未知狀態"))
        self._status_indicator.setStyleSheet(f"color: {color};")
        self._status_text.setText(text)


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
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
        else:
            self.setText(f"無法載入{self._title}")


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
        self._result_text.setMaximumHeight(200)
        self._result_text.setReadOnly(True)

        layout.addWidget(title)
        layout.addWidget(self._result_text)
        self.setLayout(layout)

    def update_result(self, result: dict) -> None:
        """Render detection result details into the text panel."""
        text = "=== 檢測結果 ===\n"
        text += f"狀態: {result.get('status', 'N/A')}\n"
        text += f"產品: {result.get('product', 'N/A')}\n"
        text += f"區域: {result.get('area', 'N/A')}\n"
        text += f"類型: {result.get('inference_type', 'N/A')}\n"
        text += f"模型路徑: {result.get('ckpt_path', 'N/A')}\n"

        anomaly_score = result.get("anomaly_score")
        if anomaly_score or anomaly_score in (0, 0.0):
            text += f"異常分數: {anomaly_score}\n"

        detections = result.get("detections", [])
        if detections:
            text += f"檢測到的項目數量: {len(detections)}\n"
            for idx, det in enumerate(detections[:3]):
                text += f"  - 項目{idx + 1}: {det}\n"

        missing = result.get("missing_items")
        if missing:
            text += f"缺失項目: {missing}\n"
            try:
                if isinstance(missing, list) and missing:
                    text += "缺失項目列表:\n"
                    text += "\n".join(f"  - {item}" for item in missing) + "\n"
            except Exception:
                pass

        text += "==================="
        self._result_text.setPlainText(text)
