from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.gui.widgets import BigStatusLabel, FailReasonLabel, ResultDisplayWidget, SessionStatsWidget

if TYPE_CHECKING:
    from core.types import DetectionResult


class InfoPanel(QWidget):
    """
    Panel displaying system status, pass/fail indicator, detailed results,
    session statistics, and filterable system log.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._log_filter_active = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setSpacing(6)

        # ── 大字狀態燈號 ──────────────────────────────────────────────
        self.big_status_label = BigStatusLabel()
        layout.addWidget(self.big_status_label)

        # ── FAIL 原因列 ──────────────────────────────────────────────
        self.fail_reason_label = FailReasonLabel()
        layout.addWidget(self.fail_reason_label)

        # ── 當班統計 ─────────────────────────────────────────────────
        self.session_stats = SessionStatsWidget()
        layout.addWidget(self.session_stats)

        # ── 詳細結果 ─────────────────────────────────────────────────
        self.result_widget = ResultDisplayWidget()
        layout.addWidget(self.result_widget)

        # ── 系統日誌（含過濾工具列）──────────────────────────────────
        log_group = QGroupBox("系統日誌")
        log_layout = QVBoxLayout()
        log_layout.setSpacing(4)

        # 工具列
        toolbar = QHBoxLayout()
        self._filter_chk = QCheckBox("僅顯示警告 / 錯誤")
        self._filter_chk.setToolTip("勾選後只顯示 WARN / ERROR / FAIL 等級的訊息")
        self._filter_chk.toggled.connect(self._on_filter_toggled)
        toolbar.addWidget(self._filter_chk)
        toolbar.addStretch()

        self._clear_log_btn = QPushButton("清除日誌")
        self._clear_log_btn.setFixedWidth(80)
        self._clear_log_btn.clicked.connect(self._clear_log)
        toolbar.addWidget(self._clear_log_btn)
        log_layout.addLayout(toolbar)

        self.log_text = QTextEdit()
        self.log_text.setFont(QFont("Consolas", 8))
        self.log_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
            """
        )
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        self.setLayout(layout)

        # Internal buffer for filter re-rendering
        self._log_buffer: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_result(self, result: DetectionResult) -> None:
        """Update status indicators, fail reason, session stats, and result detail."""
        self.result_widget.update_result(result)
        self.fail_reason_label.update_from_result(result)
        self.session_stats.record_result(result.status)

    def append_log(self, message: str) -> None:
        """Append a log line, respecting the active filter."""
        self._log_buffer.append(message)
        if self._log_filter_active and not self._is_important(message):
            return
        self.log_text.append(message)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_important(message: str) -> bool:
        """Return True if the message should always be shown (WARN/ERROR level)."""
        upper = message.upper()
        return any(kw in upper for kw in ("WARN", "ERROR", "FAIL", "CRITICAL", "EXCEPTION", "斷線", "失敗"))

    def _on_filter_toggled(self, active: bool) -> None:
        self._log_filter_active = active
        # Re-render the visible log from buffer
        self.log_text.clear()
        for msg in self._log_buffer:
            if not active or self._is_important(msg):
                self.log_text.append(msg)

    def _clear_log(self) -> None:
        self._log_buffer.clear()
        self.log_text.clear()
