from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
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

        # ── 連續 NG 警示橫幅（平時隱藏）────────────────────────────────
        self._alert_banner = QFrame()
        self._alert_banner.setFrameShape(QFrame.StyledPanel)
        self._alert_banner.setStyleSheet(
            "QFrame { background-color: #f8d7da; border: 2px solid #f5c6cb; border-radius: 4px; padding: 4px; }"
        )
        banner_layout = QHBoxLayout()
        banner_layout.setContentsMargins(8, 4, 8, 4)
        self._alert_text = QLabel()
        self._alert_text.setFont(QFont("Microsoft JhengHei", 10, QFont.Bold))
        self._alert_text.setStyleSheet("color: #721c24; background: transparent; border: none;")
        self._alert_text.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        banner_layout.addWidget(self._alert_text, stretch=1)
        dismiss_btn = QPushButton("✕")
        dismiss_btn.setFixedSize(24, 24)
        dismiss_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; color: #721c24; font-weight: bold; }"
            "QPushButton:hover { background: #f5c6cb; border-radius: 3px; }"
        )
        dismiss_btn.setToolTip("關閉警示")
        dismiss_btn.clicked.connect(self._dismiss_alert)
        banner_layout.addWidget(dismiss_btn)
        self._alert_banner.setLayout(banner_layout)
        self._alert_banner.setVisible(False)
        layout.addWidget(self._alert_banner)

        # ── 當班統計 ─────────────────────────────────────────────────
        self.session_stats = SessionStatsWidget()
        self.session_stats.consecutive_fail_reached.connect(self._show_alert)
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
        # Auto-hide the consecutive-FAIL banner the moment a PASS arrives.
        if result.status == "PASS":
            self._dismiss_alert()

    def _show_alert(self, count: int) -> None:
        self._alert_text.setText(f"⚠ 警告：已連續 {count} 次 NG！請確認產線狀況。")
        self._alert_banner.setVisible(True)

    def _dismiss_alert(self) -> None:
        self._alert_banner.setVisible(False)

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
