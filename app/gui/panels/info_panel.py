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

from app.gui.i18n import normalize_language, tr
from app.gui.widgets import (
    BigStatusLabel,
    FailReasonLabel,
    OperatorGuidanceCard,
    ResultDisplayWidget,
    SessionStatsWidget,
)

if TYPE_CHECKING:
    from core.types import DetectionResult


class InfoPanel(QWidget):
    """Right-side status, result detail, statistics, and debug log panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._log_filter_active = False
        self._language = "en"
        self._log_buffer: list[str] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setSpacing(8)

        self.big_status_label = BigStatusLabel()
        layout.addWidget(self.big_status_label)

        self.operator_guidance_card = OperatorGuidanceCard()
        layout.addWidget(self.operator_guidance_card)

        self.fail_reason_label = FailReasonLabel()
        layout.addWidget(self.fail_reason_label)

        self._alert_banner = QFrame()
        self._alert_banner.setFrameShape(QFrame.StyledPanel)
        self._alert_banner.setStyleSheet(
            "QFrame { background-color: #f8d7da; border: 2px solid #f5c6cb; "
            "border-radius: 4px; padding: 4px; }"
        )
        banner_layout = QHBoxLayout()
        banner_layout.setContentsMargins(8, 4, 8, 4)
        self._alert_text = QLabel()
        self._alert_text.setFont(QFont("Microsoft JhengHei", 10, QFont.Bold))
        self._alert_text.setStyleSheet("color: #721c24; background: transparent; border: none;")
        self._alert_text.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        banner_layout.addWidget(self._alert_text, stretch=1)

        self._dismiss_alert_btn = QPushButton("x")
        self._dismiss_alert_btn.setFixedSize(24, 24)
        self._dismiss_alert_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; color: #721c24; "
            "font-weight: bold; }"
            "QPushButton:hover { background: #f5c6cb; border-radius: 3px; }"
        )
        self._dismiss_alert_btn.clicked.connect(self._dismiss_alert)
        banner_layout.addWidget(self._dismiss_alert_btn)
        self._alert_banner.setLayout(banner_layout)
        self._alert_banner.setVisible(False)
        layout.addWidget(self._alert_banner)

        self.session_stats = SessionStatsWidget()
        self.session_stats.consecutive_fail_reached.connect(self._show_alert)
        layout.addWidget(self.session_stats)

        self.result_widget = ResultDisplayWidget()
        layout.addWidget(self.result_widget)

        self.debug_log_toggle = QPushButton("Debug Log >")
        self.debug_log_toggle.setObjectName("secondaryAction")
        self.debug_log_toggle.setCheckable(True)
        self.debug_log_toggle.toggled.connect(self._set_log_visible)
        layout.addWidget(self.debug_log_toggle)

        self.log_group = QGroupBox("Debug Log")
        log_layout = QVBoxLayout()
        log_layout.setSpacing(4)

        toolbar = QHBoxLayout()
        self._filter_chk = QCheckBox()
        self._filter_chk.toggled.connect(self._on_filter_toggled)
        toolbar.addWidget(self._filter_chk)
        toolbar.addStretch()

        self._clear_log_btn = QPushButton()
        self._clear_log_btn.setFixedWidth(90)
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
        self.log_group.setLayout(log_layout)
        self.log_group.setVisible(False)
        layout.addWidget(self.log_group)

        self.setLayout(layout)
        self.set_language(self._language)

    def update_result(self, result: DetectionResult) -> None:
        """Update status indicators, fail reason, stats, and detail text."""
        self.result_widget.update_result(result)
        self.fail_reason_label.update_from_result(result)
        self.operator_guidance_card.update_from_result(result)
        self.session_stats.record_result(result.status)
        if result.status == "PASS":
            self._dismiss_alert()

    def append_log(self, message: str) -> None:
        """Append a log line, respecting the active filter."""
        self._log_buffer.append(message)
        if self._log_filter_active and not self._is_important(message):
            return
        self.log_text.append(message)

    def set_language(self, language: str) -> None:
        """Update visible right-panel labels."""
        self._language = normalize_language(language)
        self.debug_log_toggle.setText(
            tr(self._language, "debug_log_open")
            if self.log_group.isVisible()
            else tr(self._language, "debug_log_closed")
        )
        self.log_group.setTitle(tr(self._language, "debug_log"))
        self._filter_chk.setText(tr(self._language, "filter_log"))
        self._filter_chk.setToolTip(tr(self._language, "filter_log"))
        self._clear_log_btn.setText(tr(self._language, "clear_log"))
        self._dismiss_alert_btn.setToolTip("Dismiss alert" if self._language == "en" else "關閉警示")

        self.fail_reason_label.set_language(self._language)
        self.operator_guidance_card.set_language(self._language)
        self.session_stats.set_language(self._language)
        self.result_widget.set_language(self._language)

        if self._alert_banner.isVisible():
            self._show_alert(self.session_stats.consecutive_fails)

    def _show_alert(self, count: int) -> None:
        self._alert_text.setText(tr(self._language, "consecutive_fail").format(count=count))
        self._alert_banner.setVisible(True)

    def _dismiss_alert(self) -> None:
        self._alert_banner.setVisible(False)

    @staticmethod
    def _is_important(message: str) -> bool:
        """Return True if the message should always be shown."""
        upper = message.upper()
        return any(
            keyword in upper
            for keyword in (
                "WARN",
                "WARNING",
                "ERROR",
                "FAIL",
                "CRITICAL",
                "EXCEPTION",
                "斷線",
                "失敗",
            )
        )

    def _on_filter_toggled(self, active: bool) -> None:
        self._log_filter_active = active
        self.log_text.clear()
        for message in self._log_buffer:
            if not active or self._is_important(message):
                self.log_text.append(message)

    def _clear_log(self) -> None:
        self._log_buffer.clear()
        self.log_text.clear()

    def _set_log_visible(self, visible: bool) -> None:
        self.log_group.setVisible(visible)
        self.debug_log_toggle.setText(
            tr(self._language, "debug_log_open")
            if visible
            else tr(self._language, "debug_log_closed")
        )
