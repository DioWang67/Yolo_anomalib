"""Info panel: status indicators, pipeline dashboard, detection results, logs.

Layout (top → bottom):
1. BigStatusLabel         — large PASS/FAIL/RUNNING indicator
2. Pipeline Dashboard     — real-time counters (Captured / Dropped / Saved / Queue)
3. ResultDisplayWidget    — detection detail table
4. System Log             — scrolling log text
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
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

from app.gui.widgets import BigStatusLabel, ResultDisplayWidget

if TYPE_CHECKING:
    from core.types import DetectionResult


# ======================================================================
# Reusable metric card widget
# ======================================================================

class _MetricCard(QFrame):
    """A single dashboard metric with title + large value."""

    _BASE_STYLE = """
        _MetricCard {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #f8f9fa, stop:1 #e9ecef
            );
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 8px;
        }
    """

    _ALERT_STYLE = """
        _MetricCard {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #fff5f5, stop:1 #ffe3e3
            );
            border: 2px solid #e03131;
            border-radius: 8px;
            padding: 8px;
        }
    """

    def __init__(
        self,
        title: str,
        icon: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(self._BASE_STYLE)
        self.setMinimumWidth(110)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        # Title row
        title_label = QLabel(f"{icon}  {title}" if icon else title)
        title_label.setFont(QFont("Microsoft JhengHei UI", 9))
        title_label.setStyleSheet("color: #495057; border: none; background: transparent;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Value
        self.value_label = QLabel("--")
        self.value_label.setFont(QFont("Consolas", 22, QFont.Bold))
        self.value_label.setStyleSheet("color: #212529; border: none; background: transparent;")
        self.value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.value_label)

    def set_value(self, value: int | str, *, alert: bool = False) -> None:
        """Update the displayed value.  Set ``alert=True`` for red styling."""
        self.value_label.setText(str(value))
        if alert:
            self.setStyleSheet(self._ALERT_STYLE)
            self.value_label.setStyleSheet("color: #c92a2a; border: none; background: transparent;")
        else:
            self.setStyleSheet(self._BASE_STYLE)
            self.value_label.setStyleSheet("color: #212529; border: none; background: transparent;")


# ======================================================================
# Info Panel
# ======================================================================

class InfoPanel(QWidget):
    """Panel displaying system status, pipeline dashboard, results, and logs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()

        # --- 1. Big status indicator ---
        self.big_status_label = BigStatusLabel()
        layout.addWidget(self.big_status_label)

        # --- 2. Pipeline dashboard ---
        self._dashboard_group = self._build_dashboard()
        layout.addWidget(self._dashboard_group)

        # --- 3. Detection result details ---
        self.result_widget = ResultDisplayWidget()
        layout.addWidget(self.result_widget)

        # --- 4. System log ---
        log_group = QGroupBox("系統日誌")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setFont(QFont("Consolas", 8))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
        """)
        self.log_text.setReadOnly(True)

        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Dashboard builder
    # ------------------------------------------------------------------

    def _build_dashboard(self) -> QGroupBox:
        """Create the real-time pipeline metrics panel."""
        group = QGroupBox("📊 管線即時儀表板")
        group.setFont(QFont("Microsoft JhengHei UI", 10, QFont.Bold))
        group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #339af0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #1971c2;
            }
        """)

        grid = QGridLayout()
        grid.setSpacing(8)

        self._card_captured = _MetricCard("取像數量", "📷")
        self._card_dropped  = _MetricCard("丟幀數量", "⚠️")
        self._card_saved    = _MetricCard("存檔數量", "💾")
        self._card_queue    = _MetricCard("佇列積壓", "📦")

        grid.addWidget(self._card_captured, 0, 0)
        grid.addWidget(self._card_dropped,  0, 1)
        grid.addWidget(self._card_saved,    1, 0)
        grid.addWidget(self._card_queue,    1, 1)

        group.setLayout(grid)
        return group

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_result(self, result: DetectionResult) -> None:
        """Update all status indicators with the new result."""
        self.result_widget.update_result(result)

    def update_dashboard(self, stats: dict[str, Any]) -> None:
        """Refresh the pipeline dashboard counters.

        Args:
            stats: Dictionary from ``DetectionSystem.pipeline_stats()``.
                Expected keys: ``frames_captured``, ``frames_dropped``,
                ``tasks_saved``, ``inference_queue_size``, ``io_queue_size``.
        """
        captured = stats.get("frames_captured", 0)
        dropped  = stats.get("frames_dropped", 0)
        saved    = stats.get("tasks_saved", 0)
        inf_q    = stats.get("inference_queue_size", 0)
        io_q     = stats.get("io_queue_size", 0)

        self._card_captured.set_value(captured)
        self._card_dropped.set_value(dropped, alert=(dropped > 0))
        self._card_saved.set_value(saved)
        self._card_queue.set_value(f"{inf_q} / {io_q}")

    def reset_dashboard(self) -> None:
        """Reset all dashboard counters to initial state."""
        self._card_captured.set_value("--")
        self._card_dropped.set_value("--")
        self._card_saved.set_value("--")
        self._card_queue.set_value("-- / --")

    def append_log(self, message: str) -> None:
        self.log_text.append(message)
