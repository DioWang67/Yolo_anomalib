from __future__ import annotations

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QGroupBox, QTextEdit, QVBoxLayout, QWidget
from typing import TYPE_CHECKING
from app.gui.widgets import BigStatusLabel, ResultDisplayWidget, StatusWidget

if TYPE_CHECKING:
    from core.types import DetectionResult


class InfoPanel(QWidget):
    """
    Panel displaying system status, large pass/fail indicator, detailed results, and logs.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()

        self.big_status_label = BigStatusLabel()
        layout.addWidget(self.big_status_label)

        self.result_widget = ResultDisplayWidget()
        layout.addWidget(self.result_widget)

        log_group = QGroupBox("系統日誌")
        log_layout = QVBoxLayout()

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

    def update_result(self, result: DetectionResult) -> None:
        """Update all status indicators with the new result."""
        self.result_widget.update_result(result)
        
        # Also update the big status label here for convenience/redundancy
        # although MainWindow also does it. It's good practice for the panel
        # to manage its sub-components if possible, but we'll stick to the
        # specific request of 'update_result' wrapper.
        pass

    def append_log(self, message: str) -> None:
        self.log_text.append(message)
