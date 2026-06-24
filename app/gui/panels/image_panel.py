import numpy as np
from PyQt5.QtWidgets import QGroupBox, QTabWidget, QVBoxLayout, QWidget

from app.gui.i18n import normalize_language, tr
from app.gui.widgets import AutoPhaseBanner, ImageViewer


class ImagePanel(QGroupBox):
    """Central inspection image viewer with original, processed, and result tabs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Inspection Viewer", parent)
        self._language = "en"
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Always-visible Auto Mode phase banner (hidden outside Auto Mode).
        # Sits above the tabs so operators see the phase regardless of which
        # image tab is currently selected.
        self.auto_phase_banner = AutoPhaseBanner()
        layout.addWidget(self.auto_phase_banner)

        self.image_tabs = QTabWidget()
        self.image_tabs.setStyleSheet(
            """
            QTabWidget::pane {
                border: 1px solid #1f2933;
                background: #111827;
                border-radius: 6px;
            }
            QTabBar::tab {
                background: #e5e7eb;
                color: #374151;
                padding: 7px 14px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #111827;
                color: #ffffff;
            }
            """
        )

        self.original_image = ImageViewer("Original image")
        self.image_tabs.addTab(self.original_image, "Original")

        self.processed_image = ImageViewer("Processed image")
        self.image_tabs.addTab(self.processed_image, "Processed")

        self.result_image = ImageViewer("Result image")
        self.image_tabs.addTab(self.result_image, "Result")

        layout.addWidget(self.image_tabs)
        self.setLayout(layout)
        self.set_language(self._language)

    def update_image(self, image: np.ndarray) -> None:
        """Update the live preview image."""
        self.original_image.display_image(image)

    def clear_all(self) -> None:
        """Clear all image viewers."""
        self.original_image.clear()
        self.processed_image.clear()
        self.result_image.clear()

    def set_language(self, language: str) -> None:
        """Update visible viewer labels."""
        self._language = normalize_language(language)
        self.setTitle(tr(self._language, "viewer"))
        self.image_tabs.setTabText(0, tr(self._language, "original"))
        self.image_tabs.setTabText(1, tr(self._language, "processed"))
        self.image_tabs.setTabText(2, tr(self._language, "result"))
        self.original_image.set_language(self._language)
        self.processed_image.set_language(self._language)
        self.result_image.set_language(self._language)
        self.original_image.set_title(tr(self._language, "original_image"))
        self.processed_image.set_title(tr(self._language, "processed_image"))
        self.result_image.set_title(tr(self._language, "result_image"))
