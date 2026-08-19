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

        self.processed_image = ImageViewer("Processed image")

        self.result_image = ImageViewer("Result image")
        self._show_original_tab = True
        self._show_processed_tab = True
        self._rebuild_tabs()

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

    def set_optional_tabs_visible(
        self, *, show_original: bool, show_processed: bool
    ) -> None:
        """Show or hide optional image tabs while keeping the result tab visible.

        Args:
            show_original: Whether the original-image tab is visible.
            show_processed: Whether the processed-image tab is visible.

        Returns:
            None.
        """
        self._show_original_tab = bool(show_original)
        self._show_processed_tab = bool(show_processed)
        self._rebuild_tabs()
        self.set_language(self._language)

    def _rebuild_tabs(self) -> None:
        """Rebuild tab order from current visibility preferences."""
        current_widget = self.image_tabs.currentWidget()
        self.image_tabs.clear()
        if self._show_original_tab:
            self.image_tabs.addTab(self.original_image, tr(self._language, "original"))
        if self._show_processed_tab:
            self.image_tabs.addTab(self.processed_image, tr(self._language, "processed"))
        self.image_tabs.addTab(self.result_image, tr(self._language, "result"))

        if current_widget is not None:
            index = self.image_tabs.indexOf(current_widget)
            if index >= 0:
                self.image_tabs.setCurrentIndex(index)

    def set_language(self, language: str) -> None:
        """Update visible viewer labels."""
        self._language = normalize_language(language)
        self.setTitle(tr(self._language, "viewer"))
        tab_labels = (
            (self.original_image, "original"),
            (self.processed_image, "processed"),
            (self.result_image, "result"),
        )
        for widget, key in tab_labels:
            index = self.image_tabs.indexOf(widget)
            if index >= 0:
                self.image_tabs.setTabText(index, tr(self._language, key))
        self.original_image.set_language(self._language)
        self.processed_image.set_language(self._language)
        self.result_image.set_language(self._language)
        self.original_image.set_title(tr(self._language, "original_image"))
        self.processed_image.set_title(tr(self._language, "processed_image"))
        self.result_image.set_title(tr(self._language, "result_image"))
