from PyQt5.QtWidgets import QGroupBox, QTabWidget, QVBoxLayout, QWidget
import numpy as np
from app.gui.widgets import ImageViewer


class ImagePanel(QGroupBox):
    """
    Panel displaying the images (Original, Processed, Result) in tabs.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("影像預覽", parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()
        self.image_tabs = QTabWidget()
        
        self.original_image = ImageViewer("原始影像")
        self.image_tabs.addTab(self.original_image, "原始")

        self.processed_image = ImageViewer("處理後影像")
        self.image_tabs.addTab(self.processed_image, "處理後")

        self.result_image = ImageViewer("結果影像")
        self.image_tabs.addTab(self.result_image, "結果")

        layout.addWidget(self.image_tabs)
        self.setLayout(layout)

    def update_image(self, image: np.ndarray) -> None:
        """Update the live preview image (defaults to original/input view)."""
        self.original_image.display_image(image)

    def clear_all(self) -> None:
        """Clear all image viewers."""
        # Using the fact that ImageViewer inherits QLabel and we can probably setText or something if we expose it
        # But ImageViewer has set_image and no clear method in previous code, let's assume we can clear them.
        # Looking at widgets.py, ImageViewer inherits QLabel. clear() is a QLabel method.
        # But widgets.py ImageViewer sets text in init. 
        # Let's verify widgets.py content in my mind.. yes it has set_image.
        # I'll just clear the text or set to default. 
        # Actually in main_window.py lines 424-426 calling .clear() which is QLabel's clear.
        self.original_image.clear()
        self.processed_image.clear()
        self.result_image.clear()
        # Restore default text?
        self.original_image.setText("等待原始影像")
        self.processed_image.setText("等待處理後影像")
        self.result_image.setText("等待結果影像")
