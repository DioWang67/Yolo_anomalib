"""Full-width host page for PIN-protected engineering controls."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.gui.i18n import normalize_language, tr
from app.gui.widgets import ImageViewer


class EngineeringSettingsPage(QWidget):
    """Host one persistent set of engineering controls in the main workspace."""

    back_to_inspection_requested = pyqtSignal()

    def __init__(
        self,
        engineering_controls: QScrollArea,
        *,
        language: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("EngineeringSettingsPage")
        self._language = normalize_language(language)
        self.engineering_controls = engineering_controls
        self.engineering_controls.setParent(self)
        self.engineering_controls.setVisible(True)
        self.engineering_controls.setMaximumHeight(16_777_215)
        self.engineering_controls.setFocusPolicy(Qt.NoFocus)
        self._target = ("", "")

        self.setStyleSheet(
            "QWidget#EngineeringSettingsPage {"
            "background:#f3f6fa;"
            "font-family:'Segoe UI','Microsoft JhengHei';font-size:10pt;}"
            "QFrame#engineeringPageHeader {"
            "background:white;border:1px solid #dce3ec;border-radius:9px;}"
            "QPushButton#engineeringBackButton {"
            "background:transparent;color:#34506b;border:0;padding:8px 12px;"
            "font-weight:600;}"
            "QPushButton#engineeringBackButton:hover {"
            "background:#edf3f8;border-radius:6px;}"
            "QFrame#engineeringPreviewCard {"
            "background:white;border:1px solid #dce3ec;border-radius:9px;}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(12)
        layout.addWidget(self._build_header())
        layout.addWidget(self._build_preview_card())
        layout.addWidget(self.engineering_controls, 1)
        self.set_language(self._language)

    def _build_header(self) -> QFrame:
        header = QFrame(self)
        header.setObjectName("engineeringPageHeader")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(10, 8, 14, 8)
        layout.setSpacing(12)

        self.back_button = QPushButton(header)
        self.back_button.setObjectName("engineeringBackButton")
        self.back_button.clicked.connect(self.back_to_inspection_requested.emit)
        layout.addWidget(self.back_button)

        divider = QFrame(header)
        divider.setFrameShape(QFrame.VLine)
        divider.setStyleSheet("color:#dce3ec;")
        layout.addWidget(divider)

        title_column = QVBoxLayout()
        title_column.setSpacing(2)
        self.title_label = QLabel(header)
        self.title_label.setStyleSheet(
            "font-size:15pt;font-weight:700;color:#1f3347;border:0;"
        )
        title_column.addWidget(self.title_label)
        self.hint_label = QLabel(header)
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(
            "color:#5b6b7c;font-size:9pt;border:0;"
        )
        title_column.addWidget(self.hint_label)
        layout.addLayout(title_column, 1)

        self.security_badge = QLabel(header)
        self.security_badge.setAlignment(Qt.AlignCenter)
        self.security_badge.setStyleSheet(
            "background:#eaf3ff;color:#245b8f;border:1px solid #b8d4ef;"
            "border-radius:11px;padding:4px 10px;font-weight:600;"
        )
        layout.addWidget(self.security_badge)
        return header

    def _build_preview_card(self) -> QFrame:
        card = QFrame(self)
        card.setObjectName("engineeringPreviewCard")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(18)

        context_column = QVBoxLayout()
        context_column.setSpacing(6)
        self.preview_title_label = QLabel(card)
        self.preview_title_label.setStyleSheet(
            "font-size:12pt;font-weight:700;color:#1f3347;border:0;"
        )
        context_column.addWidget(self.preview_title_label)

        self.preview_hint_label = QLabel(card)
        self.preview_hint_label.setWordWrap(True)
        self.preview_hint_label.setStyleSheet(
            "color:#5b6b7c;font-size:9pt;border:0;"
        )
        context_column.addWidget(self.preview_hint_label)

        self.preview_target_label = QLabel(card)
        self.preview_target_label.setStyleSheet(
            "background:#edf6ed;color:#246b36;border:1px solid #bad8bf;"
            "border-radius:9px;padding:4px 9px;font-weight:600;"
        )
        context_column.addWidget(self.preview_target_label, 0, Qt.AlignLeft)
        context_column.addStretch()
        layout.addLayout(context_column, 1)

        self.preview_viewer = ImageViewer()
        self.preview_viewer.setObjectName("engineeringPreview")
        self.preview_viewer.setMinimumSize(320, 180)
        self.preview_viewer.setMaximumSize(420, 230)
        layout.addWidget(self.preview_viewer, 2)
        return card

    @pyqtSlot(object)
    def update_preview(self, image: object) -> None:
        """Render a frame routed through the existing GUI-thread image signal."""
        self.preview_viewer.display_image(image)

    def clear_preview(self) -> None:
        """Remove a stale frame when no live inspection source is active."""
        self.preview_viewer.clear()

    def set_target(self, product: str, area: str) -> None:
        """Update the target context shown beside the calibration preview."""
        self._target = (str(product).strip(), str(area).strip())
        self._update_target_label()

    def configure_tab_order(self, widgets: tuple[QWidget, ...]) -> None:
        """Follow visual order instead of construction order in the grid."""
        focus_chain = (self.back_button, *widgets)
        for current, following in zip(
            focus_chain[:-1],
            focus_chain[1:],
            strict=True,
        ):
            QWidget.setTabOrder(current, following)

    def _update_target_label(self) -> None:
        product, area = self._target
        if product and area:
            text = tr(self._language, "engineer_preview_target").format(
                product=product,
                area=area,
            )
        else:
            text = tr(self._language, "engineer_preview_target_missing")
        self.preview_target_label.setText(text)

    def set_language(self, language: str) -> None:
        """Update page chrome while preserving all engineering control state."""
        self._language = normalize_language(language)
        self.back_button.setText(tr(self._language, "back_to_inspection"))
        self.title_label.setText(tr(self._language, "engineer_page_title"))
        self.hint_label.setText(tr(self._language, "engineer_page_hint"))
        self.security_badge.setText(tr(self._language, "engineer_page_badge"))
        self.preview_title_label.setText(
            tr(self._language, "engineer_preview_title")
        )
        self.preview_hint_label.setText(
            tr(self._language, "engineer_preview_hint")
        )
        preview_title = tr(self._language, "engineer_preview_image")
        self.preview_viewer.set_language(self._language)
        self.preview_viewer.set_title(preview_title)
        self._update_target_label()
