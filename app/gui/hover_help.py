"""Compact, accessible hover help for dense Qt workflows."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFocusEvent
from PyQt5.QtWidgets import QLabel, QToolTip, QWidget


class HoverHelpBadge(QLabel):
    """Show explanatory copy only on hover or keyboard focus."""

    def __init__(
        self,
        help_text: str,
        *,
        language: str = "zh_TW",
        parent: QWidget | None = None,
    ) -> None:
        normalized_help = str(help_text).strip()
        if not normalized_help:
            raise ValueError("Hover help text must not be empty.")
        label = (
            "ⓘ 提示"
            if str(language).lower().startswith("zh")
            else "ⓘ Help"
        )
        super().__init__(label, parent)
        self.setObjectName("HoverHelpBadge")
        self.setProperty("hoverHelp", True)
        self.setToolTip(normalized_help)
        self.setToolTipDuration(20_000)
        self.setCursor(Qt.WhatsThisCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName(label)
        self.setAccessibleDescription(normalized_help)
        self.setStyleSheet(
            'QLabel[hoverHelp="true"] {'
            "color:#174a7e;background:#eef6ff;border:1px solid #b8d3ee;"
            "border-radius:10px;padding:2px 8px;font-weight:bold;}"
        )

    def focusInEvent(self, event: QFocusEvent) -> None:  # noqa: N802
        """Expose the same help to keyboard-only users."""
        super().focusInEvent(event)
        QToolTip.showText(
            self.mapToGlobal(self.rect().bottomLeft()),
            self.toolTip(),
            self,
        )

    def focusOutEvent(self, event: QFocusEvent) -> None:  # noqa: N802
        super().focusOutEvent(event)
        QToolTip.hideText()
