"""Qt-native evidence cards: exact crops, fixed ROI and measured cell differences."""

from __future__ import annotations

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

_NAMES = {"red": "紅線", "green": "綠線", "orange": "橘線", "yellow": "黃線", "black": "黑線"}

#: ΔE76 spread across the 4x4 cells below which the grid says nothing the
#: single worst-cell number does not already say.
UNIFORM_CELL_TOLERANCE = 0.5


def color_label(color: str) -> str:
    """One wire name for every view; the table used to print raw ``red``."""
    return _NAMES.get(str(color).casefold(), str(color))


def position_label(position: int, color: str) -> str:
    return f"位置 {position} {color_label(color)}"


class CropView(QWidget):
    def __init__(self, image, bbox, roi, parent=None):
        super().__init__(parent)
        self.setMinimumSize(96, 112)
        self._image = None
        self._bbox, self._roi = bbox, roi
        if image is not None:
            rgb = image[:, :, ::-1].copy()
            height, width = rgb.shape[:2]
            self._image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#eef2f6"))
        if self._image is None:
            painter.setPen(QColor("#66788a"))
            painter.drawText(self.rect(), Qt.AlignCenter, "尚無影像")
            return
        scale = min((self.width() - 12) / self._image.width(), (self.height() - 12) / self._image.height())
        width, height = self._image.width() * scale, self._image.height() * scale
        left, top = (self.width() - width) / 2, (self.height() - height) / 2
        painter.drawImage(QRectF(left, top, width, height), self._image)
        if self._roi and self._bbox:
            x1, y1, x2, y2 = self._roi
            painter.setPen(QPen(QColor("#00e5ff"), 2))
            painter.drawRect(
                QRectF(
                    left + (x1 - self._bbox[0]) * scale,
                    top + (y1 - self._bbox[1]) * scale,
                    (x2 - x1) * scale,
                    (y2 - y1) * scale,
                )
            )


class GoldenEvidenceView(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.show_previews([])

    def show_previews(self, previews):
        old = self.takeWidget()
        if old is not None:
            old.deleteLater()
        host = QWidget()
        layout = QGridLayout(host)
        if not previews:
            label = QLabel("尚無可顯示的影像證據。建立基準後，這裡會顯示各位置的實拍對照。")
            label.setWordWrap(True)
            layout.addWidget(label, 0, 0)
        # Failures first. With five positions in a three-wide grid the last two
        # sit below the fold, so a banner saying "check the marked position"
        # could point at a card the operator has to scroll to find.
        ordered = sorted(previews, key=lambda p: not (p.get("row") or {}).get("reasons"))
        for index, preview in enumerate(ordered):
            layout.addWidget(self._card(preview), index // 3, index % 3)
        layout.setRowStretch((len(ordered) + 2) // 3, 1)
        self.setWidget(host)

    def _card(self, preview):
        card = QFrame()
        card.setObjectName("evidenceCard")
        card.setStyleSheet("QFrame#evidenceCard {background: white; border: 1px solid #d8e2ec; border-radius: 6px;}")
        layout = QVBoxLayout(card)
        row = preview["row"]
        state = "待檢查" if row is None else ("需處理" if row["reasons"] else "穩定")
        title = QLabel(f"{preview['position']} · {color_label(preview['color'])}　{state}")
        title.setStyleSheet("font-weight: bold; color: " + ("#b42318" if row and row["reasons"] else "#243b53"))
        layout.addWidget(title)
        images = QHBoxLayout()
        for kind, caption in (
            ("reference", "基準實拍（第 1 張）"),
            ("current", f"本次最差（第 {preview['frame']} 張）" if preview["frame"] else "等待本次取樣"),
        ):
            column = QVBoxLayout()
            label = QLabel(caption)
            label.setAlignment(Qt.AlignCenter)
            column.addWidget(label)
            column.addWidget(CropView(preview[f"{kind}_image"], preview[f"{kind}_bbox"], preview["roi"]))
            images.addLayout(column)
        layout.addLayout(images)
        if row:
            summary = QLabel(
                f"色差 {row['delta_e']:.2f} / {preview['delta_e_limit']:.2f}　"
                f"波動 {row['jitter']:.2f} / {preview['repeatability_limit']:.2f}"
            )
            layout.addWidget(summary)
            layout.addLayout(self._heatmap(preview))
            reason = QLabel("；".join(row["reasons"]) or "各區域均在設定範圍內")
            reason.setWordWrap(True)
            layout.addWidget(reason)
        return card

    def _heatmap(self, preview):
        """Sixteen cells only when they disagree.

        The grid exists to localise a partial shift. Lighting drift moves every
        cell together, and then it printed the same number sixteen times and
        pushed the remaining positions off screen.
        """
        grid = QGridLayout()
        grid.setSpacing(2)
        cells = [value for values in (preview["heatmap"] or []) for value in values]
        if not cells:
            return grid
        limit = preview["delta_e_limit"]
        if max(cells) - min(cells) <= UNIFORM_CELL_TOLERANCE:
            label = QLabel(f"16 個取樣區域一致偏移 {max(cells):.1f}")
            label.setStyleSheet(
                "background: %s; color: #243b53; padding: 4px;"
                % ("#fee4e2" if max(cells) > limit else "#e3f3eb")
            )
            grid.addWidget(label, 0, 0)
            return grid
        for y, values in enumerate(preview["heatmap"]):
            for x, value in enumerate(values):
                label = QLabel(f"{value:.1f}")
                label.setAlignment(Qt.AlignCenter)
                color = "#fee4e2" if value > limit else "#e3f3eb"
                label.setStyleSheet(f"background: {color}; color: #243b53; padding: 2px;")
                grid.addWidget(label, y, x)
        return grid
