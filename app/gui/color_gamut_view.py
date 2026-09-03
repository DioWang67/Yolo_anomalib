"""One row of visual evidence per colour: the crop, and where its pixels sit.

Built to be read, not parsed. Each row carries the crop the station measured
with the measured region marked on it, and today's pixels plotted inside the
envelope the baseline recorded -- so a colour drifting toward its neighbour
looks different from one losing saturation, which a single falling percentage
cannot show.

The numbers stay, small, on the right. They are the audit trail; the picture is
the reading.
"""

from __future__ import annotations

from PyQt5.QtCore import QRect, QSize, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QSizePolicy, QWidget

from core.services.color_preflight import ColorPreflightColor
from core.services.color_preflight_samples import (
    AXIS_SATURATION_VALUE,
    ColorGamutSample,
)

#: The crop and the cloud are the reading, so they get the room. Everything
#: else on the row is a caption.
_ROW_HEIGHT = 108
_THUMB_WIDTH = 74
_PLOT_WIDTH = 190
_PAD = 10
#: Room for the swatch, the colour name and its read/expected count.
_LABEL_WIDTH = 104
#: Room for the state word, the retention bar and the figures under it.
_FIGURES_WIDTH = 168
#: Density grid for the cloud; roughly three screen pixels per cell at the
#: plot's size, which keeps a cluster solid without turning it into blocks.
_DENSITY_BINS_X = 62
_DENSITY_BINS_Y = 30


def _qimage_from_bgr(crop) -> QImage:
    """Wrap an OpenCV BGR array as a QImage, copied so it owns its buffer."""
    height, width = crop.shape[:2]
    rgb = crop[:, :, ::-1].copy()
    return QImage(
        rgb.data, width, height, 3 * width, QImage.Format_RGB888
    ).copy()


class ColorGamutRow(QWidget):
    """Paint one colour's crop, cloud and figures on a single line."""

    def __init__(
        self,
        *,
        reading: ColorPreflightColor,
        sample: ColorGamutSample | None,
        state_text: str,
        stroke: str,
        wash: str,
        ink: str,
        ink_faint: str,
        rule: str,
        surface: str,
        ground: str,
        swatch: QPixmap | None,
        retention_floor: float,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._reading = reading
        self._sample = sample
        self._state_text = state_text
        self._stroke = QColor(stroke)
        self._wash = QColor(wash)
        self._ink = QColor(ink)
        self._ink_faint = QColor(ink_faint)
        self._rule = QColor(rule)
        self._surface = QColor(surface)
        self._ground = QColor(ground)
        self._swatch = swatch
        self._retention_floor = retention_floor
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(_ROW_HEIGHT)
        if sample is not None and sample.measured_box is not None:
            self.setToolTip(
                f"{reading.color} · {sample.axis}\n"
                f"measured {sample.measured_box}"
            )

    def sizeHint(self) -> QSize:  # noqa: D102
        return QSize(
            _LABEL_WIDTH + _THUMB_WIDTH + _PLOT_WIDTH + _FIGURES_WIDTH + _PAD * 5,
            _ROW_HEIGHT,
        )

    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: D102
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            body = self.rect().adjusted(0, 0, -1, -1)
            painter.setPen(QPen(self._rule))
            painter.setBrush(
                self._wash if not self._reading.is_ok else self._surface
            )
            painter.drawRoundedRect(body, 6, 6)

            x = body.left() + _PAD
            self._paint_label(painter, x, body)
            x += _LABEL_WIDTH
            self._paint_thumbnail(painter, x, body)
            x += _THUMB_WIDTH + _PAD
            self._paint_plot(painter, x, body)
            x += _PLOT_WIDTH + _PAD
            self._paint_figures(painter, x, body)
        finally:
            painter.end()

    def _paint_label(self, painter: QPainter, x: int, body: QRect) -> None:
        if self._swatch is not None:
            painter.drawPixmap(x, body.top() + 30, self._swatch)
        name_rect = QRect(x + 26, body.top() + 28, _LABEL_WIDTH - 28, 20)
        font = painter.font()
        font.setPointSize(11)
        font.setBold(not self._reading.is_ok)
        painter.setFont(font)
        painter.setPen(self._ink)
        painter.drawText(name_rect, Qt.AlignLeft | Qt.AlignVCenter, self._reading.color)

        font.setPointSize(8)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(self._ink_faint)
        painter.drawText(
            QRect(x + 26, body.top() + 50, _LABEL_WIDTH - 28, 18),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"{self._reading.observed_count} / {self._reading.expected_count}",
        )

    def _paint_thumbnail(self, painter: QPainter, x: int, body: QRect) -> None:
        sample = self._sample
        slot = QRect(x, body.top() + 10, _THUMB_WIDTH, _ROW_HEIGHT - 21)
        painter.setPen(QPen(self._rule))
        painter.setBrush(self._ground)
        painter.drawRect(slot)
        if sample is None or sample.crop_bgr is None:
            return
        image = _qimage_from_bgr(sample.crop_bgr)
        # Nearest-neighbour on purpose: a wire crop is 25 pixels across, and
        # smoothing it invents detail that was never measured.
        scaled = image.scaled(
            slot.size(), Qt.KeepAspectRatio, Qt.FastTransformation
        )
        target = QRect(
            slot.left() + (slot.width() - scaled.width()) // 2,
            slot.top() + (slot.height() - scaled.height()) // 2,
            scaled.width(),
            scaled.height(),
        )
        painter.drawImage(target, scaled)
        box = sample.measured_box
        if box is None:
            return
        # The saved crop is the whole detection box; this outline is the part
        # the colour check actually used.
        crop_h, crop_w = sample.crop_bgr.shape[:2]
        scale_x = scaled.width() / crop_w
        scale_y = scaled.height() / crop_h
        box_rect = QRect(
            target.left() + int(box[0] * scale_x),
            target.top() + int(box[1] * scale_y),
            max(1, int((box[2] - box[0]) * scale_x)),
            max(1, int((box[3] - box[1]) * scale_y)),
        )
        # Veil the pixels inside the box that are *not* this colour. The wires
        # run diagonally through an axis-aligned box, so part of it is board --
        # and "the box is half background" has to be visible rather than
        # inferred from a coverage number nobody reads.
        mask = sample.hit_mask
        if mask is not None and mask.size:
            veil = QImage(
                mask.shape[1], mask.shape[0], QImage.Format_ARGB32
            )
            veil.fill(Qt.transparent)
            miss = QColor(15, 22, 25, 150).rgba()
            for row in range(mask.shape[0]):
                for column in range(mask.shape[1]):
                    if not mask[row, column]:
                        veil.setPixel(column, row, miss)
            painter.drawImage(
                box_rect,
                veil.scaled(
                    box_rect.size(), Qt.IgnoreAspectRatio, Qt.FastTransformation
                ),
            )
        painter.setBrush(Qt.NoBrush)
        pen = QPen(QColor(255, 255, 255, 220))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRect(box_rect)

    def _paint_plot(self, painter: QPainter, x: int, body: QRect) -> None:
        plot = QRect(x, body.top() + 10, _PLOT_WIDTH, _ROW_HEIGHT - 21)
        painter.setPen(QPen(self._rule))
        painter.setBrush(self._surface)
        painter.drawRect(plot)
        sample = self._sample
        if sample is None:
            return
        (x_lo, x_hi), (y_lo, y_hi) = sample.plot_window()
        x_channel, y_channel = sample.axis_channels()
        x_span = max(1e-6, x_hi - x_lo)
        y_span = max(1e-6, y_hi - y_lo)

        def to_point(values) -> tuple[int, int]:
            px = plot.left() + int(
                plot.width()
                * min(1.0, max(0.0, (values[x_channel] - x_lo) / x_span))
            )
            # Screen y grows downward; the second channel should read upward.
            py = plot.bottom() - int(
                plot.height()
                * min(1.0, max(0.0, (values[y_channel] - y_lo) / y_span))
            )
            return px, py

        def to_rect(low, high) -> QRect | None:
            if low is None or high is None:
                return None
            x1, y1 = to_point(low)
            x2, y2 = to_point(high)
            return QRect(
                min(x1, x2),
                min(y1, y2),
                max(1, abs(x2 - x1)),
                max(1, abs(y2 - y1)),
            )

        envelope = to_rect(sample.envelope_min, sample.envelope_max)
        core = to_rect(sample.core_min, sample.core_max)
        if core is not None:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(self._stroke.red(), self._stroke.green(), self._stroke.blue(), 34))
            painter.drawRect(core)
        if envelope is not None:
            pen = QPen(self._stroke)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(envelope)

        density = sample.density(_DENSITY_BINS_X, _DENSITY_BINS_Y)
        if density is not None:
            rows, columns = density.shape
            cell_w = plot.width() / columns
            cell_h = plot.height() / rows
            painter.setPen(Qt.NoPen)
            for row_index in range(rows):
                for column_index in range(columns):
                    weight = float(density[row_index, column_index])
                    if weight <= 0.0:
                        continue
                    painter.setBrush(
                        QColor(31, 41, 51, int(20 + 215 * weight))
                    )
                    painter.drawRect(
                        QRect(
                            plot.left() + int(column_index * cell_w),
                            plot.top() + int(row_index * cell_h),
                            max(1, int(cell_w + 1)),
                            max(1, int(cell_h + 1)),
                        )
                    )

        # Two characters instead of a sentence: which pair of channels this
        # plot is showing, since it is chosen per colour.
        painter.setPen(QPen(self._ink_faint))
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)
        painter.drawText(
            plot.adjusted(4, 2, -4, -2),
            Qt.AlignTop | Qt.AlignRight,
            "S×V" if sample.axis == AXIS_SATURATION_VALUE else "H×S",
        )

        if sample.baseline_mean is not None:
            cx, cy = to_point(sample.baseline_mean)
            pen = QPen(self._stroke)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(cx - 4, cy, cx + 4, cy)
            painter.drawLine(cx, cy - 4, cx, cy + 4)

    def _paint_figures(self, painter: QPainter, x: int, body: QRect) -> None:
        reading = self._reading
        font = painter.font()
        font.setPointSize(9)
        font.setBold(not reading.is_ok)
        painter.setFont(font)
        painter.setPen(self._stroke if not reading.is_ok else self._ink)
        painter.drawText(
            QRect(x, body.top() + 20, _FIGURES_WIDTH, 18),
            Qt.AlignLeft | Qt.AlignVCenter,
            self._state_text,
        )

        bar = QRect(x, body.top() + 44, _FIGURES_WIDTH - 12, 10)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._ground)
        painter.drawRoundedRect(bar, 3, 3)
        retention = reading.retention
        if retention is not None:
            filled = max(0.0, min(1.0, float(retention)))
            if filled > 0:
                painter.setBrush(self._stroke)
                painter.drawRoundedRect(
                    QRect(
                        bar.left(),
                        bar.top(),
                        max(2, int(bar.width() * filled)),
                        bar.height(),
                    ),
                    3,
                    3,
                )
            if 0.0 < self._retention_floor < 1.0:
                tick = bar.left() + int(bar.width() * self._retention_floor)
                pen = QPen(self._ink)
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawLine(tick, bar.top() - 2, tick, bar.bottom() + 2)

        font.setPointSize(8)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(self._ink_faint)
        # Retention past the reference is good news and needs no precision, so
        # it stops counting rather than printing 786% and reading as a fault.
        if retention is None:
            shown = "--"
        elif retention > 1.5:
            shown = "＞150%"
        else:
            shown = f"{retention:.0%}"
        figures = shown
        if reading.margin is not None:
            figures += f"　{reading.margin:+.3f}"
            if reading.reference_margin is not None:
                figures += f" / {reading.reference_margin:+.3f}"
        hit = self._sample.hit_fraction if self._sample is not None else None
        if hit is not None:
            figures += f"　命中 {hit:.0%}"
        painter.drawText(
            QRect(x, body.top() + 58, _FIGURES_WIDTH, 18),
            Qt.AlignLeft | Qt.AlignVCenter,
            figures,
        )
