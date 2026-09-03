"""The visual evidence behind a pre-shift reading: crops and colour clouds.

A margin is one number standing in for a distribution. It answers "how much
room is left" but not "room in which direction", and at a machine the second
question is the one that gets acted on: a colour drifting toward its neighbour
looks nothing like a colour losing saturation, and both show up as the same
falling percentage.

So this module produces, per colour, the two things an operator can read
directly:

* **the crop that was measured** -- the saved detection crop with the region
  the colour check actually used marked on it, because the saved crop is the
  whole bounding box while the measurement uses the station's ROI inset and
  then a centred sub-crop; and
* **where its pixels sit inside the baseline's envelope** -- today's cloud
  against the recorded min/max box, the 10th-to-90th percentile core, and the
  baseline mean.

Both geometries come from the production helpers (``extract_bbox_roi`` and
``center_crop_by_ratio``) rather than being recomputed here. A picture of the
wrong region is worse than no picture, and this file existing at all is only
justified while it shows what the line measured.

No Qt here: this is measurement, and the painting lives in the GUI layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.color_sampling import center_crop_by_ratio
from core.services.slot_roi import ColorRoiPolicy, extract_bbox_roi

#: Hue is what separates red from orange from yellow, so it is the axis worth
#: watching -- unless the baseline barely constrains it, which is what an
#: achromatic colour like black looks like. Above this span (of 180 OpenCV hue
#: degrees) hue says nothing about that colour and saturation/value do.
_WIDE_HUE_SPAN = 60.0

#: At most this many pixels are kept per colour for the cloud. A wire crop is a
#: few thousand pixels; plotting all of them is slower and no more legible.
_MAX_CLOUD_POINTS = 1500

AXIS_HUE_SATURATION = "hue_saturation"
AXIS_SATURATION_VALUE = "saturation_value"

#: Channel index pairs (x, y) for each axis choice, in OpenCV HSV order.
_AXIS_CHANNELS = {
    AXIS_HUE_SATURATION: (0, 1),
    AXIS_SATURATION_VALUE: (1, 2),
}
#: Full scale of each channel, in OpenCV HSV units.
_CHANNEL_SCALE = (180.0, 255.0, 255.0)

#: Breathing room around the plotted window, as a fraction of its span, so the
#: envelope's edges are inside the frame rather than on it.
_WINDOW_PAD_FRACTION = 0.18
#: A floor on the window's span, as a fraction of the channel. Without it a
#: very tight envelope zooms until sensor noise looks like drift.
_MIN_WINDOW_FRACTION = 0.08


@dataclass(frozen=True)
class ColorGamutSample:
    """One colour's measured pixels against the envelope it is held to."""

    color: str
    #: The saved detection crop, BGR, exactly as the station stored it.
    crop_bgr: np.ndarray | None
    #: The region of ``crop_bgr`` the colour check measured, as
    #: ``(x1, y1, x2, y2)``. ``None`` when the policy rejected the box.
    measured_box: tuple[int, int, int, int] | None
    #: Measured pixels as HSV rows, thinned for plotting.
    cloud_hsv: np.ndarray | None
    #: Which two channels to plot, and why -- see ``_WIDE_HUE_SPAN``.
    axis: str
    envelope_min: tuple[float, float, float] | None
    envelope_max: tuple[float, float, float] | None
    core_min: tuple[float, float, float] | None
    core_max: tuple[float, float, float] | None
    baseline_mean: tuple[float, float, float] | None
    #: Per-pixel mask over the measured region: True where the pixel falls
    #: inside the baseline's recorded envelope. Shaped like the region, so it
    #: can be drawn over the marked box -- which is how "the box is half
    #: background" becomes visible instead of inferred.
    hit_mask: np.ndarray | None = None
    #: Pixels the runtime actually divides by. For a chromatic colour that is
    #: the saturation-gated subset; for black it is the whole region, because
    #: black is scored on the whole crop against its learned coverage.
    counted_mask: np.ndarray | None = None

    @property
    def has_cloud(self) -> bool:
        return self.cloud_hsv is not None and len(self.cloud_hsv) > 0

    @property
    def hit_fraction(self) -> float | None:
        """Share of what the runtime divides by that is inside the envelope.

        The denominator is the runtime's, not the whole region: a chromatic
        colour is scored over the saturation-gated pixels while black is scored
        over the whole crop, so reporting one share for both would describe a
        calculation the line does not perform for four colours out of five.

        Not the checker's ``coverage_mean`` either: that was measured with the
        calibration's dominant-hue mask, and this is a plain envelope test, so
        the two are not comparable and are never shown as if they were. What
        this is good for is its own trend -- computed the same way every shift,
        a falling share means the marked box is catching less of the wire than
        it used to, which is what a part shifting in the fixture looks like.
        """
        if self.hit_mask is None or self.hit_mask.size == 0:
            return None
        counted = self.counted_mask
        if counted is None:
            return float(self.hit_mask.mean())
        total = int(counted.sum())
        if total == 0:
            return None
        return float((self.hit_mask & counted).sum()) / total

    def axis_channels(self) -> tuple[int, int]:
        return _AXIS_CHANNELS.get(self.axis, _AXIS_CHANNELS[AXIS_HUE_SATURATION])

    def axis_scale(self) -> tuple[float, float]:
        x_channel, y_channel = self.axis_channels()
        return _CHANNEL_SCALE[x_channel], _CHANNEL_SCALE[y_channel]

    def plot_window(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """The value range each axis should span, as ``((x0, x1), (y0, y1))``.

        Not the channel's full range. Red and orange live within a few degrees
        of hue out of 180, so a full-scale plot draws their envelope as a speck
        against an empty chart -- and the question being asked is where the
        cloud sits *relative to that envelope*, which needs both filling the
        frame. The window covers the envelope and the bulk of today's cloud, so
        a drift outside the envelope stays visible instead of being clipped
        into the border.
        """
        return (
            self._axis_window(self.axis_channels()[0]),
            self._axis_window(self.axis_channels()[1]),
        )

    def density(self, bins_x: int = 56, bins_y: int = 36) -> np.ndarray | None:
        """Normalised 2D density of the cloud over ``plot_window``.

        Density rather than a scatter of equal dots, because the measured
        region is a wire against a board: after the runtime's saturation pass
        the background is still there, and an equal-weight dot makes two
        hundred background pixels look exactly as important as two hundred
        wire pixels. The colour check judges the dominant colour, so the
        picture has to show where the mass is -- a dark cluster with the
        background as haze around it -- rather than treating every pixel as
        one vote.

        Rows are the y axis, top row highest value, so it can be drawn
        straight down the screen.
        """
        if not self.has_cloud:
            return None
        x_channel, y_channel = self.axis_channels()
        (x_lo, x_hi), (y_lo, y_hi) = self.plot_window()
        if x_hi <= x_lo or y_hi <= y_lo:
            return None
        counts, _, _ = np.histogram2d(
            self.cloud_hsv[:, x_channel],
            self.cloud_hsv[:, y_channel],
            bins=(bins_x, bins_y),
            range=((x_lo, x_hi), (y_lo, y_hi)),
        )
        peak = counts.max()
        if peak <= 0:
            return None
        # Log scaling: a wire cluster can be two orders of magnitude denser
        # than the haze, and on a linear scale the haze disappears entirely --
        # which is the evidence that the crop caught something other than the
        # wire.
        scaled = np.log1p(counts) / np.log1p(peak)
        return np.flipud(scaled.T)

    def _axis_window(self, channel: int) -> tuple[float, float]:
        scale = _CHANNEL_SCALE[channel]
        lows: list[float] = []
        highs: list[float] = []
        for low, high in (
            (self.envelope_min, self.envelope_max),
            (self.core_min, self.core_max),
        ):
            if low is not None and high is not None:
                lows.append(low[channel])
                highs.append(high[channel])
        if self.has_cloud:
            column = self.cloud_hsv[:, channel]
            # Percentiles, not min/max: one stray pixel from a highlight
            # should not zoom the whole plot out and flatten the difference
            # this is here to show.
            lows.append(float(np.percentile(column, 2)))
            highs.append(float(np.percentile(column, 98)))
        if not lows:
            return 0.0, scale
        low = min(lows)
        high = max(highs)
        span = max(high - low, scale * _MIN_WINDOW_FRACTION)
        pad = span * _WINDOW_PAD_FRACTION
        return max(0.0, low - pad), min(scale, high + pad)


def _triple(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


def _axis_for(
    envelope_min: tuple[float, float, float] | None,
    envelope_max: tuple[float, float, float] | None,
) -> str:
    """Choose the axis pair the baseline actually constrains for this colour.

    Black's recorded hue runs almost the full circle, so plotting hue for it
    would draw a box around the whole chart and imply a tolerance that does not
    exist. Its saturation and value are what the envelope pins down.
    """
    if envelope_min is None or envelope_max is None:
        return AXIS_HUE_SATURATION
    if envelope_max[0] - envelope_min[0] > _WIDE_HUE_SPAN:
        return AXIS_SATURATION_VALUE
    return AXIS_HUE_SATURATION


def _envelope_mask(
    hsv: np.ndarray,
    low: tuple[float, float, float] | None,
    high: tuple[float, float, float] | None,
) -> np.ndarray | None:
    """Per-pixel test against the recorded envelope, hue seam included.

    A stored ``hue_min`` above ``hue_max`` is how an envelope that straddles
    the 0/179 seam is written down; testing it as a plain range would reject
    every red pixel on one side of the seam.
    """
    if low is None or high is None:
        return None
    hue = hsv[:, :, 0]
    if low[0] <= high[0]:
        hue_ok = (hue >= low[0]) & (hue <= high[0])
    else:
        hue_ok = (hue >= low[0]) | (hue <= high[0])
    return (
        hue_ok
        & (hsv[:, :, 1] >= low[1])
        & (hsv[:, :, 1] <= high[1])
        & (hsv[:, :, 2] >= low[2])
        & (hsv[:, :, 2] <= high[2])
    )


def _crops_by_detection(crop_paths: Sequence[str]) -> dict[int, Path]:
    """Index the saved crops by the detection they came from.

    The station names each crop ``..._<class>_<index>.png``, and it writes one
    only for a usable bounding box -- so pairing crops to colour measurements
    by list position mispairs every crop after the first skipped detection, and
    shows one wire's picture beside another wire's numbers. Keying on the index
    in the name cannot drift that way.
    """
    indexed: dict[int, Path] = {}
    for raw in crop_paths:
        path = Path(str(raw))
        parts = path.stem.rsplit("_", 2)
        if len(parts) < 3:
            continue
        try:
            indexed[int(parts[-1])] = path
        except ValueError:
            continue
    return indexed


def _crop_class(path: Path) -> str:
    parts = path.stem.rsplit("_", 2)
    return parts[-2] if len(parts) >= 3 else ""


def _thin(rows: np.ndarray, limit: int = _MAX_CLOUD_POINTS) -> np.ndarray:
    if len(rows) <= limit:
        return rows
    # Even stride rather than a random sample: the same crop should draw the
    # same cloud every time it is opened.
    step = int(np.ceil(len(rows) / limit))
    return rows[::step]


def measured_region(
    crop_bgr: np.ndarray,
    roi_policy: ColorRoiPolicy,
    center_margin_ratio: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Return the pixels the colour check measured, and where they came from.

    The saved crop is the whole detection box, so the station's inset and the
    centred sub-crop are applied here -- through the same two helpers the
    runtime uses, because a marked region that does not match the measurement
    would be a picture of a lie.
    """
    if not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
        return None
    height, width = crop_bgr.shape[:2]
    inset = extract_bbox_roi(
        crop_bgr, [0, 0, width, height], policy=roi_policy
    )
    if inset is None or inset.size == 0:
        return None
    inset_x = int(round(width * roi_policy.inset_x_ratio))
    inset_y = int(round(height * roi_policy.inset_y_ratio))
    measured = center_crop_by_ratio(inset, center_margin_ratio)
    if measured.size == 0:
        return None
    # center_crop_by_ratio trims the same fraction from each edge, and returns
    # the input untouched when it cannot; recovering the offset from the shapes
    # keeps this in step with it rather than restating its rule.
    margin_y = (inset.shape[0] - measured.shape[0]) // 2
    margin_x = (inset.shape[1] - measured.shape[1]) // 2
    x1 = inset_x + margin_x
    y1 = inset_y + margin_y
    return measured, (x1, y1, x1 + measured.shape[1], y1 + measured.shape[0])


def build_gamut_samples(
    *,
    crop_paths: Sequence[str],
    detections: Sequence[Mapping[str, object]],
    color_items: Sequence[Mapping[str, object]],
    baseline_summary: Mapping[str, object],
    roi_policy: ColorRoiPolicy,
    center_margin_ratio: float,
    sat_threshold: float = 0.0,
) -> dict[str, ColorGamutSample]:
    """Build one sample per colour that was read, keyed by colour name.

    Crops are matched to colour measurements by position: the station writes
    one crop per detection in detection order, and the colour check reports one
    item per detection in the same order. A colour read twice keeps the crop
    whose measured region is smallest -- the tighter box is the one with least
    room to spare, and this panel is about what is closest to the edge.
    """
    envelopes = {
        str(name).casefold(): value
        for name, value in baseline_summary.items()
        if isinstance(value, Mapping)
    }
    crops = _crops_by_detection(crop_paths)
    samples: dict[str, ColorGamutSample] = {}
    for index, item in enumerate(color_items):
        observed = str(item.get("best_color") or "").strip()
        if not observed:
            continue
        crop_bgr: np.ndarray | None = None
        path = crops.get(index)
        if path is not None and path.is_file():
            # The crop is named for the detector's class, which is what the
            # colour check was asked about; a disagreement means these two
            # records are not about the same detection, and showing the picture
            # anyway would be worse than showing none.
            declared = str(
                item.get("class_name") or item.get("class") or ""
            ).strip()
            crop_class = _crop_class(path)
            if not declared or crop_class.casefold() == declared.casefold():
                crop_bgr = cv2.imread(str(path))
        region = (
            measured_region(crop_bgr, roi_policy, center_margin_ratio)
            if crop_bgr is not None
            else None
        )
        stats = envelopes.get(observed.casefold(), {})
        envelope_min = _triple(stats.get("hsv_min"))
        envelope_max = _triple(stats.get("hsv_max"))
        axis = _axis_for(envelope_min, envelope_max)

        cloud = None
        measured_box = None
        hit_mask = None
        counted_mask = None
        if region is not None:
            measured, measured_box = region
            hsv = cv2.cvtColor(measured, cv2.COLOR_BGR2HSV)
            rows = hsv.reshape(-1, 3).astype(np.float32)
            # Mirror the runtime's own first pass over these pixels: it drops
            # everything below the saturation threshold before scoring a
            # chromatic colour (``StatsColorChecker.check``), and scores black
            # on the unfiltered crop because black *is* the desaturated case.
            # Without this the cloud is mostly the board behind the wire, and a
            # picture of the background is worse than no picture.
            if axis != AXIS_SATURATION_VALUE and sat_threshold > 0:
                kept = rows[rows[:, 1] >= float(sat_threshold)]
                if len(kept):
                    rows = kept
            cloud = _thin(rows)
            hit_mask = _envelope_mask(hsv, envelope_min, envelope_max)
            counted_mask = (
                np.ones(hsv.shape[:2], dtype=bool)
                if axis == AXIS_SATURATION_VALUE or sat_threshold <= 0
                else hsv[:, :, 1] >= float(sat_threshold)
            )
        sample = ColorGamutSample(
            color=observed,
            crop_bgr=crop_bgr,
            measured_box=measured_box,
            cloud_hsv=cloud,
            axis=axis,
            envelope_min=envelope_min,
            envelope_max=envelope_max,
            core_min=_triple(stats.get("hsv_p10")),
            core_max=_triple(stats.get("hsv_p90")),
            baseline_mean=_triple(stats.get("hsv_mean")),
            hit_mask=hit_mask,
            counted_mask=counted_mask,
        )
        existing = samples.get(observed)
        if existing is None or _box_area(measured_box) < _box_area(
            existing.measured_box
        ):
            samples[observed] = sample
    return samples


def _box_area(box: tuple[int, int, int, int] | None) -> int:
    if box is None:
        return 1 << 30
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])
