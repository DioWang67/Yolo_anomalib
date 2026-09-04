from __future__ import annotations

import numpy as np
import pytest

from core.services.color_preflight_samples import (
    AXIS_HUE_SATURATION,
    AXIS_SATURATION_VALUE,
    ColorGamutSample,
    build_gamut_samples,
    measured_region,
)


def _solid(width: int, height: int, bgr: tuple[int, int, int]) -> np.ndarray:
    return np.full((height, width, 3), bgr, dtype=np.uint8)


def test_measured_region_is_the_whole_detection_box() -> None:
    """v6 measures the whole box, restricted to its largest connected match,
    not a fixed geometric sub-crop -- so there is no smaller region to mark."""
    crop = _solid(100, 50, (10, 20, 30))

    result = measured_region(crop)

    assert result is not None
    measured, box = result
    assert measured is crop
    assert box == (0, 0, 100, 50)


def test_measured_region_rejects_an_empty_crop() -> None:
    assert measured_region(np.zeros((0, 0, 3), dtype=np.uint8)) is None


def test_an_achromatic_envelope_is_plotted_on_saturation_and_value() -> None:
    """Black's recorded hue spans the circle, so hue would imply a tolerance.

    Drawing a box around the whole chart for a colour the baseline does not
    constrain by hue would be a picture of a tolerance that does not exist.
    """
    black = ColorGamutSample(
        color="Black",
        crop_bgr=None,
        measured_box=None,
        cloud_hsv=None,
        axis="",
        envelope_min=(0.0, 0.0, 9.0),
        envelope_max=(174.0, 77.0, 52.0),
        core_min=None,
        core_max=None,
        baseline_mean=None,
    )
    yellow = ColorGamutSample(
        color="Yellow",
        crop_bgr=None,
        measured_box=None,
        cloud_hsv=None,
        axis="",
        envelope_min=(18.0, 125.0, 116.0),
        envelope_max=(30.0, 202.0, 255.0),
        core_min=None,
        core_max=None,
        baseline_mean=None,
    )

    from core.services.color_preflight_samples import _axis_for

    assert _axis_for(black.envelope_min, black.envelope_max) == (
        AXIS_SATURATION_VALUE
    )
    assert _axis_for(yellow.envelope_min, yellow.envelope_max) == (
        AXIS_HUE_SATURATION
    )


def _sample(cloud: np.ndarray | None, **kwargs) -> ColorGamutSample:
    base = dict(
        color="Red",
        crop_bgr=None,
        measured_box=None,
        cloud_hsv=cloud,
        axis=AXIS_HUE_SATURATION,
        envelope_min=(2.0, 120.0, 90.0),
        envelope_max=(10.0, 210.0, 250.0),
        core_min=None,
        core_max=None,
        baseline_mean=(6.0, 180.0, 200.0),
    )
    base.update(kwargs)
    return ColorGamutSample(**base)


def test_the_plot_window_frames_the_envelope_rather_than_the_channel() -> None:
    """Red lives in a few degrees of hue out of 180.

    On a full-scale axis its envelope is a speck against an empty chart, and
    the question being asked -- where the cloud sits relative to that envelope
    -- becomes unreadable.
    """
    cloud = np.array([[6.0, 180.0, 200.0]] * 50, dtype=np.float32)

    (x_lo, x_hi), _y = _sample(cloud).plot_window()

    assert x_lo < 2.0 and x_hi > 10.0
    # Nowhere near the full 0..180 hue range.
    assert x_hi - x_lo < 40.0


def test_a_drift_outside_the_envelope_stays_inside_the_window() -> None:
    """Clipping the evidence into the border would hide the finding."""
    cloud = np.array([[24.0, 180.0, 200.0]] * 50, dtype=np.float32)

    (x_lo, x_hi), _y = _sample(cloud).plot_window()

    assert x_lo <= 2.0
    assert x_hi >= 24.0


def test_one_stray_highlight_does_not_zoom_the_plot_out() -> None:
    cloud = np.array(
        [[6.0, 180.0, 200.0]] * 200 + [[170.0, 10.0, 250.0]],
        dtype=np.float32,
    )

    (x_lo, x_hi), _y = _sample(cloud).plot_window()

    assert x_hi < 60.0


def test_density_finds_the_dominant_cluster_under_background_haze() -> None:
    """The colour check judges the dominant colour, so the plot must too.

    Equal-weight dots make two hundred background pixels look exactly as
    important as two hundred wire pixels; density is what separates them.
    """
    wire = np.array([[6.0, 180.0, 200.0]] * 300, dtype=np.float32)
    haze = np.array(
        [[float(hue), 40.0, 120.0] for hue in range(2, 30)], dtype=np.float32
    )
    sample = _sample(np.vstack([wire, haze]))

    density = sample.density(bins_x=32, bins_y=16)

    assert density is not None
    assert density.max() == pytest.approx(1.0)
    # The peak is one cell, and the haze registers without rivalling it.
    peak_cells = int((density > 0.9).sum())
    assert peak_cells == 1
    assert 0.0 < float(density[density < 0.9].max()) < 0.9


def test_no_cloud_means_no_density_rather_than_an_empty_grid() -> None:
    assert _sample(None).density() is None


def test_background_is_dropped_for_a_chromatic_colour_only() -> None:
    """Mirrors the runtime: it filters by saturation before scoring a colour.

    Black is the exception because black *is* the desaturated case, and
    filtering it would throw away the pixels being judged.
    """
    crop = np.zeros((20, 20, 3), dtype=np.uint8)
    # Half saturated red, half near-grey.
    crop[:10] = (40, 40, 200)
    crop[10:] = (120, 120, 125)
    items = [{"best_color": "Red"}, {"best_color": "Black"}]
    summary = {
        "Red": {"hsv_min": [0, 120, 90], "hsv_max": [10, 255, 255]},
        "Black": {"hsv_min": [0, 0, 9], "hsv_max": [174, 77, 52]},
    }

    filtered = build_gamut_samples(
        crop_paths=[],
        detections=[],
        color_items=items,
        baseline_summary=summary,
        sat_threshold=60.0,
    )

    # No crops on disk, so nothing to sample -- the point here is that the
    # call is well-formed for both colours and neither raises.
    assert set(filtered) == {"Red", "Black"}
    assert filtered["Red"].axis == AXIS_HUE_SATURATION
    assert filtered["Black"].axis == AXIS_SATURATION_VALUE


def _write_crop(directory, index: int, klass: str, bgr) -> str:
    import cv2

    path = directory / f"yolo_Cable1_A_120000_{klass}_{index}.png"
    cv2.imwrite(str(path), bgr)
    return str(path)


def test_a_skipped_crop_does_not_shift_every_later_picture(tmp_path) -> None:
    """The station writes a crop only for a usable box.

    Pairing by list position then shows one wire's picture beside another
    wire's numbers, which is the most misleading thing this panel could do.
    """
    red = _solid(20, 40, (30, 30, 200))
    green = _solid(20, 40, (30, 160, 40))
    # Detection 1 produced no crop; 0 and 2 did.
    crop_paths = [
        _write_crop(tmp_path, 0, "Red", red),
        _write_crop(tmp_path, 2, "Green", green),
    ]
    items = [
        {"best_color": "Red", "class_name": "Red"},
        {"best_color": "Yellow", "class_name": "Yellow"},
        {"best_color": "Green", "class_name": "Green"},
    ]
    summary = {
        "Red": {"hsv_min": [0, 100, 80], "hsv_max": [10, 255, 255]},
        "Yellow": {"hsv_min": [18, 125, 116], "hsv_max": [30, 202, 255]},
        "Green": {"hsv_min": [78, 100, 40], "hsv_max": [96, 255, 200]},
    }

    samples = build_gamut_samples(
        crop_paths=crop_paths,
        detections=[],
        color_items=items,
        baseline_summary=summary,
    )

    assert samples["Red"].crop_bgr is not None
    assert samples["Green"].crop_bgr is not None
    # The gap stays a gap rather than borrowing the next crop.
    assert samples["Yellow"].crop_bgr is None
    # And Green really is green, not Yellow's numbers over Green's picture.
    assert samples["Green"].crop_bgr[..., 1].mean() > 100


def test_a_crop_naming_another_class_is_not_shown(tmp_path) -> None:
    """Two records that disagree are not about the same detection."""
    crop_paths = [_write_crop(tmp_path, 0, "Green", _solid(20, 40, (30, 160, 40)))]
    items = [{"best_color": "Red", "class_name": "Red"}]

    samples = build_gamut_samples(
        crop_paths=crop_paths,
        detections=[],
        color_items=items,
        baseline_summary={"Red": {"hsv_min": [0, 100, 80], "hsv_max": [10, 255, 255]}},
    )

    assert samples["Red"].crop_bgr is None


def test_the_hit_mask_says_how_much_of_the_box_is_the_colour(tmp_path) -> None:
    """The wires run diagonally through an axis-aligned box.

    Part of the measured region is board, and the share that is not the colour
    has to be visible rather than inferred from a number nobody reads.
    """
    crop = _solid(20, 40, (250, 250, 250))
    # A saturated red stripe down the middle third.
    crop[:, 7:13] = (30, 30, 200)
    crop_paths = [_write_crop(tmp_path, 0, "Red", crop)]
    items = [{"best_color": "Red", "class_name": "Red"}]

    samples = build_gamut_samples(
        crop_paths=crop_paths,
        detections=[],
        color_items=items,
        baseline_summary={
            "Red": {"hsv_min": [0, 100, 80], "hsv_max": [10, 255, 255]}
        },
        sat_threshold=20.0,
    )

    sample = samples["Red"]
    assert sample.hit_mask is not None
    assert sample.hit_mask.shape == (40, 20)
    # The stripe is 6 of 20 columns, so a full-width box is mostly background
    # -- but the runtime divides a chromatic colour by its saturation-gated
    # pixels, and a white board does not clear that gate. Against the
    # denominator the line actually uses, the stripe is nearly all of it.
    assert sample.hit_fraction > 0.9
    # Against the whole region it would have read as a third.
    assert 0.2 < float(sample.hit_mask.mean()) < 0.45


def test_an_envelope_across_the_hue_seam_still_matches(tmp_path) -> None:
    """A stored hue_min above hue_max is how a wrapped envelope is written."""
    from core.services.color_preflight_samples import _envelope_mask

    hsv = np.zeros((2, 2, 3), dtype=np.float32)
    hsv[0, 0] = (178.0, 200.0, 200.0)
    hsv[0, 1] = (3.0, 200.0, 200.0)
    hsv[1, 0] = (90.0, 200.0, 200.0)
    hsv[1, 1] = (0.0, 10.0, 10.0)

    mask = _envelope_mask(hsv, (175.0, 100.0, 80.0), (8.0, 255.0, 255.0))

    assert mask is not None
    # Both sides of the seam count; the green and the dark pixel do not.
    assert mask.tolist() == [[True, True], [False, False]]


def test_black_is_measured_over_the_whole_region_not_the_gated_subset(
    tmp_path,
) -> None:
    """Black is scored on the whole crop against its learned coverage.

    Gating it by saturation would throw away the pixels being judged, so the
    denominator reported for black has to be the whole region -- describing a
    calculation the line does not perform would be worse than no figure.
    """
    crop = _solid(20, 40, (200, 200, 200))
    crop[:, 6:14] = (20, 20, 22)
    crop_paths = [_write_crop(tmp_path, 0, "Black", crop)]
    items = [{"best_color": "Black", "class_name": "Black"}]

    samples = build_gamut_samples(
        crop_paths=crop_paths,
        detections=[],
        color_items=items,
        baseline_summary={
            "Black": {"hsv_min": [0, 0, 9], "hsv_max": [174, 77, 52]}
        },
        sat_threshold=20.0,
    )

    sample = samples["Black"]
    assert sample.counted_mask is not None
    assert bool(sample.counted_mask.all())
    # 8 of 20 columns are the wire, and that is the share reported.
    assert 0.3 < sample.hit_fraction < 0.5
