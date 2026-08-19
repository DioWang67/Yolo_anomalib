import json
from types import SimpleNamespace

import numpy as np
import pytest

from core.color_qc_enhanced import ColorQCEnhanced
from core.services.color_checker import (
    COLOR_CHECK_EVALUATED_STATUS,
    COLOR_CHECK_NO_DETECTIONS_STATUS,
    ColorCheckerService,
)
from core.stats_color_checker import (
    COLOR_CONF_THRESHOLDS,
    DEFAULT_RATIO_THRESHOLD,
    StatsColorChecker,
)


class StubColorChecker:
    def __init__(
        self,
        *,
        best_color: str,
        is_ok: bool = True,
        supported_colors=("Black", "Orange"),
    ) -> None:
        self.supported_colors = supported_colors
        self.allowed_colors: list[list[str] | None] = []
        self._result = SimpleNamespace(
            best_color=best_color,
            diff=0.60,
            threshold=0.75,
            is_ok=is_ok,
        )

    def check(self, _image, allowed_colors=None):
        self.allowed_colors.append(allowed_colors)
        return self._result


def _check_detection(
    *,
    detected_class: str,
    best_color: str,
    is_ok: bool = True,
    candidates=("Black", "Orange"),
    supported_colors=("Black", "Orange"),
    generic_classes=None,
):
    service = ColorCheckerService()
    checker = StubColorChecker(
        best_color=best_color,
        is_ok=is_ok,
        supported_colors=supported_colors,
    )
    service._checker = checker
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    result = service.check_items(
        frame=frame,
        processed_image=frame,
        detections=[{"class": detected_class, "bbox": [0, 0, 10, 10]}],
        candidates=candidates,
        generic_classes=generic_classes,
    )
    return result, checker


def test_color_label_mismatch_fails_even_when_distance_is_within_threshold():
    result, _checker = _check_detection(
        detected_class="Black",
        best_color="Orange",
    )

    assert result.is_ok is False
    assert result.items[0].is_ok is False
    assert result.items[0].class_name == "Black"
    assert result.items[0].best_color == "Orange"


def test_matching_color_label_passes_when_distance_is_within_threshold():
    result, _checker = _check_detection(
        detected_class="Black",
        best_color="black",
    )

    assert result.is_ok is True
    assert result.items[0].is_ok is True


def test_generic_detector_class_uses_color_threshold_result():
    result, checker = _check_detection(
        detected_class="LED",
        best_color="Orange",
    )

    assert result.is_ok is True
    assert result.items[0].is_ok is True
    assert checker.allowed_colors == [["Black", "Orange"]]


def test_generic_detector_class_is_not_used_as_candidate_when_candidates_are_none():
    result, checker = _check_detection(
        detected_class="LED",
        best_color="Orange",
        candidates=None,
    )

    assert result.is_ok is True
    assert result.items[0].is_ok is True
    assert checker.allowed_colors == [None]


def test_generic_detector_candidate_is_removed_before_checker_call():
    result, checker = _check_detection(
        detected_class="LED",
        best_color="Orange",
        candidates=["LED"],
    )

    assert result.is_ok is True
    assert result.items[0].is_ok is True
    assert checker.allowed_colors == [None]


def test_generic_detector_with_only_unsupported_color_candidates_fails_closed():
    result, checker = _check_detection(
        detected_class="LED",
        best_color="Orange",
        candidates=["Purple"],
        supported_colors=("Orange",),
    )

    assert checker.allowed_colors == [None]
    assert result.is_ok is False
    assert result.items[0].is_ok is False


def test_unknown_detector_class_is_not_implicitly_treated_as_generic():
    result, checker = _check_detection(
        detected_class="Lamp",
        best_color="Orange",
        candidates=None,
        supported_colors=("Orange",),
    )

    assert checker.allowed_colors == [None]
    assert result.is_ok is False
    assert result.items[0].is_ok is False


@pytest.mark.parametrize(
    ("candidates", "expected_allowed"),
    ((["Green"], None), (["Green", "Orange"], ["Orange"])),
)
def test_configured_color_missing_from_model_fails_closed(
    candidates,
    expected_allowed,
):
    result, checker = _check_detection(
        detected_class="Green",
        best_color="Orange",
        candidates=candidates,
        supported_colors=("Orange",),
    )

    assert checker.allowed_colors == [expected_allowed]
    assert result.is_ok is False
    assert result.items[0].is_ok is False


def test_custom_generic_detector_class_can_be_declared_explicitly():
    result, checker = _check_detection(
        detected_class="Lamp",
        best_color="Orange",
        candidates=["Lamp"],
        supported_colors=("Orange",),
        generic_classes=["Lamp"],
    )

    assert checker.allowed_colors == [None]
    assert result.is_ok is True


@pytest.mark.parametrize("candidates", (None, ["LED"]))
def test_generic_detector_uses_all_colors_with_production_checker(
    tmp_path,
    candidates,
):
    model_path = tmp_path / "color.json"
    model_path.write_text(
        json.dumps(
            {
                "config": {
                    "hist_bins": [1, 1, 1],
                    "default_hist_thr": 0.25,
                },
                "colors": {
                    "Orange": {
                        "avg_color_hist": [1.0],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    service = ColorCheckerService()
    service._checker = ColorQCEnhanced.from_json(model_path)
    frame = np.full((20, 20, 3), 255, dtype=np.uint8)

    result = service.check_items(
        frame=frame,
        processed_image=frame,
        detections=[{"class": "LED", "bbox": [0, 0, 10, 10]}],
        candidates=candidates,
    )

    assert service._checker.supported_colors == ("Orange",)
    assert result.is_ok is True
    assert result.items[0].best_color == "Orange"


def test_matching_color_label_still_fails_when_distance_exceeds_threshold():
    result, _checker = _check_detection(
        detected_class="Black",
        best_color="Black",
        is_ok=False,
    )

    assert result.is_ok is False
    assert result.items[0].is_ok is False


def test_evaluated_detections_report_the_evaluated_status():
    result, _checker = _check_detection(
        detected_class="Black",
        best_color="black",
    )

    assert result.status == COLOR_CHECK_EVALUATED_STATUS
    assert result.to_dict()["status"] == COLOR_CHECK_EVALUATED_STATUS


# --- zero detections must not be decided by the background --------------------


@pytest.mark.parametrize("detections", ([], None))
def test_zero_detections_fails_closed_instead_of_checking_full_frame(detections):
    """A frame with no ROI carries no color evidence, so it must not PASS."""
    service = ColorCheckerService()
    # This stub would happily accept anything handed to it; if the full-frame
    # fallback were still in place the background would decide the verdict.
    checker = StubColorChecker(best_color="Black", is_ok=True)
    service._checker = checker
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    result = service.check_items(
        frame=frame,
        processed_image=frame,
        detections=detections,
        candidates=["Black"],
    )

    assert result.is_ok is False
    assert result.status == COLOR_CHECK_NO_DETECTIONS_STATUS
    assert result.items == []
    assert checker.allowed_colors == [], "full frame must never be color-checked"


def test_zero_detections_fails_closed_with_production_checker(tmp_path):
    """Same contract against the real checker and a saturated background."""
    model_path = tmp_path / "color.json"
    model_path.write_text(
        json.dumps(
            {
                "config": {"hist_bins": [1, 1, 1], "default_hist_thr": 0.25},
                "colors": {"Orange": {"avg_color_hist": [1.0]}},
            }
        ),
        encoding="utf-8",
    )
    service = ColorCheckerService()
    service._checker = ColorQCEnhanced.from_json(model_path)
    # A uniform frame that the single-bin model matches perfectly (diff == 0).
    frame = np.full((20, 20, 3), 255, dtype=np.uint8)

    result = service.check_items(
        frame=frame,
        processed_image=frame,
        detections=[],
        candidates=["Orange"],
    )

    assert result.is_ok is False
    assert result.status == COLOR_CHECK_NO_DETECTIONS_STATUS
    assert result.items == []


def test_zero_detections_result_serializes_the_no_detections_status():
    service = ColorCheckerService()
    service._checker = StubColorChecker(best_color="Black", is_ok=True)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    payload = service.check_items(
        frame=frame,
        processed_image=frame,
        detections=[],
    ).to_dict()

    assert payload["is_ok"] is False
    assert payload["status"] == COLOR_CHECK_NO_DETECTIONS_STATUS
    assert payload["items"] == []


# --- override application must fail loudly ------------------------------------


def _write_color_qc_model(tmp_path):
    model_path = tmp_path / "color_qc.json"
    model_path.write_text(
        json.dumps(
            {
                "config": {"hist_bins": [1, 1, 1], "default_hist_thr": 0.25},
                "colors": {"Orange": {"avg_color_hist": [1.0]}},
            }
        ),
        encoding="utf-8",
    )
    return model_path


def _write_stats_model(tmp_path):
    stats_path = tmp_path / "color_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "summary": {
                    "red": {
                        "hsv_min": [0, 0, 0],
                        "hsv_max": [180, 255, 255],
                        "lab_min": [0, 0, 0],
                        "lab_max": [255, 255, 255],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return stats_path


def test_color_qc_threshold_override_failure_propagates_from_ensure_loaded(tmp_path):
    service = ColorCheckerService()

    with pytest.raises(RuntimeError, match="color threshold overrides"):
        service.ensure_loaded(
            str(_write_color_qc_model(tmp_path)),
            overrides={"Orange": "not-a-number"},
            checker_type="color_qc",
        )


def test_color_qc_rule_override_failure_propagates_from_ensure_loaded(tmp_path):
    service = ColorCheckerService()

    with pytest.raises(RuntimeError, match="color rule overrides"):
        service.ensure_loaded(
            str(_write_color_qc_model(tmp_path)),
            rules_overrides={"Orange": "not-a-mapping"},
            checker_type="color_qc",
        )


def test_stats_threshold_override_failure_propagates_from_ensure_loaded(tmp_path):
    service = ColorCheckerService()
    stats_path = str(_write_stats_model(tmp_path))
    service.ensure_loaded(stats_path, checker_type="stats")

    with pytest.raises(RuntimeError, match="color threshold overrides"):
        service.ensure_loaded(
            stats_path,
            overrides={"red": "not-a-number"},
            checker_type="stats",
        )


def test_stats_default_threshold_failure_propagates_from_ensure_loaded(tmp_path):
    service = ColorCheckerService()

    with pytest.raises(RuntimeError, match="default color threshold"):
        service.ensure_loaded(
            str(_write_stats_model(tmp_path)),
            checker_type="stats",
            default_threshold="not-a-number",
        )


def test_valid_overrides_still_apply_for_color_qc(tmp_path):
    service = ColorCheckerService()
    service.ensure_loaded(
        str(_write_color_qc_model(tmp_path)),
        overrides={"orange": 0.9},
        rules_overrides={"orange": {"v_p50_min": 10}},
        checker_type="color_qc",
    )

    assert service.is_ready() is True
    assert service._checker.model.colors[0].hist_thr == pytest.approx(0.9)
    assert service._checker._color_rules_overrides == {"orange": {"v_p50_min": 10.0}}


def test_valid_overrides_still_apply_for_stats(tmp_path):
    service = ColorCheckerService()
    service.ensure_loaded(
        str(_write_stats_model(tmp_path)),
        overrides={"Red": 0.42},
        checker_type="stats",
        default_threshold=0.5,
    )

    assert service.is_ready() is True
    assert service._checker._color_thresholds["red"] == pytest.approx(0.42)
    assert service._checker._default_threshold == pytest.approx(0.5)


class TestCheckerOverrideContracts:
    """The checkers themselves must not swallow malformed override values."""

    def test_color_qc_threshold_override_rejects_bad_value(self, tmp_path):
        checker = ColorQCEnhanced.from_json(_write_color_qc_model(tmp_path))

        with pytest.raises(ValueError, match="Invalid color threshold override"):
            checker.apply_threshold_overrides({"Orange": "abc"})

    def test_color_qc_threshold_override_is_all_or_nothing(self, tmp_path):
        checker = ColorQCEnhanced.from_json(_write_color_qc_model(tmp_path))
        original = checker.model.colors[0].hist_thr

        with pytest.raises(ValueError):
            checker.apply_threshold_overrides({"Orange": "abc"})

        assert checker.model.colors[0].hist_thr == original

    def test_color_qc_threshold_override_skips_unknown_color(self, tmp_path, caplog):
        checker = ColorQCEnhanced.from_json(_write_color_qc_model(tmp_path))

        with caplog.at_level("WARNING"):
            checker.apply_threshold_overrides({"Purple": 0.5})

        assert "not in loaded model" in caplog.text

    def test_color_qc_rule_override_rejects_non_mapping_entry(self, tmp_path):
        checker = ColorQCEnhanced.from_json(_write_color_qc_model(tmp_path))

        with pytest.raises(TypeError, match="must be a mapping"):
            checker.apply_color_rules_overrides({"Orange": ["v_p50_min", 10]})

    def test_color_qc_rule_override_rejects_bad_value(self, tmp_path):
        checker = ColorQCEnhanced.from_json(_write_color_qc_model(tmp_path))

        with pytest.raises(ValueError, match="Invalid color rule"):
            checker.apply_color_rules_overrides({"Orange": {"v_p50_min": "abc"}})

    def test_color_qc_rule_override_keeps_explicit_none(self, tmp_path):
        checker = ColorQCEnhanced.from_json(_write_color_qc_model(tmp_path))

        checker.apply_color_rules_overrides({"Orange": {"v_p50_min": None}})

        assert checker._color_rules_overrides == {"orange": {"v_p50_min": None}}

    def test_stats_threshold_override_rejects_bad_value(self, tmp_path):
        checker = StatsColorChecker.from_json(_write_stats_model(tmp_path))

        with pytest.raises(ValueError, match="Invalid color threshold override"):
            checker.apply_threshold_overrides({"red": "abc"})

    def test_stats_threshold_override_is_all_or_nothing(self, tmp_path):
        checker = StatsColorChecker.from_json(_write_stats_model(tmp_path))
        original = dict(checker._color_thresholds)

        with pytest.raises(ValueError):
            checker.apply_threshold_overrides({"red": 0.9, "green": "abc"})

        assert checker._color_thresholds == original

    def test_stats_default_threshold_rejects_bad_value(self, tmp_path):
        checker = StatsColorChecker.from_json(_write_stats_model(tmp_path))

        with pytest.raises(ValueError, match="Invalid default color threshold"):
            checker.set_default_threshold("abc")


# --- cross-product runtime configuration isolation ----------------------------
#
# A checker instance is cached and reused whenever model_path and checker_type
# are unchanged, so every ensure_loaded call must produce an effective
# configuration built from (immutable baseline + this invocation's overrides) —
# never from whatever the previously inspected product left behind.


def _write_multicolor_qc_model(tmp_path):
    model_path = tmp_path / "qc_multi.json"
    model_path.write_text(
        json.dumps(
            {
                "config": {"hist_bins": [1, 1, 1], "default_hist_thr": 0.25},
                "colors": {
                    "Red": {"avg_color_hist": [1.0], "hist_thr": 0.20},
                    "Green": {"avg_color_hist": [1.0], "hist_thr": 0.30},
                },
            }
        ),
        encoding="utf-8",
    )
    return model_path


def _write_multicolor_stats_model(tmp_path):
    stats_path = tmp_path / "stats_multi.json"
    stats_path.write_text(
        json.dumps(
            {
                "summary": {
                    color: {
                        "hsv_min": [0, 0, 0],
                        "hsv_max": [180, 255, 255],
                        "lab_min": [0, 0, 0],
                        "lab_max": [255, 255, 255],
                    }
                    for color in ("red", "green")
                }
            }
        ),
        encoding="utf-8",
    )
    return stats_path


def _qc_thresholds(service):
    return {entry.name: entry.hist_thr for entry in service._checker.model.colors}


class TestStatsCrossProductIsolation:
    """StatsColorChecker must not carry one product's thresholds into the next."""

    def test_absent_override_returns_to_baseline(self, tmp_path):
        service = ColorCheckerService()
        path = str(_write_multicolor_stats_model(tmp_path))

        service.ensure_loaded(path, overrides={"red": 0.50}, checker_type="stats")
        assert service._checker._color_thresholds["red"] == pytest.approx(0.50)

        service.ensure_loaded(path, overrides=None, checker_type="stats")

        assert service._checker._color_thresholds["red"] == pytest.approx(
            COLOR_CONF_THRESHOLDS["red"]
        )

    def test_partial_override_returns_untouched_colors_to_baseline(self, tmp_path):
        service = ColorCheckerService()
        path = str(_write_multicolor_stats_model(tmp_path))

        service.ensure_loaded(
            path, overrides={"red": 0.50, "green": 0.40}, checker_type="stats"
        )
        service.ensure_loaded(path, overrides={"red": 0.30}, checker_type="stats")

        assert service._checker._color_thresholds["red"] == pytest.approx(0.30)
        assert service._checker._color_thresholds["green"] == pytest.approx(
            COLOR_CONF_THRESHOLDS["green"]
        )

    def test_repeated_override_replaces_rather_than_accumulates(self, tmp_path):
        service = ColorCheckerService()
        path = str(_write_multicolor_stats_model(tmp_path))

        service.ensure_loaded(path, overrides={"red": 0.50}, checker_type="stats")
        service.ensure_loaded(path, overrides={"red": 0.30}, checker_type="stats")

        assert service._checker._color_thresholds["red"] == pytest.approx(0.30)

    def test_absent_default_threshold_returns_to_baseline(self, tmp_path):
        service = ColorCheckerService()
        path = str(_write_multicolor_stats_model(tmp_path))

        service.ensure_loaded(path, checker_type="stats", default_threshold=0.60)
        assert service._checker._default_threshold == pytest.approx(0.60)

        service.ensure_loaded(path, checker_type="stats", default_threshold=None)

        assert service._checker._default_threshold == pytest.approx(
            DEFAULT_RATIO_THRESHOLD
        )

    def test_switching_products_does_not_reload_the_model(self, tmp_path, monkeypatch):
        service = ColorCheckerService()
        path = str(_write_multicolor_stats_model(tmp_path))
        loads = []
        original = StatsColorChecker.from_json

        def counting_from_json(*args, **kwargs):
            loads.append(args[0] if args else kwargs.get("stats_path"))
            return original(*args, **kwargs)

        monkeypatch.setattr(
            "core.services.color_checker.StatsColorChecker.from_json",
            counting_from_json,
        )

        service.ensure_loaded(path, overrides={"red": 0.50}, checker_type="stats")
        first_instance = service._checker
        service.ensure_loaded(path, overrides={"green": 0.40}, checker_type="stats")

        assert len(loads) == 1, "differing overrides must not trigger a model reload"
        assert service._checker is first_instance
        # ...and the cached instance still switched configuration correctly.
        assert service._checker._color_thresholds["green"] == pytest.approx(0.40)
        assert service._checker._color_thresholds["red"] == pytest.approx(
            COLOR_CONF_THRESHOLDS["red"]
        )

    def test_invalid_override_leaves_baseline_not_partial_state(self, tmp_path):
        service = ColorCheckerService()
        path = str(_write_multicolor_stats_model(tmp_path))
        service.ensure_loaded(path, overrides={"red": 0.50}, checker_type="stats")

        with pytest.raises(RuntimeError):
            service.ensure_loaded(
                path,
                overrides={"green": 0.40, "red": "not-a-number"},
                checker_type="stats",
            )

        # Neither the rejected batch nor the previous product's value survives.
        assert service._checker._color_thresholds["green"] == pytest.approx(
            COLOR_CONF_THRESHOLDS["green"]
        )
        assert service._checker._color_thresholds["red"] == pytest.approx(
            COLOR_CONF_THRESHOLDS["red"]
        )

    def test_baseline_module_constant_is_never_mutated(self, tmp_path):
        snapshot = dict(COLOR_CONF_THRESHOLDS)
        service = ColorCheckerService()
        path = str(_write_multicolor_stats_model(tmp_path))

        service.ensure_loaded(
            path, overrides={"red": 0.99, "green": 0.98}, checker_type="stats"
        )

        assert COLOR_CONF_THRESHOLDS == snapshot
        assert service._checker._baseline_color_thresholds == snapshot

    def test_baseline_snapshot_is_not_aliased_to_live_thresholds(self, tmp_path):
        checker = StatsColorChecker.from_json(_write_multicolor_stats_model(tmp_path))

        checker.apply_runtime_configuration(color_thresholds={"red": 0.99})

        assert checker._baseline_color_thresholds["red"] == pytest.approx(
            COLOR_CONF_THRESHOLDS["red"]
        )


class TestColorQcCrossProductIsolation:
    """ColorQCEnhanced must not carry one product's thresholds into the next."""

    def test_absent_override_returns_to_model_baseline(self, tmp_path):
        service = ColorCheckerService()
        path = str(_write_multicolor_qc_model(tmp_path))

        service.ensure_loaded(
            path, overrides={"Red": 0.50, "Green": 0.40}, checker_type="color_qc"
        )
        assert _qc_thresholds(service) == {"Red": 0.50, "Green": 0.40}

        service.ensure_loaded(path, overrides=None, checker_type="color_qc")

        assert _qc_thresholds(service) == {"Red": 0.20, "Green": 0.30}

    def test_partial_override_returns_untouched_colors_to_baseline(self, tmp_path):
        service = ColorCheckerService()
        path = str(_write_multicolor_qc_model(tmp_path))

        service.ensure_loaded(
            path, overrides={"Red": 0.50, "Green": 0.40}, checker_type="color_qc"
        )
        service.ensure_loaded(path, overrides={"Red": 0.11}, checker_type="color_qc")

        assert _qc_thresholds(service) == {"Red": 0.11, "Green": 0.30}

    def test_rules_are_cleared_when_next_product_supplies_none(self, tmp_path):
        service = ColorCheckerService()
        path = str(_write_multicolor_qc_model(tmp_path))

        service.ensure_loaded(
            path,
            rules_overrides={"Red": {"v_p50_min": 10}},
            checker_type="color_qc",
        )
        assert service._checker._color_rules_overrides == {"red": {"v_p50_min": 10.0}}

        service.ensure_loaded(path, rules_overrides={}, checker_type="color_qc")

        assert service._checker._color_rules_overrides == {}

    def test_rules_are_cleared_when_next_product_supplies_nothing(self, tmp_path):
        service = ColorCheckerService()
        path = str(_write_multicolor_qc_model(tmp_path))

        service.ensure_loaded(
            path,
            rules_overrides={"Red": {"v_p50_min": 10}},
            checker_type="color_qc",
        )
        service.ensure_loaded(path, checker_type="color_qc")

        assert service._checker._color_rules_overrides == {}

    def test_switching_products_does_not_reload_the_model(self, tmp_path, monkeypatch):
        service = ColorCheckerService()
        path = str(_write_multicolor_qc_model(tmp_path))
        loads = []
        original = ColorQCEnhanced.from_json

        def counting_from_json(*args, **kwargs):
            loads.append(args[0] if args else kwargs.get("path"))
            return original(*args, **kwargs)

        monkeypatch.setattr(
            "core.services.color_checker.ColorQCEnhanced.from_json",
            counting_from_json,
        )

        service.ensure_loaded(path, overrides={"Red": 0.50}, checker_type="color_qc")
        first_instance = service._checker
        service.ensure_loaded(path, overrides={"Green": 0.40}, checker_type="color_qc")

        assert len(loads) == 1, "differing overrides must not trigger a model reload"
        assert service._checker is first_instance
        assert _qc_thresholds(service) == {"Red": 0.20, "Green": 0.40}

    def test_invalid_override_leaves_baseline_not_partial_state(self, tmp_path):
        service = ColorCheckerService()
        path = str(_write_multicolor_qc_model(tmp_path))
        service.ensure_loaded(path, overrides={"Red": 0.50}, checker_type="color_qc")

        with pytest.raises(RuntimeError):
            service.ensure_loaded(
                path,
                overrides={"Green": 0.40, "Red": "not-a-number"},
                checker_type="color_qc",
            )

        assert _qc_thresholds(service) == {"Red": 0.20, "Green": 0.30}

    def test_invalid_rules_leave_baseline_not_partial_state(self, tmp_path):
        service = ColorCheckerService()
        path = str(_write_multicolor_qc_model(tmp_path))
        service.ensure_loaded(
            path,
            overrides={"Red": 0.50},
            rules_overrides={"Red": {"v_p50_min": 10}},
            checker_type="color_qc",
        )

        with pytest.raises(RuntimeError):
            service.ensure_loaded(
                path,
                overrides={"Red": 0.33},
                rules_overrides={"Red": "not-a-mapping"},
                checker_type="color_qc",
            )

        assert _qc_thresholds(service) == {"Red": 0.20, "Green": 0.30}
        assert service._checker._color_rules_overrides == {}

    def test_baseline_snapshot_survives_runtime_mutation(self, tmp_path):
        checker = ColorQCEnhanced.from_json(_write_multicolor_qc_model(tmp_path))
        assert checker._baseline_hist_thr == (0.20, 0.30)

        # Mutate through both the merging and the replacing entry points.
        checker.apply_threshold_overrides({"Red": 0.99})
        checker.apply_runtime_configuration(color_thresholds={"Green": 0.98})

        assert checker._baseline_hist_thr == (0.20, 0.30)

        checker.reset_runtime_configuration()

        assert [entry.hist_thr for entry in checker.model.colors] == [0.20, 0.30]
