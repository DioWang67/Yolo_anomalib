import logging
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from core.config import DetectionConfig
from core.pipeline.context import DetectionContext
from core.pipeline.registry import PipelineEnv
from core.pipeline.steps import ColorCheckStep, CountCheckStep, PositionCheckStep, SaveResultsStep, SequenceCheckStep
from core.services.color_checker import ColorCheckerService


@pytest.fixture
def mock_env():
    """Provides a mock PipelineEnv for testing steps."""
    env = Mock(spec=PipelineEnv)
    env.logger = logging.getLogger(__name__)
    env.result_sink = MagicMock()
    env.config = DetectionConfig(weights="dummy.pt")  # Add required argument
    return env


@pytest.fixture
def base_context():
    """Provides a basic DetectionContext for tests."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    return DetectionContext(
        product="TestProduct",
        area="TestArea",
        inference_type="yolo",
        frame=frame,
        processed_image=frame,  # Add required argument
        result={"detections": [], "missing_items": []},
        status="PASS",
        config=DetectionConfig(weights="dummy.pt"),
    )


class TestPositionCheckStep:
    def test_run_with_no_detections(self, mock_env, base_context):
        """Test that status remains PASS when there are no detections to check."""
        step = PositionCheckStep(mock_env.logger, base_context.product, base_context.area)
        step.run(base_context)
        assert base_context.status == "PASS"

    def test_run_with_all_positions_ok(self, mock_env, base_context):
        """Test that status remains PASS when all detections have position_status 'OK'."""
        base_context.result["detections"] = [
            {"class": "J1", "position_status": "OK"},
            {"class": "LED1", "position_status": "OK"},
        ]
        step = PositionCheckStep(mock_env.logger, base_context.product, base_context.area)
        step.run(base_context)
        assert base_context.status == "PASS"

    def test_run_with_one_position_wrong_fails_context(self, mock_env, base_context):
        """Test that status becomes FAIL if a detection's position is outside tolerance."""
        base_context.status = "PASS"
        base_context.config.position_config = {
            "TestProduct": {
                "TestArea": {
                    "enabled": True,
                    "expected_boxes": {"LED1": {"x1": 40, "y1": 40, "x2": 60, "y2": 60}},
                    "tolerance": 10,
                    "tolerance_unit": "pixel",
                }
            }
        }
        base_context.result["detections"] = [
            {"class": "J1", "cx": 10, "cy": 10},  # No config, will be marked UNEXPECTED but should not fail the step
            {"class": "LED1", "cx": 99, "cy": 99}, # Has config, is outside tolerance
        ]
        # Force the step to run even if the global config flag isn't set
        step = PositionCheckStep(mock_env.logger, base_context.product, base_context.area, options={'force': True})
        step.run(base_context)
        assert base_context.status == "FAIL"

    def test_run_with_missing_items_fails_context(self, mock_env, base_context):
        """Test that status becomes FAIL if there are missing items."""
        base_context.status = "PASS"
        base_context.config.position_config = {
            "TestProduct": {"TestArea": {"enabled": True}}
        }
        base_context.result["missing_items"] = ["J2"]
        step = PositionCheckStep(mock_env.logger, base_context.product, base_context.area)
        step.run(base_context)
        assert base_context.status == "FAIL"

    def test_run_does_not_change_existing_fail_status(self, mock_env, base_context):
        """Test that an existing FAIL status is not incorrectly changed to PASS."""
        base_context.status = "FAIL"
        base_context.result["detections"] = [
            {"class": "J1", "position_status": "OK"},
        ]
        step = PositionCheckStep(mock_env.logger, base_context.product, base_context.area)
        step.run(base_context)
        assert base_context.status == "FAIL"


class TestSaveResultsStep:
    def test_run_calls_result_sink_save(self, mock_env, base_context):
        """Test that the step correctly calls the result_sink's save method."""
        mock_sink = mock_env.result_sink
        mock_sink.save.return_value = {
            "original_path": "path/to/orig.jpg",
            "annotated_path": "path/to/anno.jpg",
        }

        step = SaveResultsStep(mock_sink, mock_env.logger)
        step.run(base_context)

        # Verify that save was called once
        mock_sink.save.assert_called_once()

        # Verify the context's save_result is populated
        assert base_context.save_result is not None
        assert base_context.save_result["original_path"] == "path/to/orig.jpg"

    def test_run_handles_yolo_parameters_correctly(self, mock_env, base_context):
        """Test that parameters for a YOLO inference are passed correctly to the sink."""
        base_context.inference_type = "yolo"
        base_context.result = {
            "detections": [{"class": "J1"}],
            "missing_items": ["J2"],
            "result_frame": np.zeros((50, 50, 3), dtype=np.uint8),
        }
        base_context.color_result = {"is_ok": False}

        mock_sink = mock_env.result_sink
        step = SaveResultsStep(mock_sink, mock_env.logger)
        step.run(base_context)

        # Check that save was called with the correct keyword arguments
        mock_sink.save.assert_called_with(
            frame=base_context.frame,
            processed_image=base_context.processed_image,
            detections=base_context.result["detections"],
            status=base_context.status,
            detector="yolo",
            missing_items=base_context.result["missing_items"],
            anomaly_score=None,
            heatmap_path=None,
            product=base_context.product,
            area=base_context.area,
            ckpt_path=None,
            color_result=base_context.color_result,
            sequence_check=None,
        )

    def test_run_handles_anomalib_parameters_correctly(self, mock_env, base_context):
        """Test that parameters for an Anomalib inference are passed correctly to the sink."""
        base_context.inference_type = "anomalib"
        base_context.result = {
            "anomaly_score": 0.85,
            "output_path": "path/to/heatmap.jpg",
            "ckpt_path": "path/to/model.ckpt",
        }

        mock_sink = mock_env.result_sink
        step = SaveResultsStep(mock_sink, mock_env.logger)
        step.run(base_context)

        # Check that save was called with the correct keyword arguments
        mock_sink.save.assert_called_with(
            frame=base_context.frame,
            detections=[],
            status=base_context.status,
            detector="anomalib",
            missing_items=[],
            processed_image=base_context.processed_image,
            anomaly_score=0.85,
            heatmap_path="path/to/heatmap.jpg",
            ckpt_path="path/to/model.ckpt",
            product=base_context.product,
            area=base_context.area,
            color_result=None,
            sequence_check=None,
        )

    def test_run_with_save_disabled_does_not_call_sink(self, mock_env, base_context):
        """Test that the sink is not called if the step is disabled in config."""
        mock_sink = mock_env.result_sink

        step = SaveResultsStep(mock_sink, mock_env.logger, options={"enabled": False})

        step.run(base_context)

@pytest.fixture
def mock_color_service():
    service = MagicMock(spec=ColorCheckerService)
    # Mock check_items to return a result object
    def fake_check_items(**kwargs):
        res = MagicMock()
        res.items = []
        res.to_dict.return_value = {"is_ok": True, "items": []}
        return res
    service.check_items.side_effect = fake_check_items
    return service

class TestColorCheckStep:
    def test_run_calls_color_service(self, mock_env, base_context, mock_color_service):
        step = ColorCheckStep(mock_color_service, mock_env.logger)
        step.run(base_context)
        mock_color_service.check_items.assert_called_once()
        assert "color_result" in base_context.__dict__ or hasattr(base_context, "color_result")

    def test_run_updates_status_on_fail(self, mock_env, base_context, mock_color_service):
        mock_it = MagicMock()
        mock_it.is_ok = False
        mock_it.index = 0
        mock_it.class_name = "LED"
        mock_it.best_color = "red"
        mock_it.diff = 10.0
        mock_it.threshold = 5.0

        mock_res = MagicMock()
        mock_res.items = [mock_it]
        mock_res.to_dict.return_value = {"is_ok": False, "items": [{"is_ok": False}]}

        # Override side_effect to use our mock_res
        mock_color_service.check_items.side_effect = None
        mock_color_service.check_items.return_value = mock_res

        step = ColorCheckStep(mock_color_service, mock_env.logger)
        step.run(base_context)
        assert base_context.status == "FAIL"

class TestCountCheckStep:
    def test_run_pass(self, mock_env, base_context):
        base_context.config.expected_items = {"TestProduct": {"TestArea": ["LED", "J1"]}}
        base_context.result["detections"] = [{"class": "LED"}, {"class": "J1"}]
        step = CountCheckStep(mock_env.logger, base_context.product, base_context.area)
        step.run(base_context)
        assert base_context.status == "PASS"

    def test_run_fail_missing(self, mock_env, base_context):
        base_context.config.expected_items = {"TestProduct": {"TestArea": ["LED", "J1"]}}
        base_context.result["detections"] = [{"class": "LED"}] # J1 missing
        step = CountCheckStep(mock_env.logger, base_context.product, base_context.area)
        step.run(base_context)
        assert base_context.status == "FAIL"
        assert "J1" in base_context.result["missing_items"]

# ---------------------------------------------------------------------------
# Position validation unit tests (PositionValidator directly)
# ---------------------------------------------------------------------------
from core.position_validator import PositionValidator, _base_class_name


def _make_pos_config(
    expected_boxes: dict,
    tolerance: float = 5.0,
    tolerance_unit: str = "pixel",
    mode: str = "center",
    imgsz: int = 640,
    enabled: bool = True,
) -> DetectionConfig:
    """Build a DetectionConfig with position_config wired up."""
    cfg = DetectionConfig(weights="dummy.pt")
    cfg.position_config = {
        "P": {
            "A": {
                "enabled": enabled,
                "expected_boxes": expected_boxes,
                "tolerance": tolerance,
                "tolerance_unit": tolerance_unit,
                "mode": mode,
                "imgsz": imgsz,
            }
        }
    }
    return cfg


def _det(cls: str, cx: float, cy: float, bbox=None) -> dict:
    """Shorthand for a detection dict."""
    d: dict[str, Any] = {
        "class": cls,
        "cx": cx,
        "cy": cy,
        "confidence": 0.9,
    }
    if bbox is not None:
        d["bbox"] = bbox
    return d


class TestBaseClassNameInference:
    def test_plain(self):
        assert _base_class_name("LED") == "LED"

    def test_indexed(self):
        assert _base_class_name("LED#0") == "LED"
        assert _base_class_name("LED#12") == "LED"

    def test_non_numeric_hash(self):
        assert _base_class_name("C#Sharp") == "C#Sharp"


class TestEuclideanDistance:
    """Verify center mode uses Euclidean (not Chebyshev) distance."""

    def test_diagonal_offset_euclidean(self):
        """A point at (7,7) from center is ~9.9px away (Euclidean), not 7 (Chebyshev)."""
        boxes = {"W": {"x1": 90, "y1": 90, "x2": 110, "y2": 110}}  # center (100, 100)
        cfg = _make_pos_config(boxes, tolerance=9.0, tolerance_unit="pixel")
        v = PositionValidator(cfg, "P", "A")
        dets = v.validate([_det("W", 107.0, 107.0)])
        # Euclidean: sqrt(7^2+7^2) ≈ 9.899 > 9 → WRONG
        assert dets[0]["position_status"] == "WRONG"

    def test_diagonal_offset_within_tolerance(self):
        boxes = {"W": {"x1": 90, "y1": 90, "x2": 110, "y2": 110}}
        cfg = _make_pos_config(boxes, tolerance=10.0, tolerance_unit="pixel")
        v = PositionValidator(cfg, "P", "A")
        dets = v.validate([_det("W", 107.0, 107.0)])
        # Euclidean: ~9.899 < 10 → CORRECT
        assert dets[0]["position_status"] == "CORRECT"

    def test_axial_offset(self):
        """Pure horizontal offset: Euclidean == axial distance."""
        boxes = {"W": {"x1": 90, "y1": 90, "x2": 110, "y2": 110}}
        cfg = _make_pos_config(boxes, tolerance=5.0, tolerance_unit="pixel")
        v = PositionValidator(cfg, "P", "A")
        dets = v.validate([_det("W", 106.0, 100.0)])
        # Euclidean: 6 > 5 → WRONG
        assert dets[0]["position_status"] == "WRONG"


class TestIoUMode:
    """Test IoU-based position validation."""

    def test_perfect_overlap(self):
        boxes = {"W": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}}
        cfg = _make_pos_config(boxes, tolerance=50.0, tolerance_unit="pixel", mode="iou")
        v = PositionValidator(cfg, "P", "A")
        # Detection bbox exactly matches expected → IoU=1.0, error=0.0
        dets = v.validate([_det("W", 150.0, 150.0, bbox=[100, 100, 200, 200])])
        assert dets[0]["position_status"] == "CORRECT"
        assert dets[0]["position_error"] == pytest.approx(0.0, abs=0.01)

    def test_no_overlap(self):
        boxes = {"W": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}}
        cfg = _make_pos_config(boxes, tolerance=50.0, tolerance_unit="pixel", mode="iou")
        v = PositionValidator(cfg, "P", "A")
        # Detection bbox has zero overlap → IoU=0.0, error=1.0
        dets = v.validate([_det("W", 400.0, 400.0, bbox=[300, 300, 400, 400])])
        assert dets[0]["position_status"] == "WRONG"
        assert dets[0]["position_error"] == pytest.approx(1.0, abs=0.01)

    def test_partial_overlap(self):
        boxes = {"W": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}}
        cfg = _make_pos_config(boxes, tolerance=0.4, tolerance_unit="pixel", mode="iou")
        v = PositionValidator(cfg, "P", "A")
        # 50% overlap on x-axis → IoU = 50*100 / (100*100 + 100*100 - 50*100) ≈ 0.333
        # error = 1 - 0.333 ≈ 0.667 > 0.4 → WRONG
        dets = v.validate([_det("W", 175.0, 150.0, bbox=[150, 100, 250, 200])])
        assert dets[0]["position_status"] == "WRONG"


class TestBboxModeDeprecation:
    """Legacy 'bbox' mode should be treated as 'center'."""

    def test_bbox_treated_as_center(self):
        boxes = {"W": {"x1": 90, "y1": 90, "x2": 110, "y2": 110}}
        cfg = _make_pos_config(boxes, tolerance=15.0, tolerance_unit="pixel", mode="bbox")
        v = PositionValidator(cfg, "P", "A")
        assert v.mode == "center"
        dets = v.validate([_det("W", 100.0, 100.0)])
        assert dets[0]["position_status"] == "CORRECT"


class TestPerClassTolerance:
    """Per-class tolerance override via expected_box['tolerance']."""

    def test_tighter_per_class_causes_fail(self):
        boxes = {
            "W": {
                "x1": 90, "y1": 90, "x2": 110, "y2": 110,
                "tolerance": 0.5,  # per-class: 0.5% of 640 = 3.2px
            }
        }
        cfg = _make_pos_config(boxes, tolerance=20.0)  # global: 20px
        v = PositionValidator(cfg, "P", "A")
        # 5px offset > 3.2px per-class → WRONG
        dets = v.validate([_det("W", 105.0, 100.0)])
        assert dets[0]["position_status"] == "WRONG"

    def test_generous_per_class_allows_pass(self):
        boxes = {
            "W": {
                "x1": 90, "y1": 90, "x2": 110, "y2": 110,
                "tolerance": 50.0,  # per-class pixel
            }
        }
        cfg = _make_pos_config(boxes, tolerance=1.0, tolerance_unit="pixel")  # global: 1px
        v = PositionValidator(cfg, "P", "A")
        # 15px offset; per-class = 50px (pixel unit) → CORRECT
        dets = v.validate([_det("W", 115.0, 100.0)])
        assert dets[0]["position_status"] == "CORRECT"


class TestMultiInstanceMatching:
    """Multi-instance matching with #N indexed keys."""

    def test_two_instances_both_correct(self):
        boxes = {
            "LED#0": {"x1": 90, "y1": 90, "x2": 110, "y2": 110},
            "LED#1": {"x1": 290, "y1": 290, "x2": 310, "y2": 310},
        }
        cfg = _make_pos_config(boxes, tolerance=15.0, tolerance_unit="pixel")
        v = PositionValidator(cfg, "P", "A")
        dets = v.validate([
            _det("LED", 100.0, 100.0),
            _det("LED", 300.0, 300.0),
        ])
        assert all(d["position_status"] == "CORRECT" for d in dets)

    def test_two_instances_one_wrong(self):
        boxes = {
            "LED#0": {"x1": 95, "y1": 95, "x2": 105, "y2": 105},
            "LED#1": {"x1": 295, "y1": 295, "x2": 305, "y2": 305},
        }
        cfg = _make_pos_config(boxes, tolerance=5.0, tolerance_unit="pixel")
        v = PositionValidator(cfg, "P", "A")
        dets = v.validate([
            _det("LED", 100.0, 100.0),  # correct
            _det("LED", 500.0, 500.0),  # way off
        ])
        statuses = [d["position_status"] for d in dets]
        assert "CORRECT" in statuses
        assert "WRONG" in statuses

    def test_greedy_matches_nearest(self):
        """Order of detections shouldn't matter — greedy pairs nearest."""
        boxes = {
            "X#0": {"x1": 90, "y1": 90, "x2": 110, "y2": 110},   # center (100, 100)
            "X#1": {"x1": 290, "y1": 290, "x2": 310, "y2": 310},  # center (300, 300)
        }
        cfg = _make_pos_config(boxes, tolerance=15.0, tolerance_unit="pixel")
        v = PositionValidator(cfg, "P", "A")
        # Reversed order — detection near X#1 first
        dets = v.validate([
            _det("X", 300.0, 300.0),
            _det("X", 100.0, 100.0),
        ])
        assert all(d["position_status"] == "CORRECT" for d in dets)


class TestPrecomputedCenter:
    """Verify that cx/cy in expected_boxes are preferred over bbox midpoint."""

    def test_uses_precomputed_center(self):
        boxes = {
            "W": {
                "x1": 90, "y1": 90, "x2": 110, "y2": 110,  # midpoint (100, 100)
                "cx": 95.0, "cy": 95.0,  # precomputed center differs
            }
        }
        cfg = _make_pos_config(boxes, tolerance=3.0, tolerance_unit="pixel")
        v = PositionValidator(cfg, "P", "A")
        # Detection at (95, 95) — exactly at precomputed center → distance 0
        dets = v.validate([_det("W", 95.0, 95.0)])
        assert dets[0]["position_status"] == "CORRECT"
        assert dets[0]["position_error"] == pytest.approx(0.0, abs=0.01)

    def test_would_fail_without_precomputed(self):
        """Same scenario but if midpoint (100,100) were used, dist=7.07 > 3."""
        boxes = {
            "W": {
                "x1": 90, "y1": 90, "x2": 110, "y2": 110,
                "cx": 95.0, "cy": 95.0,
            }
        }
        cfg = _make_pos_config(boxes, tolerance=3.0, tolerance_unit="pixel")
        v = PositionValidator(cfg, "P", "A")
        # Detection at (100, 100) — 7.07px from precomputed (95,95) → WRONG
        dets = v.validate([_det("W", 100.0, 100.0)])
        assert dets[0]["position_status"] == "WRONG"


class TestUnexpectedDetection:
    """Detection class not in expected_boxes → UNEXPECTED."""

    def test_unknown_class(self):
        boxes = {"W": {"x1": 90, "y1": 90, "x2": 110, "y2": 110}}
        cfg = _make_pos_config(boxes, tolerance=10.0, tolerance_unit="pixel")
        v = PositionValidator(cfg, "P", "A")
        dets = v.validate([_det("Alien", 100.0, 100.0)])
        assert dets[0]["position_status"] == "UNEXPECTED"


class TestRegionMode:
    """Test region mode: point-to-rectangle edge distance."""

    def test_point_inside_region_passes(self):
        boxes = {"W": {"x1": 50, "y1": 50, "x2": 150, "y2": 150}}
        cfg = _make_pos_config(boxes, tolerance=5.0, tolerance_unit="pixel", mode="region")
        v = PositionValidator(cfg, "P", "A")
        # Point inside box → edge_distance = 0 < 5 → CORRECT
        dets = v.validate([_det("W", 100.0, 100.0)])
        assert dets[0]["position_status"] == "CORRECT"
        assert dets[0]["position_error"] == pytest.approx(0.0)

    def test_point_outside_region_fails(self):
        boxes = {"W": {"x1": 50, "y1": 50, "x2": 150, "y2": 150}}
        cfg = _make_pos_config(boxes, tolerance=5.0, tolerance_unit="pixel", mode="region")
        v = PositionValidator(cfg, "P", "A")
        # Point at (160, 100) → edge_distance = 10 > 5 → WRONG
        dets = v.validate([_det("W", 160.0, 100.0)])
        assert dets[0]["position_status"] == "WRONG"

    def test_point_on_edge_passes(self):
        boxes = {"W": {"x1": 50, "y1": 50, "x2": 150, "y2": 150}}
        cfg = _make_pos_config(boxes, tolerance=1.0, tolerance_unit="pixel", mode="region")
        v = PositionValidator(cfg, "P", "A")
        # Point exactly on edge → edge_distance = 0
        dets = v.validate([_det("W", 50.0, 100.0)])
        assert dets[0]["position_status"] == "CORRECT"


class TestMissingCxCyHandling:
    """Detection with missing or invalid center coordinates."""

    def test_missing_cx_cy_marks_unknown(self):
        boxes = {"W": {"x1": 90, "y1": 90, "x2": 110, "y2": 110}}
        cfg = _make_pos_config(boxes, tolerance=10.0, tolerance_unit="pixel")
        v = PositionValidator(cfg, "P", "A")
        bad_det: dict[str, Any] = {"class": "W", "confidence": 0.9}
        dets = v.validate([bad_det])
        assert dets[0]["position_status"] == "UNKNOWN"

    def test_invalid_cx_cy_marks_error(self):
        boxes = {"W": {"x1": 90, "y1": 90, "x2": 110, "y2": 110}}
        cfg = _make_pos_config(boxes, tolerance=10.0, tolerance_unit="pixel")
        v = PositionValidator(cfg, "P", "A")
        bad_det: dict[str, Any] = {"class": "W", "cx": "abc", "cy": "xyz", "confidence": 0.9}
        dets = v.validate([bad_det])
        assert dets[0]["position_status"] == "ERROR"

    def test_out_of_bounds_marks_invalid(self):
        boxes = {"W": {"x1": 90, "y1": 90, "x2": 110, "y2": 110}}
        cfg = _make_pos_config(boxes, tolerance=10.0, tolerance_unit="pixel")
        v = PositionValidator(cfg, "P", "A")
        dets = v.validate([_det("W", -10.0, 100.0)])
        assert dets[0]["position_status"] == "INVALID"


class TestGreedyAssignEdgeCases:
    """Edge cases for _greedy_assign()."""

    def test_empty_candidates(self):
        boxes = {
            "LED#0": {"x1": 90, "y1": 90, "x2": 110, "y2": 110},
            "LED#1": {"x1": 290, "y1": 290, "x2": 310, "y2": 310},
        }
        cfg = _make_pos_config(boxes, tolerance=10.0, tolerance_unit="pixel")
        v = PositionValidator(cfg, "P", "A")
        # No detections for LED → both unassigned
        dets = v.validate([_det("Other", 100.0, 100.0)])
        assert dets[0]["position_status"] == "UNEXPECTED"

    def test_more_detections_than_keys(self):
        boxes = {"LED#0": {"x1": 95, "y1": 95, "x2": 105, "y2": 105}}
        cfg = _make_pos_config(boxes, tolerance=10.0, tolerance_unit="pixel")
        v = PositionValidator(cfg, "P", "A")
        dets = v.validate([
            _det("LED", 100.0, 100.0),
            _det("LED", 200.0, 200.0),  # extra, no matching key
        ])
        statuses = [d["position_status"] for d in dets]
        assert "CORRECT" in statuses

    def test_candidate_without_cx_cy_skipped_in_matching(self):
        boxes = {
            "LED#0": {"x1": 95, "y1": 95, "x2": 105, "y2": 105},
        }
        cfg = _make_pos_config(boxes, tolerance=10.0, tolerance_unit="pixel")
        v = PositionValidator(cfg, "P", "A")
        bad_det: dict[str, Any] = {"class": "LED", "confidence": 0.9}
        dets = v.validate([bad_det])
        assert dets[0]["position_status"] == "UNKNOWN"


class TestIoUFallback:
    """IoU mode edge cases."""

    def test_iou_without_bbox_falls_back(self):
        boxes = {"W": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}}
        cfg = _make_pos_config(boxes, tolerance=50.0, tolerance_unit="pixel", mode="iou")
        v = PositionValidator(cfg, "P", "A")
        # Detection without bbox → falls back to edge distance
        dets = v.validate([_det("W", 150.0, 150.0)])  # no bbox
        assert dets[0]["position_status"] in {"CORRECT", "WRONG"}

    def test_iou_zero_area_box(self):
        """Zero-area expected box should not crash."""
        boxes = {"W": {"x1": 100, "y1": 100, "x2": 100, "y2": 100}}
        cfg = _make_pos_config(boxes, tolerance=50.0, tolerance_unit="pixel", mode="iou")
        v = PositionValidator(cfg, "P", "A")
        dets = v.validate([_det("W", 100.0, 100.0, bbox=[90, 90, 110, 110])])
        assert dets[0]["position_status"] in {"CORRECT", "WRONG"}


class TestComputeIoU:
    """Direct unit tests for _compute_iou static method."""

    def test_perfect(self):
        assert PositionValidator._compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_no_overlap(self):
        assert PositionValidator._compute_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_half_overlap(self):
        iou = PositionValidator._compute_iou((0, 0, 10, 10), (5, 0, 15, 10))
        # Intersection: 5*10=50, Union: 100+100-50=150
        assert iou == pytest.approx(50 / 150, abs=0.01)

    def test_zero_area(self):
        assert PositionValidator._compute_iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


class TestPointToRectDistance:
    """Direct unit tests for _point_to_rect_distance."""

    def test_inside(self):
        assert PositionValidator._point_to_rect_distance(5, 5, 0, 0, 10, 10) == 0.0

    def test_on_edge(self):
        assert PositionValidator._point_to_rect_distance(0, 5, 0, 0, 10, 10) == 0.0

    def test_outside_right(self):
        assert PositionValidator._point_to_rect_distance(15, 5, 0, 0, 10, 10) == pytest.approx(5.0)

    def test_outside_corner(self):
        dist = PositionValidator._point_to_rect_distance(13, 14, 0, 0, 10, 10)
        assert dist == pytest.approx((3**2 + 4**2)**0.5)


class TestGetSummary:
    """Test PositionValidator.get_summary()."""

    def test_summary_counts(self):
        boxes = {
            "A": {"x1": 90, "y1": 90, "x2": 110, "y2": 110},
            "B": {"x1": 290, "y1": 290, "x2": 310, "y2": 310},
        }
        cfg = _make_pos_config(boxes, tolerance=5.0, tolerance_unit="pixel")
        v = PositionValidator(cfg, "P", "A")
        dets = v.validate([
            _det("A", 100.0, 100.0),  # CORRECT
            _det("B", 500.0, 500.0),  # WRONG
            _det("C", 50.0, 50.0),    # UNEXPECTED
        ])
        summary = v.get_summary(dets)
        assert summary["total"] == 3
        assert summary["correct"] == 1
        assert summary["wrong"] == 1
        assert summary["unexpected"] == 1
        assert summary["tolerance_px"] == 5.0
        assert summary["imgsz"] == 640


class TestEvaluateStatus:
    """Test PositionValidator.evaluate_status()."""

    def test_pass_no_issues(self):
        boxes = {"A": {"x1": 90, "y1": 90, "x2": 110, "y2": 110}}
        cfg = _make_pos_config(boxes, tolerance=10.0, tolerance_unit="pixel")
        v = PositionValidator(cfg, "P", "A")
        dets = v.validate([_det("A", 100.0, 100.0)])
        assert v.evaluate_status(dets, []) == "PASS"

    def test_fail_with_missing(self):
        boxes = {"A": {"x1": 90, "y1": 90, "x2": 110, "y2": 110}}
        cfg = _make_pos_config(boxes, tolerance=10.0, tolerance_unit="pixel")
        v = PositionValidator(cfg, "P", "A")
        dets = v.validate([_det("A", 100.0, 100.0)])
        assert v.evaluate_status(dets, ["B"]) == "FAIL"

    def test_fail_with_wrong_position(self):
        boxes = {"A": {"x1": 90, "y1": 90, "x2": 110, "y2": 110}}
        cfg = _make_pos_config(boxes, tolerance=1.0, tolerance_unit="pixel")
        v = PositionValidator(cfg, "P", "A")
        dets = v.validate([_det("A", 200.0, 200.0)])  # WRONG
        assert v.evaluate_status(dets, []) == "FAIL"


class TestGetImageSize:
    """Test _get_image_size fallback chain."""

    def test_from_area_config(self):
        boxes = {"W": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}
        cfg = _make_pos_config(boxes, imgsz=1024)
        v = PositionValidator(cfg, "P", "A")
        assert v.imgsz == 1024

    def test_list_imgsz(self):
        cfg = DetectionConfig(weights="dummy.pt")
        cfg.position_config = {
            "P": {"A": {
                "enabled": True,
                "expected_boxes": {"W": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
                "tolerance": 5.0,
                "imgsz": [512, 512],
            }}
        }
        v = PositionValidator(cfg, "P", "A")
        assert v.imgsz == 512

    def test_fallback_to_640(self):
        cfg = DetectionConfig(weights="dummy.pt")
        cfg.position_config = {
            "P": {"A": {
                "enabled": True,
                "expected_boxes": {},
                "tolerance": 5.0,
            }}
        }
        v = PositionValidator(cfg, "P", "A")
        assert v.imgsz == 640


class TestPerClassTolerancePixelUnit:
    """Per-class tolerance with pixel unit."""

    def test_pixel_unit_per_class(self):
        boxes = {
            "W": {
                "x1": 90, "y1": 90, "x2": 110, "y2": 110,
                "tolerance": 20.0,  # 20 pixels when unit=pixel
            }
        }
        cfg = _make_pos_config(boxes, tolerance=1.0, tolerance_unit="pixel")
        v = PositionValidator(cfg, "P", "A")
        # 15px offset; per-class 20px (pixel) → CORRECT
        dets = v.validate([_det("W", 115.0, 100.0)])
        assert dets[0]["position_status"] == "CORRECT"


class TestSequenceCheckStep:
    def test_run_pass_l2r(self, mock_env, base_context):
        base_context.result["detections"] = [
            {"class": "A", "bbox": [10, 0, 20, 0]},
            {"class": "B", "bbox": [30, 0, 40, 0]},
        ]
        step = SequenceCheckStep(mock_env.logger, base_context.product, base_context.area,
                                 options={"expected": ["A", "B"]})
        step.run(base_context)
        assert base_context.status == "PASS"

    def test_run_fail_order(self, mock_env, base_context):
        base_context.result["detections"] = [
            {"class": "B", "bbox": [10, 0, 20, 0]},
            {"class": "A", "bbox": [30, 0, 40, 0]},
        ]
        step = SequenceCheckStep(mock_env.logger, base_context.product, base_context.area,
                                 options={"expected": ["A", "B"]})
        step.run(base_context)
        assert base_context.status == "FAIL"

    def test_run_fail_length(self, mock_env, base_context):
        base_context.result["detections"] = [
            {"class": "A", "bbox": [10, 0, 20, 0]},
        ]
        step = SequenceCheckStep(mock_env.logger, base_context.product, base_context.area,
                                 options={"expected": ["A", "B"]})
        step.run(base_context)
        assert base_context.status == "FAIL"
        assert base_context.result["sequence_check"]["reason"] == "length_mismatch"
