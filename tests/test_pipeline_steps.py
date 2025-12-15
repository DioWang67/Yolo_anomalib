import logging
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from core.config import DetectionConfig
from core.pipeline.context import DetectionContext
from core.pipeline.registry import PipelineEnv
from core.pipeline.steps import PositionCheckStep, SaveResultsStep


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
        )

    def test_run_with_save_disabled_does_not_call_sink(self, mock_env, base_context):
        """Test that the sink is not called if the step is disabled in config."""
        mock_sink = mock_env.result_sink

        step = SaveResultsStep(mock_sink, mock_env.logger, options={"enabled": False})

        step.run(base_context)

        mock_sink.save.assert_not_called()
        assert base_context.save_result is None
