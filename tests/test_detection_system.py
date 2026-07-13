"""Integration tests for DetectionSystem.

Heavy dependencies (cv2, torch, ultralytics, anomalib) are mocked via
pytest fixtures to avoid importing real GPU libraries in CI.  All mocks
are scoped to this module and cleaned up automatically so they do NOT
pollute other test files.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is in path
sys.path.append(os.getcwd())

# ---------------------------------------------------------------------------
# Module-scoped autouse fixture: install mocks BEFORE any test in this file
# runs, and restore originals after ALL tests in this file finish.
# ---------------------------------------------------------------------------
_MOCKED_MODULES = [
    "cv2",
    "torch", "torch.cuda", "torch.cuda.amp",
    "ultralytics", "ultralytics.utils", "ultralytics.utils.plotting",
    "anomalib", "anomalib.config", "anomalib.data", "anomalib.engine",
    "MVS_camera_control",
]


@pytest.fixture(autouse=True, scope="module")
def _mock_heavy_deps():
    """Temporarily replace heavy native modules with MagicMock objects."""
    originals = {k: sys.modules.get(k) for k in _MOCKED_MODULES}

    mock_torch = MagicMock()
    mock_torch.__path__ = []
    mock_torch.cuda = MagicMock()
    mock_torch.cuda.amp = MagicMock()

    mock_ultralytics = MagicMock()
    mock_ultralytics.__path__ = []

    sys.modules["cv2"] = MagicMock()
    sys.modules["torch"] = mock_torch
    sys.modules["torch.cuda"] = mock_torch.cuda
    sys.modules["torch.cuda.amp"] = mock_torch.cuda.amp
    sys.modules["ultralytics"] = mock_ultralytics
    sys.modules["ultralytics.utils"] = MagicMock()
    sys.modules["ultralytics.utils.plotting"] = MagicMock()
    sys.modules["anomalib"] = MagicMock()
    sys.modules["anomalib.config"] = MagicMock()
    sys.modules["anomalib.data"] = MagicMock()
    sys.modules["anomalib.engine"] = MagicMock()
    sys.modules["MVS_camera_control"] = MagicMock()

    yield  # ── tests run here ──

    # Restore originals
    for key, original in originals.items():
        if original is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = original


# Import AFTER the fixture installs mocks (pytest evaluates fixtures before
# test collection within the module when using autouse + module scope).
# We use a lazy import helper so the real import happens inside each test.
_DetectionSystem = None


def _get_detection_system():
    global _DetectionSystem
    if _DetectionSystem is None:
        from core.detection_system import DetectionSystem
        _DetectionSystem = DetectionSystem
    return _DetectionSystem


class TestDetectionSystemIntegration(unittest.TestCase):
    def setUp(self):
        DetectionSystem = _get_detection_system()
        with patch("core.detection_system.DetectionConfig.from_yaml") as mock_cfg:
            mock_cfg.return_value = MagicMock(
                output_dir="Result",
                max_cache_size=5,
                buffer_limit=10,
            )
            self.system = DetectionSystem(config_path=None)
            self.system.camera = MagicMock()
            self.system.camera.capture_frame.return_value = MagicMock()

    def tearDown(self):
        import time
        try:
            if self.system.pipeline_running:
                self.system.stop_pipeline(timeout=2.0)
        except Exception:
            pass
        try:
            self.system.shutdown()
        except Exception:
            pass
        # Allow daemon threads from AsyncPipelineManager to fully terminate
        time.sleep(0.3)

    def test_pipeline_lifecycle(self):
        """Verify that start_pipeline and stop_pipeline orchestrate workers correctly."""
        self.system.load_model_configs = MagicMock()
        self.system._prepare_resources = MagicMock()
        self.system.run_inference = MagicMock(
            return_value={"status": "PASS", "detections": []}
        )

        self.system.start_pipeline("LED", "A", "yolo")
        self.assertTrue(self.system.pipeline_running)

        mgr = self.system._pipeline
        self.assertIsNotNone(mgr._acq_worker)
        self.assertIsNotNone(mgr._inf_worker)
        self.assertIsNotNone(mgr._sto_worker)
        self.assertTrue(mgr._acq_worker.is_alive())

        self.system.stop_pipeline(timeout=1.0)
        self.assertFalse(self.system.pipeline_running)

    def test_shutdown_cleanup(self):
        """Verify that shutdown() calls stop_pipeline() when pipeline is running."""
        self.system.stop_pipeline = MagicMock()
        self.system._pipeline._active = True
        self.system.shutdown()
        self.system.stop_pipeline.assert_called_once()

    def test_pipeline_stats(self):
        """Verify stats reporting."""
        self.system.load_model_configs = MagicMock()
        self.system._prepare_resources = MagicMock()
        self.system.run_inference = MagicMock(
            return_value={"status": "PASS", "detections": []}
        )
        self.system.start_pipeline("LED", "A", "yolo")
        stats = self.system.pipeline_stats()
        self.assertEqual(stats["pipeline_running"], True)
        self.assertIn("frames_captured", stats)
        self.system.stop_pipeline()

    def test_run_inference_delegates_fusion_runner(self):
        """Verify fusion inference is delegated to the extracted runner."""
        expected_result = {"status": "PASS", "detections": []}
        runner_instance = MagicMock()
        runner_instance.run.return_value = expected_result

        with patch("core.detection_system.FusionInferenceRunner") as runner_cls:
            runner_cls.return_value = runner_instance

            result = self.system.run_inference(
                MagicMock(), "LED", "A", "fusion", MagicMock()
            )

        self.assertEqual(result, expected_result)
        runner_cls.assert_called_once_with(
            self.system.model_manager, self.system.config, self.system.result_sink
        )
        runner_instance.run.assert_called_once()
        adjust_callback = runner_instance.run.call_args.kwargs[
            "adjust_anomalib_output_path"
        ]
        self.assertIs(adjust_callback.__self__, self.system)
        self.assertIs(
            adjust_callback.__func__,
            self.system._adjust_anomalib_output_path.__func__,
        )

    def test_prepare_resources_loads_color_overrides_before_color_checker(self):
        """Verify DetectionSystem wires model-level color overrides into color checker."""
        self.system.load_model_configs = MagicMock()
        self.system.config.enable_color_check = True
        self.system.config.color_model_path = "models/color.pkl"
        self.system.config.color_checker_type = "color_qc"
        self.system.config.color_score_threshold = 0.7
        self.system.color_override_loader = MagicMock()
        self.system.color_override_loader.load.return_value = (
            {"red": 0.91},
            {"red": {"min_area": 3}},
            {"yellow_h_min": 18},
        )
        self.system.color_service = MagicMock()
        run_logger = MagicMock()

        self.system._prepare_resources("LED", "A", "yolo", run_logger)

        self.system.color_override_loader.load.assert_called_once_with(
            self.system.config,
            "LED",
            "A",
            "yolo",
            self.system.logger.logger,
        )
        self.system.color_service.ensure_loaded.assert_called_once_with(
            "models/color.pkl",
            overrides={"red": 0.91},
            rules_overrides={"red": {"min_area": 3}},
            checker_type="color_qc",
            default_threshold=0.7,
            decision_tuning={"yellow_h_min": 18},
        )

    def test_apply_camera_settings_pushes_exposure_and_gain_once(self):
        """Per-model exposure/gain applied on change, skipped when unchanged."""
        self.system.camera = MagicMock()
        self.system.camera.is_initialized = True
        self.system.config.exposure_time = "51170.0000"
        self.system.config.gain = "23.0"

        self.system._apply_camera_settings_from_config()
        self.system.camera.set_exposure.assert_called_once_with(51170.0)
        self.system.camera.set_gain.assert_called_once_with(23.0)

        # Same values again -> no redundant hardware calls (hot-path guard)
        self.system._apply_camera_settings_from_config()
        self.system.camera.set_exposure.assert_called_once()
        self.system.camera.set_gain.assert_called_once()

        # Changed exposure -> re-applied
        self.system.config.exposure_time = "42000.0000"
        self.system._apply_camera_settings_from_config()
        self.assertEqual(self.system.camera.set_exposure.call_count, 2)

    def test_prepare_auto_inspection_settles_camera_before_preview_frames(self):
        """Auto preview must start only after per-model camera settings are ready."""
        self.system.load_model_configs = MagicMock()
        self.system._validate_runtime_for_current_model = MagicMock()
        self.system._prepare_resources = MagicMock()
        self.system._ensure_camera_settings_ready_for_capture = MagicMock()

        self.system.prepare_auto_inspection("Cable1", "A", "yolo")

        self.system.load_model_configs.assert_called_once_with("Cable1", "A", "yolo")
        self.system._validate_runtime_for_current_model.assert_called_once()
        self.system._prepare_resources.assert_called_once()
        self.assertFalse(self.system._prepare_resources.call_args.kwargs["load_model_config"])
        self.system._ensure_camera_settings_ready_for_capture.assert_called_once()

    def test_apply_camera_settings_noop_when_camera_uninitialized(self):
        self.system.camera = MagicMock()
        self.system.camera.is_initialized = False
        self.system.config.exposure_time = "1000"
        self.system.config.gain = "1.0"

        self.system._apply_camera_settings_from_config()
        self.system.camera.set_exposure.assert_not_called()
        self.system.camera.set_gain.assert_not_called()

    def test_apply_camera_settings_retries_after_hardware_rejection(self):
        """A transient hardware rejection is retried before acquisition starts."""
        self.system.camera = MagicMock()
        self.system.camera.is_initialized = True
        self.system.camera.set_exposure.side_effect = [False, True]
        self.system.camera.set_gain.return_value = True
        self.system.config.exposure_time = "22380.0000"
        self.system.config.gain = "23.0"

        with patch("core.detection_system.time.sleep"):
            self.system._apply_camera_settings_from_config()

        self.assertEqual(self.system.camera.set_exposure.call_count, 2)
        self.assertEqual(self.system.camera.set_gain.call_count, 2)
        self.assertEqual(
            self.system._applied_camera_settings,
            ("22380.0000", "23.0"),
        )

    def test_camera_settings_discard_stale_frames_before_first_inspection(self):
        """Frames queued before a model exposure update must not reach inference."""
        self.system.camera = MagicMock()
        self.system.camera.is_initialized = True
        self.system.camera.set_exposure.return_value = True
        self.system.camera.set_gain.return_value = True
        self.system.camera.clear_image_buffer.return_value = True
        self.system.config.exposure_time = "22380.0000"
        self.system.config.gain = "23.0"

        with patch("core.detection_system.time.sleep"):
            self.system._ensure_camera_settings_ready_for_capture()

        self.system.camera.set_exposure.assert_called_once_with(22380.0)
        self.system.camera.set_gain.assert_called_once_with(23.0)
        self.system.camera.clear_image_buffer.assert_called_once()
        self.system.camera.capture_frame.assert_not_called()
        self.assertFalse(self.system._camera_settings_need_settle)

    def test_camera_settings_continue_when_transition_frames_are_unavailable(self):
        """An empty transition buffer must not abort an otherwise retryable inspection."""
        self.system.camera = MagicMock()
        self.system.camera.is_initialized = True
        self.system.camera.set_exposure.return_value = True
        self.system.camera.set_gain.return_value = True
        self.system.camera.clear_image_buffer.return_value = False
        self.system.camera.capture_frame.return_value = None
        self.system.config.exposure_time = "22380.0000"
        self.system.config.gain = "23.0"

        with patch("core.detection_system.time.sleep"):
            self.system._ensure_camera_settings_ready_for_capture()

        self.assertEqual(
            self.system.camera.capture_frame.call_count,
            self.system._CAMERA_SETTINGS_SETTLE_FRAMES,
        )
        self.assertFalse(self.system._camera_settings_need_settle)

    def test_resolve_output_dir_rejects_project_escape(self):
        from core.security import SecurityError

        self.system.config.output_dir = ".."

        with self.assertRaises(SecurityError):
            self.system._resolve_output_dir()

    def test_refresh_result_sink_updates_config_when_output_dir_unchanged(self):
        """Model config switches must update the existing result sink."""
        sink = MagicMock()
        self.system.result_sink = sink
        self.system._sink_base_dir = self.system._resolve_output_dir()
        active_config = MagicMock(
            output_dir=str(self.system._sink_base_dir),
            buffer_limit=10,
        )
        self.system.config = active_config

        self.system._refresh_result_sink()

        sink.update_config.assert_called_once_with(active_config)


if __name__ == "__main__":
    unittest.main()
