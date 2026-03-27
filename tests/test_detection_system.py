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
        self.system._run_inference = MagicMock(
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
        self.system._run_inference = MagicMock(
            return_value={"status": "PASS", "detections": []}
        )
        self.system.start_pipeline("LED", "A", "yolo")
        stats = self.system.pipeline_stats()
        self.assertEqual(stats["pipeline_running"], True)
        self.assertIn("frames_captured", stats)
        self.system.stop_pipeline()


if __name__ == "__main__":
    unittest.main()
