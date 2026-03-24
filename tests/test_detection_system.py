
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import queue
import threading

# Ensure project root is in path
sys.path.append(os.getcwd())

# Mock heavy dependencies before importing DetectionSystem
mock_cv2 = MagicMock()
mock_torch = MagicMock()
mock_torch.__path__ = []  # Make it look like a package
mock_torch.cuda = MagicMock()
mock_torch.cuda.amp = MagicMock()

sys.modules['cv2'] = mock_cv2
sys.modules['torch'] = mock_torch
sys.modules['torch.cuda'] = mock_torch.cuda
sys.modules['torch.cuda.amp'] = mock_torch.cuda.amp
mock_ultralytics = MagicMock()
mock_ultralytics.__path__ = []
sys.modules['ultralytics'] = mock_ultralytics
sys.modules['ultralytics.utils'] = MagicMock()
sys.modules['ultralytics.utils.plotting'] = MagicMock()
sys.modules['anomalib'] = MagicMock()
sys.modules['anomalib.config'] = MagicMock()
sys.modules['anomalib.data'] = MagicMock()
sys.modules['anomalib.engine'] = MagicMock()
sys.modules['MVS_camera_control'] = MagicMock()

from core.detection_system import DetectionSystem
from core.types import DetectionTask

class TestDetectionSystemIntegration(unittest.TestCase):
    def setUp(self):
        with patch('core.detection_system.DetectionConfig.from_yaml') as mock_cfg:
            mock_cfg.return_value = MagicMock(
                output_dir="Result",
                max_cache_size=5,
                buffer_limit=10,
            )
            self.system = DetectionSystem(config_path=None)
            self.system.camera = MagicMock()
            self.system.camera.capture_frame.return_value = MagicMock()
            
    def test_pipeline_lifecycle(self):
        """Verify that start_pipeline and stop_pipeline orchestrate workers correctly."""
        self.system._prepare_resources = MagicMock()
        self.system._run_inference = MagicMock(return_value={"status": "PASS", "detections": []})

        # Start pipeline
        self.system.start_pipeline("LED", "A", "yolo")
        self.assertTrue(self.system.pipeline_running)

        # Pipeline manager should have active workers
        mgr = self.system._pipeline
        self.assertIsNotNone(mgr._acq_worker)
        self.assertIsNotNone(mgr._inf_worker)
        self.assertIsNotNone(mgr._sto_worker)
        self.assertTrue(mgr._acq_worker.is_alive())

        # Stop pipeline
        self.system.stop_pipeline(timeout=1.0)
        self.assertFalse(self.system.pipeline_running)

    def test_shutdown_cleanup(self):
        """Verify that shutdown() calls stop_pipeline() when pipeline is running."""
        self.system.stop_pipeline = MagicMock()
        # Simulate active pipeline via the manager
        self.system._pipeline._active = True
        self.system.shutdown()
        self.system.stop_pipeline.assert_called_once()

    def test_pipeline_stats(self):
        """Verify stats reporting."""
        self.system._prepare_resources = MagicMock()
        self.system.start_pipeline("LED", "A", "yolo")
        stats = self.system.pipeline_stats()
        self.assertEqual(stats["pipeline_running"], True)
        self.assertIn("frames_captured", stats)
        self.system.stop_pipeline()

if __name__ == "__main__":
    unittest.main()
