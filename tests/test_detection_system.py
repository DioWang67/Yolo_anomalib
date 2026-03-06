
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
sys.modules['ultralytics'] = MagicMock()
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
            mock_cfg.return_value = MagicMock(output_dir="Result", max_cache_size=5)
            self.system = DetectionSystem(config_path=None)
            self.system.camera = MagicMock()
            self.system.camera.capture_frame.return_value = MagicMock()
            
    def test_pipeline_lifecycle(self):
        """Verify that start_pipeline and stop_pipeline orchestrate workers correctly."""
        # Mock _prepare_resources and _run_inference to avoid actual backend logic
        self.system._prepare_resources = MagicMock()
        self.system._run_inference = MagicMock(return_value={"status": "PASS", "detections": []})
        
        # Start pipeline
        self.system.start_pipeline("LED", "A", "yolo")
        self.assertTrue(self.system.pipeline_running)
        self.assertIsNotNone(self.system._acq_worker)
        self.assertIsNotNone(self.system._inf_worker)
        self.assertIsNotNone(self.system._sto_worker)
        
        # Verify workers are running
        self.assertTrue(self.system._acq_worker.is_alive())
        self.assertTrue(self.system._inf_worker.is_alive())
        self.assertTrue(self.system._sto_worker.is_alive())
        
        # Stop pipeline
        self.system.stop_pipeline(timeout=1.0)
        self.assertFalse(self.system.pipeline_running)
        self.assertIsNone(self.system._acq_worker)
        
    def test_shutdown_cleanup(self):
        """Verify that shutdown() calls stop_pipeline()."""
        self.system.stop_pipeline = MagicMock()
        self.system._pipeline_active = True
        self.system.shutdown()
        self.system.stop_pipeline.assert_called_once()

    def test_pipeline_stats(self):
        """Verify stats reporting."""
        self.system.start_pipeline("LED", "A", "yolo")
        stats = self.system.pipeline_stats()
        self.assertEqual(stats["pipeline_running"], True)
        self.assertIn("frames_captured", stats)
        self.system.stop_pipeline()

if __name__ == "__main__":
    unittest.main()
