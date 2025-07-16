# logger.py
import logging
from datetime import datetime
import os

class DetectionLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self._setup_logger()

    def _setup_logger(self):
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(self.log_dir, f"detection_{date_str}.log")
        os.makedirs(self.log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_detection(self, status: str, detections: list):
        self.logger.info(f"Detection Status: {status}")
        for det in detections:
            self.logger.info(
                f"Class: {det['class']}, Confidence: {det['confidence']:.2f}"
            )

    def log_anomaly(self, status: str, anomaly_score: float):
        self.logger.info(f"Anomaly Detection Status: {status}")
        self.logger.info(f"Anomaly Score: {anomaly_score:.4f}")