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
        
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            root_logger.setLevel(logging.INFO)
            root_logger.addHandler(file_handler)
            root_logger.addHandler(stream_handler)
        self.logger = logging.getLogger(__name__)

    def log_detection(self, status: str, detections: list):
        self.logger.info(f"Detection Status: {status}")
        # Aggregate by class to reduce log noise
        try:
            from collections import defaultdict
            agg = defaultdict(list)
            for det in detections or []:
                cls = det.get('class')
                conf = float(det.get('confidence', 0.0))
                agg[cls].append(conf)
            if not agg:
                return
            for cls, confs in agg.items():
                cnt = len(confs)
                mx = max(confs)
                mn = min(confs)
                avg = sum(confs) / cnt if cnt else 0.0
                self.logger.info(
                    f"Class {cls}: x{cnt}, max={mx:.2f}, min={mn:.2f}, avg={avg:.2f}"
                )
        except Exception:
            # Fallback to verbose per-detection logs if aggregation fails
            for det in detections:
                self.logger.info(
                    f"Class: {det['class']}, Confidence: {det['confidence']:.2f}"
                )

    def log_anomaly(self, status: str, anomaly_score: float):
        self.logger.info(f"Anomaly Detection Status: {status}")
        self.logger.info(f"Anomaly Score: {anomaly_score:.4f}")
