# detector.py
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from .utils import ImageUtils


class YOLODetector:
    def __init__(self, model: YOLO, config):
        self.model = model
        self.config = config
        self.colors = colors
        self.image_utils = ImageUtils()
        self.Annotator = Annotator

    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        # Keep BGR throughout; ImageUtils handles letterbox to target size
        resized_img = self.image_utils.letterbox(frame, size=self.config.imgsz)
        return resized_img

    @staticmethod
    def iou(box1: list[int], box2: list[int]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return intersection_area / float(box1_area + box2_area - intersection_area)

    @staticmethod
    def check_missing_items(
        expected_items: list[str], detected_class_names: set[str]
    ) -> set[str]:
        return set(expected_items) - detected_class_names

    def process_detections(
        self,
        predictions,
        processed_image: np.ndarray,
        original_image: np.ndarray,
        expected_items: list[str],
    ) -> tuple[np.ndarray, list[dict[str, Any]], list[str]]:
        try:
            result = predictions[0]
            if not hasattr(result, "boxes"):
                raise AttributeError("boxes attribute missing")
            boxes = result.boxes
            detections: list[dict[str, Any]] = []
            detected_items = set()
            from collections import Counter

            det_counter: Counter = Counter()

            if boxes is not None and len(boxes) > 0:
                names = self.model.names
                items = []
                if hasattr(boxes, "xyxy"):
                    xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
                    confs = boxes.conf.detach().cpu().numpy()
                    clss = boxes.cls.detach().cpu().numpy().astype(int)
                    items = [
                        (coords, float(conf), int(cid))
                        for coords, conf, cid in zip(xyxy, confs, clss)
                    ]
                else:
                    for box in boxes:
                        if not all(
                            hasattr(box, attr) for attr in ("xyxy", "conf", "cls")
                        ):
                            raise AttributeError(
                                "box missing expected attributes")
                        coords = box.xyxy
                        for attr in ("detach", "cpu", "numpy"):
                            if hasattr(coords, attr):
                                coords = getattr(coords, attr)()
                        coords_array = np.asarray(coords).reshape(-1)
                        if coords_array.size < 4:
                            raise ValueError("invalid bbox shape")
                        coords_int = coords_array[:4].astype(int)
                        items.append(
                            (
                                coords_int,
                                float(box.conf),
                                int(box.cls),
                            )
                        )
                for coords, conf, cid in items:
                    x1, y1, x2, y2 = map(int, coords[:4])
                    cname = (
                        names[int(cid)] if isinstance(
                            names, dict) else names[int(cid)]
                    )
                    det = {
                        "class": cname,
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2],
                        "class_id": int(cid),
                    }
                    detections.append(det)
                    detected_items.add(cname)
                    det_counter[cname] += 1

            # Draw annotations on processed image
            result_frame = processed_image.copy()
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                color = self.colors(det["class_id"], True)
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{det['class']}: {det['confidence']:.2f}"
                self.image_utils.draw_label(
                    result_frame, label, (x1, y1 - 10), color)

            # Enforce counts (treat expected_items as multiset)
            exp_items = [str(x).strip() for x in (expected_items or [])]
            exp_counter: Counter = Counter(exp_items)
            missing_items: list[str] = []
            for name, need in exp_counter.items():
                have = int(det_counter.get(name, 0))
                if have < need:
                    missing_items.extend([name] * (need - have))
            return result_frame, detections, missing_items
        except Exception as e:
            raise RuntimeError(f"處理檢測結果失敗: {str(e)}")

    def draw_results(
        self, frame: np.ndarray, status: str, detections: list[dict]
    ) -> np.ndarray:
        result_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class']} {det['confidence']:.2f}"
            color = self.colors(det["class_id"], True)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                result_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
        color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
        cv2.putText(
            result_frame,
            f"Status: {status}",
            (230, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
        return result_frame
