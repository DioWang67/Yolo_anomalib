from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class SavePathBundle:
    base_path: str
    detector_prefix: str
    timestamp: str
    image_name: str
    original_path: str
    preprocessed_path: str
    annotated_path: str
    cropped_dir: str


class ResultPathManager:
    """Utility for building deterministic result paths."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir

    def ensure_base(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)

    def get_annotated_path(
        self,
        status: str,
        detector: str,
        product: Optional[str],
        area: Optional[str],
        anomaly_score: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> str:
        bundle = self.build_paths(
            status=status,
            detector=detector,
            product=product,
            area=area,
            anomaly_score=anomaly_score,
            timestamp=timestamp,
        )
        os.makedirs(os.path.dirname(bundle.annotated_path), exist_ok=True)
        return bundle.annotated_path

    def build_paths(
        self,
        *,
        status: str,
        detector: str,
        product: Optional[str],
        area: Optional[str],
        anomaly_score: Optional[float],
        timestamp: Optional[datetime] = None,
    ) -> SavePathBundle:
        timestamp_dt = timestamp or datetime.now()
        date_folder = timestamp_dt.strftime("%Y%m%d")
        ts = timestamp_dt.strftime("%H%M%S")
        detector_prefix = (detector or "").lower()
        product = product or "unknown"
        area = area or "unknown"

        base_path = os.path.join(
            self.base_dir, date_folder, product, area, status)
        original_dir = os.path.join(base_path, "original", detector_prefix)
        preprocessed_dir = os.path.join(
            base_path, "preprocessed", detector_prefix)
        annotated_dir = os.path.join(base_path, "annotated", detector_prefix)
        cropped_dir = os.path.join(base_path, "cropped", detector_prefix)

        if detector_prefix == "anomalib" and anomaly_score is not None:
            image_name = (
                f"{detector_prefix}_{product}_{area}_{ts}_{anomaly_score:.4f}.jpg"
            )
        else:
            image_name = (
                f"{detector_prefix}_{product}_{area}_{ts}.jpg"
                if product and area
                else f"{detector_prefix}_{ts}.jpg"
            )

        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(preprocessed_dir, exist_ok=True)
        os.makedirs(annotated_dir, exist_ok=True)
        os.makedirs(cropped_dir, exist_ok=True)

        original_path = os.path.join(original_dir, image_name)
        preprocessed_path = os.path.join(preprocessed_dir, image_name)
        annotated_path = os.path.join(annotated_dir, image_name)

        return SavePathBundle(
            base_path=base_path,
            detector_prefix=detector_prefix,
            timestamp=ts,
            image_name=image_name,
            original_path=original_path,
            preprocessed_path=preprocessed_path,
            annotated_path=annotated_path,
            cropped_dir=cropped_dir,
        )
