from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


def compute_change_metrics(
    curr_gray: np.ndarray,
    prev_gray: np.ndarray,
    threshold: int,
) -> Tuple[float, float]:
    """Compute normalized mean diff and change ratio between two grayscale frames."""
    if curr_gray.shape != prev_gray.shape:
        raise ValueError("Frame shape mismatch for compute_change_metrics().")

    diff = cv2.absdiff(curr_gray, prev_gray).astype(np.float32)
    mean_diff_norm = float(diff.mean() / 255.0)
    changed_ratio = float((diff >= threshold).mean())
    return mean_diff_norm, changed_ratio


@dataclass
class AdaptiveCalibrator:
    """Estimate noise level and suggest a dynamic threshold (mu + k * sigma)."""

    warmup_frames: int = 20
    k_sigma: float = 3.0

    _count: int = 0
    _sum: float = 0.0
    _sqsum: float = 0.0
    _ready: bool = False
    _suggest_thr: int = 25

    def update(self, curr_gray: np.ndarray, prev_gray: Optional[np.ndarray]) -> None:
        if prev_gray is None or prev_gray.shape != curr_gray.shape:
            return

        diff = cv2.absdiff(curr_gray, prev_gray).astype(np.float32)
        dmean = float(diff.mean())
        self._count += 1
        self._sum += dmean
        self._sqsum += dmean * dmean

        if self._count >= self.warmup_frames:
            mu = self._sum / self._count
            var = max(self._sqsum / self._count - mu * mu, 0.0)
            sigma = var ** 0.5
            suggested = int(round(mu + self.k_sigma * sigma))
            self._suggest_thr = int(min(max(suggested, 2), 255))
            self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def suggested_threshold(self) -> int:
        return self._suggest_thr
