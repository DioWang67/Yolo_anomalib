#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""led_qc_enhanced.py

Standalone LED color quality checker for the "advanced" model format
that contains top-level keys like colors/config/version.

Key features implemented:
- Load advanced JSON model (colors/config/...)
- Compute HSV 3D histogram on masked region (V > 30)
- Compare with each color's avg_color_hist using L1 distance
- Pick best-matching color and decide pass/fail by threshold
- Optional white-specific rule using S/V percentiles

This module is self-contained and does not depend on the simple model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import colorsys

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


# -------------------- Data structures --------------------
@dataclass
class LEDQCAdvancedResult:
    best_color: str
    diff: float
    threshold: float
    is_ok: bool
    scores: List[Tuple[str, float]]
    metrics: Dict[str, float]


@dataclass
class _ColorEntry:
    name: str
    avg_color_hist: List[float]
    hist_thr: Optional[float] = None


@dataclass
class _Model:
    hist_bins: Tuple[int, int, int]
    default_hist_thr: float
    colors: List[_ColorEntry]
    # White-specific config (optional)
    white_s_p90_max: Optional[float] = None
    white_v_p50_min: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "_Model":
        cfg = data.get("config", {})
        bins = cfg.get("hist_bins", [12, 12, 12])
        if not (isinstance(bins, list) and len(bins) == 3):
            bins = [12, 12, 12]
        default_hist_thr = float(cfg.get("default_hist_thr", 0.25))

        # Optional white-specific knobs
        white_s_p90_max = cfg.get("white_s_p90_max", None)
        white_v_p50_min = cfg.get("white_v_p50_min", None)
        if white_s_p90_max is not None:
            white_s_p90_max = float(white_s_p90_max)
        if white_v_p50_min is not None:
            white_v_p50_min = float(white_v_p50_min)

        colors_field = data.get("colors", {}) or {}
        colors: List[_ColorEntry] = []
        for name, c in colors_field.items():
            avg_hist = c.get("avg_color_hist")
            if not isinstance(avg_hist, list) or not avg_hist:
                continue
            hist_thr = c.get("hist_thr")
            colors.append(
                _ColorEntry(
                    name=name,
                    avg_color_hist=list(avg_hist),
                    hist_thr=float(hist_thr) if hist_thr is not None else None,
                )
            )

        if not colors:
            raise ValueError("advanced color model has no valid avg_color_hist entries")

        return cls(
            hist_bins=(int(bins[0]), int(bins[1]), int(bins[2])),
            default_hist_thr=default_hist_thr,
            colors=colors,
            white_s_p90_max=white_s_p90_max,
            white_v_p50_min=white_v_p50_min,
        )


def _compute_hsv3d_hist(image_bgr, bins: Tuple[int, int, int]) -> Tuple[List[float], Dict[str, float]]:
    """Compute normalized HSV 3D histogram on pixels with V>30.

    Returns (hist, metrics) where metrics includes S p90 and V p50 for optional white checks.
    Accepts numpy array (H,W,3 BGR) or nested lists [[b,g,r], ...].
    """
    hb, sb, vb = bins

    if np is not None:
        try:
            import cv2  # type: ignore
            img = image_bgr
            if not isinstance(img, np.ndarray):
                img = np.array(image_bgr, dtype=np.uint8)  # type: ignore
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = hsv[:, :, 2] > 30
            if mask.any():
                h = hsv[:, :, 0][mask].astype(np.float32)
                s = hsv[:, :, 1][mask].astype(np.float32)
                v = hsv[:, :, 2][mask].astype(np.float32)
                # Percentiles for white rule
                s_p90 = float(np.percentile(s, 90))
                v_p50 = float(np.percentile(v, 50))
                # Indices
                h_idx = (h * (hb / 180.0)).astype(np.int32)
                s_idx = (s * (sb / 256.0)).astype(np.int32)
                v_idx = (v * (vb / 256.0)).astype(np.int32)
                h_idx = np.clip(h_idx, 0, hb - 1)
                s_idx = np.clip(s_idx, 0, sb - 1)
                v_idx = np.clip(v_idx, 0, vb - 1)
                idx = (h_idx * sb * vb + s_idx * vb + v_idx).ravel()
                hist = np.bincount(idx, minlength=hb * sb * vb).astype(np.float32)
                if hist.sum() > 0:
                    hist /= hist.sum()
                return hist.tolist(), {"s_p90": s_p90, "v_p50": v_p50}
            else:
                total_bins = hb * sb * vb
                return [0.0] * total_bins, {"s_p90": 0.0, "v_p50": 0.0}
        except Exception:
            pass

    # Python fallback (slower)
    pixels: List[Tuple[float, float, float]] = []
    s_vals: List[float] = []
    v_vals: List[float] = []
    if np is not None and isinstance(image_bgr, np.ndarray):
        img = image_bgr.reshape(-1, 3)
        for b, g, r in img:
            r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0
            h_val, s_val, v_val = colorsys.rgb_to_hsv(r_n, g_n, b_n)
            if v_val * 255.0 > 30:
                pixels.append((h_val, s_val, v_val))
                s_vals.append(s_val * 255.0)
                v_vals.append(v_val * 255.0)
    else:
        for row in image_bgr:
            for b, g, r in row:
                r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0
                h_val, s_val, v_val = colorsys.rgb_to_hsv(r_n, g_n, b_n)
                if v_val * 255.0 > 30:
                    pixels.append((h_val, s_val, v_val))
                    s_vals.append(s_val * 255.0)
                    v_vals.append(v_val * 255.0)

    total_bins = hb * sb * vb
    hist = [0] * total_bins
    for h_val, s_val, v_val in pixels:
        h_idx = int(h_val * 360.0 // (360 / hb)) % hb
        s_idx = min(int(s_val * sb), sb - 1)
        v_idx = min(int(v_val * vb), vb - 1)
        idx = h_idx * sb * vb + s_idx * vb + v_idx
        hist[idx] += 1
    ssum = float(sum(hist))
    if ssum > 0:
        hist = [h / ssum for h in hist]

    def _percentile(vals: List[float], p: float) -> float:
        if not vals:
            return 0.0
        vs = sorted(vals)
        k = max(0, min(len(vs) - 1, int(round((p / 100.0) * (len(vs) - 1)))))
        return float(vs[k])

    return hist, {"s_p90": _percentile(s_vals, 90.0), "v_p50": _percentile(v_vals, 50.0)}


def _l1(a: List[float], b: List[float]) -> float:
    if np is not None:
        aa = np.asarray(a, dtype=np.float32)
        bb = np.asarray(b, dtype=np.float32)
        sa = aa.sum(); sb = bb.sum()
        if sa > 0: aa = aa / sa
        if sb > 0: bb = bb / sb
        return float(np.sum(np.abs(aa - bb)))
    sa = sum(a); sb = sum(b)
    if sa > 0:
        a = [x / sa for x in a]
    if sb > 0:
        b = [x / sb for x in b]
    return sum(abs(x - y) for x, y in zip(a, b))


class LEDQCEnhanced:
    """Enhanced LED QC using the advanced JSON model."""

    def __init__(self, model: _Model) -> None:
        self.model = model

    @classmethod
    def from_json(cls, path: Any) -> "LEDQCEnhanced":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "colors" not in data:
            raise ValueError("advanced model json must contain 'colors'")
        model = _Model.from_dict(data)
        return cls(model)

    def check(self, image_bgr, allowed_colors: Optional[Iterable[str]] = None) -> LEDQCAdvancedResult:
        hist, metrics = _compute_hsv3d_hist(image_bgr, self.model.hist_bins)

        allowed = set(c.lower() for c in allowed_colors) if allowed_colors else None

        best_name = ""
        best_diff = float("inf")
        best_thr = self.model.default_hist_thr
        scores: List[Tuple[str, float]] = []
        for c in self.model.colors:
            if allowed and c.name.lower() not in allowed:
                continue
            d = _l1(hist, c.avg_color_hist)
            scores.append((c.name, float(d)))
            if d < best_diff:
                best_diff = float(d)
                best_name = c.name
                best_thr = float(c.hist_thr) if c.hist_thr is not None else float(self.model.default_hist_thr)

        if not scores:
            raise ValueError("no colors to compare (check allowed_colors filter)")

        # Base decision by histogram threshold
        is_ok = best_diff <= best_thr

        # Optional white rule: if predicted White, enforce S/V percentile constraints
        if best_name.lower() == "white":
            s_p90 = metrics.get("s_p90", 0.0)
            v_p50 = metrics.get("v_p50", 0.0)
            s_ok = True if self.model.white_s_p90_max is None else (s_p90 <= self.model.white_s_p90_max)
            v_ok = True if self.model.white_v_p50_min is None else (v_p50 >= self.model.white_v_p50_min)
            # Logging reasons when white rules fail (for debugging)
            try:
                import logging
                lg = logging.getLogger(__name__)
                if not (best_diff <= best_thr):
                    lg.info("Color hist FAIL: color=%s diff=%.2f > thr=%.2f", best_name, best_diff, best_thr)
                if not s_ok:
                    lg.info(
                        "White rule FAIL (S): s_p90=%.1f > max=%.1f", s_p90, (self.model.white_s_p90_max if self.model.white_s_p90_max is not None else float('nan'))
                    )
                if not v_ok:
                    lg.info(
                        "White rule FAIL (V): v_p50=%.1f < min=%.1f", v_p50, (self.model.white_v_p50_min if self.model.white_v_p50_min is not None else float('nan'))
                    )
            except Exception:
                pass
            is_ok = is_ok and s_ok and v_ok
        else:
            # Non-white: if histogram fails, log the reason
            if not (best_diff <= best_thr):
                try:
                    import logging
                    logging.getLogger(__name__).info(
                        "Color hist FAIL: color=%s diff=%.2f > thr=%.2f",
                        best_name, best_diff, best_thr,
                    )
                except Exception:
                    pass

        return LEDQCAdvancedResult(
            best_color=best_name,
            diff=float(best_diff),
            threshold=float(best_thr),
            is_ok=bool(is_ok),
            scores=sorted(scores, key=lambda x: x[1]),
            metrics=metrics,
        )

    # --- runtime overrides ---
    def apply_threshold_overrides(self, overrides: Dict[str, float]) -> None:
        """Override per-color histogram thresholds from a mapping (case-insensitive)."""
        if not overrides:
            return
        try:
            # map by lower-cased name
            by_name = {c.name.lower(): c for c in self.model.colors}
            for name, thr in overrides.items():
                key = str(name).lower()
                if key in by_name:
                    try:
                        by_name[key].hist_thr = float(thr)
                    except Exception:
                        continue
        except Exception:
            pass

    def apply_white_overrides(self, overrides: Dict[str, Optional[float]]) -> None:
        """Override white-specific rules: s_p90_max, v_p50_min. Use None to disable."""
        try:
            smax = overrides.get("s_p90_max") if isinstance(overrides, dict) else None
            vmin = overrides.get("v_p50_min") if isinstance(overrides, dict) else None
            if smax is not None:
                try:
                    self.model.white_s_p90_max = float(smax)
                except Exception:
                    self.model.white_s_p90_max = None
            if vmin is not None:
                try:
                    self.model.white_v_p50_min = float(vmin)
                except Exception:
                    self.model.white_v_p50_min = None
        except Exception:
            pass


__all__ = [
    "LEDQCEnhanced",
    "LEDQCAdvancedResult",
]

