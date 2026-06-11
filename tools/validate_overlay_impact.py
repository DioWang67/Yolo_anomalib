"""A/B validation: does removing the camera FPS overlay change inference results?

Field frames captured by MVSCamera.get_frame() have a green "FPS: x.x" text
burned into the top-left corner before inference (see review issue C3). Before
removing that overlay from the capture path, this script verifies on recorded
field images that a clean frame produces the same inspection outcome.

For every image the script runs the full DetectionSystem.detect() twice:

* **A (overlay)**  — the recorded image as-is (text already burned in).
* **B (clean)**    — the same image with the green text inpainted away.

and reports status changes, detection-set changes, and bbox center drift.

Usage::

    python tools/validate_overlay_impact.py ^
        --images Result/20260608/PCBA1/A/PASS/original/yolo ^
        --images Result/20260608/PCBA1/A/DETECTION_FAIL/original/yolo ^
        --product PCBA1 --area A --type yolo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running as `python tools/validate_overlay_impact.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Overlay text is drawn at (10, 30) with FONT_HERSHEY_SIMPLEX scale 1 in pure
# green; a second "Auto Exposure" line may appear at (10, 60). Confine the
# mask search to a generous corner window so board content is never touched.
_CORNER_H = 100
_CORNER_W = 520


def remove_overlay(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Inpaint the pure-green overlay text in the top-left corner.

    Returns:
        (clean_image, masked_pixel_count). A zero count means no overlay was
        found (image was already clean).
    """
    corner = image[:_CORNER_H, :_CORNER_W]
    b, g, r = corner[..., 0].astype(int), corner[..., 1].astype(int), corner[..., 2].astype(int)
    # putText uses (0, 255, 0); JPEG smears edges, so accept a tolerance band.
    mask_corner = ((g > 160) & (b < 120) & (r < 120) & (g - b > 80) & (g - r > 80)).astype(np.uint8)
    masked = int(mask_corner.sum())
    if masked == 0:
        return image, 0
    mask_corner = cv2.dilate(mask_corner, np.ones((5, 5), np.uint8))
    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    full_mask[:_CORNER_H, :_CORNER_W] = mask_corner
    clean = cv2.inpaint(image, full_mask, 3, cv2.INPAINT_TELEA)
    return clean, masked


def _detection_summary(result) -> dict[str, object]:
    classes = sorted(item.label for item in result.items)
    centers = {
        f"{item.label}#{i}": (
            (item.bbox_xyxy[0] + item.bbox_xyxy[2]) / 2,
            (item.bbox_xyxy[1] + item.bbox_xyxy[3]) / 2,
        )
        for i, item in enumerate(result.items)
    }
    return {
        "status": result.status,
        "classes": classes,
        "missing": sorted(result.missing_items or []),
        "anomaly_score": result.anomaly_score,
        "centers": centers,
    }


def _max_center_drift(a: dict, b: dict) -> float:
    drift = 0.0
    for key, (ax, ay) in a["centers"].items():
        if key in b["centers"]:
            bx, by = b["centers"][key]
            drift = max(drift, float(np.hypot(ax - bx, ay - by)))
    return drift


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", action="append", required=True,
                        help="Folder of original field images (repeatable)")
    parser.add_argument("--product", required=True)
    parser.add_argument("--area", required=True)
    parser.add_argument("--type", dest="infer_type", default="yolo")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max images per folder (0 = all)")
    args = parser.parse_args()

    # Validation must not touch the camera: detection runs on file frames only.
    import core.detection_system as ds
    ds.DetectionSystem.initialize_camera = lambda self: None  # type: ignore[method-assign]
    from core.detection_system import DetectionSystem

    system = DetectionSystem()
    rows: list[dict[str, object]] = []
    changed = 0

    try:
        for folder in args.images:
            paths = sorted(Path(folder).glob("*.jpg")) + sorted(Path(folder).glob("*.png"))
            if args.limit:
                paths = paths[: args.limit]
            recorded = next(
                (part for part in Path(folder).parts
                 if part in {"PASS", "FAIL", "DETECTION_FAIL", "INFERENCE_ERROR"}),
                "?",
            )
            for path in paths:
                image = cv2.imread(str(path))
                if image is None:
                    print(f"SKIP unreadable: {path}")
                    continue
                clean, masked_px = remove_overlay(image)

                res_overlay = system.detect(
                    args.product, args.area, args.infer_type, frame=image
                )
                res_clean = system.detect(
                    args.product, args.area, args.infer_type, frame=clean
                )
                a = _detection_summary(res_overlay)
                b = _detection_summary(res_clean)
                same_status = a["status"] == b["status"]
                same_classes = a["classes"] == b["classes"]
                drift = _max_center_drift(a, b)
                if not (same_status and same_classes):
                    changed += 1
                rows.append({
                    "image": path.name,
                    "recorded": recorded,
                    "masked_px": masked_px,
                    "overlay": a,
                    "clean": b,
                    "same_status": same_status,
                    "same_classes": same_classes,
                    "max_drift_px": round(drift, 2),
                })
                flag = "OK " if same_status and same_classes else "DIFF"
                score_a = a["anomaly_score"]
                score_b = b["anomaly_score"]
                score_txt = (
                    f" score {score_a:.4f}->{score_b:.4f}"
                    if isinstance(score_a, float) and isinstance(score_b, float)
                    else ""
                )
                print(
                    f"[{flag}] {path.name} recorded={recorded} "
                    f"overlay={a['status']} clean={b['status']} "
                    f"classes_same={same_classes} drift={drift:.2f}px "
                    f"masked_px={masked_px}{score_txt}"
                )
                if not same_classes:
                    print(f"       overlay classes: {a['classes']}")
                    print(f"       clean   classes: {b['classes']}")
                if a["missing"] != b["missing"]:
                    print(f"       missing overlay={a['missing']} clean={b['missing']}")
    finally:
        system.shutdown()

    print()
    print(f"Total: {len(rows)} images, outcome changes: {changed}")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
