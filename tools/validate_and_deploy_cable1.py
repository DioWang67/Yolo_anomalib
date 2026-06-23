#!/usr/bin/env python
"""Validate a candidate Cable1 YOLO model, then optionally deploy it.

Acceptance test uses the real production failure frame (6 wires, where the
shipped model found only 4-5 and confused orange/red). A candidate model must
detect all expected wires at the production confidence threshold before it is
allowed to replace the deployed weights.

Usage:
    python tools/validate_and_deploy_cable1.py --weights path/to/best.onnx
    python tools/validate_and_deploy_cable1.py --weights path/to/best.onnx --deploy
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_TARGET = REPO_ROOT / "models" / "Cable1" / "A" / "yolo" / "weights" / "best.onnx"
# Most recent known production failure frame (orange->red, red missing).
DEFAULT_TEST_IMAGE = (
    REPO_ROOT
    / "Result" / "20260623" / "Cable1" / "A" / "DETECTION_FAIL"
    / "original" / "yolo" / "yolo_Cable1_A_102539.jpg"
)
EXPECTED_WIRES = 6
PROD_CONF = 0.4
PROD_IOU = 0.45


def run_detections(weights: str, image_path: Path, conf: float) -> list[tuple]:
    from ultralytics import YOLO

    model = YOLO(weights, task="detect")
    names = model.names
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"cannot read test image: {image_path}")
    res = model(img, conf=conf, iou=PROD_IOU, imgsz=640, verbose=False)[0]
    boxes = res.boxes
    rows = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        rows.append((x1, names[int(boxes.cls[i])], float(boxes.conf[i])))
    rows.sort()
    return rows


def report(weights: str, image_path: Path) -> bool:
    print(f"weights: {weights}")
    print(f"test image: {image_path}")
    passed = True
    for conf in (PROD_CONF, 0.25):
        rows = run_detections(weights, image_path, conf)
        seq = " -> ".join(c for _, c, _ in rows)
        print(f"\n[conf>={conf}] {len(rows)} wires (left->right): {seq}")
        for _, cls, cf in rows:
            print(f"    {cls:8s} {cf:.3f}")
        if conf == PROD_CONF and len(rows) < EXPECTED_WIRES:
            print(
                f"  ! FAIL gate: only {len(rows)}/{EXPECTED_WIRES} wires at "
                f"production conf {PROD_CONF}"
            )
            passed = False
    return passed


def deploy(weights: Path) -> None:
    if not DEPLOY_TARGET.parent.is_dir():
        raise FileNotFoundError(f"deploy dir missing: {DEPLOY_TARGET.parent}")
    if DEPLOY_TARGET.exists():
        backup = DEPLOY_TARGET.with_suffix(f".onnx.bak.{time.strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(DEPLOY_TARGET, backup)
        print(f"backed up current weights -> {backup}")
    shutil.copy2(weights, DEPLOY_TARGET)
    print(f"deployed {weights} -> {DEPLOY_TARGET}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="candidate .onnx weights")
    parser.add_argument("--test-image", default=str(DEFAULT_TEST_IMAGE))
    parser.add_argument("--deploy", action="store_true", help="deploy if gate passes")
    parser.add_argument(
        "--force", action="store_true", help="deploy even if the gate fails"
    )
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[error] weights not found: {weights}", file=sys.stderr)
        return 2

    passed = report(weights, Path(args.test_image))
    print("\n" + ("GATE PASSED" if passed else "GATE FAILED"))

    if args.deploy:
        if passed or args.force:
            deploy(weights)
        else:
            print("not deploying (gate failed); rerun with --force to override")
            return 1
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
