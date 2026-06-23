#!/usr/bin/env python
"""Run the full Cable1 pipeline over known-good boards to measure over-kill.

Loads the deployed model + color/sequence pipeline and reports PASS/FAIL per
image so we can see whether correctly-wired boards are wrongly rejected.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.detection_system import DetectionSystem  # noqa: E402


def main(image_dir: str) -> int:
    files = sorted(glob.glob(str(Path(image_dir) / "*.jpg")) + glob.glob(str(Path(image_dir) / "*.png")))
    if not files:
        print(f"no images in {image_dir}")
        return 1

    system = DetectionSystem()
    passed = 0
    try:
        print(f"{'image':40s} {'status':16s} observed_sequence / fail reason")
        print("-" * 100)
        for f in files:
            frame = cv2.imread(f)
            if frame is None:
                print(f"{Path(f).name:40s} UNREADABLE")
                continue
            res = system.detect("Cable1", "A", "yolo", frame=frame)
            status = str(res.status)
            seq = res.sequence_check or {}
            observed = seq.get("observed")
            detail = ""
            if status == "PASS":
                passed += 1
            else:
                bits = []
                if observed is not None and not seq.get("is_ok", True):
                    bits.append(f"seq={observed}")
                cc = res.color_check or {}
                if not cc.get("is_ok", True):
                    fails = [i.get("class_name") for i in cc.get("items", []) if not i.get("is_ok")]
                    bits.append(f"color_fail={fails}")
                if res.missing_items:
                    bits.append(f"missing={res.missing_items}")
                detail = "  ".join(bits)
            print(f"{Path(f).name:40s} {status:16s} {detail}")
        print("-" * 100)
        print(f"PASS {passed}/{len(files)}  ({passed/len(files)*100:.0f}%)")
    finally:
        system.shutdown()
    return 0


if __name__ == "__main__":
    image_dir = sys.argv[1] if len(sys.argv) > 1 else str(
        REPO_ROOT.parent / "Yolo11_auto_train" / "data" / "Cable1" / "raw" / "images"
    )
    raise SystemExit(main(image_dir))
