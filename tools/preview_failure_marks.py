"""Render the annotated overlay on a sample frame for visual review.

Not part of the product: this is the manual check for how the failure emphasis
layer reads on a real board, since pixel assertions cannot judge legibility.

Usage:
    python tools/preview_failure_marks.py <source-image> <output-image>
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.services.results.annotations import annotate_yolo_frame  # noqa: E402
from core.utils import ImageUtils  # noqa: E402

DETECTIONS = [
    {
        "bbox": [185, 368, 266, 432],
        "class": "J6",
        "class_id": 0,
        "confidence": 0.94,
        "source_index": 0,
        "position_status": "CORRECT",
    },
    {
        "bbox": [318, 368, 410, 432],
        "class": "Black",
        "class_id": 1,
        "confidence": 0.91,
        "source_index": 1,
        "position_status": "CORRECT",
    },
    {
        "bbox": [128, 276, 196, 356],
        "class": "screw",
        "class_id": 2,
        "confidence": 0.88,
        "source_index": 2,
        "position_status": "CORRECT",
    },
    {
        "bbox": [396, 274, 464, 350],
        "class": "screw",
        "class_id": 2,
        "confidence": 0.86,
        "source_index": 3,
        "position_status": "WRONG",
        "position_error": 18.4,
        "position_offset": {"dx": 16.0, "dy": -9.0},
        "position_expected_center": {"cx": 414.0, "cy": 321.0},
        "position_expected_box": {"x1": 380.0, "y1": 283.0, "x2": 448.0, "y2": 359.0},
    },
    {
        "bbox": [520, 170, 600, 266],
        "class": "J9B",
        "class_id": 4,
        "confidence": 0.82,
        "source_index": 4,
        "position_status": "UNEXPECTED",
    },
    {
        "bbox": [8, 455, 62, 502],
        "class": "R97",
        "class_id": 5,
        "confidence": 0.79,
        "source_index": 5,
        "position_status": "CORRECT",
    },
]

COLOR_RESULT = {
    "is_ok": False,
    "items": [
        {"index": 0, "class_name": "J6", "best_color": "J6", "diff": 0.31,
         "threshold": 0.75, "is_ok": True},
        {"index": 1, "class_name": "Black", "best_color": "Orange", "diff": 0.62,
         "threshold": 0.75, "is_ok": False},
        {"index": 2, "class_name": "screw", "best_color": "screw", "diff": 0.28,
         "threshold": 0.70, "is_ok": True},
        {"index": 3, "class_name": "screw", "best_color": "screw", "diff": 0.35,
         "threshold": 0.70, "is_ok": True},
        {"index": 4, "class_name": "J9B", "best_color": "J9B", "diff": 0.40,
         "threshold": 0.75, "is_ok": True},
        {"index": 5, "class_name": "R97", "best_color": "R97", "diff": 0.22,
         "threshold": 0.70, "is_ok": True},
    ],
}

MISSING_LOCATIONS = [
    {"class": "D14", "expected_key": "D14", "bbox": [286, 466, 320, 512],
     "reason": "missing"},
]


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    source, destination = Path(sys.argv[1]), Path(sys.argv[2])
    frame = cv2.imread(str(source))
    if frame is None:
        print(f"cannot read {source}")
        return 1

    annotate_yolo_frame(
        ImageUtils(),
        frame,
        DETECTIONS,
        COLOR_RESULT,
        "DETECTION_FAIL",
        missing_items=["D14"],
        missing_locations=MISSING_LOCATIONS,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), frame)
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
