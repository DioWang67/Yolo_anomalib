"""Open the button-based inference case review window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.gui.review_cases_dialog import run_review_dialog


def main(argv: list[str] | None = None) -> int:
    """Parse paths and launch the review UI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="Result")
    parser.add_argument("--manifest", default="review_manifest.csv")
    parser.add_argument("--training-data", default="../Yolo11_auto_train/data")
    parser.add_argument("--product")
    parser.add_argument("--area")
    args = parser.parse_args(argv)
    run_review_dialog(
        result_root=args.result_root,
        manifest_path=args.manifest,
        training_data_dir=args.training_data,
        product=args.product,
        area=args.area,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
