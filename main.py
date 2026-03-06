"""程式命令列進入點，可直接執行推論或啟動互動模式。"""

import argparse
import logging
import os
import sys

from app.cli import run_cli
from core.detection_system import DetectionSystem
from core.logging_config import configure_logging

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore


def setup_logging() -> None:
    configure_logging()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(
        description="YOLO/Anomalib detection runner")
    parser.add_argument("--product", type=str, help="產品機種", required=False)
    parser.add_argument("--area", type=str, help="區域", required=False)
    parser.add_argument(
        "--type",
        dest="infer_type",
        type=str,
        choices=["yolo", "anomalib"],
        help="推理類型",
        required=False,
    )
    parser.add_argument(
        "--image", type=str, help="指定輸入影像路徑（可選）", required=False
    )
    args = parser.parse_args()

    system = DetectionSystem()
    try:
        if args.product and args.area and args.infer_type:
            frame = None
            if args.image:
                _logger = logging.getLogger(__name__)

                # --- Guard: OpenCV must be available ---
                if cv2 is None:
                    _logger.error(
                        "缺少 OpenCV (cv2)，無法讀取 --image 檔案。"
                        "請安裝 opencv-python 後重試。"
                    )
                    sys.exit(1)

                # --- Validate image path ---
                image_path = args.image
                try:
                    from core.security import SecurityError, path_validator
                    image_path = str(
                        path_validator.validate_path(args.image, must_exist=True)
                    )
                except ImportError:
                    # Security module unavailable — fall back to basic check
                    if not os.path.exists(args.image):
                        _logger.error("影像檔案不存在: %s", args.image)
                        sys.exit(1)
                except SecurityError as e:
                    _logger.error("影像路徑安全驗證失敗: %s", e)
                    sys.exit(1)
                except FileNotFoundError:
                    _logger.error("影像檔案不存在: %s", args.image)
                    sys.exit(1)

                # --- Read image ---
                frame = cv2.imread(image_path)
                if frame is None:
                    _logger.error("影像讀取失敗 (檔案可能損壞或格式不支援): %s", image_path)
                    sys.exit(1)

            result = system.detect(
                args.product, args.area, args.infer_type, frame=frame
            )
            from core.format_result import format_detection_result
            print(format_detection_result(result))
            if result.status == "ERROR":
                sys.exit(1)
        else:
            # 進入互動模式
            run_cli(system)
    finally:
        system.shutdown()
