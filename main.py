import sys
import logging
from datetime import datetime

from core.detection_system import DetectionSystem
from app.cli import run_cli
import argparse
import os
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"logs/detection_{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
    )
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="YOLO/Anomalib detection runner")
    parser.add_argument("--product", type=str, help="產品機種", required=False)
    parser.add_argument("--area", type=str, help="區域", required=False)
    parser.add_argument("--type", dest="infer_type", type=str, choices=["yolo", "anomalib"], help="推理類型", required=False)
    parser.add_argument("--image", type=str, help="指定輸入影像路徑（可選）", required=False)
    args = parser.parse_args()

    system = DetectionSystem()
    try:
        if args.product and args.area and args.infer_type:
            frame = None
            if args.image:
                if cv2 is None:
                    logging.getLogger(__name__).error("缺少 OpenCV，無法讀取 --image 檔案")
                elif not os.path.exists(args.image):
                    logging.getLogger(__name__).error(f"影像檔案不存在: {args.image}")
                else:
                    frame = cv2.imread(args.image)
                    if frame is None:
                        logging.getLogger(__name__).error(f"影像讀取失敗: {args.image}")

            result = system.detect(args.product, args.area, args.infer_type, frame=frame)
            print("\n=== 檢測結果 ===")
            print(f"狀態: {result.get('status', '')}")
            print(f"機種: {result.get('product', '')}")
            print(f"區域: {result.get('area', '')}")
            print(f"類型: {result.get('inference_type', '')}")
            print(f"檢查點: {result.get('ckpt_path', '')}")
            print(f"異常分數: {result.get('anomaly_score', '')}")
            print(f"偵測項目: {result.get('detections', [])}")
            print(f"缺少項目: {result.get('missing_items', [])}")
            print(f"原始影像: {result.get('original_image_path', '')}")
            print(f"預處理影像: {result.get('preprocessed_image_path', '')}")
            print(f"熱度圖: {result.get('heatmap_path', '')}")
            print(f"裁切影像: {result.get('cropped_paths', [])}")
            color_info = result.get('color_check')
            if color_info:
                status_text = 'PASS' if color_info.get('is_ok') else 'FAIL'
                diff_val = color_info.get('diff')
                print(f"顏色檢測: {status_text}, 差異: {diff_val}")
            else:
                print("顏色檢測: 未執行")
            print("====================\n")
        else:
            # 進入互動模式
            run_cli(system)
    finally:
        system.shutdown()
