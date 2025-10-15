"""互動式命令列介面，可在無 GUI 環境下執行檢測。"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_cli(system) -> None:
    """Run an interactive shell to execute detections."""
    logger = system.logger
    logger.logger.info("檢測系統已啟動，等待使用者輸入...")

    models_base = os.path.join(PROJECT_ROOT, "models")
    if not os.path.isdir(models_base):
        logger.logger.error(f"找不到 models 目錄: {models_base}")
        return

    available_products = [
        d
        for d in os.listdir(models_base)
        if os.path.isdir(os.path.join(models_base, d))
    ]
    if not available_products:
        logger.logger.error("models 資料夾中未找到任何機種")
        return

    print(f"可用機種: {', '.join(available_products)}")
    while True:
        product = input("請輸入要檢測的機種 (輸入 'quit' 離開): ").strip()
        if product.lower() == "quit":
            logger.logger.info("離開檢測系統")
            system.shutdown()
            return
        if product not in available_products:
            print(f"無此機種: {product}，可用: {', '.join(available_products)}")
            continue
        break

    product_dir = os.path.join(models_base, product)
    available_areas = [
        d
        for d in os.listdir(product_dir)
        if os.path.isdir(os.path.join(product_dir, d))
    ]

    while True:
        print(f"可用區域: {', '.join(available_areas)}")
        cmd = input("請輸入檢測指令 (格式: area,inference_type 或 quit): ").strip()
        if cmd.lower() == "quit":
            logger.logger.info("離開檢測系統")
            system.shutdown()
            break

        try:
            parts = cmd.split(",")
            if len(parts) != 2:
                print("指令格式錯誤，應為 area,inference_type")
                continue

            area, inference_type = parts[0].strip(), parts[1].strip().lower()
            if area not in available_areas:
                print(f"無此區域: {area}，可用: {', '.join(available_areas)}")
                continue
            if inference_type not in ["yolo", "anomalib"]:
                print("推理類型只能是: yolo 或 anomalib")
                continue

            config_path = os.path.join(
                models_base, product, area, inference_type, "config.yaml"
            )
            if not os.path.exists(config_path):
                print(f"模型設定不存在: {config_path}")
                continue

            result = system.detect(product, area, inference_type)
            status = result.get("status", "")
            error_msg = result.get("error") or result.get("error_message", "")
            print("\n=== 檢測結果 ===")
            print(f"狀態 {status}")
            print(f"機種: {result.get('product', '')}")
            print(f"站點: {result.get('area', '')}")
            print(f"類型: {result.get('inference_type', '')}")
            if error_msg:
                print(f"錯誤訊息: {error_msg}")
            if status == "ERROR":
                print("====================\n")
                continue
            print(f"檢查點 {result.get('ckpt_path', '')}")
            print(f"異常分數: {result.get('anomaly_score', '')}")
            print(f"檢測項目: {result.get('detections', [])}")
            print(f"缺少項目: {result.get('missing_items', [])}")
            print(f"原始影像: {result.get('original_image_path', '')}")
            print(f"預處理影像 {result.get('preprocessed_image_path', '')}")
            print(f"熱度圖 {result.get('heatmap_path', '')}")
            print(f"裁切影像: {result.get('cropped_paths', [])}")
            color_info = result.get("color_check")
            if color_info:
                status_text = "PASS" if color_info.get("is_ok") else "FAIL"
                diff_val = color_info.get("diff")
                print(f"顏色檢測: {status_text}, 差異: {diff_val}")
            else:
                print("顏色檢測: 未執行")
            print("====================\n")

        except KeyboardInterrupt:
            print("\n中止，正在關閉...")
            system.shutdown()
            break
        except Exception as e:
            logger.logger.error(f"指令處理失敗: {str(e)}")


