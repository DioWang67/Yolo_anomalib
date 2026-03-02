"""共用的檢測結果格式化工具，供 CLI / main.py 使用。"""

from __future__ import annotations

from core.types import DetectionResult


def format_detection_result(result: DetectionResult) -> str:
    """Format a DetectionResult into a human-readable string.

    Args:
        result: The DetectionResult returned by ``DetectionSystem.detect()``.

    Returns:
        Multi-line string suitable for console output.
    """
    lines: list[str] = []

    lines.append("\n=== 檢測結果 ===")
    lines.append(f"狀態 {result.status}")
    lines.append(f"機種: {result.product}")
    lines.append(f"站點: {result.area}")
    lines.append(f"類型: {result.inference_type}")

    if result.error:
        lines.append(f"錯誤訊息: {result.error}")

    if result.status == "ERROR":
        lines.append("====================\n")
        return "\n".join(lines)

    lines.append(f"檢查點 {result.ckpt_path}")
    lines.append(f"異常分數: {result.anomaly_score}")
    lines.append(f"檢測項目: {result.detections}")
    lines.append(f"缺少項目: {result.missing_items}")
    lines.append(f"原始影像: {result.original_image_path}")
    lines.append(f"預處理影像 {result.preprocessed_image_path}")
    lines.append(f"熱度圖 {result.heatmap_path}")
    lines.append(f"裁切影像: {result.cropped_paths}")

    color_info = result.color_check
    if color_info:
        status_text = "PASS" if color_info.get("is_ok") else "FAIL"
        diff_val = color_info.get("diff")
        lines.append(f"顏色檢測: {status_text}, 差異: {diff_val}")
    else:
        lines.append("顏色檢測: 未執行")

    lines.append("====================\n")
    return "\n".join(lines)
