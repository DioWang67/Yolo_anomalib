"""共用的檢測結果格式化工具，供 CLI / main.py 使用。"""

from __future__ import annotations

from typing import Any


def format_detection_result(result: dict[str, Any]) -> str:
    """Format a detection result dict into a human-readable string.

    Args:
        result: The result dict returned by ``DetectionSystem.detect()``.

    Returns:
        Multi-line string suitable for console output.
    """
    lines: list[str] = []
    status = result.get("status", "")
    error_msg = result.get("error") or result.get("error_message", "")

    lines.append("\n=== 檢測結果 ===")
    lines.append(f"狀態 {status}")
    lines.append(f"機種: {result.get('product', '')}")
    lines.append(f"站點: {result.get('area', '')}")
    lines.append(f"類型: {result.get('inference_type', '')}")

    if error_msg:
        lines.append(f"錯誤訊息: {error_msg}")

    if status == "ERROR":
        lines.append("====================\n")
        return "\n".join(lines)

    lines.append(f"檢查點 {result.get('ckpt_path', '')}")
    lines.append(f"異常分數: {result.get('anomaly_score', '')}")
    lines.append(f"檢測項目: {result.get('detections', [])}")
    lines.append(f"缺少項目: {result.get('missing_items', [])}")
    lines.append(f"原始影像: {result.get('original_image_path', '')}")
    lines.append(f"預處理影像 {result.get('preprocessed_image_path', '')}")
    lines.append(f"熱度圖 {result.get('heatmap_path', '')}")
    lines.append(f"裁切影像: {result.get('cropped_paths', [])}")

    color_info = result.get("color_check")
    if color_info:
        status_text = "PASS" if color_info.get("is_ok") else "FAIL"
        diff_val = color_info.get("diff")
        lines.append(f"顏色檢測: {status_text}, 差異: {diff_val}")
    else:
        lines.append("顏色檢測: 未執行")

    lines.append("====================\n")
    return "\n".join(lines)
