from __future__ import annotations

from datetime import datetime
from typing import Any

from .excel_buffer import format_excel_row


def build_excel_row(
    columns: list[str],
    *,
    timestamp: datetime,
    status: str,
    detector: str,
    product: str | None,
    area: str | None,
    detections: list[dict[str, Any]],
    missing_items: list[str],
    anomaly_score: float | None,
    annotated_path: str,
    original_path: str,
    preprocessed_path: str,
    heatmap_path: str,
    cropped_paths: list[str],
    ckpt_path: str | None,
    color_result: dict[str, Any] | None,
    test_id: int,
) -> list[Any]:
    """Compose an Excel row according to the configured column order."""
    confidence_scores = (
        ";".join(f"{det['class']}:{det['confidence']:.2f}" for det in detections)
        if detections
        else ""
    )

    error_message = ""
    if status != "PASS":
        error_message = (
            f"缺失項目: {', '.join(missing_items)}"
            if missing_items
            else "異常分數超出門檻"
        )

    color_status = ""
    diff_value = ""
    if color_result:
        color_status = "PASS" if color_result.get("is_ok", False) else "FAIL"
        diff_value = ";".join(
            f"{item.get('diff', 0):.2f}" for item in color_result.get("items", [])
        )

    row_dict = {
        columns[0]: timestamp,
        columns[1]: test_id,
        columns[2]: product or "",
        columns[3]: area or "",
        columns[4]: detector,
        columns[5]: status,
        columns[6]: confidence_scores,
        columns[7]: anomaly_score if anomaly_score is not None else "",
        columns[8]: color_status,
        columns[9]: diff_value,
        columns[10]: error_message,
        columns[11]: annotated_path,
        columns[12]: original_path,
        columns[13]: preprocessed_path,
        columns[14]: heatmap_path,
        columns[15]: ";".join(cropped_paths) if cropped_paths else "",
        columns[16]: ckpt_path or "",
    }
    return format_excel_row(columns, row_dict)
