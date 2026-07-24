from __future__ import annotations

import cv2
import numpy as np

from core.visualization import draw_detection_overlay


def test_dense_detection_overlay_keeps_full_labels_in_legend(monkeypatch):
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = [
        {
            "bbox": [150 + index * 32, 350, 177 + index * 32, 400],
            "class": class_name,
            "class_id": index,
            "confidence": 0.95 - index * 0.01,
        }
        for index, class_name in enumerate(
            ["Red", "Green", "Orange", "Yellow", "Black", "Black"]
        )
    ]
    text_calls: list[tuple[str, tuple[int, int]]] = []
    original_put_text = cv2.putText

    def record_put_text(image, text, origin, *args, **kwargs):
        text_calls.append((text, origin))
        return original_put_text(image, text, origin, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record_put_text)

    draw_detection_overlay(
        frame,
        detections,
        lambda detection: (0, 200, 0),
    )

    box_tags = [(text, origin) for text, origin in text_calls if text.startswith("#") and " " not in text]
    legend_labels = [(text, origin) for text, origin in text_calls if " " in text]
    assert [text for text, _ in box_tags] == [f"#{index}" for index in range(6)]
    assert all(origin[1] < 120 for _, origin in legend_labels)
    assert all(origin[1] >= 330 for _, origin in box_tags)
