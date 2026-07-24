from __future__ import annotations

from datetime import datetime

from core.services.results.path_manager import ResultPathManager


def test_same_timestamp_still_produces_unique_inspection_paths(tmp_path):
    manager = ResultPathManager(str(tmp_path))
    timestamp = datetime(2026, 7, 22, 12, 34, 56, 123456)

    first = manager.build_paths(
        status="PASS",
        detector="yolo",
        product="Cable1",
        area="A",
        anomaly_score=None,
        timestamp=timestamp,
    )
    second = manager.build_paths(
        status="PASS",
        detector="yolo",
        product="Cable1",
        area="A",
        anomaly_score=None,
        timestamp=timestamp,
    )

    assert first.inspection_id != second.inspection_id
    assert first.image_name != second.image_name
    assert first.original_path != second.original_path
    assert "123456" in first.image_name
