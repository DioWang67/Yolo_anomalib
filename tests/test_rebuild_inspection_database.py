import json

from core.services.inspection_repository import InspectionRepository
from tools.rebuild_inspection_database import rebuild_inspection_database


def test_rebuild_indexes_existing_snapshots_and_reports_invalid_files(tmp_path):
    result_root = tmp_path / "Result"
    metadata = result_root / "20260720" / "Cable1" / "A" / "FAIL" / "metadata" / "yolo"
    metadata.mkdir(parents=True)
    (metadata / "valid_config_snapshot.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-07-20T10:00:00",
                "status": "FAIL",
                "detector": "yolo",
                "product": "Cable1",
                "area": "A",
            }
        ),
        encoding="utf-8",
    )
    (metadata / "invalid_config_snapshot.json").write_text(
        "not-json", encoding="utf-8"
    )

    indexed_count, errors = rebuild_inspection_database(result_root)

    assert indexed_count == 1
    assert len(errors) == 1
    records = InspectionRepository(
        result_root / "inspection_records.sqlite3"
    ).query("SELECT product, station FROM inspections")
    assert records == [{"product": "Cable1", "station": "A"}]
