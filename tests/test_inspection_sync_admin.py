from pathlib import Path

from core.services.inspection_repository import InspectionRepository
from tools.inspection_sync_admin import main


def test_sync_admin_reports_local_outbox(
    tmp_path: Path,
    capsys,
) -> None:
    result_root = tmp_path / "Result"
    repository = InspectionRepository(
        result_root / "inspection_records.sqlite3"
    )
    repository.upsert_snapshot(
        {
            "timestamp": "2026-07-29T12:00:00+08:00",
            "status": "PASS",
            "detector": "yolo",
            "product": "Cable1",
            "equipment": {"station": "A"},
            "artifacts": {},
        },
        snapshot_path=result_root / "result.json",
    )

    exit_code = main(["--result-root", str(result_root)])

    assert exit_code == 0
    assert "pending=1" in capsys.readouterr().out
