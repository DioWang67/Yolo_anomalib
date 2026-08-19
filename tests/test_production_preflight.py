from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.services.inspection_repository import InspectionRepository
from tools.production_preflight import (
    ProductionPreflightSettings,
    load_settings,
    main,
    run_preflight,
)

_NOW = datetime(2026, 7, 29, tzinfo=timezone.utc)


def _database(result_root: Path) -> Path:
    database_path = result_root / "inspection_records.sqlite3"
    InspectionRepository(database_path).upsert_snapshot(
        {
            "timestamp": _NOW.isoformat(),
            "status": "PASS",
            "detector": "yolo",
            "product": "Cable1",
            "equipment": {"station": "A"},
            "artifacts": {},
        },
        snapshot_path=result_root / "result.json",
    )
    return database_path


def test_preflight_runs_verified_backup_restore_drill(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "Result"
    _database(result_root)

    checks = run_preflight(
        ProductionPreflightSettings(
            result_root=result_root,
            min_free_disk_mb=0,
        ),
        backup_restore_drill=True,
        now=_NOW,
    )

    by_name = {check.name: check for check in checks}
    assert by_name["result_folder_writable"].status == "PASS"
    assert by_name["free_disk"].status == "PASS"
    assert by_name["database_integrity"].status == "PASS"
    assert by_name["backup_restore_drill"].status == "PASS"
    backup_detail, backup_sha256 = by_name["backup_restore_drill"].detail.split(
        " sha256=",
        maxsplit=1,
    )
    backup_path = Path(backup_detail.removeprefix("verified_backup="))
    assert backup_path.is_file()
    assert backup_sha256 == hashlib.sha256(backup_path.read_bytes()).hexdigest()
    assert list(result_root.glob(".preflight-*.tmp")) == []


def test_preflight_rejects_backup_that_changes_during_restore_drill(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_root = tmp_path / "Result"
    _database(result_root)
    real_sha256 = hashlib.sha256
    call_count = 0

    def changing_sha256(path: Path) -> str:
        nonlocal call_count
        call_count += 1
        digest = real_sha256(path.read_bytes()).hexdigest()
        return digest if call_count == 1 else "0" * 64

    monkeypatch.setattr("tools.production_preflight._sha256_file", changing_sha256)

    checks = run_preflight(
        ProductionPreflightSettings(result_root=result_root, min_free_disk_mb=0),
        backup_restore_drill=True,
        now=_NOW,
    )

    drill = next(check for check in checks if check.name == "backup_restore_drill")
    assert drill.status == "FAIL"
    assert "changed during the restore drill" in drill.detail


def test_preflight_requires_sync_token_when_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("COMPANY_TOKEN", raising=False)
    checks = run_preflight(
        ProductionPreflightSettings(
            result_root=tmp_path,
            min_free_disk_mb=0,
            sync_enabled=True,
            sync_endpoint="https://company.example/inspections",
            sync_token_env="COMPANY_TOKEN",
        ),
        now=_NOW,
    )

    sync = next(check for check in checks if check.name == "company_sync_configuration")
    assert sync.status == "FAIL"
    assert "COMPANY_TOKEN" in sync.detail


def test_preflight_can_reach_all_pass_release_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_root = tmp_path / "Result"
    _database(result_root)
    monkeypatch.setenv("COMPANY_TOKEN", "secret")

    checks = run_preflight(
        ProductionPreflightSettings(
            result_root=result_root,
            min_free_disk_mb=0,
            sync_enabled=True,
            sync_endpoint="https://company.example/inspections",
            sync_token_env="COMPANY_TOKEN",
        ),
        backup_restore_drill=True,
        now=datetime.now(timezone.utc),
    )

    assert {check.status for check in checks} == {"PASS"}


def test_load_settings_reads_global_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
global:
  min_free_disk_mb: 2048
  inspection_backup_interval_hours: 12
  inspection_sync_enabled: true
  inspection_sync_endpoint: https://company.example/inspections
  inspection_sync_api_token_env: COMPANY_TOKEN
""",
        encoding="utf-8",
    )

    settings = load_settings(tmp_path / "Result", config_path=config_path)

    assert settings.min_free_disk_mb == 2048
    assert settings.backup_interval_hours == 12
    assert settings.sync_enabled is True
    assert settings.sync_token_env == "COMPANY_TOKEN"


def test_load_settings_rejects_string_boolean_that_could_bypass_https_policy(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'inspection_sync_enabled: true\ninspection_sync_allow_insecure_http: "false"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inspection_sync_allow_insecure_http"):
        load_settings(tmp_path / "Result", config_path=config_path)


def test_preflight_rejects_materially_future_dated_backup(tmp_path: Path) -> None:
    result_root = tmp_path / "Result"
    backup_dir = result_root / "database_backups"
    backup_dir.mkdir(parents=True)
    backup_path = backup_dir / "inspection_records.future.sqlite3.bak"
    backup_path.write_bytes(b"not-used-by-recency-check")
    future_timestamp = _NOW.timestamp() + 3600
    os.utime(backup_path, (future_timestamp, future_timestamp))

    checks = run_preflight(
        ProductionPreflightSettings(result_root=result_root, min_free_disk_mb=0),
        now=_NOW,
    )

    recent_backup = next(check for check in checks if check.name == "recent_database_backup")
    assert recent_backup.status == "WARN"
    assert "timestamp is in the future" in recent_backup.detail


def test_strict_cli_blocks_unresolved_warnings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("COMPANY_TOKEN", "secret")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "min_free_disk_mb: 0\n"
        "inspection_sync_enabled: true\n"
        "inspection_sync_endpoint: https://company.example/inspections\n"
        "inspection_sync_api_token_env: COMPANY_TOKEN\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--result-root",
            os.fspath(tmp_path / "Result"),
            "--config",
            os.fspath(config_path),
            "--strict",
        ]
    )

    assert exit_code == 1


def test_output_json_records_auditable_preflight_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_root = tmp_path / "Result"
    _database(result_root)
    monkeypatch.setenv("COMPANY_TOKEN", "secret")
    config_path = tmp_path / "config.yaml"
    config_bytes = (
        b"min_free_disk_mb: 0\n"
        b"inspection_sync_enabled: true\n"
        b"inspection_sync_endpoint: https://company.example/inspections\n"
        b"inspection_sync_api_token_env: COMPANY_TOKEN\n"
    )
    config_path.write_bytes(config_bytes)
    output_path = tmp_path / "evidence" / "preflight.json"
    evidence_now = datetime.now(timezone.utc)

    exit_code = main(
        [
            "--result-root",
            os.fspath(result_root),
            "--config",
            os.fspath(config_path),
            "--backup-restore-drill",
            "--strict",
            "--output-json",
            os.fspath(output_path),
        ],
        now=evidence_now,
    )

    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["created_at"] == evidence_now.isoformat()
    assert report["started_at"] == evidence_now.isoformat()
    assert report["completed_at"] == evidence_now.isoformat()
    assert report["result_root"] == os.fspath(result_root.resolve())
    assert report["config"] == {
        "path": os.fspath(config_path.resolve()),
        "sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    assert report["backup_restore_drill"] is True
    assert report["strict"] is True
    assert report["blocked"] is False
    assert report["checks"]
    assert {check["status"] for check in report["checks"]} == {"PASS"}
    assert output_path.read_bytes().endswith(b"\n")
    assert list(output_path.parent.glob(f".{output_path.name}.*.tmp")) == []


def test_output_json_preserves_failed_checks_and_blocked_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("COMPANY_TOKEN", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "min_free_disk_mb: 0\n"
        "inspection_sync_enabled: true\n"
        "inspection_sync_endpoint: https://company.example/inspections\n"
        "inspection_sync_api_token_env: COMPANY_TOKEN\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "preflight.json"

    exit_code = main(
        [
            "--result-root",
            os.fspath(tmp_path / "Result"),
            "--config",
            os.fspath(config_path),
            "--output-json",
            os.fspath(output_path),
        ],
        now=_NOW,
    )

    assert exit_code == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["blocked"] is True
    assert {(check["name"], check["status"]) for check in report["checks"]} >= {("company_sync_configuration", "FAIL")}


def test_output_json_keeps_json_stdout_list_compatible(
    tmp_path: Path,
    capsys,
) -> None:
    output_path = tmp_path / "preflight.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("min_free_disk_mb: 0\n", encoding="utf-8")

    exit_code = main(
        [
            "--result-root",
            os.fspath(tmp_path / "Result"),
            "--config",
            os.fspath(config_path),
            "--json",
            "--output-json",
            os.fspath(output_path),
        ],
        now=_NOW,
    )

    assert exit_code == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(stdout_payload, list)
    assert stdout_payload == report["checks"]


def test_config_error_does_not_replace_existing_evidence_report(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- invalid\n- config\n", encoding="utf-8")
    output_path = tmp_path / "preflight.json"
    output_path.write_text("previous verified report\n", encoding="utf-8")

    exit_code = main(
        [
            "--result-root",
            os.fspath(tmp_path / "Result"),
            "--config",
            os.fspath(config_path),
            "--output-json",
            os.fspath(output_path),
        ],
        now=_NOW,
    )

    assert exit_code == 1
    assert "[FAIL] preflight_configuration:" in capsys.readouterr().out
    assert output_path.read_text(encoding="utf-8") == ("previous verified report\n")


def test_atomic_output_failure_preserves_destination_and_cleans_temp_file(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    output_path = tmp_path / "preflight.json"
    output_path.write_text("previous verified report\n", encoding="utf-8")

    def deny_replace(_source: Path, _destination: Path) -> None:
        raise PermissionError("replacement denied")

    monkeypatch.setattr("tools.production_preflight.os.replace", deny_replace)

    exit_code = main(
        [
            "--result-root",
            os.fspath(tmp_path / "Result"),
            "--output-json",
            os.fspath(output_path),
        ],
        now=_NOW,
    )

    assert exit_code == 1
    assert "[FAIL] preflight_report: replacement denied" in (capsys.readouterr().out)
    assert output_path.read_text(encoding="utf-8") == ("previous verified report\n")
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []


def test_backup_change_while_serializing_report_preserves_previous_evidence(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    result_root = tmp_path / "Result"
    _database(result_root)
    output_path = tmp_path / "preflight.json"
    previous_report = b"previous verified report\n"
    output_path.write_bytes(previous_report)
    real_dump = json.dump
    changed_backup: Path | None = None

    def dump_then_change_backup(payload, handle, **kwargs) -> None:
        nonlocal changed_backup
        real_dump(payload, handle, **kwargs)
        backup_check = next(
            check for check in payload["checks"] if check["name"] == "backup_restore_drill"
        )
        backup_detail, _separator, _sha256 = backup_check["detail"].rpartition(
            " sha256="
        )
        changed_backup = Path(backup_detail.removeprefix("verified_backup="))
        changed_backup.write_bytes(b"changed after restore verification")

    monkeypatch.setattr(
        "tools.production_preflight.json.dump",
        dump_then_change_backup,
    )

    exit_code = main(
        [
            "--result-root",
            os.fspath(result_root),
            "--backup-restore-drill",
            "--output-json",
            os.fspath(output_path),
        ],
        now=_NOW,
    )

    assert changed_backup is not None
    assert exit_code == 1
    assert "Verified backup changed before evidence emission" in capsys.readouterr().out
    assert output_path.read_bytes() == previous_report
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []


def test_config_change_while_serializing_report_preserves_previous_evidence(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("min_free_disk_mb: 0\n", encoding="utf-8")
    output_path = tmp_path / "preflight.json"
    previous_report = b"previous verified report\n"
    output_path.write_bytes(previous_report)
    real_dump = json.dump

    def dump_then_change_config(payload, handle, **kwargs) -> None:
        real_dump(payload, handle, **kwargs)
        config_path.write_text("min_free_disk_mb: 1\n", encoding="utf-8")

    monkeypatch.setattr(
        "tools.production_preflight.json.dump",
        dump_then_change_config,
    )

    exit_code = main(
        [
            "--result-root",
            os.fspath(tmp_path / "Result"),
            "--config",
            os.fspath(config_path),
            "--output-json",
            os.fspath(output_path),
        ],
        now=_NOW,
    )

    assert exit_code == 1
    assert "Production config changed while preflight was running" in (
        capsys.readouterr().out
    )
    assert output_path.read_bytes() == previous_report
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []


def test_output_json_cannot_overwrite_config_before_preflight_writes(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "config.yaml"
    original = b"min_free_disk_mb: 0\n"
    config_path.write_bytes(original)
    result_root = tmp_path / "Result"

    exit_code = main(
        [
            "--result-root",
            os.fspath(result_root),
            "--config",
            os.fspath(config_path),
            "--output-json",
            os.fspath(config_path),
        ],
        now=_NOW,
    )

    assert exit_code == 1
    assert "cannot overwrite the config" in capsys.readouterr().out
    assert config_path.read_bytes() == original
    assert not result_root.exists()


def test_output_json_cannot_overwrite_any_result_data(
    tmp_path: Path,
    capsys,
) -> None:
    result_root = tmp_path / "Result"
    database_path = _database(result_root)
    with sqlite3.connect(database_path) as connection:
        before = connection.execute("PRAGMA integrity_check").fetchone()

    exit_code = main(
        [
            "--result-root",
            os.fspath(result_root),
            "--output-json",
            os.fspath(database_path),
        ],
        now=_NOW,
    )

    assert exit_code == 1
    assert "outside the result root" in capsys.readouterr().out
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == before


def test_output_json_rejects_symbolic_link_before_preflight_writes(
    tmp_path: Path,
    capsys,
) -> None:
    target = tmp_path / "existing-report.json"
    original = b"verified evidence\n"
    target.write_bytes(original)
    link = tmp_path / "report-link.json"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")
    result_root = tmp_path / "Result"

    exit_code = main(
        [
            "--result-root",
            os.fspath(result_root),
            "--output-json",
            os.fspath(link),
        ],
        now=_NOW,
    )

    assert exit_code == 1
    assert "cannot be a symbolic link" in capsys.readouterr().out
    assert target.read_bytes() == original
    assert not result_root.exists()
