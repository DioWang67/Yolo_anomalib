"""Production preflight for result storage, recovery, and server sync."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.services.inspection_database import (
    InspectionDatabaseError,
    InspectionDatabaseManager,
)
from core.station_data import resolve_result_root


@dataclass(frozen=True)
class ProductionCheck:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class ProductionPreflightSettings:
    result_root: Path
    min_free_disk_mb: int = 1024
    backup_interval_hours: int = 24
    sync_enabled: bool = False
    sync_endpoint: str = ""
    sync_token_env: str = "YOLO11_INSPECTION_SYNC_TOKEN"
    allow_insecure_http: bool = False


def load_settings(
    result_root: str | Path,
    *,
    config_path: str | Path | None = None,
) -> ProductionPreflightSettings:
    payload: dict[str, Any] = {}
    if config_path is not None:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Production config must contain a YAML mapping.")
        global_payload = loaded.get("global")
        payload = (
            dict(global_payload)
            if isinstance(global_payload, dict)
            else dict(loaded)
        )
    return ProductionPreflightSettings(
        result_root=Path(result_root).resolve(),
        min_free_disk_mb=_bounded_int(
            payload.get("min_free_disk_mb", 1024),
            name="min_free_disk_mb",
            minimum=0,
            maximum=10_000_000,
        ),
        backup_interval_hours=_bounded_int(
            payload.get("inspection_backup_interval_hours", 24),
            name="inspection_backup_interval_hours",
            minimum=1,
            maximum=168,
        ),
        sync_enabled=bool(payload.get("inspection_sync_enabled", False)),
        sync_endpoint=str(payload.get("inspection_sync_endpoint", "") or ""),
        sync_token_env=str(
            payload.get(
                "inspection_sync_api_token_env",
                "YOLO11_INSPECTION_SYNC_TOKEN",
            )
            or ""
        ),
        allow_insecure_http=bool(
            payload.get("inspection_sync_allow_insecure_http", False)
        ),
    )


def run_preflight(
    settings: ProductionPreflightSettings,
    *,
    backup_restore_drill: bool = False,
    now: datetime | None = None,
) -> tuple[ProductionCheck, ...]:
    current = _aware_utc(now)
    result_root = settings.result_root
    checks: list[ProductionCheck] = []
    checks.append(_check_result_folder(result_root))
    checks.append(
        _check_disk_capacity(result_root, settings.min_free_disk_mb)
    )
    database_path = result_root / "inspection_records.sqlite3"
    checks.append(_check_database(database_path))
    if backup_restore_drill:
        checks.append(_run_backup_restore_drill(database_path))
    else:
        checks.append(
            ProductionCheck(
                "backup_restore_drill",
                "WARN",
                "Not run. Re-run with --backup-restore-drill before release.",
            )
        )
    checks.append(
        _check_latest_backup(
            database_path,
            current,
            timedelta(hours=settings.backup_interval_hours),
        )
    )
    checks.append(_check_sync_settings(settings))
    return tuple(checks)


def _check_result_folder(result_root: Path) -> ProductionCheck:
    try:
        result_root.mkdir(parents=True, exist_ok=True)
        probe = result_root / f".preflight-{uuid.uuid4().hex}.tmp"
        with probe.open("x", encoding="utf-8") as handle:
            handle.write("production-preflight")
            handle.flush()
            os.fsync(handle.fileno())
        probe.unlink()
    except OSError as exc:
        return ProductionCheck(
            "result_folder_writable",
            "FAIL",
            f"{result_root}: {exc}",
        )
    return ProductionCheck(
        "result_folder_writable",
        "PASS",
        str(result_root),
    )


def _check_disk_capacity(
    result_root: Path,
    minimum_mb: int,
) -> ProductionCheck:
    try:
        free_mb = shutil.disk_usage(result_root).free // (1024 * 1024)
    except OSError as exc:
        return ProductionCheck("free_disk", "FAIL", str(exc))
    status = "PASS" if free_mb >= minimum_mb else "FAIL"
    return ProductionCheck(
        "free_disk",
        status,
        f"free={free_mb:,} MiB required={minimum_mb:,} MiB",
    )


def _check_database(database_path: Path) -> ProductionCheck:
    if not database_path.is_file():
        return ProductionCheck(
            "database_integrity",
            "WARN",
            "No inspection database exists yet; perform one saved inspection.",
        )
    try:
        InspectionDatabaseManager(database_path).check_integrity()
    except (InspectionDatabaseError, OSError) as exc:
        return ProductionCheck("database_integrity", "FAIL", str(exc))
    return ProductionCheck("database_integrity", "PASS", str(database_path))


def _check_latest_backup(
    database_path: Path,
    now: datetime,
    interval: timedelta,
) -> ProductionCheck:
    backup_dir = database_path.parent / "database_backups"
    try:
        backups = tuple(
            path
            for path in backup_dir.glob("*.sqlite3.bak")
            if path.is_file()
        )
        latest = max(backups, key=lambda path: path.stat().st_mtime)
    except ValueError:
        return ProductionCheck(
            "recent_database_backup",
            "WARN",
            "No verified database backup exists.",
        )
    except OSError as exc:
        return ProductionCheck("recent_database_backup", "FAIL", str(exc))
    modified = datetime.fromtimestamp(latest.stat().st_mtime, timezone.utc)
    age = now - modified
    status = "PASS" if age <= interval else "WARN"
    return ProductionCheck(
        "recent_database_backup",
        status,
        f"path={latest} age_hours={age.total_seconds() / 3600:.1f}",
    )


def _check_sync_settings(
    settings: ProductionPreflightSettings,
) -> ProductionCheck:
    if not settings.sync_enabled:
        return ProductionCheck(
            "company_sync_configuration",
            "WARN",
            "Company synchronization is disabled.",
        )
    parsed = urlparse(settings.sync_endpoint.strip())
    local_http = parsed.scheme == "http" and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }
    secure = (
        bool(parsed.hostname)
        and parsed.scheme in {"http", "https"}
        and (
            parsed.scheme == "https"
            or local_http
            or settings.allow_insecure_http
        )
    )
    if not secure:
        return ProductionCheck(
            "company_sync_configuration",
            "FAIL",
            "Sync endpoint is missing or does not meet HTTPS policy.",
        )
    if not settings.sync_token_env or not os.environ.get(
        settings.sync_token_env
    ):
        return ProductionCheck(
            "company_sync_configuration",
            "FAIL",
            "Sync token environment variable is not set: "
            f"{settings.sync_token_env or '<empty>'}",
        )
    return ProductionCheck(
        "company_sync_configuration",
        "PASS",
        f"endpoint={settings.sync_endpoint} token_env={settings.sync_token_env}",
    )


def _run_backup_restore_drill(database_path: Path) -> ProductionCheck:
    if not database_path.is_file():
        return ProductionCheck(
            "backup_restore_drill",
            "FAIL",
            "Cannot run recovery drill before the inspection database exists.",
        )
    try:
        backup = InspectionDatabaseManager(database_path).backup(
            reason="production_preflight"
        )
        with tempfile.TemporaryDirectory(
            prefix=".yolo11-restore-drill-",
            dir=database_path.parent,
        ) as temporary_dir:
            drill_path = Path(temporary_dir) / "inspection_records.sqlite3"
            drill_manager = InspectionDatabaseManager(drill_path)
            drill_manager.restore(backup.path)
            drill_manager.check_integrity()
    except (InspectionDatabaseError, OSError, ValueError) as exc:
        return ProductionCheck("backup_restore_drill", "FAIL", str(exc))
    return ProductionCheck(
        "backup_restore_drill",
        "PASS",
        f"verified_backup={backup.path}",
    )


def _bounded_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be an integer.")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not minimum <= normalized <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}."
        )
    return normalized


def _aware_utc(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if not isinstance(current, datetime) or current.tzinfo is None:
        raise ValueError("Production preflight clock must be timezone-aware.")
    return current.astimezone(timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--config")
    parser.add_argument("--backup-restore-drill", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat WARN checks as release blockers.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(
            resolve_result_root(args.result_root),
            config_path=args.config,
        )
        checks = run_preflight(
            settings,
            backup_restore_drill=args.backup_restore_drill,
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"[FAIL] preflight_configuration: {exc}")
        return 1
    if args.as_json:
        print(
            json.dumps(
                [asdict(check) for check in checks],
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for check in checks:
            print(f"[{check.status}] {check.name}: {check.detail}")
    blocked = any(check.status == "FAIL" for check in checks)
    if args.strict:
        blocked = blocked or any(check.status == "WARN" for check in checks)
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
