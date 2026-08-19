"""Production preflight for result storage, recovery, and server sync."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import uuid
from collections.abc import Callable
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

PREFLIGHT_REPORT_SCHEMA_VERSION = 1
BACKUP_CLOCK_SKEW_TOLERANCE = timedelta(minutes=5)


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


@dataclass(frozen=True)
class ProductionConfigEvidence:
    path: Path | None
    sha256: str | None


def load_settings(
    result_root: str | Path,
    *,
    config_path: str | Path | None = None,
) -> ProductionPreflightSettings:
    settings, _ = _load_settings_with_evidence(
        result_root,
        config_path=config_path,
    )
    return settings


def _load_settings_with_evidence(
    result_root: str | Path,
    *,
    config_path: str | Path | None = None,
) -> tuple[ProductionPreflightSettings, ProductionConfigEvidence]:
    payload, evidence = _load_config(config_path)
    settings = _settings_from_payload(result_root, payload)
    return settings, evidence


def _load_config(
    config_path: str | Path | None,
) -> tuple[dict[str, Any], ProductionConfigEvidence]:
    if config_path is None:
        return {}, ProductionConfigEvidence(path=None, sha256=None)

    candidate = Path(config_path)
    if candidate.is_symlink():
        raise ValueError("Production config cannot be a symbolic link.")
    path = candidate.resolve()
    raw_config = path.read_bytes()
    loaded = yaml.safe_load(raw_config.decode("utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Production config must contain a YAML mapping.")
    global_payload = loaded.get("global")
    payload = dict(global_payload) if isinstance(global_payload, dict) else dict(loaded)
    return payload, ProductionConfigEvidence(
        path=path,
        sha256=hashlib.sha256(raw_config).hexdigest(),
    )


def _settings_from_payload(
    result_root: str | Path,
    payload: dict[str, Any],
) -> ProductionPreflightSettings:
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
        sync_enabled=_strict_bool(
            payload.get("inspection_sync_enabled", False),
            name="inspection_sync_enabled",
        ),
        sync_endpoint=str(payload.get("inspection_sync_endpoint", "") or ""),
        sync_token_env=str(
            payload.get(
                "inspection_sync_api_token_env",
                "YOLO11_INSPECTION_SYNC_TOKEN",
            )
            or ""
        ),
        allow_insecure_http=_strict_bool(
            payload.get("inspection_sync_allow_insecure_http", False),
            name="inspection_sync_allow_insecure_http",
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
    checks.append(_check_disk_capacity(result_root, settings.min_free_disk_mb))
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
        backups = tuple(path for path in backup_dir.glob("*.sqlite3.bak") if path.is_file())
        latest = max(backups, key=lambda path: path.stat().st_mtime)
        modified = datetime.fromtimestamp(latest.stat().st_mtime, timezone.utc)
    except ValueError:
        return ProductionCheck(
            "recent_database_backup",
            "WARN",
            "No verified database backup exists.",
        )
    except OSError as exc:
        return ProductionCheck("recent_database_backup", "FAIL", str(exc))
    age = now - modified
    if age < -BACKUP_CLOCK_SKEW_TOLERANCE:
        return ProductionCheck(
            "recent_database_backup",
            "WARN",
            f"Backup timestamp is in the future: path={latest} modified={modified.isoformat()}",
        )
    age = max(age, timedelta())
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
        and (parsed.scheme == "https" or local_http or settings.allow_insecure_http)
    )
    if not secure:
        return ProductionCheck(
            "company_sync_configuration",
            "FAIL",
            "Sync endpoint is missing or does not meet HTTPS policy.",
        )
    if not settings.sync_token_env or not os.environ.get(settings.sync_token_env):
        return ProductionCheck(
            "company_sync_configuration",
            "FAIL",
            f"Sync token environment variable is not set: {settings.sync_token_env or '<empty>'}",
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
        backup = InspectionDatabaseManager(database_path).backup(reason="production_preflight")
        backup_sha256 = _sha256_file(backup.path)
        with tempfile.TemporaryDirectory(
            prefix=".yolo11-restore-drill-",
            dir=database_path.parent,
        ) as temporary_dir:
            drill_path = Path(temporary_dir) / "inspection_records.sqlite3"
            drill_manager = InspectionDatabaseManager(drill_path)
            drill_manager.restore(backup.path)
            drill_manager.check_integrity()
        if _sha256_file(backup.path) != backup_sha256:
            raise ValueError("Verified backup changed during the restore drill.")
    except (InspectionDatabaseError, OSError, ValueError) as exc:
        return ProductionCheck("backup_restore_drill", "FAIL", str(exc))
    return ProductionCheck(
        "backup_restore_drill",
        "PASS",
        f"verified_backup={backup.path} sha256={backup_sha256}",
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
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return normalized


def _strict_bool(value: object, *, name: str) -> bool:
    """Accept only YAML booleans at the security-sensitive config boundary."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a YAML boolean (true or false).")
    return value


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
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Atomically write a schema-versioned preflight evidence report.",
    )
    return parser


def _is_blocked(
    checks: tuple[ProductionCheck, ...],
    *,
    strict: bool,
) -> bool:
    blocking_statuses = {"FAIL", "WARN"} if strict else {"FAIL"}
    return any(check.status in blocking_statuses for check in checks)


def _build_evidence_report(
    *,
    started_at: datetime,
    completed_at: datetime,
    settings: ProductionPreflightSettings,
    config: ProductionConfigEvidence,
    backup_restore_drill: bool,
    strict: bool,
    checks: tuple[ProductionCheck, ...],
    blocked: bool,
) -> dict[str, Any]:
    return {
        "schema_version": PREFLIGHT_REPORT_SCHEMA_VERSION,
        "created_at": completed_at.isoformat(),
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "result_root": str(settings.result_root),
        "config": {
            "path": str(config.path) if config.path is not None else None,
            "sha256": config.sha256,
        },
        "backup_restore_drill": backup_restore_drill,
        "strict": strict,
        "checks": [asdict(check) for check in checks],
        "blocked": blocked,
    }


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
    *,
    before_replace: Callable[[], None] | None = None,
) -> None:
    if path.is_symlink():
        raise ValueError("Preflight report destination cannot be a symbolic link.")
    destination = path.expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if before_replace is not None:
            before_replace()
        os.replace(temporary_path, destination)
        temporary_path = None
    except (OSError, TypeError, ValueError):
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_report_destination(
    path: Path,
    *,
    settings: ProductionPreflightSettings,
    config: ProductionConfigEvidence,
) -> Path:
    """Keep an evidence output away from live station data and its config."""
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ValueError("Preflight report destination cannot be a symbolic link.")
    destination = candidate.resolve(strict=False)
    result_root = settings.result_root.resolve(strict=False)
    try:
        destination.relative_to(result_root)
    except ValueError:
        pass
    else:
        raise ValueError(f"Preflight report destination must be outside the result root: {result_root}")
    if config.path is not None and destination == config.path.resolve(strict=False):
        raise ValueError("Preflight report destination cannot overwrite the config.")
    if destination.suffix.lower() != ".json":
        raise ValueError("Preflight report destination must use a .json suffix.")
    return destination


def _verify_config_evidence(config: ProductionConfigEvidence) -> None:
    """Fail when the config changes between parsing and evidence emission."""
    if config.path is None:
        return
    if config.path.is_symlink() or not config.path.is_file():
        raise ValueError("Production config became missing or unsafe during preflight.")
    if _sha256_file(config.path) != config.sha256:
        raise ValueError("Production config changed while preflight was running.")


def _verify_backup_evidence(checks: tuple[ProductionCheck, ...]) -> None:
    """Re-verify the exact backup bytes immediately before reporting them."""
    for check in checks:
        if check.name != "backup_restore_drill" or check.status != "PASS":
            continue
        path_text, separator, expected_sha256 = check.detail.rpartition(" sha256=")
        if not separator or not path_text.startswith("verified_backup="):
            raise ValueError("Backup restore evidence has an invalid format.")
        path = Path(path_text.removeprefix("verified_backup="))
        if path.is_symlink() or not path.is_file():
            raise ValueError("Verified backup became missing or unsafe.")
        if _sha256_file(path) != expected_sha256:
            raise ValueError("Verified backup changed before evidence emission.")


def main(
    argv: list[str] | None = None,
    *,
    now: datetime | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    report_destination: Path | None = None
    try:
        current = _aware_utc(now)
        settings, config_evidence = _load_settings_with_evidence(
            resolve_result_root(args.result_root),
            config_path=args.config,
        )
        if args.output_json is not None:
            report_destination = _validate_report_destination(
                args.output_json,
                settings=settings,
                config=config_evidence,
            )
        checks = run_preflight(
            settings,
            backup_restore_drill=args.backup_restore_drill,
            now=current,
        )
        _verify_backup_evidence(checks)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"[FAIL] preflight_configuration: {exc}")
        return 1
    completed_at = current if now is not None else _aware_utc(None)
    blocked = _is_blocked(checks, strict=args.strict)
    if report_destination is not None:
        report = _build_evidence_report(
            started_at=current,
            completed_at=completed_at,
            settings=settings,
            config=config_evidence,
            backup_restore_drill=args.backup_restore_drill,
            strict=args.strict,
            checks=checks,
            blocked=blocked,
        )

        def verify_report_sources() -> None:
            _verify_config_evidence(config_evidence)
            _verify_backup_evidence(checks)

        try:
            verify_report_sources()
            _validate_report_destination(
                report_destination,
                settings=settings,
                config=config_evidence,
            )
            _write_json_atomic(
                report_destination,
                report,
                before_replace=verify_report_sources,
            )
        except (OSError, TypeError, ValueError) as exc:
            print(f"[FAIL] preflight_report: {exc}")
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
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
