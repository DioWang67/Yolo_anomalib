"""Immutable deployable color profiles composed from a base model and revisions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from core.services.inspection_release_models import InspectionReleaseError
from core.services.inspection_release_store import sha256_file
from core.station_data import StationDataPaths, load_station_data_paths
from tools.color_calibration_service import canonical_sha256
from tools.color_configuration_revisions import ColorConfigurationRevision

COLOR_PROFILE_SCHEMA_VERSION = 1


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class ColorRevisionBinding:
    """One exact per-color calibration revision inside a profile."""

    scope_hash: str
    threshold_key: str
    revision_id: str
    display_version: str
    config_path: str
    config_sha256: str
    canonical_config_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "scope_hash": self.scope_hash,
            "threshold_key": self.threshold_key,
            "revision_id": self.revision_id,
            "display_version": self.display_version,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "canonical_config_sha256": self.canonical_config_sha256,
        }


@dataclass(frozen=True)
class ColorProfilePackage:
    """One immutable, complete color-check configuration."""

    package_id: str
    display_version: str
    product: str
    area: str
    model_type: str
    checker_type: str
    colors: tuple[str, ...]
    color_model_path: Path
    color_model_sha256: str
    manifest_path: Path
    manifest_sha256: str
    revisions: tuple[ColorRevisionBinding, ...]

    @property
    def revision_overrides(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (binding.scope_hash, binding.revision_id) for binding in self.revisions
            )
        )

    @property
    def summary(self) -> str:
        overrides = ", ".join(
            f"{item.threshold_key}: {item.display_version}" for item in self.revisions
        )
        base = f"{len(self.colors)} 色基準"
        return f"{base}｜{overrides}" if overrides else base


class ColorProfileStore:
    """Create deterministic append-only packages without changing live config."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        discovered_paths = load_station_data_paths(self.root)
        self._station_paths: StationDataPaths | None = (
            discovered_paths if discovered_paths.color_profiles == self.root else None
        )

    def create(
        self,
        *,
        product: str,
        area: str,
        model_type: str,
        model_config_path: str | Path,
        project_root: str | Path,
        revisions: Sequence[ColorConfigurationRevision] = (),
        color_model_override: str | Path | None = None,
    ) -> ColorProfilePackage | None:
        """Snapshot the complete base model and exact non-conflicting revisions."""
        config_path = Path(model_config_path).expanduser().resolve()
        config = self._read_model_config(config_path)
        if not bool(config.get("enable_color_check")):
            return None
        if color_model_override is not None:
            color_model = Path(color_model_override).expanduser().resolve()
            if color_model.is_symlink() or not color_model.is_file():
                raise InspectionReleaseError("選取的完整顏色基準候選不存在或不安全。")
        else:
            color_model_value = str(config.get("color_model_path") or "").strip()
            if not color_model_value:
                raise InspectionReleaseError(
                    "顏色檢查已啟用，但模型設定缺少 color_model_path。"
                )
            color_model = self._resolve_color_model(
                color_model_value,
                config_path=config_path,
                project_root=Path(project_root).expanduser().resolve(),
            )
        checker_type = str(config.get("color_checker_type") or "stats").strip().lower()
        model_payload = self._read_color_model(color_model)
        colors = self._extract_colors(model_payload)
        revision_bindings = self._revision_bindings(
            revisions,
            checker_type=checker_type,
            product=str(product).strip(),
            area=str(area).strip(),
            model_type=str(model_type).strip().lower(),
        )
        identity = {
            "schema_version": COLOR_PROFILE_SCHEMA_VERSION,
            "product": str(product).strip(),
            "area": str(area).strip(),
            "model_type": str(model_type).strip().lower(),
            "checker_type": checker_type,
            "colors": list(colors),
            "color_model_sha256": sha256_file(color_model),
            "revisions": [
                {
                    "scope_hash": item.scope_hash,
                    "threshold_key": item.threshold_key,
                    "revision_id": item.revision_id,
                    "config_sha256": item.config_sha256,
                }
                for item in revision_bindings
            ],
        }
        package_id = hashlib.sha256(_canonical_json(identity)).hexdigest()[:24]
        destination = self.root / package_id
        if destination.is_dir():
            return self.load(destination / "manifest.json")

        staging = self.root / f".{package_id}.{uuid4().hex}.tmp"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            model_snapshot = staging / "color_model.json"
            shutil.copyfile(color_model, model_snapshot)
            manifest_payload = {
                **identity,
                "package_id": package_id,
                "display_version": (
                    f"{checker_type}-{len(colors)}color-{package_id[:8]}"
                ),
                "source_color_model_path": str(color_model),
                "color_model_path": "color_model.json",
                "revisions": [item.to_dict() for item in revision_bindings],
            }
            (staging / "manifest.json").write_bytes(_canonical_json(manifest_payload))
            self.root.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(staging, destination)
            except OSError:
                if not destination.is_dir():
                    raise
            return self.load(destination / "manifest.json")
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def load(self, manifest_path: str | Path) -> ColorProfilePackage:
        manifest = Path(manifest_path).expanduser().resolve()
        try:
            manifest.relative_to(self.root)
        except ValueError as exc:
            raise InspectionReleaseError("顏色方案 manifest 超出允許目錄。") from exc
        if manifest.is_symlink() or not manifest.is_file():
            raise InspectionReleaseError("顏色方案 manifest 不存在。")
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InspectionReleaseError("顏色方案 manifest 無法讀取。") from exc
        if payload.get("schema_version") != COLOR_PROFILE_SCHEMA_VERSION:
            raise InspectionReleaseError("不支援的顏色方案格式。")
        package_id = str(payload.get("package_id") or "")
        if manifest.parent.name != package_id:
            raise InspectionReleaseError("顏色方案目錄與 package ID 不一致。")
        identity = {
            "schema_version": COLOR_PROFILE_SCHEMA_VERSION,
            "product": str(payload.get("product") or ""),
            "area": str(payload.get("area") or ""),
            "model_type": str(payload.get("model_type") or ""),
            "checker_type": str(payload.get("checker_type") or ""),
            "colors": list(payload.get("colors") or ()),
            "color_model_sha256": str(payload.get("color_model_sha256") or ""),
            "revisions": [
                {
                    "scope_hash": str(item["scope_hash"]),
                    "threshold_key": str(item["threshold_key"]),
                    "revision_id": str(item["revision_id"]),
                    "config_sha256": str(item["config_sha256"]),
                }
                for item in payload.get("revisions") or ()
            ],
        }
        expected_package_id = hashlib.sha256(_canonical_json(identity)).hexdigest()[:24]
        if expected_package_id != package_id:
            raise InspectionReleaseError("顏色方案 package ID 驗證失敗。")
        model_path = (manifest.parent / str(payload["color_model_path"])).resolve()
        try:
            model_path.relative_to(manifest.parent)
        except ValueError as exc:
            raise InspectionReleaseError("顏色基準檔超出方案目錄。") from exc
        expected_model_sha = str(payload.get("color_model_sha256") or "")
        if model_path.is_symlink() or sha256_file(model_path) != expected_model_sha:
            raise InspectionReleaseError("顏色方案基準檔完整性驗證失敗。")
        revisions = tuple(
            ColorRevisionBinding(
                scope_hash=str(item["scope_hash"]),
                threshold_key=str(item["threshold_key"]),
                revision_id=str(item["revision_id"]),
                display_version=str(item["display_version"]),
                config_path=str(self._resolve_external_path(str(item["config_path"]))),
                config_sha256=str(item["config_sha256"]),
                canonical_config_sha256=str(item["canonical_config_sha256"]),
            )
            for item in payload.get("revisions") or ()
        )
        for binding in revisions:
            if sha256_file(Path(binding.config_path)) != binding.config_sha256:
                raise InspectionReleaseError(
                    f"{binding.threshold_key} 校正修訂完整性驗證失敗。"
                )
        return ColorProfilePackage(
            package_id=package_id,
            display_version=str(payload.get("display_version") or package_id),
            product=str(payload.get("product") or ""),
            area=str(payload.get("area") or ""),
            model_type=str(payload.get("model_type") or ""),
            checker_type=str(payload.get("checker_type") or ""),
            colors=tuple(str(value) for value in payload.get("colors") or ()),
            color_model_path=model_path,
            color_model_sha256=expected_model_sha,
            manifest_path=manifest,
            manifest_sha256=sha256_file(manifest),
            revisions=revisions,
        )

    def _resolve_external_path(self, path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if path.exists() or self._station_paths is None:
            return path.resolve()
        return self._station_paths.relocate_legacy_path(path)

    @staticmethod
    def _read_model_config(path: Path) -> dict[str, Any]:
        if path.is_symlink() or not path.is_file():
            raise InspectionReleaseError("模型設定快照不存在。")
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise InspectionReleaseError("模型設定快照無法讀取。") from exc
        if not isinstance(payload, dict):
            raise InspectionReleaseError("模型設定快照格式錯誤。")
        return payload

    @staticmethod
    def _resolve_color_model(
        value: str,
        *,
        config_path: Path,
        project_root: Path,
    ) -> Path:
        raw = Path(value).expanduser()
        candidates = (
            (raw,)
            if raw.is_absolute()
            else (
                project_root / raw,
                config_path.parent / raw,
            )
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file() and not resolved.is_symlink():
                return resolved
        raise InspectionReleaseError(f"完整顏色基準檔不存在：{value}")

    @staticmethod
    def _read_color_model(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InspectionReleaseError("完整顏色基準檔無法讀取。") from exc
        if not isinstance(payload, dict):
            raise InspectionReleaseError("完整顏色基準檔格式錯誤。")
        return payload

    @staticmethod
    def _extract_colors(payload: dict[str, Any]) -> tuple[str, ...]:
        summary = payload.get("summary")
        if not isinstance(summary, dict) or not summary:
            raise InspectionReleaseError("顏色基準檔缺少可辨識的 summary 色別。")
        return tuple(str(key) for key in summary)

    @staticmethod
    def _revision_bindings(
        revisions: Sequence[ColorConfigurationRevision],
        *,
        checker_type: str,
        product: str,
        area: str,
        model_type: str,
    ) -> tuple[ColorRevisionBinding, ...]:
        bindings: list[ColorRevisionBinding] = []
        seen_thresholds: set[str] = set()
        for revision in revisions:
            scope = revision.scope
            if (
                scope.product,
                scope.area,
                scope.model_type,
            ) != (product, area, model_type):
                raise InspectionReleaseError(
                    "校正修訂與模型的產品、區域或推論類型不一致。"
                )
            if scope.checker_type != checker_type:
                raise InspectionReleaseError("校正修訂與顏色檢查器類型不一致。")
            key = scope.threshold_key.casefold()
            if key in seen_thresholds:
                raise InspectionReleaseError(
                    f"{scope.threshold_key} 同時選取了多個校正修訂。"
                )
            seen_thresholds.add(key)
            try:
                payload = json.loads(revision.config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise InspectionReleaseError(
                    f"{scope.threshold_key} 校正修訂無法讀取。"
                ) from exc
            if canonical_sha256(payload) != revision.new_config_sha256:
                raise InspectionReleaseError(
                    f"{scope.threshold_key} 校正修訂 checksum 不一致。"
                )
            bindings.append(
                ColorRevisionBinding(
                    scope_hash=scope.scope_hash,
                    threshold_key=scope.threshold_key,
                    revision_id=revision.revision_id,
                    display_version=revision.display_version,
                    config_path=str(revision.config_path.resolve()),
                    config_sha256=sha256_file(revision.config_path),
                    canonical_config_sha256=revision.new_config_sha256,
                )
            )
        return tuple(sorted(bindings, key=lambda item: item.threshold_key.casefold()))
