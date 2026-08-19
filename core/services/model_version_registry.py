"""Discover and safely activate deployed inference model versions."""

from __future__ import annotations

import getpass
import hashlib
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from core.version_utils import parse_model_version, version_to_string

SUPPORTED_WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".engine",
    ".onnx",
    ".pt",
    ".pth",
    ".torchscript",
}

# These values belong to the physical station and must not be rolled back with
# a model trained on another day.
STATION_LOCAL_FIELDS = {
    "buffer_limit",
    "calibration",
    "exposure_time",
    "flush_interval",
    "gain",
    "jpeg_quality",
    "light_brightness",
    "max_crops_per_frame",
    "output_dir",
    "png_compression",
    "save_annotated",
    "save_crops",
    "save_fail_only",
    "save_original",
    "save_processed",
}


class ModelVersionRegistryError(RuntimeError):
    """Raised when model history cannot be read or safely changed."""


class ModelVersionBusyError(ModelVersionRegistryError):
    """Raised when training or another operator is changing the same model."""


@dataclass(frozen=True)
class ModelVersionRecord:
    """One deployable model artifact and its traceability metadata.

    Args:
        product: Product name from the models directory.
        area: Inspection area/station name.
        model_type: Inference backend, such as ``yolo`` or ``anomalib``.
        version: Semantic version or ``legacy`` when it cannot be determined.
        weight_path: Absolute artifact path.
        is_current: Whether ``config.yaml`` currently points at this artifact.
        trained_at: Training completion time when known; otherwise file time.
        deployed_at: Original deployment time when recorded.
        activated_at: Most recent activation time when recorded.
        training_time_inferred: True when ``trained_at`` came from file mtime.
        evaluation_metrics: Offline evaluation metrics stored during deployment.
        dataset_hash: Dataset identity stored during training.
        training_config_hash: Training configuration identity.
        weight_sha256: Expected artifact digest, when available.
        config_snapshot_path: Version-matched inference config, when available.
        file_size: Artifact size in bytes, or zero for a missing file.
        warning: Human-readable integrity/metadata warning.
    """

    product: str
    area: str
    model_type: str
    version: str
    weight_path: Path
    is_current: bool
    trained_at: datetime | None
    deployed_at: datetime | None
    activated_at: datetime | None
    training_time_inferred: bool
    evaluation_metrics: dict[str, Any] = field(default_factory=dict)
    dataset_hash: str = ""
    training_config_hash: str = ""
    weight_sha256: str = ""
    config_snapshot_path: Path | None = None
    file_size: int = 0
    warning: str = ""

    @property
    def exists(self) -> bool:
        """Return whether the referenced artifact is a non-empty file."""
        return self.weight_path.is_file() and self.file_size > 0

    @property
    def has_config_snapshot(self) -> bool:
        """Return whether a version-matched inference config is available."""
        return bool(
            self.config_snapshot_path
            and self.config_snapshot_path.is_file()
        )

    @property
    def identity(self) -> tuple[str, str, str, str]:
        """Return a stable identity used to refresh a selected table row."""
        return (self.product, self.area, self.model_type, self.weight_path.name)


class ModelVersionRegistry:
    """Read model version history and atomically change the active version.

    Args:
        models_root: Inference ``models`` directory.
        lock_timeout: Seconds to wait for the training deploy lock.

    Raises:
        ModelVersionRegistryError: If ``models_root`` is invalid.
    """

    def __init__(self, models_root: str | Path, *, lock_timeout: float = 3.0) -> None:
        self.models_root = Path(models_root).expanduser().resolve()
        self.project_root = self.models_root.parent
        self.lock_timeout = max(float(lock_timeout), 0.0)
        if not self.models_root.is_dir():
            raise ModelVersionRegistryError(
                f"找不到模型目錄：{self.models_root}"
            )

    def list_versions(
        self,
        *,
        product: str | None = None,
        area: str | None = None,
        model_type: str | None = None,
    ) -> list[ModelVersionRecord]:
        """Return all discoverable versions, including legacy artifacts.

        Args:
            product: Optional exact product filter.
            area: Optional exact station filter.
            model_type: Optional exact inference backend filter.

        Returns:
            Records sorted by product/station/type, current state, and time.
        """
        records: list[ModelVersionRecord] = []
        for target_dir in self._iter_target_dirs():
            target_product = target_dir.parent.parent.name
            target_area = target_dir.parent.name
            target_type = target_dir.name
            if product and target_product != product:
                continue
            if area and target_area != area:
                continue
            if model_type and target_type != model_type:
                continue
            records.extend(
                self._read_target_versions(
                    target_product,
                    target_area,
                    target_type,
                    target_dir,
                )
            )

        return sorted(
            records,
            key=lambda item: (
                item.product.casefold(),
                item.area.casefold(),
                item.model_type.casefold(),
                not item.is_current,
                -item.trained_at.timestamp() if item.trained_at else float("inf"),
                item.weight_path.name.casefold(),
            ),
        )

    def activate(
        self,
        record: ModelVersionRecord,
        *,
        allow_incomplete: bool = False,
        operator: str | None = None,
    ) -> ModelVersionRecord:
        """Atomically point a target config at a selected historical version.

        Station-local camera/output fields are preserved. A version-matched
        config snapshot is required unless ``allow_incomplete`` is explicitly
        set for a legacy artifact.

        Args:
            record: Version returned by :meth:`list_versions`.
            allow_incomplete: Permit a legacy weight-only switch.
            operator: Optional audit name; defaults to the Windows user.

        Returns:
            Refreshed record marked as current.

        Raises:
            ModelVersionRegistryError: If validation or publishing fails.
            ModelVersionBusyError: If training currently holds the deploy lock.
        """
        target_dir = self._target_dir(record.product, record.area, record.model_type)
        lock_dir = self._acquire_target_lock(target_dir)
        try:
            refreshed = self._find_record(record.identity)
            if refreshed.is_current:
                return refreshed
            self._validate_activation_target(refreshed, target_dir)
            if not refreshed.has_config_snapshot and not allow_incomplete:
                raise ModelVersionRegistryError(
                    "此舊版本沒有配套的模型設定快照，無法直接安全切換。"
                )
            if refreshed.weight_sha256:
                actual_hash = _sha256_file(refreshed.weight_path)
                if actual_hash.lower() != refreshed.weight_sha256.lower():
                    raise ModelVersionRegistryError(
                        "模型檔案雜湊不符，可能已損壞或被替換，已取消切換。"
                    )

            config_path = target_dir / "config.yaml"
            current_config = _load_yaml_mapping(config_path, required=True)
            next_config = (
                _load_yaml_mapping(refreshed.config_snapshot_path, required=True)
                if refreshed.has_config_snapshot
                else dict(current_config)
            )
            for field_name in STATION_LOCAL_FIELDS:
                if field_name in current_config:
                    next_config[field_name] = current_config[field_name]
            self._set_config_weight(
                next_config,
                refreshed,
                self._portable_weight_path(refreshed.weight_path),
            )
            if refreshed.version != "legacy":
                next_config["model_version"] = refreshed.version
            else:
                next_config.pop("model_version", None)

            current_file = self._current_weight_path(target_dir, current_config)
            activated_at = datetime.now().astimezone().isoformat(timespec="seconds")
            operator_name = (operator or getpass.getuser() or "unknown").strip()
            backup_path = self._backup_current_config(
                target_dir,
                current_config,
                current_file.name if current_file else "unknown",
            )
            event = {
                "activated_at": activated_at,
                "operator": operator_name,
                "product": refreshed.product,
                "area": refreshed.area,
                "model_type": refreshed.model_type,
                "from_file": current_file.name if current_file else "",
                "to_file": refreshed.weight_path.name,
                "version": refreshed.version,
                "config_backup": str(backup_path.relative_to(target_dir).as_posix()),
                "legacy_weight_only": not refreshed.has_config_snapshot,
            }
            history_path = target_dir / "activation_history.yaml"
            history_existed = history_path.is_file()
            history = _load_yaml_mapping(history_path)
            events = history.get("events") if isinstance(history.get("events"), list) else []
            next_history = {"schema_version": 1, "events": [*events, event]}

            selected_manifest = self._manifest_for_record(refreshed)
            selected_manifest.update(
                {
                    "schema_version": 1,
                    "product": refreshed.product,
                    "area": refreshed.area,
                    "model_type": refreshed.model_type,
                    "deployed_version": refreshed.version,
                    "deployed_file": refreshed.weight_path.name,
                    "activated_at": activated_at,
                    "activated_by": operator_name,
                    "activation_source": "operator_version_manager",
                    "previous_file": current_file.name if current_file else "",
                }
            )
            manifest_path = target_dir / "deployment_manifest.yaml"
            manifest_existed = manifest_path.is_file()
            previous_manifest = _load_yaml_mapping(manifest_path)
            try:
                _write_yaml_atomic(history_path, next_history)
                _write_yaml_atomic(manifest_path, selected_manifest)
                # Config is the public pointer and is deliberately published last.
                _write_yaml_atomic(config_path, next_config)
            except OSError as exc:
                if history_existed:
                    _write_yaml_atomic(history_path, history)
                else:
                    history_path.unlink(missing_ok=True)
                if manifest_existed:
                    _write_yaml_atomic(manifest_path, previous_manifest)
                else:
                    manifest_path.unlink(missing_ok=True)
                raise ModelVersionRegistryError(f"模型版本切換失敗：{exc}") from exc

            return self._find_record(refreshed.identity)
        finally:
            self._release_target_lock(lock_dir)

    def previous_version(
        self, product: str, area: str, model_type: str
    ) -> ModelVersionRecord | None:
        """Return the most appropriate version for a one-click rollback.

        Args:
            product: Product name.
            area: Inspection station.
            model_type: Inference backend.

        Returns:
            Previously active record, or the next older artifact when no audit
            history exists. Returns ``None`` when rollback is unavailable.
        """
        records = self.list_versions(
            product=product, area=area, model_type=model_type
        )
        current = next((item for item in records if item.is_current), None)
        if current is None:
            return None
        history = _load_yaml_mapping(
            self._target_dir(product, area, model_type) / "activation_history.yaml"
        )
        events = history.get("events") if isinstance(history.get("events"), list) else []
        for event in reversed(events):
            if not isinstance(event, dict):
                continue
            if str(event.get("to_file") or "") != current.weight_path.name:
                continue
            previous_name = str(event.get("from_file") or "")
            previous = next(
                (item for item in records if item.weight_path.name == previous_name),
                None,
            )
            if previous and previous.exists:
                return previous

        historical = [item for item in records if not item.is_current and item.exists]
        if not historical:
            return None
        current_time = current.trained_at
        older = [
            item
            for item in historical
            if current_time is None
            or item.trained_at is None
            or item.trained_at <= current_time
        ]
        return (older or historical)[0]

    def _iter_target_dirs(self) -> Iterator[Path]:
        for product_dir in sorted(self.models_root.iterdir()):
            if not product_dir.is_dir() or product_dir.name.startswith("."):
                continue
            for area_dir in sorted(product_dir.iterdir()):
                if not area_dir.is_dir() or area_dir.name.startswith("."):
                    continue
                for target_dir in sorted(area_dir.iterdir()):
                    if (
                        target_dir.is_dir()
                        and not target_dir.name.startswith(".")
                        and (target_dir / "config.yaml").is_file()
                    ):
                        yield target_dir

    def _read_target_versions(
        self,
        product: str,
        area: str,
        model_type: str,
        target_dir: Path,
    ) -> list[ModelVersionRecord]:
        config = _load_yaml_mapping(target_dir / "config.yaml")
        current_path = self._current_weight_path(target_dir, config)
        current_manifest = _load_yaml_mapping(target_dir / "deployment_manifest.yaml")
        activation_times = self._activation_times(target_dir)
        weights_dir = (target_dir / "weights").resolve()
        candidates: dict[Path, dict[str, Any]] = {}
        if weights_dir.is_dir():
            for weight_path in weights_dir.iterdir():
                if not _is_weight_candidate(weight_path, current_path):
                    continue
                manifest_path = weight_path.with_name(
                    f"{weight_path.name}.manifest.yaml"
                )
                candidates[weight_path.resolve()] = _load_yaml_mapping(manifest_path)

        if current_path is not None:
            resolved_current = current_path.resolve()
            candidates.setdefault(resolved_current, {})
        current_file = str(current_manifest.get("deployed_file") or "")
        if current_file:
            manifest_weight = (weights_dir / current_file).resolve()
            candidates.setdefault(manifest_weight, {})

        records: list[ModelVersionRecord] = []
        for weight_path, sidecar in candidates.items():
            metadata = dict(sidecar)
            if not metadata and current_file == weight_path.name:
                metadata = dict(current_manifest)
            parsed_version = parse_model_version(weight_path.name)
            version = (
                version_to_string(parsed_version)
                if parsed_version
                else str(metadata.get("deployed_version") or "").strip()
            )
            is_current = bool(
                current_path is not None and weight_path == current_path.resolve()
            )
            if not version and is_current:
                version = str(config.get("model_version") or "").strip()
            version = version or "legacy"

            file_time = (
                datetime.fromtimestamp(weight_path.stat().st_mtime).astimezone()
                if weight_path.is_file()
                else None
            )
            trained_at = _parse_datetime(metadata.get("trained_at"))
            inferred = trained_at is None and file_time is not None
            trained_at = trained_at or file_time
            deployed_at = _parse_datetime(
                metadata.get("deployed_at") or metadata.get("deployed_date")
            )
            snapshot_path = self._config_snapshot_path(
                target_dir, weight_path, metadata
            )
            warning_parts: list[str] = []
            file_size = weight_path.stat().st_size if weight_path.is_file() else 0
            if not weight_path.is_file() or file_size <= 0:
                warning_parts.append("模型檔案不存在或為空")
            if snapshot_path is None or not snapshot_path.is_file():
                warning_parts.append("缺少版本設定快照")
            if not metadata:
                warning_parts.append("缺少訓練與部署紀錄")
            records.append(
                ModelVersionRecord(
                    product=product,
                    area=area,
                    model_type=model_type,
                    version=version,
                    weight_path=weight_path,
                    is_current=is_current,
                    trained_at=trained_at,
                    deployed_at=deployed_at,
                    activated_at=activation_times.get(weight_path.name),
                    training_time_inferred=inferred,
                    evaluation_metrics=(
                        dict(metadata.get("evaluation_metrics"))
                        if isinstance(metadata.get("evaluation_metrics"), dict)
                        else {}
                    ),
                    dataset_hash=str(metadata.get("dataset_hash") or ""),
                    training_config_hash=str(
                        metadata.get("training_config_hash") or ""
                    ),
                    weight_sha256=str(metadata.get("weight_sha256") or ""),
                    config_snapshot_path=snapshot_path,
                    file_size=file_size,
                    warning="；".join(warning_parts),
                )
            )
        return records

    def _activation_times(self, target_dir: Path) -> dict[str, datetime]:
        history = _load_yaml_mapping(target_dir / "activation_history.yaml")
        events = history.get("events") if isinstance(history.get("events"), list) else []
        result: dict[str, datetime] = {}
        for event in events:
            if not isinstance(event, dict):
                continue
            filename = str(event.get("to_file") or "")
            activated_at = _parse_datetime(event.get("activated_at"))
            if filename and activated_at:
                result[filename] = activated_at
        return result

    def _config_snapshot_path(
        self,
        target_dir: Path,
        weight_path: Path,
        metadata: dict[str, Any],
    ) -> Path | None:
        configured = str(metadata.get("config_snapshot") or "").strip()
        if configured:
            candidate = Path(configured)
            candidate = candidate if candidate.is_absolute() else target_dir / candidate
        else:
            candidate = target_dir / "versions" / f"{weight_path.name}.config.yaml"
        resolved = candidate.resolve()
        if not resolved.is_relative_to(target_dir.resolve()):
            return None
        return resolved if resolved.is_file() else None

    def _current_weight_path(
        self, target_dir: Path, config: dict[str, Any]
    ) -> Path | None:
        configured = str(config.get("weights") or "").strip()
        if not configured and target_dir.name.lower() == "anomalib":
            product = target_dir.parent.parent.name
            area = target_dir.parent.name
            anomalib_config = config.get("anomalib_config")
            models = (
                anomalib_config.get("models")
                if isinstance(anomalib_config, dict)
                else None
            )
            product_models = models.get(product) if isinstance(models, dict) else None
            area_model = (
                product_models.get(area)
                if isinstance(product_models, dict)
                else None
            )
            if isinstance(area_model, dict):
                configured = str(
                    area_model.get("ckpt_path")
                    or area_model.get("weights")
                    or area_model.get("model_path")
                    or ""
                ).strip()
        if not configured:
            return None
        path = Path(configured).expanduser()
        if path.is_absolute():
            return path.resolve()
        project_candidate = (self.project_root / path).resolve()
        target_candidate = (target_dir / path).resolve()
        if project_candidate.exists() or str(path).replace("\\", "/").startswith("models/"):
            return project_candidate
        return target_candidate

    @staticmethod
    def _set_config_weight(
        config: dict[str, Any], record: ModelVersionRecord, portable_path: str
    ) -> None:
        """Set the backend-specific runtime artifact path in a config mapping."""
        if record.model_type.lower() != "anomalib":
            config["weights"] = portable_path
            return
        anomalib_config = config.get("anomalib_config")
        if not isinstance(anomalib_config, dict):
            anomalib_config = {}
            config["anomalib_config"] = anomalib_config
        models = anomalib_config.get("models")
        if not isinstance(models, dict):
            models = {}
            anomalib_config["models"] = models
        product_models = models.get(record.product)
        if not isinstance(product_models, dict):
            product_models = {}
            models[record.product] = product_models
        area_model = product_models.get(record.area)
        if not isinstance(area_model, dict):
            area_model = {}
            product_models[record.area] = area_model
        area_model["ckpt_path"] = portable_path
        config.pop("weights", None)

    def _target_dir(self, product: str, area: str, model_type: str) -> Path:
        target_dir = (self.models_root / product / area / model_type).resolve()
        if not target_dir.is_relative_to(self.models_root):
            raise ModelVersionRegistryError("模型目標路徑超出 models 目錄。")
        if not (target_dir / "config.yaml").is_file():
            raise ModelVersionRegistryError(f"找不到模型設定：{target_dir}")
        return target_dir

    def _find_record(
        self, identity: tuple[str, str, str, str]
    ) -> ModelVersionRecord:
        product, area, model_type, filename = identity
        for item in self.list_versions(
            product=product, area=area, model_type=model_type
        ):
            if item.weight_path.name == filename:
                return item
        raise ModelVersionRegistryError(f"找不到模型版本：{filename}")

    def _validate_activation_target(
        self, record: ModelVersionRecord, target_dir: Path
    ) -> None:
        weights_dir = (target_dir / "weights").resolve()
        if not record.weight_path.resolve().is_relative_to(weights_dir):
            raise ModelVersionRegistryError("模型檔案不在指定工位的 weights 目錄內。")
        if record.weight_path.suffix.lower() not in SUPPORTED_WEIGHT_SUFFIXES:
            raise ModelVersionRegistryError("不支援此模型檔案格式。")
        if not record.exists:
            raise ModelVersionRegistryError("模型檔案不存在或為空，無法切換。")

    def _portable_weight_path(self, weight_path: Path) -> str:
        try:
            return weight_path.resolve().relative_to(self.project_root).as_posix()
        except ValueError as exc:
            raise ModelVersionRegistryError("模型檔案不在推理專案內。") from exc

    def _backup_current_config(
        self,
        target_dir: Path,
        config: dict[str, Any],
        current_filename: str,
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_name = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in current_filename
        )
        backup_path = (
            target_dir / "versions" / "config_backups" / f"{timestamp}_{safe_name}.yaml"
        )
        _write_yaml_atomic(backup_path, config)
        return backup_path

    def _manifest_for_record(self, record: ModelVersionRecord) -> dict[str, Any]:
        sidecar_path = record.weight_path.with_name(
            f"{record.weight_path.name}.manifest.yaml"
        )
        metadata = _load_yaml_mapping(sidecar_path)
        if record.config_snapshot_path:
            metadata["config_snapshot"] = str(
                record.config_snapshot_path.relative_to(
                    self._target_dir(record.product, record.area, record.model_type)
                ).as_posix()
            )
        if record.weight_sha256:
            metadata["weight_sha256"] = record.weight_sha256
        return metadata

    def _acquire_target_lock(self, target_dir: Path) -> Path:
        lock_dir = target_dir / ".deploy.lock"
        deadline = time.monotonic() + self.lock_timeout
        while True:
            try:
                lock_dir.mkdir()
                return lock_dir
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise ModelVersionBusyError(
                        "模型正在訓練部署或被其他人切換，請稍後再試。"
                    ) from None
                time.sleep(0.1)

    @staticmethod
    def _release_target_lock(lock_dir: Path) -> None:
        try:
            lock_dir.rmdir()
        except FileNotFoundError:
            pass


def _is_weight_candidate(path: Path, current_path: Path | None) -> bool:
    if not path.is_file() or path.name.startswith(".") or ".bak." in path.name:
        return False
    if path.suffix.lower() not in SUPPORTED_WEIGHT_SUFFIXES:
        return False
    is_current = bool(
        current_path is not None and path.resolve() == current_path.resolve()
    )
    normalized_name = path.name.lower()
    if normalized_name.endswith(".training.pt") and not is_current:
        return False
    if path.stem.lower() in {"best", "last"} and not is_current:
        return False
    return True


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    for parser in (
        lambda raw: datetime.fromisoformat(raw.replace("Z", "+00:00")),
        lambda raw: datetime.strptime(raw, "%Y%m%d"),
        lambda raw: datetime.strptime(raw, "%Y-%m-%d %H:%M"),
    ):
        try:
            parsed = parser(text)
            return parsed.astimezone() if parsed.tzinfo else parsed.astimezone()
        except ValueError:
            continue
    return None


def _load_yaml_mapping(path: Path | None, *, required: bool = False) -> dict[str, Any]:
    if path is None or not path.is_file():
        if required:
            raise ModelVersionRegistryError(f"找不到必要設定檔：{path}")
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        if required:
            raise ModelVersionRegistryError(f"無法讀取設定檔 {path}：{exc}") from exc
        return {}
    if not isinstance(payload, dict):
        if required:
            raise ModelVersionRegistryError(f"設定檔不是 YAML mapping：{path}")
        return {}
    return dict(payload)


def _write_yaml_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
