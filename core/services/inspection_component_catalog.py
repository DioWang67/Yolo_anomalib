"""Read-only catalog joining AI model and color-configuration versions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from core.services.color_baseline_recalibration import (
    ALGORITHM_VERSION,
    ColorBaselineCandidateStore,
)
from core.services.color_profile_store import ColorProfileStore
from core.services.inspection_release_store import (
    InspectionReleaseStore,
    sha256_file,
)
from core.services.model_version_registry import ModelVersionRegistry
from tools.color_calibration_service import ColorCalibrationError
from tools.color_configuration_revisions import (
    ColorConfigurationRevisionStore,
)


class InspectionComponentCatalogError(RuntimeError):
    """A component version catalog could not be loaded safely."""


@dataclass(frozen=True)
class InspectionComponentRecord:
    """One immutable component version exposed to engineering workflows."""

    component_id: str
    category: str
    component_type: str
    product: str
    area: str
    inference_type: str
    version: str
    status: str
    created_at: str
    integrity: str
    source_path: Path
    detail: str

    @property
    def can_compose(self) -> bool:
        return self.status != "REVOKED" and self.integrity in {"VERIFIED", "WARNING"}


class InspectionComponentCatalog:
    """Compose existing registries into one non-mutating version inventory."""

    def __init__(
        self,
        *,
        models_root: str | Path,
        color_revisions_root: str | Path,
        color_profiles_root: str | Path | None = None,
        color_baselines_root: str | Path | None = None,
        inspection_releases_root: str | Path | None = None,
    ) -> None:
        self.models_root = Path(models_root).expanduser().resolve()
        self.color_revisions_root = Path(color_revisions_root).expanduser().resolve()
        self.color_profiles_root = (
            Path(color_profiles_root).expanduser().resolve()
            if color_profiles_root is not None
            else self.models_root.parent / ".color_profiles"
        )
        self.color_baselines_root = (
            Path(color_baselines_root).expanduser().resolve()
            if color_baselines_root is not None
            else self.models_root.parent / ".color_baselines"
        )
        self.inspection_releases_root = (
            Path(inspection_releases_root).expanduser().resolve() if inspection_releases_root is not None else None
        )

    def list_components(self) -> tuple[InspectionComponentRecord, ...]:
        (
            deployed_models,
            deployed_colors,
            deployed_color_bases,
            deployed_profiles,
        ) = self._deployed_components()
        records = [
            *self._model_components(deployed_models),
            *self._base_color_components(deployed_color_bases),
            *self._candidate_color_base_components(deployed_color_bases),
            *self._profile_components(deployed_profiles),
            *self._color_revision_components(deployed_colors),
        ]
        return tuple(
            sorted(
                records,
                key=lambda item: (
                    item.product.casefold(),
                    item.area.casefold(),
                    item.category,
                    item.component_type.casefold(),
                    item.created_at,
                    item.version,
                ),
                reverse=True,
            )
        )

    def _model_components(
        self,
        deployed_models: set[tuple[str, str, str, str]],
    ) -> list[InspectionComponentRecord]:
        try:
            versions = ModelVersionRegistry(self.models_root).list_versions()
        except (OSError, RuntimeError, ValueError) as exc:
            raise InspectionComponentCatalogError(f"無法載入模型版本：{exc}") from exc
        records: list[InspectionComponentRecord] = []
        for version in versions:
            if not version.exists:
                integrity = "MISSING"
            elif not version.has_config_snapshot:
                integrity = "INCOMPLETE"
            elif version.warning:
                integrity = "WARNING"
            else:
                integrity = "VERIFIED"
            timestamp = version.trained_at or version.deployed_at or version.activated_at
            details = {
                "artifact": version.weight_path.name,
                "config_snapshot": (str(version.config_snapshot_path) if version.config_snapshot_path else ""),
                "evaluation": version.evaluation_metrics,
                "warning": version.warning,
            }
            records.append(
                InspectionComponentRecord(
                    component_id=(
                        f"model:{version.product}:{version.area}:"
                        f"{version.model_type}:{version.version}:"
                        f"{version.weight_path.name}"
                    ),
                    category="AI_MODEL",
                    component_type=version.model_type,
                    product=version.product,
                    area=version.area,
                    inference_type=version.model_type,
                    version=version.version,
                    status=(
                        "DEPLOYED"
                        if (
                            version.product,
                            version.area,
                            version.model_type,
                            version.version,
                        )
                        in deployed_models
                        else ("DEFAULT" if version.is_current else "HISTORY")
                    ),
                    created_at=timestamp.isoformat() if timestamp else "",
                    integrity=integrity,
                    source_path=version.weight_path,
                    detail=json.dumps(
                        details,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ),
                )
            )
        return records

    def _candidate_color_base_components(
        self,
        deployed_color_bases: set[str],
    ) -> list[InspectionComponentRecord]:
        if not self.color_baselines_root.is_dir():
            return []
        try:
            candidates = ColorBaselineCandidateStore(self.color_baselines_root).list_candidates()
        except (OSError, RuntimeError, ValueError) as exc:
            raise InspectionComponentCatalogError(f"無法載入顏色基準候選：{exc}") from exc
        records: list[InspectionComponentRecord] = []
        for candidate in candidates:
            report = json.loads(candidate.report_path.read_text(encoding="utf-8"))
            details = {
                "checker": "stats",
                "colors": list(candidate.colors),
                "color_count": len(candidate.colors),
                "sha256": candidate.color_model_sha256,
                "candidate_id": candidate.candidate_id,
                "candidate_status": candidate.status,
                "algorithm": candidate.algorithm,
                "report_path": str(candidate.report_path),
                "color_reports": report.get("color_reports") or [],
                "limitations": report.get("limitations") or [],
                "role": "BASELINE_CANDIDATE",
            }
            records.append(
                InspectionComponentRecord(
                    component_id=(f"color-base-candidate:{candidate.candidate_id}"),
                    category="COLOR_BASE",
                    component_type="stats_color",
                    product=candidate.product,
                    area=candidate.area,
                    inference_type=candidate.model_type,
                    version=candidate.display_version,
                    status=("DEPLOYED" if candidate.color_model_sha256 in deployed_color_bases else "HISTORY"),
                    created_at=candidate.created_at,
                    integrity=(
                        "INCOMPATIBLE"
                        if candidate.algorithm != ALGORITHM_VERSION
                        else ("VERIFIED" if candidate.status == "READY" else "WARNING")
                    ),
                    source_path=candidate.color_model_path,
                    detail=json.dumps(
                        details,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )
            )
        return records

    def _color_revision_components(
        self,
        deployed_colors: set[str],
    ) -> list[InspectionComponentRecord]:
        if not self.color_revisions_root.is_dir():
            return []
        store = ColorConfigurationRevisionStore(root=self.color_revisions_root)
        records: list[InspectionComponentRecord] = []
        for scope_root in sorted(store.root.iterdir()):
            if not scope_root.is_dir() or scope_root.name == "active" or scope_root.name.startswith("."):
                continue
            try:
                scope = store.scope_for_hash(scope_root.name)
                active = store.read_active_pointer(scope)
                active_id = str(active.get("revision_id") or "") if active else ""
                for revision in store.list_revisions(scope):
                    revoked = store.is_revoked(revision)
                    if revoked:
                        status = "REVOKED"
                    elif revision.revision_id in deployed_colors:
                        status = "DEPLOYED"
                    elif revision.revision_id == active_id:
                        status = "DEFAULT"
                    else:
                        status = "HISTORY"
                    details = {
                        "checker": scope.checker_type,
                        "threshold": scope.threshold_key,
                        "operator": revision.operator,
                        "reason": revision.approval_reason,
                        "metrics": dict(revision.metrics),
                        "revision_id": revision.revision_id,
                    }
                    records.append(
                        InspectionComponentRecord(
                            component_id=f"color:{revision.revision_id}",
                            category="COLOR_REVISION",
                            component_type=f"{scope.threshold_key.title()} 門檻",
                            product=scope.product,
                            area=scope.area,
                            inference_type=scope.model_type,
                            version=revision.display_version,
                            status=status,
                            created_at=revision.created_at.isoformat(),
                            integrity="VERIFIED",
                            source_path=revision.config_path,
                            detail=json.dumps(
                                details,
                                ensure_ascii=False,
                                sort_keys=True,
                                default=str,
                            ),
                        )
                    )
            except (OSError, ColorCalibrationError, ValueError) as exc:
                raise InspectionComponentCatalogError(f"無法載入顏色版本 {scope_root.name}：{exc}") from exc
        return records

    def _base_color_components(
        self,
        deployed_profiles: set[str],
    ) -> list[InspectionComponentRecord]:
        records: list[InspectionComponentRecord] = []
        for config_path in sorted(self.models_root.glob("*/*/*/config.yaml")):
            try:
                config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                if (
                    not isinstance(config, dict)
                    or not config.get("enable_color_check")
                    or not config.get("color_model_path")
                ):
                    continue
                relative = config_path.relative_to(self.models_root)
                product, area, inference_type = relative.parts[:3]
                color_model = self._resolve_color_model(
                    str(config["color_model_path"]),
                    config_path,
                )
                payload = json.loads(color_model.read_text(encoding="utf-8"))
                summary = payload.get("summary")
                if not isinstance(summary, dict) or not summary:
                    continue
                digest = sha256_file(color_model)
                checker_type = str(config.get("color_checker_type") or "stats").strip().lower()
                details = {
                    "checker": checker_type,
                    "colors": list(summary),
                    "color_count": len(summary),
                    "sha256": digest,
                    "role": "BASELINE",
                }
                records.append(
                    InspectionComponentRecord(
                        component_id=(f"color-base:{product}:{area}:{inference_type}:{digest[:12]}"),
                        category="COLOR_BASE",
                        component_type=f"{checker_type}_color",
                        product=product,
                        area=area,
                        inference_type=inference_type,
                        version=f"base-{digest[:8]}",
                        status=("DEPLOYED" if digest in deployed_profiles else "DEFAULT"),
                        created_at="",
                        integrity="VERIFIED",
                        source_path=color_model,
                        detail=json.dumps(
                            details,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    )
                )
            except (
                OSError,
                ValueError,
                TypeError,
                json.JSONDecodeError,
                yaml.YAMLError,
            ) as exc:
                raise InspectionComponentCatalogError(f"無法載入完整顏色基準 {config_path}：{exc}") from exc
        return records

    def _profile_components(
        self,
        deployed_profiles: set[str],
    ) -> list[InspectionComponentRecord]:
        if not self.color_profiles_root.is_dir():
            return []
        store = ColorProfileStore(self.color_profiles_root)
        records: list[InspectionComponentRecord] = []
        for manifest in sorted(self.color_profiles_root.glob("*/manifest.json")):
            try:
                profile = store.load(manifest)
            except (OSError, RuntimeError, ValueError) as exc:
                raise InspectionComponentCatalogError(f"無法載入顏色方案 {manifest.parent.name}：{exc}") from exc
            details = {
                "checker": profile.checker_type,
                "colors": list(profile.colors),
                "color_count": len(profile.colors),
                "overrides": {item.threshold_key: item.display_version for item in profile.revisions},
                "package_id": profile.package_id,
                "role": "PROFILE",
            }
            records.append(
                InspectionComponentRecord(
                    component_id=f"color-profile:{profile.package_id}",
                    category="COLOR_PROFILE",
                    component_type=f"{profile.checker_type}_color",
                    product=profile.product,
                    area=profile.area,
                    inference_type=profile.model_type,
                    version=profile.display_version,
                    status=("DEPLOYED" if profile.package_id in deployed_profiles else "HISTORY"),
                    created_at="",
                    integrity="VERIFIED",
                    source_path=profile.manifest_path,
                    detail=json.dumps(
                        details,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )
            )
        return records

    def _resolve_color_model(
        self,
        value: str,
        config_path: Path,
    ) -> Path:
        raw = Path(value).expanduser()
        candidates = (
            (raw,)
            if raw.is_absolute()
            else (
                self.models_root.parent / raw,
                config_path.parent / raw,
            )
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file() and not resolved.is_symlink():
                return resolved
        raise InspectionComponentCatalogError(f"完整顏色基準不存在：{value}")

    def _deployed_components(
        self,
    ) -> tuple[
        set[tuple[str, str, str, str]],
        set[str],
        set[str],
        set[str],
    ]:
        models: set[tuple[str, str, str, str]] = set()
        colors: set[str] = set()
        color_bases: set[str] = set()
        profiles: set[str] = set()
        root = self.inspection_releases_root
        if root is None or not root.is_dir():
            return models, colors, color_bases, profiles
        try:
            store = InspectionReleaseStore(root)
            for release in store.list_releases():
                pointer = store.active_pointer(release.scope)
                if not pointer or str(pointer.get("release_id") or "") != release.release_id:
                    continue
                for component in release.components:
                    if component.kind in {"yolo", "anomalib"}:
                        models.add(
                            (
                                release.scope.product,
                                release.scope.area,
                                component.kind,
                                component.version,
                            )
                        )
                    if component.role == "color_check":
                        colors.update(dict(component.revision_overrides).values())
                        if component.artifact_sha256:
                            color_bases.add(component.artifact_sha256)
                        package_id = str(dict(component.metadata).get("package_id") or "")
                        if package_id:
                            profiles.add(package_id)
        except (OSError, RuntimeError, ValueError) as exc:
            raise InspectionComponentCatalogError(f"無法解析目前檢測組合：{exc}") from exc
        return models, colors, color_bases, profiles
