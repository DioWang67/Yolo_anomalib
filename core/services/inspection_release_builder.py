"""Build deterministic inspection releases from acceptance matrix evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from core.services.color_profile_store import (
    ColorProfilePackage,
    ColorProfileStore,
)
from core.services.inspection_release_models import (
    ComponentBinding,
    InspectionRelease,
    InspectionReleaseError,
    InspectionScope,
    ReleaseStatus,
    ValidationEvidence,
    template_for_inference_type,
)
from core.services.inspection_release_store import sha256_file
from core.services.model_version_registry import ModelVersionRecord
from tools.color_calibration_service import canonical_sha256
from tools.color_configuration_revisions import (
    ColorConfigurationRevision,
    ColorConfigurationRevisionStore,
)


def build_release_from_matrix(
    report_path: str | Path,
    *,
    combination_id: str,
    display_version: str,
    operator: str,
    reason: str,
    status: ReleaseStatus = ReleaseStatus.TESTED,
    release_id: str | None = None,
    created_at: str | None = None,
) -> InspectionRelease:
    """Bind one matrix combination to exact model and color artifacts."""
    report = Path(report_path).expanduser().resolve()
    if report.is_symlink() or not report.is_file():
        raise InspectionReleaseError(f"Acceptance report is unavailable: {report}")
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InspectionReleaseError("Acceptance report is invalid.") from exc
    if payload.get("schema_version") != 1:
        raise InspectionReleaseError("Unsupported acceptance matrix schema.")
    combination = _find(payload.get("combinations"), "combination_id", combination_id)
    model = _find(
        payload.get("model_variants"),
        "variant_id",
        str(combination.get("model_variant_id") or ""),
    )
    color = _find(
        payload.get("color_variants"),
        "variant_id",
        str(combination.get("color_variant_id") or ""),
    )
    if bool(color.get("include_active_revisions")):
        raise InspectionReleaseError("A symbolic active color pointer cannot be published; select an exact revision.")
    product = str(payload.get("product") or "")
    area = str(payload.get("area") or "")
    inference_type = str(payload.get("inference_type") or "")
    template = template_for_inference_type(inference_type)
    scope = InspectionScope(product, area, template.template_id)
    components = _model_components(model, template.inference_type)
    revision_overrides = color.get("revision_overrides") or {}
    if not isinstance(revision_overrides, Mapping):
        raise InspectionReleaseError("Color revision overrides are invalid.")
    project_root = Path(str(model.get("models_root") or "")).expanduser().resolve().parent
    profile = _profile_from_matrix(
        model,
        product=product,
        area=area,
        inference_type=inference_type,
        project_root=project_root,
        revision_overrides=revision_overrides,
        color_model_path=str(color.get("color_model_path") or ""),
        color_model_sha256=str(color.get("color_model_sha256") or ""),
    )
    if profile is not None:
        components.append(_profile_component(profile))
    elif revision_overrides:
        components.append(
            _color_component(
                color,
                revision_overrides,
                project_root / ".color_revisions",
            )
        )
    validation = ValidationEvidence(
        report_path=str(report),
        report_sha256=sha256_file(report),
        run_id=str(payload.get("run_id") or ""),
        sample_count=int(payload.get("sample_count") or 0),
        combination_id=combination_id,
        metrics=tuple(sorted((combination.get("metrics") or {}).items())),
        color_metrics=tuple(sorted((combination.get("color_metrics") or {}).items())),
    )
    return InspectionRelease(
        release_id=release_id or str(uuid4()),
        display_version=display_version,
        scope=scope,
        components=tuple(components),
        status=status,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        operator=operator,
        reason=reason,
        validation=validation,
    )


def build_validated_release_from_matrix(
    draft: InspectionRelease,
    report_path: str | Path,
    *,
    combination_id: str,
    status: ReleaseStatus = ReleaseStatus.TESTED,
) -> InspectionRelease:
    """Project exact matrix evidence onto the same immutable draft identity."""
    if draft.status is not ReleaseStatus.DRAFT:
        raise InspectionReleaseError("只有 DRAFT 組合可以套用快速驗收結果。")
    if status not in {ReleaseStatus.TESTED, ReleaseStatus.BLOCKED}:
        raise InspectionReleaseError("驗收結果狀態必須是 TESTED 或 BLOCKED。")
    report = Path(report_path).expanduser().resolve()
    if report.is_symlink() or not report.is_file():
        raise InspectionReleaseError(f"驗收報告不存在或不安全：{report}")
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InspectionReleaseError("驗收報告無法讀取。") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise InspectionReleaseError("不支援的驗收矩陣格式。")
    report_scope = InspectionScope(
        str(payload.get("product") or ""),
        str(payload.get("area") or ""),
        template_for_inference_type(str(payload.get("inference_type") or "")).template_id,
    )
    if report_scope != draft.scope:
        raise InspectionReleaseError("驗收報告的產品、區域或檢測類型不符。")
    combination = _find(payload.get("combinations"), "combination_id", combination_id)
    model = _find(
        payload.get("model_variants"),
        "variant_id",
        str(combination.get("model_variant_id") or ""),
    )
    color = _find(
        payload.get("color_variants"),
        "variant_id",
        str(combination.get("color_variant_id") or ""),
    )
    _verify_matrix_model_matches_draft(draft, model)
    _verify_matrix_color_matches_draft(draft, color)
    sample_count = int(payload.get("sample_count") or 0)
    if sample_count <= 0:
        raise InspectionReleaseError("驗收報告沒有已執行的人工確認樣本。")
    metrics = combination.get("metrics") or {}
    color_metrics = combination.get("color_metrics") or {}
    if not isinstance(metrics, Mapping) or not isinstance(color_metrics, Mapping):
        raise InspectionReleaseError("驗收報告的指標格式無效。")
    validation = ValidationEvidence(
        report_path=str(report),
        report_sha256=sha256_file(report),
        run_id=str(payload.get("run_id") or ""),
        sample_count=sample_count,
        combination_id=combination_id,
        metrics=tuple(sorted((str(key), value) for key, value in metrics.items())),
        color_metrics=tuple(
            sorted((str(key), value) for key, value in color_metrics.items())
        ),
    )
    return replace(draft, status=status, validation=validation)


def build_draft_release(
    model: ModelVersionRecord,
    *,
    display_version: str,
    operator: str,
    reason: str,
    color_revision: ColorConfigurationRevision | None = None,
    color_profile: ColorProfilePackage | None = None,
    release_id: str | None = None,
    created_at: str | None = None,
) -> InspectionRelease:
    """Compose an unvalidated but reproducible model/color combination."""
    if not model.exists:
        raise InspectionReleaseError(f"Selected model artifact is unavailable: {model.weight_path}")
    if not model.has_config_snapshot or model.config_snapshot_path is None:
        raise InspectionReleaseError("Selected model has no version-matched config snapshot.")
    artifact_sha = sha256_file(model.weight_path)
    if model.weight_sha256 and artifact_sha != model.weight_sha256.lower():
        raise InspectionReleaseError("Selected model artifact checksum mismatch.")
    template = template_for_inference_type(model.model_type)
    scope = InspectionScope(model.product, model.area, template.template_id)
    role = "primary_detector" if model.model_type.lower() == "yolo" else "anomaly_detector"
    components = [
        ComponentBinding(
            component_id=f"{model.model_type.lower()}-model",
            kind=model.model_type.lower(),
            role=role,
            version=model.version,
            artifact_path=str(model.weight_path.resolve()),
            artifact_sha256=artifact_sha,
            config_path=str(model.config_snapshot_path.resolve()),
            config_sha256=sha256_file(model.config_snapshot_path),
            metadata=tuple(
                sorted(
                    {
                        "source": "engineering_composer",
                        "is_current_model_pointer": model.is_current,
                    }.items()
                )
            ),
        )
    ]
    if color_revision is not None and color_profile is not None:
        raise InspectionReleaseError("不能同時使用舊式單色修訂與完整顏色方案。")
    if color_profile is not None:
        components.append(_profile_component(color_profile))
    elif color_revision is not None:
        revision_scope = color_revision.scope
        if (
            revision_scope.product,
            revision_scope.area,
            revision_scope.model_type,
        ) != (model.product, model.area, model.model_type):
            raise InspectionReleaseError("Color revision does not match the selected model scope.")
        try:
            color_config = json.loads(color_revision.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InspectionReleaseError("Selected color revision config is unreadable.") from exc
        if canonical_sha256(color_config) != color_revision.new_config_sha256.lower():
            raise InspectionReleaseError("Selected color revision checksum mismatch.")
        config_sha = sha256_file(color_revision.config_path)
        components.append(
            ComponentBinding(
                component_id="stats-color",
                kind="stats_color",
                role="color_check",
                version=color_revision.display_version,
                config_path=str(color_revision.config_path.resolve()),
                config_sha256=config_sha,
                revision_overrides=(
                    (
                        revision_scope.scope_hash,
                        color_revision.revision_id,
                    ),
                ),
                metadata=tuple(
                    sorted(
                        {
                            "threshold_key": revision_scope.threshold_key,
                            "source": "engineering_composer",
                        }.items()
                    )
                ),
            )
        )
    evidence = ValidationEvidence(
        report_path="",
        report_sha256="",
        run_id="UNVALIDATED",
        sample_count=0,
        combination_id="",
        metrics=(
            ("confirmed", 0),
            ("errors", None),
            ("fn", None),
            ("fp", None),
        ),
        color_metrics=(
            ("errors", None),
            ("escape_rate", None),
            ("overkill_rate", None),
        ),
    )
    return InspectionRelease(
        release_id=release_id or str(uuid4()),
        display_version=display_version,
        scope=scope,
        components=tuple(components),
        status=ReleaseStatus.DRAFT,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        operator=operator,
        reason=reason,
        validation=evidence,
    )


def _find(items: Any, key: str, value: str) -> Mapping[str, Any]:
    if not isinstance(items, list):
        raise InspectionReleaseError("Acceptance report list is invalid.")
    for item in items:
        if isinstance(item, Mapping) and str(item.get(key) or "") == value:
            return item
    raise InspectionReleaseError(f"Acceptance report does not contain {value!r}.")


def _verify_matrix_model_matches_draft(
    draft: InspectionRelease,
    model: Mapping[str, Any],
) -> None:
    if draft.scope.inference_type == "fusion":
        raise InspectionReleaseError("Fusion 組合不能使用單模型快速驗收報告。")
    role = (
        "primary_detector"
        if draft.scope.inference_type == "yolo"
        else "anomaly_detector"
    )
    component = draft.component_for_role(role)
    identity = model.get("identity") or {}
    if component is None or not isinstance(identity, Mapping):
        raise InspectionReleaseError("驗收報告缺少選取組合的模型識別。")
    expected = (
        component.version,
        component.artifact_sha256.lower(),
        component.config_sha256.lower(),
        str(Path(component.artifact_path).expanduser().resolve()),
        str(Path(component.config_path).expanduser().resolve()),
    )
    actual = (
        str(identity.get("version") or ""),
        str(identity.get("sha256") or "").lower(),
        str(identity.get("runtime_config_sha256") or "").lower(),
        str(Path(str(model.get("weight_path") or "")).expanduser().resolve()),
        str(Path(str(model.get("config_path") or "")).expanduser().resolve()),
    )
    if actual != expected:
        raise InspectionReleaseError(
            "驗收報告使用的模型、權重或 config 不是選取的組合版本。"
        )


def _verify_matrix_color_matches_draft(
    draft: InspectionRelease,
    color: Mapping[str, Any],
) -> None:
    if bool(color.get("include_active_revisions")):
        raise InspectionReleaseError("快速驗收不得使用執行當下的 Active 顏色指標。")
    raw_overrides = color.get("revision_overrides") or {}
    if not isinstance(raw_overrides, Mapping):
        raise InspectionReleaseError("驗收報告的逐色修訂格式無效。")
    actual_overrides = {
        str(key): str(value) for key, value in raw_overrides.items()
    }
    component = draft.component_for_role("color_check")
    raw_color_path = str(color.get("color_model_path") or "").strip()
    actual_color_path = (
        str(Path(raw_color_path).expanduser().resolve()) if raw_color_path else ""
    )
    actual_color_sha256 = str(color.get("color_model_sha256") or "").lower()
    if component is None:
        if actual_overrides or actual_color_path or actual_color_sha256:
            raise InspectionReleaseError(
                "驗收報告套用了選取組合未綁定的顏色版本。"
            )
        return
    expected_color_path = (
        str(Path(component.artifact_path).expanduser().resolve())
        if component.artifact_path
        else ""
    )
    if (
        actual_overrides != dict(component.revision_overrides)
        or actual_color_path != expected_color_path
        or actual_color_sha256 != component.artifact_sha256.lower()
    ):
        raise InspectionReleaseError(
            "驗收報告使用的顏色基準或逐色修訂不是選取的組合版本。"
        )


def _model_components(model: Mapping[str, Any], inference_type: str) -> list[ComponentBinding]:
    identity = model.get("identity") or {}
    if not isinstance(identity, Mapping):
        raise InspectionReleaseError("Model identity is invalid.")
    config_path = Path(str(model.get("config_path") or "")).expanduser().resolve()
    weight_path = Path(str(model.get("weight_path") or "")).expanduser().resolve()
    if inference_type == "fusion":
        raise InspectionReleaseError("Fusion reports must identify YOLO and Anomalib artifacts separately.")
    kind = inference_type
    role = "primary_detector" if kind == "yolo" else "anomaly_detector"
    return [
        ComponentBinding(
            component_id=f"{kind}-model",
            kind=kind,
            role=role,
            version=str(identity.get("version") or ""),
            artifact_path=str(weight_path),
            artifact_sha256=str(identity.get("sha256") or "").lower(),
            config_path=str(config_path),
            config_sha256=str(identity.get("runtime_config_sha256") or "").lower(),
            metadata=(("variant_id", str(model.get("variant_id") or "")),),
        )
    ]


def _color_component(
    color: Mapping[str, Any],
    overrides: Mapping[str, Any],
    revisions_root: Path,
) -> ComponentBinding:
    config_paths: list[Path] = []
    normalized: list[tuple[str, str]] = []
    for scope_hash, revision_id in sorted(overrides.items()):
        scope_text = str(scope_hash)
        revision_text = str(revision_id)
        config_paths.append(revisions_root / scope_text / revision_text / "config.json")
        normalized.append((scope_text, revision_text))
    if len(config_paths) != 1:
        raise InspectionReleaseError("The first release format supports one color scope per component.")
    config_path = config_paths[0].resolve()
    label = str(color.get("label") or "")
    return ComponentBinding(
        component_id="stats-color",
        kind="stats_color",
        role="color_check",
        version=label.split("/")[-1].strip() or str(color.get("variant_id") or ""),
        config_path=str(config_path),
        config_sha256=sha256_file(config_path),
        revision_overrides=tuple(normalized),
        metadata=(("variant_id", str(color.get("variant_id") or "")),),
    )


def _profile_from_matrix(
    model: Mapping[str, Any],
    *,
    product: str,
    area: str,
    inference_type: str,
    project_root: Path,
    revision_overrides: Mapping[str, Any],
    color_model_path: str = "",
    color_model_sha256: str = "",
) -> ColorProfilePackage | None:
    config_path = Path(str(model.get("config_path") or "")).expanduser().resolve()
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise InspectionReleaseError("Acceptance matrix model config is unreadable.") from exc
    if not isinstance(config, Mapping) or not config.get("enable_color_check") or not config.get("color_model_path"):
        return None
    revision_store = ColorConfigurationRevisionStore(root=project_root / ".color_revisions")
    revisions: list[ColorConfigurationRevision] = []
    for scope_hash, revision_id in sorted(revision_overrides.items()):
        scope = revision_store.scope_for_hash(str(scope_hash))
        revisions.append(revision_store.resolve_revision(scope, str(revision_id)))
    color_model_override: Path | None = None
    if color_model_path:
        color_model_override = Path(color_model_path).expanduser().resolve()
        if color_model_override.is_symlink() or not color_model_override.is_file():
            raise InspectionReleaseError("Acceptance matrix color baseline is unavailable.")
        actual_sha256 = sha256_file(color_model_override)
        if color_model_sha256 and actual_sha256 != color_model_sha256.lower():
            raise InspectionReleaseError("Acceptance matrix color baseline checksum mismatch.")
    return ColorProfileStore(project_root / ".color_profiles").create(
        product=product,
        area=area,
        model_type=inference_type,
        model_config_path=config_path,
        project_root=project_root,
        revisions=revisions,
        color_model_override=color_model_override,
    )


def _profile_component(
    profile: ColorProfilePackage,
) -> ComponentBinding:
    return ComponentBinding(
        component_id="stats-color",
        kind="stats_color",
        role="color_check",
        version=profile.display_version,
        artifact_path=str(profile.color_model_path.resolve()),
        artifact_sha256=profile.color_model_sha256,
        config_path=str(profile.manifest_path.resolve()),
        config_sha256=profile.manifest_sha256,
        revision_overrides=profile.revision_overrides,
        metadata=tuple(
            sorted(
                {
                    "checker_type": profile.checker_type,
                    "colors": ",".join(profile.colors),
                    "package_id": profile.package_id,
                    "profile_summary": profile.summary,
                    "source": "color_profile_package",
                }.items()
            )
        ),
    )
