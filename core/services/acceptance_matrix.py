"""Read-only YOLO × color-revision acceptance matrix execution.

The runner deliberately separates three concerns:

* immutable input validation;
* production-equivalent inference for one explicit combination;
* append-only report persistence.

It never updates the acceptance manifest, color active pointers, or deployed
model files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import yaml

from core.services.color_baseline_recalibration import (
    ALGORITHM_VERSION,
    ColorBaselineCandidateStore,
)
from core.services.color_profile_store import ColorProfileStore
from core.services.inspection_release_models import (
    InspectionRelease,
    InspectionReleaseError,
)
from core.services.model_acceptance import (
    AcceptanceDataError,
    AcceptanceInferenceOutcome,
    AcceptanceInferenceService,
    AcceptanceMetrics,
    AcceptanceRecord,
    ModelIdentity,
    calculate_acceptance_metrics,
    load_acceptance_manifest,
    load_model_identity,
)
from core.services.model_version_registry import ModelVersionRecord
from core.station_data import load_station_data_paths
from tools.color_calibration_service import ColorCalibrationError
from tools.color_configuration_revisions import ColorConfigurationRevisionStore

MATRIX_REPORT_SCHEMA_VERSION = 1


class AcceptanceMatrixError(ValueError):
    """Raised when a matrix request or its immutable evidence is invalid."""


class AcceptanceMatrixCancelled(RuntimeError):
    """Raised when an operator cancels a matrix before a report is committed."""


@dataclass(frozen=True)
class AcceptanceModelVariant:
    """One complete models root used by production-equivalent inference."""

    variant_id: str
    label: str
    models_root: Path
    identity: ModelIdentity
    config_path: Path | None = None
    weight_path: Path | None = None


@dataclass(frozen=True)
class AcceptanceColorVariant:
    """A deterministic color configuration selection for one matrix column."""

    variant_id: str
    label: str
    revision_overrides: tuple[tuple[str, str], ...] = ()
    include_active_revisions: bool = False
    color_model_path: Path | None = None
    color_model_sha256: str = ""

    def override_mapping(self) -> dict[str, str]:
        return dict(self.revision_overrides)


@dataclass(frozen=True)
class AcceptanceMatrixRequest:
    """Validated-at-runtime inputs for one append-only matrix run."""

    project_root: Path
    global_config_path: Path
    color_revisions_root: Path
    dataset_root: Path
    manifest_path: Path
    output_root: Path
    product: str
    area: str
    inference_type: str
    model_variants: tuple[AcceptanceModelVariant, ...]
    color_variants: tuple[AcceptanceColorVariant, ...]


@dataclass(frozen=True)
class ColorAcceptanceMetrics:
    """Color-only confusion counts where ground truth is explicitly known."""

    tp: int
    fp: int
    fn: int
    tn: int
    unknown_truth: int
    errors: int

    @property
    def escape_rate(self) -> float | None:
        return _rate(self.fn, self.tp + self.fn)

    @property
    def overkill_rate(self) -> float | None:
        return _rate(self.fp, self.fp + self.tn)


@dataclass(frozen=True)
class AcceptanceMatrixCombinationResult:
    combination_id: str
    model_variant_id: str
    model_label: str
    color_variant_id: str
    color_label: str
    metrics: AcceptanceMetrics
    color_metrics: ColorAcceptanceMetrics
    average_latency_ms: float | None
    p95_latency_ms: float | None
    changed_from_reference: int
    changed_sample_ids: tuple[str, ...]
    error: str = ""


@dataclass(frozen=True)
class AcceptanceMatrixResult:
    run_id: str
    run_root: Path
    report_path: Path
    summary_csv_path: Path
    samples_csv_path: Path
    manifest_sha256: str
    sample_count: int
    combinations: tuple[AcceptanceMatrixCombinationResult, ...]


class _InferenceService(Protocol):
    def infer(
        self,
        record: AcceptanceRecord,
        image_path: str | Path,
        *,
        inference_type: str,
        cancel_cb: Callable[[], bool] | None = None,
    ) -> AcceptanceInferenceOutcome: ...

    def close(self) -> None: ...


ServiceFactory = Callable[..., _InferenceService]
ProgressCallback = Callable[[int, int, str, str], None]


def build_model_variant(
    models_root: str | Path,
    *,
    product: str,
    area: str,
    inference_type: str,
    label: str = "",
) -> AcceptanceModelVariant:
    """Validate and describe a deployable model bundle without loading weights."""
    root = Path(models_root).expanduser().resolve()
    selected_type = "yolo" if inference_type.lower() == "fusion" else inference_type
    config_path = root / product / area / selected_type / "config.yaml"
    if not config_path.is_file():
        raise AcceptanceMatrixError(f"模型組合缺少設定檔：{config_path}")
    identity = load_model_identity(root, product, area, inference_type)
    weight_path = _model_weight_path(config_path, root)
    display_label = label.strip() or f"{root.name} / {identity.version}"
    variant_id = _unique_variant_id(f"model-{identity.version}-{_short_hash(str(root).lower())}")
    return AcceptanceModelVariant(
        variant_id,
        display_label,
        root,
        identity,
        config_path.resolve(),
        weight_path,
    )


def build_registered_model_variant(
    record: ModelVersionRecord,
    *,
    models_root: str | Path,
) -> AcceptanceModelVariant:
    """Describe one historical registry record without activating it."""
    root = Path(models_root).expanduser().resolve()
    if not record.exists:
        raise AcceptanceMatrixError(f"模型權重不存在或為空：{record.weight_path}")
    if not record.has_config_snapshot or record.config_snapshot_path is None:
        raise AcceptanceMatrixError(f"模型 {record.version} 缺少配套 config 快照，不能安全測試。")
    config_path = record.config_snapshot_path.resolve()
    identity = ModelIdentity(
        version=record.version,
        sha256=record.weight_sha256.lower(),
        runtime_config_sha256=_sha256_file(config_path),
    )
    current_suffix = "（目前正式）" if record.is_current else ""
    label = f"YOLO {record.version}{current_suffix}"
    variant_id = _unique_variant_id(f"model-{record.version}-{_short_hash(record.weight_path.name)}")
    return AcceptanceModelVariant(
        variant_id=variant_id,
        label=label,
        models_root=root,
        identity=identity,
        config_path=config_path,
        weight_path=record.weight_path.resolve(),
    )


def build_release_acceptance_variants(
    release: InspectionRelease,
    *,
    project_root: str | Path,
) -> tuple[AcceptanceModelVariant, AcceptanceColorVariant]:
    """Build one exact model/color pair from an immutable inspection release."""
    inference_type = release.scope.inference_type
    if inference_type == "fusion":
        raise InspectionReleaseError(
            "Fusion 組合含多套模型；目前快速驗收契約無法同時鎖定全部模型，"
            "請使用完整驗收工具。"
        )
    model_role = (
        "primary_detector" if inference_type == "yolo" else "anomaly_detector"
    )
    model = release.component_for_role(model_role)
    if model is None or not model.artifact_path or not model.config_path:
        raise InspectionReleaseError("選取組合缺少可重現的模型權重或 config 快照。")
    root = Path(project_root).expanduser().resolve()
    data_paths = load_station_data_paths(root)
    model_variant = AcceptanceModelVariant(
        variant_id=_unique_variant_id(f"release-{release.release_id}-model"),
        label=f"{release.display_version} / {model.version}",
        models_root=data_paths.models,
        identity=ModelIdentity(
            version=model.version,
            sha256=model.artifact_sha256,
            runtime_config_sha256=model.config_sha256,
        ),
        config_path=Path(model.config_path).expanduser().resolve(),
        weight_path=Path(model.artifact_path).expanduser().resolve(),
    )
    color = release.component_for_role("color_check")
    if color is None:
        color_variant = AcceptanceColorVariant(
            variant_id=_unique_variant_id(
                f"release-{release.release_id}-color-embedded"
            ),
            label=f"{release.display_version} / 模型內建顏色設定",
        )
    else:
        color_variant = AcceptanceColorVariant(
            variant_id=_unique_variant_id(f"release-{release.release_id}-color"),
            label=f"{release.display_version} / {color.version}",
            revision_overrides=color.revision_overrides,
            include_active_revisions=False,
            color_model_path=(
                Path(color.artifact_path).expanduser().resolve()
                if color.artifact_path
                else None
            ),
            color_model_sha256=color.artifact_sha256,
        )
    return model_variant, color_variant


def discover_color_variants(
    revisions_root: str | Path,
    *,
    product: str,
    area: str,
    model_type: str,
    checker_type: str = "stats",
    baselines_root: str | Path | None = None,
    profiles_root: str | Path | None = None,
) -> tuple[AcceptanceColorVariant, ...]:
    """Discover embedded, active, and exact immutable color revisions."""
    store = ColorConfigurationRevisionStore(root=revisions_root)
    variants: list[AcceptanceColorVariant] = [
        AcceptanceColorVariant(
            variant_id="color-embedded",
            label="完整顏色基準（不套用校正修訂）",
            include_active_revisions=False,
        )
    ]
    if baselines_root is not None:
        baseline_store = ColorBaselineCandidateStore(baselines_root)
        for candidate in baseline_store.list_candidates(
            product=product,
            area=area,
            model_type=model_type,
        ):
            if candidate.algorithm != ALGORITHM_VERSION:
                continue
            variants.append(
                AcceptanceColorVariant(
                    variant_id=f"color-base-{candidate.candidate_id}",
                    label=(f"完整顏色基準 / {candidate.display_version}（{candidate.status}）"),
                    color_model_path=candidate.color_model_path,
                    color_model_sha256=candidate.color_model_sha256,
                )
            )
    if profiles_root is not None:
        profile_store = ColorProfileStore(profiles_root)
        profile_root = Path(profiles_root).expanduser().resolve()
        if profile_root.is_dir():
            for manifest in sorted(profile_root.glob("*/manifest.json")):
                profile = profile_store.load(manifest)
                if (
                    profile.product,
                    profile.area,
                    profile.model_type,
                    profile.checker_type,
                ) != (product, area, model_type, checker_type):
                    continue
                variants.append(
                    AcceptanceColorVariant(
                        variant_id=f"color-profile-{profile.package_id}",
                        label=(
                            f"完整顏色方案 / {profile.display_version}"
                            f"（{profile.summary}）"
                        ),
                        revision_overrides=profile.revision_overrides,
                        include_active_revisions=False,
                        color_model_path=profile.color_model_path,
                        color_model_sha256=profile.color_model_sha256,
                    )
                )
    if not store.root.is_dir():
        return tuple(variants)
    has_matching_active = False
    try:
        scopes = store.iter_scopes()
    except ColorCalibrationError:
        raise
    except OSError as exc:
        raise AcceptanceMatrixError(
            f"無法列舉顏色版本目錄：{store.root}"
        ) from exc
    for scope in scopes:
        try:
            if (scope.product, scope.area, scope.model_type, scope.checker_type) != (
                product,
                area,
                model_type,
                checker_type,
            ):
                continue
            has_matching_active = store.read_active_pointer(scope) is not None or has_matching_active
            for revision in store.list_revisions(scope):
                if store.is_revoked(revision):
                    continue
                variants.append(
                    AcceptanceColorVariant(
                        variant_id=_unique_variant_id(f"color-{scope.threshold_key}-{revision.display_version}"),
                        label=(f"{scope.threshold_key.title()} 門檻修訂 / {revision.display_version}"),
                        revision_overrides=((scope.scope_hash, revision.revision_id),),
                        include_active_revisions=False,
                    )
                )
        except ColorCalibrationError:
            raise
        except OSError as exc:
            raise AcceptanceMatrixError(
                f"無法讀取顏色版本目錄：{store.root / scope.scope_hash}"
            ) from exc
    if has_matching_active:
        variants.insert(
            1,
            AcceptanceColorVariant(
                variant_id="color-active",
                label="目前正式逐色校正組合",
                include_active_revisions=True,
            ),
        )
    return tuple(variants)


def run_acceptance_matrix(
    request: AcceptanceMatrixRequest,
    *,
    service_factory: ServiceFactory = AcceptanceInferenceService,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: Callable[[], bool] | None = None,
    clock: Callable[[], datetime] | None = None,
    id_generator: Callable[[], str] | None = None,
) -> AcceptanceMatrixResult:
    """Run every selected combination and atomically persist one report set."""
    normalized = _validate_request(request)
    manifest_sha256 = _sha256_file(normalized.manifest_path)
    records = _validated_records(normalized)
    total_work = len(records) * len(normalized.model_variants) * len(normalized.color_variants)
    completed_work = 0
    sample_rows: list[dict[str, Any]] = []
    combination_results: list[AcceptanceMatrixCombinationResult] = []
    reference_fingerprints: dict[str, tuple[str, str, str]] | None = None

    for model_variant in normalized.model_variants:
        for color_variant in normalized.color_variants:
            _raise_if_cancelled(cancel_callback)
            combination_id = f"{model_variant.variant_id}__{color_variant.variant_id}"
            combination_label = f"{model_variant.label} × {color_variant.label}"
            inferred_records: list[AcceptanceRecord] = []
            latencies: list[float] = []
            combination_error = ""
            service: _InferenceService | None = None
            try:
                service = service_factory(
                    project_root=normalized.project_root,
                    models_root=model_variant.models_root,
                    global_config_path=normalized.global_config_path,
                    model_identity=model_variant.identity,
                    color_revisions_root=normalized.color_revisions_root,
                    color_revision_overrides=color_variant.override_mapping(),
                    include_active_color_revisions=(color_variant.include_active_revisions),
                    model_config_overrides=(
                        {
                            (
                                normalized.product,
                                normalized.area,
                                (
                                    "yolo"
                                    if normalized.inference_type.lower() == "fusion"
                                    else normalized.inference_type.lower()
                                ),
                            ): model_variant.config_path
                        }
                        if model_variant.config_path is not None
                        else {}
                    ),
                    color_model_path_override=(color_variant.color_model_path),
                )
                for record in records:
                    _raise_if_cancelled(cancel_callback)
                    outcome = service.infer(
                        record,
                        _validated_image_path(normalized.dataset_root, record),
                        inference_type=normalized.inference_type,
                        cancel_cb=cancel_callback,
                    )
                    inferred = _record_with_outcome(record, outcome)
                    inferred_records.append(inferred)
                    if outcome.latency_ms >= 0 and not outcome.error:
                        latencies.append(outcome.latency_ms)
                    sample_rows.append(
                        _sample_row(
                            combination_id,
                            model_variant,
                            color_variant,
                            inferred,
                        )
                    )
                    completed_work += 1
                    if progress_callback is not None:
                        progress_callback(
                            completed_work,
                            total_work,
                            combination_label,
                            record.sample_id,
                        )
            except AcceptanceMatrixCancelled:
                raise
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                combination_error = str(exc)
                processed_ids = {item.sample_id for item in inferred_records}
                for record in records:
                    if record.sample_id in processed_ids:
                        continue
                    failed = replace(
                        record,
                        machine_status="ERROR",
                        machine_reasons="",
                        model_version=model_variant.identity.version,
                        model_sha256=model_variant.identity.sha256,
                        runtime_config_sha256=(model_variant.identity.runtime_config_sha256),
                        color_model_sha256=(model_variant.identity.color_model_sha256),
                        error=combination_error,
                        color_check_status="ERROR",
                        color_details_json="[]",
                    )
                    inferred_records.append(failed)
                    sample_rows.append(
                        _sample_row(
                            combination_id,
                            model_variant,
                            color_variant,
                            failed,
                        )
                    )
                    completed_work += 1
                    if progress_callback is not None:
                        progress_callback(
                            completed_work,
                            total_work,
                            combination_label,
                            record.sample_id,
                        )
            finally:
                if service is not None:
                    service.close()

            current_fingerprints = {record.sample_id: _decision_fingerprint(record) for record in inferred_records}
            if reference_fingerprints is None:
                reference_fingerprints = current_fingerprints
                changed_ids: tuple[str, ...] = ()
            else:
                changed_ids = tuple(
                    record.sample_id
                    for record in records
                    if current_fingerprints.get(record.sample_id) != reference_fingerprints.get(record.sample_id)
                )
            combination_results.append(
                AcceptanceMatrixCombinationResult(
                    combination_id=combination_id,
                    model_variant_id=model_variant.variant_id,
                    model_label=model_variant.label,
                    color_variant_id=color_variant.variant_id,
                    color_label=color_variant.label,
                    metrics=calculate_acceptance_metrics(inferred_records),
                    color_metrics=_calculate_color_metrics(inferred_records),
                    average_latency_ms=_average(latencies),
                    p95_latency_ms=_percentile_95(latencies),
                    changed_from_reference=len(changed_ids),
                    changed_sample_ids=changed_ids,
                    error=combination_error,
                )
            )

    if _sha256_file(normalized.manifest_path) != manifest_sha256:
        raise AcceptanceMatrixError("驗收標註在測試期間被修改，報告已拒絕建立；請重新執行。")
    run_time = (clock or (lambda: datetime.now(timezone.utc)))()
    if run_time.tzinfo is None:
        raise AcceptanceMatrixError("Matrix clock must return a timezone-aware datetime.")
    run_id = _create_run_id(run_time, (id_generator or (lambda: str(uuid4())))())
    return _persist_report(
        request=normalized,
        run_id=run_id,
        run_time=run_time,
        manifest_sha256=manifest_sha256,
        records=records,
        results=tuple(combination_results),
        sample_rows=sample_rows,
    )


def _validate_request(request: AcceptanceMatrixRequest) -> AcceptanceMatrixRequest:
    product = _required_value(request.product, "產品")
    area = _required_value(request.area, "區域")
    inference_type = _required_value(request.inference_type, "推論類型")
    if not request.model_variants:
        raise AcceptanceMatrixError("至少要選一個 YOLO 模型組合。")
    if not request.color_variants:
        raise AcceptanceMatrixError("至少要選一個顏色設定組合。")
    _require_unique((item.variant_id for item in request.model_variants), "YOLO variant_id")
    _require_unique((item.variant_id for item in request.color_variants), "color variant_id")
    normalized_colors: list[AcceptanceColorVariant] = []
    for color_variant in request.color_variants:
        overrides = color_variant.override_mapping()
        if len(overrides) != len(color_variant.revision_overrides):
            raise AcceptanceMatrixError(f"顏色組合含重複 scope：{color_variant.label}")
        if any(not key.strip() or not value.strip() for key, value in overrides.items()):
            raise AcceptanceMatrixError(f"顏色組合含空白版本參照：{color_variant.label}")
        color_model_path = (
            Path(color_variant.color_model_path).expanduser().resolve()
            if color_variant.color_model_path is not None
            else None
        )
        color_model_sha256 = color_variant.color_model_sha256
        if color_model_path is not None:
            if color_model_path.is_symlink() or not color_model_path.is_file():
                raise AcceptanceMatrixError(f"完整顏色基準不存在或不安全：{color_variant.label}")
            actual_sha256 = _sha256_file(color_model_path)
            if color_model_sha256 and color_model_sha256 != actual_sha256:
                raise AcceptanceMatrixError(f"完整顏色基準在選取後已變更：{color_variant.label}")
            color_model_sha256 = actual_sha256
        normalized_colors.append(
            replace(
                color_variant,
                color_model_path=color_model_path,
                color_model_sha256=color_model_sha256,
            )
        )
    normalized_models: list[AcceptanceModelVariant] = []
    for model_variant in request.model_variants:
        config_path = (
            Path(model_variant.config_path).expanduser().resolve() if model_variant.config_path is not None else None
        )
        if config_path is None or not config_path.is_file() or config_path.is_symlink():
            raise AcceptanceMatrixError(f"模型缺少安全的 config 快照：{model_variant.label}")
        actual_config_sha256 = _sha256_file(config_path)
        if (
            model_variant.identity.runtime_config_sha256
            and model_variant.identity.runtime_config_sha256 != actual_config_sha256
        ):
            raise AcceptanceMatrixError(f"模型 config 在選取後已變更：{model_variant.label}")
        weight_path = (
            Path(model_variant.weight_path).expanduser().resolve()
            if model_variant.weight_path is not None
            else _model_weight_path(config_path, model_variant.models_root)
        )
        if not weight_path.is_file() or weight_path.is_symlink():
            raise AcceptanceMatrixError(f"模型權重不存在或不安全：{model_variant.label}")
        actual_weight_sha256 = _sha256_file(weight_path)
        if model_variant.identity.sha256 and model_variant.identity.sha256.lower() != actual_weight_sha256:
            raise AcceptanceMatrixError(f"模型權重 SHA-256 不符：{model_variant.label}")
        normalized_models.append(
            replace(
                model_variant,
                models_root=Path(model_variant.models_root).expanduser().resolve(),
                identity=replace(
                    model_variant.identity,
                    sha256=actual_weight_sha256,
                    runtime_config_sha256=actual_config_sha256,
                ),
                config_path=config_path,
                weight_path=weight_path,
            )
        )
    manifest_path = Path(request.manifest_path).expanduser().resolve()
    dataset_root = Path(request.dataset_root).expanduser().resolve()
    if not manifest_path.is_file():
        raise AcceptanceMatrixError(f"找不到驗收標註：{manifest_path}")
    if not dataset_root.is_dir():
        raise AcceptanceMatrixError(f"找不到驗收照片目錄：{dataset_root}")
    output_root = Path(request.output_root).expanduser().resolve()
    _verify_output_writable(output_root)
    return replace(
        request,
        project_root=Path(request.project_root).expanduser().resolve(),
        global_config_path=Path(request.global_config_path).expanduser().resolve(),
        color_revisions_root=(Path(request.color_revisions_root).expanduser().resolve()),
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        output_root=output_root,
        product=product,
        area=area,
        inference_type=inference_type,
        model_variants=tuple(normalized_models),
        color_variants=tuple(normalized_colors),
    )


def _validated_records(
    request: AcceptanceMatrixRequest,
) -> tuple[AcceptanceRecord, ...]:
    try:
        all_records = load_acceptance_manifest(request.manifest_path)
    except AcceptanceDataError as exc:
        raise AcceptanceMatrixError(str(exc)) from exc
    records = tuple(
        record
        for record in all_records
        if record.product == request.product and record.area == request.area and record.review_status == "confirmed"
    )
    if not records:
        raise AcceptanceMatrixError("這份驗收集沒有已確認的人工標註。")
    seen: set[str] = set()
    for record in records:
        if record.sample_id in seen:
            raise AcceptanceMatrixError(f"驗收標註含重複 sample_id：{record.sample_id}")
        seen.add(record.sample_id)
        if record.expected_verdict not in {"OK", "NG"}:
            raise AcceptanceMatrixError(f"已確認照片缺少 OK/NG 真值：{record.sample_id}")
        image_path = _validated_image_path(request.dataset_root, record)
        if _sha256_file(image_path) != record.image_sha256.lower():
            raise AcceptanceMatrixError(f"驗收照片雜湊不符：{record.sample_id}")
    return records


def _verify_output_writable(output_root: Path) -> None:
    temporary_path: Path | None = None
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".matrix-write-check.",
            dir=output_root,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        temporary_path.write_bytes(b"acceptance-matrix-write-check")
        temporary_path.unlink()
        temporary_path = None
    except OSError as exc:
        raise AcceptanceMatrixError(f"矩陣報告目錄不可寫入：{output_root}") from exc
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def _validated_image_path(dataset_root: Path, record: AcceptanceRecord) -> Path:
    image_path = (dataset_root / record.image_path).resolve()
    try:
        image_path.relative_to(dataset_root)
    except ValueError as exc:
        raise AcceptanceMatrixError(f"驗收照片路徑越界：{record.image_path}") from exc
    if not image_path.is_file() or image_path.is_symlink():
        raise AcceptanceMatrixError(f"找不到安全的驗收照片：{record.sample_id}")
    return image_path


def _record_with_outcome(record: AcceptanceRecord, outcome: AcceptanceInferenceOutcome) -> AcceptanceRecord:
    if outcome.sample_id != record.sample_id:
        raise AcceptanceMatrixError(f"推論結果 sample_id 不一致：{record.sample_id}")
    return replace(
        record,
        machine_status=outcome.machine_status,
        machine_reasons="|".join(outcome.machine_reasons),
        model_version=outcome.model_version,
        model_sha256=outcome.model_sha256,
        runtime_config_sha256=outcome.runtime_config_sha256,
        color_model_sha256=outcome.color_model_sha256,
        inference_at=outcome.inference_at,
        latency_ms=f"{outcome.latency_ms:.3f}",
        error=outcome.error,
        color_check_status=outcome.color_check_status,
        color_details_json=outcome.color_details_json,
    )


def _calculate_color_metrics(
    records: Sequence[AcceptanceRecord],
) -> ColorAcceptanceMetrics:
    tp = fp = fn = tn = unknown_truth = errors = 0
    for record in records:
        if record.machine_status == "ERROR" or record.color_check_status == "ERROR":
            errors += 1
            continue
        expected_reasons = set(filter(None, record.expected_reasons.split("|")))
        actual_color_ng: bool | None
        if "COLOR_MISMATCH" in expected_reasons:
            actual_color_ng = True
        elif record.expected_verdict == "OK":
            actual_color_ng = False
        else:
            actual_color_ng = None
        predicted_color_ng = record.color_check_status == "FAIL" or "COLOR_MISMATCH" in set(
            filter(None, record.machine_reasons.split("|"))
        )
        if actual_color_ng is None:
            unknown_truth += 1
        elif actual_color_ng and predicted_color_ng:
            tp += 1
        elif actual_color_ng:
            fn += 1
        elif predicted_color_ng:
            fp += 1
        else:
            tn += 1
    return ColorAcceptanceMetrics(tp, fp, fn, tn, unknown_truth, errors)


def _sample_row(
    combination_id: str,
    model_variant: AcceptanceModelVariant,
    color_variant: AcceptanceColorVariant,
    record: AcceptanceRecord,
) -> dict[str, Any]:
    expected_color = (
        "NG"
        if "COLOR_MISMATCH" in record.expected_reasons.split("|")
        else "OK"
        if record.expected_verdict == "OK"
        else "UNKNOWN"
    )
    predicted_color = (
        "ERROR"
        if record.color_check_status == "ERROR"
        else "NG"
        if (record.color_check_status == "FAIL" or "COLOR_MISMATCH" in record.machine_reasons.split("|"))
        else "OK"
    )
    return {
        "combination_id": combination_id,
        "model_variant_id": model_variant.variant_id,
        "model_label": model_variant.label,
        "model_version": model_variant.identity.version,
        "model_sha256": model_variant.identity.sha256,
        "color_variant_id": color_variant.variant_id,
        "color_label": color_variant.label,
        "color_revision_overrides": json.dumps(
            color_variant.override_mapping(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "color_model_sha256": color_variant.color_model_sha256,
        "sample_id": record.sample_id,
        "image_path": record.image_path,
        "image_sha256": record.image_sha256,
        "expected_verdict": record.expected_verdict,
        "expected_reasons": record.expected_reasons,
        "machine_status": record.machine_status,
        "machine_reasons": record.machine_reasons,
        "expected_color": expected_color,
        "predicted_color": predicted_color,
        "color_check_status": record.color_check_status,
        "latency_ms": record.latency_ms,
        "error": record.error,
        "color_details_json": record.color_details_json,
    }


def _persist_report(
    *,
    request: AcceptanceMatrixRequest,
    run_id: str,
    run_time: datetime,
    manifest_sha256: str,
    records: Sequence[AcceptanceRecord],
    results: tuple[AcceptanceMatrixCombinationResult, ...],
    sample_rows: Sequence[Mapping[str, Any]],
) -> AcceptanceMatrixResult:
    request.output_root.mkdir(parents=True, exist_ok=True)
    destination = request.output_root / run_id
    if destination.exists():
        raise AcceptanceMatrixError(f"矩陣報告 ID 已存在：{run_id}")
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=request.output_root))
    committed = False
    try:
        report_path = staging / "report.json"
        summary_path = staging / "summary.csv"
        samples_path = staging / "samples.csv"
        report_payload = {
            "schema_version": MATRIX_REPORT_SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": run_time.isoformat(),
            "product": request.product,
            "area": request.area,
            "inference_type": request.inference_type,
            "manifest_path": str(request.manifest_path),
            "manifest_sha256": manifest_sha256,
            "dataset_root": str(request.dataset_root),
            "sample_count": len(records),
            "reference_combination_id": (results[0].combination_id if results else ""),
            "model_variants": [_model_variant_mapping(item) for item in request.model_variants],
            "color_variants": [_color_variant_mapping(item) for item in request.color_variants],
            "combinations": [_combination_mapping(item) for item in results],
            "samples": list(sample_rows),
        }
        report_path.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_csv(summary_path, [_summary_row(item) for item in results])
        _write_csv(samples_path, sample_rows)
        os.replace(staging, destination)
        committed = True
    finally:
        if not committed:
            shutil.rmtree(staging, ignore_errors=True)
    return AcceptanceMatrixResult(
        run_id=run_id,
        run_root=destination,
        report_path=destination / "report.json",
        summary_csv_path=destination / "summary.csv",
        samples_csv_path=destination / "samples.csv",
        manifest_sha256=manifest_sha256,
        sample_count=len(records),
        combinations=results,
    )


def _combination_mapping(
    result: AcceptanceMatrixCombinationResult,
) -> dict[str, Any]:
    return {
        "combination_id": result.combination_id,
        "model_variant_id": result.model_variant_id,
        "model_label": result.model_label,
        "color_variant_id": result.color_variant_id,
        "color_label": result.color_label,
        "metrics": _acceptance_metrics_mapping(result.metrics),
        "color_metrics": _color_metrics_mapping(result.color_metrics),
        "average_latency_ms": result.average_latency_ms,
        "p95_latency_ms": result.p95_latency_ms,
        "changed_from_reference": result.changed_from_reference,
        "changed_sample_ids": list(result.changed_sample_ids),
        "error": result.error,
    }


def _summary_row(result: AcceptanceMatrixCombinationResult) -> dict[str, Any]:
    metrics = result.metrics
    color = result.color_metrics
    return {
        "combination_id": result.combination_id,
        "model_label": result.model_label,
        "color_label": result.color_label,
        "confirmed": metrics.confirmed,
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "tn": metrics.tn,
        "errors": metrics.errors,
        "accuracy": _rate(metrics.tp + metrics.tn, metrics.tp + metrics.fp + metrics.fn + metrics.tn),
        "escape_rate": metrics.escape_rate,
        "overkill_rate": metrics.overkill_rate,
        "color_tp": color.tp,
        "color_fp": color.fp,
        "color_fn": color.fn,
        "color_tn": color.tn,
        "color_unknown_truth": color.unknown_truth,
        "color_escape_rate": color.escape_rate,
        "color_overkill_rate": color.overkill_rate,
        "average_latency_ms": result.average_latency_ms,
        "p95_latency_ms": result.p95_latency_ms,
        "changed_from_reference": result.changed_from_reference,
        "error": result.error,
    }


def _model_variant_mapping(
    variant: AcceptanceModelVariant,
) -> dict[str, Any]:
    return {
        "variant_id": variant.variant_id,
        "label": variant.label,
        "models_root": str(variant.models_root),
        "config_path": (str(variant.config_path) if variant.config_path is not None else ""),
        "weight_path": (str(variant.weight_path) if variant.weight_path is not None else ""),
        "identity": asdict(variant.identity),
    }


def _color_variant_mapping(
    variant: AcceptanceColorVariant,
) -> dict[str, Any]:
    return {
        "variant_id": variant.variant_id,
        "label": variant.label,
        "revision_overrides": variant.override_mapping(),
        "include_active_revisions": variant.include_active_revisions,
        "color_model_path": (str(variant.color_model_path) if variant.color_model_path is not None else ""),
        "color_model_sha256": variant.color_model_sha256,
    }


def _acceptance_metrics_mapping(metrics: AcceptanceMetrics) -> dict[str, Any]:
    return {
        **asdict(metrics),
        "true_yield": metrics.true_yield,
        "machine_yield": metrics.machine_yield,
        "escape_rate": metrics.escape_rate,
        "overkill_rate": metrics.overkill_rate,
    }


def _color_metrics_mapping(
    metrics: ColorAcceptanceMetrics,
) -> dict[str, Any]:
    return {
        **asdict(metrics),
        "escape_rate": metrics.escape_rate,
        "overkill_rate": metrics.overkill_rate,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise AcceptanceMatrixError(f"沒有可輸出的報告資料：{path.name}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _decision_fingerprint(record: AcceptanceRecord) -> tuple[str, str, str]:
    return (
        record.machine_status,
        record.machine_reasons,
        record.color_check_status,
    )


def _create_run_id(run_time: datetime, unique_id: str) -> str:
    safe_unique_id = _unique_variant_id(unique_id)[:12]
    return f"matrix-{run_time.strftime('%Y%m%dT%H%M%SZ')}-{safe_unique_id}"


def _raise_if_cancelled(
    cancel_callback: Callable[[], bool] | None,
) -> None:
    if cancel_callback is not None and cancel_callback():
        raise AcceptanceMatrixCancelled("驗收矩陣已由操作員取消。")


def _required_value(value: str, label: str) -> str:
    normalized = str(value).strip()
    if not normalized or normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise AcceptanceMatrixError(f"{label}不可為空或包含路徑字元。")
    return normalized


def _require_unique(values: Sequence[str] | Any, label: str) -> None:
    normalized = [str(value).strip() for value in values]
    if any(not value for value in normalized):
        raise AcceptanceMatrixError(f"{label} 不可為空。")
    if len(normalized) != len(set(normalized)):
        raise AcceptanceMatrixError(f"{label} 不可重複。")


def _unique_variant_id(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "-" for character in str(value).strip()
    ).strip("-._")
    if not normalized:
        raise AcceptanceMatrixError("組合識別碼不可為空。")
    return normalized[:128]


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def _model_weight_path(config_path: Path, models_root: Path) -> Path:
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise AcceptanceMatrixError(f"無法讀取模型 config：{config_path}") from exc
    if not isinstance(payload, Mapping):
        raise AcceptanceMatrixError(f"模型 config 必須是 mapping：{config_path}")
    raw_weight = str(payload.get("weights") or "").strip()
    if not raw_weight:
        raise AcceptanceMatrixError(f"模型 config 缺少 weights：{config_path}")
    configured = Path(raw_weight).expanduser()
    candidates = (
        (configured.resolve(),)
        if configured.is_absolute()
        else (
            (models_root.resolve().parent / configured).resolve(),
            (config_path.parent / configured).resolve(),
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise AcceptanceMatrixError(f"模型 config 指向不存在的權重：{raw_weight}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _average(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile_95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return ordered[index]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator > 0 else None
