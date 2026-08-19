"""Composition root for Phase 3B/3C1 execution services."""

from __future__ import annotations

from pathlib import Path

from tools.annotation_packages import AnnotationPackageService
from tools.annotation_preparation_step import AnnotationPreparationStep
from tools.color_calibration_packages import ColorCalibrationPackageService
from tools.color_calibration_service import ColorCalibrationService
from tools.color_calibration_step import ColorCalibrationPreparationStep
from tools.color_configuration_resolver import ColorConfigurationResolver
from tools.dataset_preparation import (
    DatasetPreparationService,
    capture_dataset_source_snapshot,
)
from tools.dataset_preparation_step import DatasetPreparationStep
from tools.processing_execution import (
    BlockedStep,
    ExcludedStep,
    NoOpAnnotationStep,
    NoOpColorStep,
    NoOpReadyStep,
    ProcessingExecutionEngine,
    ProcessingStepRegistry,
)
from tools.processing_pipeline import ProcessingPlan
from tools.processing_plan_validation import ProcessingPlanValidator
from tools.processing_run_store import ProcessingRunStore


def build_processing_execution_engine(
    *,
    plan: ProcessingPlan,
    manifest_path: str | Path,
    store: ProcessingRunStore,
    validator: ProcessingPlanValidator,
    context_provider,
    dataset_step_enabled: bool,
    dataset_dry_run: bool = False,
    annotation_step_enabled: bool = False,
    color_step_enabled: bool = False,
    models_root: str | Path | None = None,
    color_revisions_root: str | Path | None = None,
) -> ProcessingExecutionEngine:
    """Build the enabled execution graph without leaking rules into QWidget."""
    registry = None
    if dataset_step_enabled or annotation_step_enabled or color_step_enabled:
        steps = []
        batch_steps = []
        if not dataset_step_enabled:
            steps.append(NoOpReadyStep())
        if not annotation_step_enabled:
            steps.append(NoOpAnnotationStep())
        if not color_step_enabled:
            steps.append(NoOpColorStep())
        steps.extend((BlockedStep(), ExcludedStep()))
    if dataset_step_enabled:
        service = DatasetPreparationService(
            artifact_root=store.artifacts_dir,
            source_manifest=manifest_path,
            source_snapshot=capture_dataset_source_snapshot(plan.records),
        )
        batch_steps.append(
            DatasetPreparationStep(service, dry_run_override=dataset_dry_run)
        )
    if annotation_step_enabled:
        annotation_service = AnnotationPackageService(
            artifact_root=store.artifacts_dir,
        )
        batch_steps.append(AnnotationPreparationStep(annotation_service))
    if color_step_enabled:
        effective_models_root = Path(models_root or (Path(manifest_path).resolve().parent / "models"))
        effective_revisions_root = Path(
            color_revisions_root
            or (Path(manifest_path).resolve().parent / ".color_revisions")
        )
        resolver = ColorConfigurationResolver(
            models_root=effective_models_root,
            revisions_root=effective_revisions_root,
        )

        def current_color_config(scope):
            resolved = resolver.resolve(scope)
            return resolved.source_path, resolved.config_sha256

        color_service = ColorCalibrationService(
            models_root=effective_models_root,
            current_config_resolver=current_color_config,
        )
        package_service = ColorCalibrationPackageService(
            artifact_root=store.artifacts_dir,
            calibration_service=color_service,
        )
        steps.append(ColorCalibrationPreparationStep(package_service))
    if dataset_step_enabled or annotation_step_enabled or color_step_enabled:
        registry = ProcessingStepRegistry(tuple(steps), batch_steps=tuple(batch_steps))
    return ProcessingExecutionEngine(
        validator=validator,
        context_provider=context_provider,
        store=store,
        step_registry=registry,
        dry_run=not (dataset_step_enabled or annotation_step_enabled or color_step_enabled),
    )
