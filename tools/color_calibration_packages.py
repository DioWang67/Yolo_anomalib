"""Immutable Phase 3C3 color calibration work packages."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from tools.color_calibration_service import (
    COLOR_ALGORITHM_VERSION,
    COLOR_CONFIG_SCHEMA_VERSION,
    ColorCalibrationError,
    ColorCalibrationGateResult,
    ColorCalibrationPreview,
    ColorCalibrationProposal,
    ColorCalibrationScope,
    ColorCalibrationService,
    canonical_sha256,
    gate_to_dict,
    preview_to_dict,
    proposal_to_dict,
    sha256_file,
)
from tools.processing_execution import CancellationToken
from tools.processing_pipeline import ProcessingPlan, ProcessingRecord, RoutingDecision, RoutingDecisionType

COLOR_PACKAGE_SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ColorPackageStatus(str, Enum):
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    PARTIAL = "PARTIAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class ColorCalibrationPackage:
    package_id: str
    plan_id: str
    report_id: str
    created_at: datetime
    operator: str
    source_manifest_sha: str
    review_revision: str
    status: ColorPackageStatus
    root: Path
    sample_ids: tuple[str, ...]
    scopes: tuple[ColorCalibrationScope, ...]
    proposals: tuple[ColorCalibrationProposal, ...]
    previews: tuple[ColorCalibrationPreview, ...]
    gates: tuple[ColorCalibrationGateResult, ...]
    diagnostics: tuple[Mapping[str, str], ...]

    @property
    def package_path(self) -> Path:
        return self.root / "package.json"


class ColorCalibrationPackageService:
    def __init__(
        self,
        *,
        artifact_root: str | Path,
        calibration_service: ColorCalibrationService,
        clock: Callable[[], datetime] | None = None,
        id_generator: Callable[[], str] | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root).resolve()
        self.calibration_service = calibration_service
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.id_generator = id_generator or (lambda: str(uuid4()))

    def create(
        self,
        plan: ProcessingPlan,
        records: Sequence[ProcessingRecord],
        decisions: Sequence[RoutingDecision],
        *,
        report_id: str,
        cancellation: CancellationToken,
    ) -> ColorCalibrationPackage:
        selected = self._color_records(records, decisions)
        if not selected:
            raise ColorCalibrationError("COLOR_PACKAGE_EMPTY", "No color-routed records were supplied.")
        package_id = _safe_id(self.id_generator(), "package")
        report_root = self.artifact_root / _safe_id(report_id, "report") / "color"
        destination = report_root / package_id
        report_root.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            loaded = load_color_calibration_package(destination / "package.json")
            if loaded.plan_id == plan.plan_id:
                return loaded
            raise ColorCalibrationError("COLOR_PACKAGE_EXISTS", f"Color package already exists: {destination}")
        cancellation.raise_if_cancelled()
        evidence, diagnostics = self.calibration_service.collect_evidence(selected)
        if not evidence:
            codes = ", ".join(sorted({str(item.get('code')) for item in diagnostics}))
            raise ColorCalibrationError("COLOR_EVIDENCE_MISSING", f"No usable color evidence was found ({codes}).")
        scopes = tuple(sorted({item.scope for item in evidence}))
        proposals: list[ColorCalibrationProposal] = []
        previews: list[ColorCalibrationPreview] = []
        gates: list[ColorCalibrationGateResult] = []
        for scope in scopes:
            cancellation.raise_if_cancelled()
            proposal, preview, gate = self.calibration_service.build(scope, evidence)
            proposals.append(proposal)
            previews.append(preview)
            gates.append(gate)
        created_at = self.clock()
        if created_at.tzinfo is None:
            raise ValueError("Color package clock must return timezone-aware time")
        staging = Path(tempfile.mkdtemp(prefix=f".{package_id}.", dir=report_root))
        package = ColorCalibrationPackage(
            package_id=package_id,
            plan_id=plan.plan_id,
            report_id=report_id,
            created_at=created_at,
            operator=plan.operator,
            source_manifest_sha=plan.source_manifest_sha,
            review_revision=plan.review_revision,
            status=ColorPackageStatus.WAITING_FOR_APPROVAL,
            root=destination,
            sample_ids=tuple(record.sample_id for record in selected),
            scopes=scopes,
            proposals=tuple(proposals),
            previews=tuple(previews),
            gates=tuple(gates),
            diagnostics=diagnostics,
        )
        try:
            for name in ("current_configs", "proposed_configs", "previews"):
                (staging / name).mkdir()
            _write_json_atomic(staging / "scopes.json", [_scope_dict(scope) for scope in scopes])
            _write_json_atomic(staging / "source_samples.json", self._source_samples(selected))
            metrics: dict[str, Any] = {}
            validation: dict[str, Any] = {}
            for proposal, preview, gate in zip(proposals, previews, gates, strict=True):
                key = proposal.scope.scope_hash
                shutil.copy2(proposal.current_config_path, staging / "current_configs" / f"{key}.yaml")
                _write_json_atomic(staging / "proposed_configs" / f"{key}.json", {
                    "schema_version": COLOR_CONFIG_SCHEMA_VERSION,
                    "scope": _scope_dict(proposal.scope),
                    "threshold_key": proposal.scope.threshold_key,
                    "checker_type": proposal.scope.checker_type,
                    "public_threshold": proposal.proposed_public_threshold,
                    "config_value": proposal.proposed_config_value,
                })
                _write_json_atomic(staging / "previews" / f"{key}.json", preview_to_dict(preview))
                metrics[key] = dict(preview.metrics)
                validation[key] = gate_to_dict(gate)
            _write_json_atomic(staging / "metrics.json", metrics)
            _write_json_atomic(staging / "validation_report.json", validation)
            _write_json_atomic(staging / "approval.json", {"schema_version": 1, "decisions": {}})
            _write_json_atomic(staging / "completion.json", {"schema_version": 1, "status": "PENDING", "revisions": [], "failures": []})
            _write_json_atomic(staging / "package.json", _package_dict(package, root=staging))
            immutable = _immutable_files(staging)
            _write_json_atomic(staging / "checksums.json", {name: sha256_file(staging / name) for name in immutable})
            cancellation.raise_if_cancelled()
            _fsync_tree(staging)
            os.replace(staging, destination)
            _fsync_directory(report_root)
            return package
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @staticmethod
    def _color_records(records: Sequence[ProcessingRecord], decisions: Sequence[RoutingDecision]) -> tuple[ProcessingRecord, ...]:
        by_id = {record.sample_id: record for record in records}
        return tuple(
            by_id[decision.sample_id]
            for decision in decisions
            if decision.decision == RoutingDecisionType.NEEDS_COLOR_CALIBRATION
            or RoutingDecisionType.NEEDS_COLOR_CALIBRATION in decision.additional_decisions
        )

    @staticmethod
    def _source_samples(records: Sequence[ProcessingRecord]) -> list[dict[str, Any]]:
        return [
            {
                "sample_id": item.sample_id,
                "source_index": item.source_index,
                "record_sha256": canonical_sha256(dict(item.fields)),
                "product": str(item.fields.get("product") or ""),
                "area": str(item.fields.get("area") or ""),
            }
            for item in records
        ]


def load_color_calibration_package(path: str | Path) -> ColorCalibrationPackage:
    package_path = Path(path).resolve()
    if package_path.name != "package.json" or not package_path.is_file():
        raise ColorCalibrationError("COLOR_PACKAGE_STALE", f"Color package is unavailable: {package_path}")
    root = package_path.parent
    _verify_checksums(root)
    try:
        raw = _read_json(package_path)
        scopes = tuple(_scope_from_dict(item) for item in raw["scopes"])
        proposals = tuple(_proposal_from_dict(item) for item in raw["proposals"])
        previews = tuple(_preview_from_dict(item) for item in raw["previews"])
        gates = tuple(_gate_from_dict(item) for item in raw["gates"])
        return ColorCalibrationPackage(
            package_id=str(raw["package_id"]), plan_id=str(raw["plan_id"]), report_id=str(raw["report_id"]),
            created_at=datetime.fromisoformat(str(raw["created_at"])), operator=str(raw["operator"]),
            source_manifest_sha=str(raw["source_manifest_sha"]), review_revision=str(raw["review_revision"]),
            status=ColorPackageStatus(str(raw["status"])), root=root,
            sample_ids=tuple(str(value) for value in raw["sample_ids"]), scopes=scopes,
            proposals=proposals, previews=previews, gates=gates,
            diagnostics=tuple(dict(item) for item in raw.get("diagnostics", ())),
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise ColorCalibrationError("COLOR_PACKAGE_STALE", f"Color package is unreadable: {package_path}") from exc


def _package_dict(package: ColorCalibrationPackage, *, root: Path) -> dict[str, Any]:
    return {
        "schema_version": COLOR_PACKAGE_SCHEMA_VERSION,
        "package_id": package.package_id,
        "plan_id": package.plan_id,
        "report_id": package.report_id,
        "created_at": package.created_at.isoformat(),
        "operator": package.operator,
        "source_manifest_sha": package.source_manifest_sha,
        "review_revision": package.review_revision,
        "sample_count": len(package.sample_ids),
        "scope_count": len(package.scopes),
        "sample_ids": list(package.sample_ids),
        "scopes": [_scope_dict(value) for value in package.scopes],
        "proposals": [proposal_to_dict(value) for value in package.proposals],
        "previews": [preview_to_dict(value) for value in package.previews],
        "gates": [gate_to_dict(value) for value in package.gates],
        "diagnostics": [dict(value) for value in package.diagnostics],
        "algorithm_version": COLOR_ALGORITHM_VERSION,
        "config_schema_version": COLOR_CONFIG_SCHEMA_VERSION,
        "status": package.status.value,
        "approved_scope_count": 0,
        "rejected_scope_count": 0,
        "completed_at": None,
        "artifact_layout_sha256": canonical_sha256(sorted(_immutable_files(root))),
    }


def _scope_dict(scope: ColorCalibrationScope) -> dict[str, str]:
    return {**asdict(scope), "scope_hash": scope.scope_hash}


def _scope_from_dict(raw: Mapping[str, Any]) -> ColorCalibrationScope:
    return ColorCalibrationScope(*(str(raw[name]) for name in ("product", "area", "model_type", "checker_type", "threshold_key")))


def _proposal_from_dict(raw: Mapping[str, Any]) -> ColorCalibrationProposal:
    from tools.color_calibration_service import ColorProposalStatus
    return ColorCalibrationProposal(
        scope=_scope_from_dict(raw["scope"]), status=ColorProposalStatus(str(raw["status"])),
        current_public_threshold=raw.get("current_public_threshold"), proposed_public_threshold=raw.get("proposed_public_threshold"),
        current_config_value=raw.get("current_config_value"), proposed_config_value=raw.get("proposed_config_value"),
        sample_count=int(raw["sample_count"]), positive_count=int(raw["positive_count"]), negative_count=int(raw["negative_count"]),
        unjudgeable_count=int(raw.get("unjudgeable_count") or 0), reasons=tuple(str(value) for value in raw.get("reasons", ())),
        current_config_path=str(raw["current_config_path"]), current_config_sha256=str(raw["current_config_sha256"]),
        algorithm_version=str(raw.get("algorithm_version") or COLOR_ALGORITHM_VERSION), config_schema_version=int(raw.get("config_schema_version") or 0),
    )


def _preview_from_dict(raw: Mapping[str, Any]) -> ColorCalibrationPreview:
    return ColorCalibrationPreview(_scope_from_dict(raw["scope"]), tuple(dict(item) for item in raw["samples"]), dict(raw["metrics"]))


def _gate_from_dict(raw: Mapping[str, Any]) -> ColorCalibrationGateResult:
    return ColorCalibrationGateResult(_scope_from_dict(raw["scope"]), bool(raw["passed"]), bool(raw["approval_allowed"]), tuple(raw["blocking_issues"]), tuple(raw["warnings"]), dict(raw["metric_deltas"]), tuple(raw["regression_samples"]))


def _immutable_files(root: Path) -> list[str]:
    mutable = {"approval.json", "completion.json", "checksums.json"}
    return sorted(str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*") if path.is_file() and path.name not in mutable)


def _verify_checksums(root: Path) -> None:
    checksums = _read_json(root / "checksums.json")
    if not isinstance(checksums, dict):
        raise ColorCalibrationError("COLOR_PACKAGE_STALE", "Color package checksums are missing.")
    for relative, expected in checksums.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ColorCalibrationError("COLOR_PACKAGE_PATH_ESCAPE", "Package checksum path escapes root.") from exc
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ColorCalibrationError("COLOR_PACKAGE_SHA_MISMATCH", f"Color package artifact changed: {relative}")


def _safe_id(value: str, kind: str) -> str:
    result = str(value).strip()
    if not _SAFE_ID.fullmatch(result):
        raise ColorCalibrationError("COLOR_PACKAGE_PATH_ESCAPE", f"Unsafe {kind} identifier.")
    return result


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_tree(root: Path) -> None:
    if os.name == "nt":
        return
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
