"""Create bounded OK-only color candidates without activating them."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from tools.color_calibration_service import (
    COLOR_CONFIG_SCHEMA_VERSION,
    ColorCalibrationError,
    ColorCalibrationScope,
    canonical_sha256,
)
from tools.color_configuration_resolver import ColorConfigurationResolver
from tools.color_configuration_revisions import (
    ColorConfigurationRevision,
    ColorConfigurationRevisionStore,
)

EXPERIMENTAL_MAX_PUBLIC_THRESHOLD_DELTA = 0.05
EXPERIMENTAL_MINIMUM_TOTAL = 30
EXPERIMENTAL_MINIMUM_OK = 5


@dataclass(frozen=True)
class ExperimentalColorScope:
    scope: ColorCalibrationScope
    sample_count: int
    ok_count: int
    ng_count: int
    current_public_threshold: float
    proposed_public_threshold: float
    corrected_ok_count: int
    evidence_sha256: str


@dataclass(frozen=True)
class ExperimentalColorCandidate:
    scope: ColorCalibrationScope
    baseline_version: str
    candidate_version: str
    candidate_revision_id: str
    current_public_threshold: float
    proposed_public_threshold: float
    corrected_ok_count: int
    sample_count: int
    active: bool = False


class ExperimentalColorCandidateService:
    """Build an immutable, inactive candidate from sufficient OK-only evidence."""

    def __init__(
        self,
        *,
        models_root: str | Path,
        revisions_root: str | Path,
        maximum_public_threshold_delta: float = EXPERIMENTAL_MAX_PUBLIC_THRESHOLD_DELTA,
    ) -> None:
        if not 0.0 < maximum_public_threshold_delta <= 0.1:
            raise ValueError("Experimental threshold delta must be within (0, 0.1].")
        self.models_root = Path(models_root).resolve()
        self.resolver = ColorConfigurationResolver(
            models_root=self.models_root,
            revisions_root=revisions_root,
        )
        self.revision_store: ColorConfigurationRevisionStore = (
            self.resolver.revision_store
        )
        self.maximum_public_threshold_delta = maximum_public_threshold_delta

    def eligible_scopes(
        self, manifest_paths: Sequence[str | Path]
    ) -> tuple[ExperimentalColorScope, ...]:
        rows = _load_feedback_rows(manifest_paths)
        grouped: dict[ColorCalibrationScope, list[Mapping[str, str]]] = {}
        for row in rows:
            if str(row.get("failure_kind") or "").strip().lower() != "threshold":
                continue
            scope = _scope_from_row(row)
            grouped.setdefault(scope, []).append(row)

        eligible: list[ExperimentalColorScope] = []
        for scope, scoped_rows in sorted(grouped.items()):
            ok_rows = [
                row
                for row in scoped_rows
                if str(row.get("actual_is_ok") or "").strip() == "1"
            ]
            ng_count = len(scoped_rows) - len(ok_rows)
            if (
                len(scoped_rows) < EXPERIMENTAL_MINIMUM_TOTAL
                or len(ok_rows) < EXPERIMENTAL_MINIMUM_OK
                or ng_count != 0
            ):
                continue
            current = _mode_float(
                _unit_float(row.get("threshold"), "threshold")
                for row in scoped_rows
            )
            observed_ok_limit = max(
                _unit_float(row.get("diff"), "diff") for row in ok_rows
            )
            proposed = min(
                1.0,
                observed_ok_limit,
                current + self.maximum_public_threshold_delta,
            )
            corrected = sum(
                current < _unit_float(row.get("diff"), "diff") <= proposed
                for row in ok_rows
            )
            if proposed <= current or corrected == 0:
                continue
            evidence_payload = [
                {
                    key: str(row.get(key) or "")
                    for key in (
                        "sample_id",
                        "item_index",
                        "image_sha256",
                        "actual_is_ok",
                        "diff",
                        "threshold",
                    )
                }
                for row in sorted(
                    scoped_rows,
                    key=lambda item: (
                        str(item.get("sample_id") or ""),
                        str(item.get("item_index") or ""),
                    ),
                )
            ]
            eligible.append(
                ExperimentalColorScope(
                    scope=scope,
                    sample_count=len(scoped_rows),
                    ok_count=len(ok_rows),
                    ng_count=ng_count,
                    current_public_threshold=current,
                    proposed_public_threshold=proposed,
                    corrected_ok_count=corrected,
                    evidence_sha256=canonical_sha256(evidence_payload),
                )
            )
        return tuple(eligible)

    def create_candidate(
        self,
        manifest_paths: Sequence[str | Path],
        scope_hash: str,
        *,
        operator: str,
        reason: str,
    ) -> ExperimentalColorCandidate:
        operator = operator.strip()
        reason = reason.strip()
        if not operator or not reason:
            raise ColorCalibrationError(
                "COLOR_APPROVAL_REQUIRED",
                "Named operator and candidate reason are required.",
            )
        eligible = self.eligible_scopes(manifest_paths)
        selected = next(
            (item for item in eligible if item.scope.scope_hash == scope_hash),
            None,
        )
        if selected is None:
            raise ColorCalibrationError(
                "COLOR_OK_ONLY_NOT_ELIGIBLE",
                "The selected scope is not eligible for an OK-only candidate.",
            )
        resolved = self.resolver.resolve(selected.scope)
        current_public, current_config = _resolved_threshold_values(
            selected.scope,
            resolved.config,
            resolved.source_path,
            fallback_public_threshold=selected.current_public_threshold,
        )
        if abs(current_public - selected.current_public_threshold) > 1e-6:
            raise ColorCalibrationError(
                "CURRENT_CONFIG_STALE",
                "The active threshold differs from the reviewed evidence.",
                retryable=True,
            )
        proposed_config = _revision_config(
            selected.scope,
            public_threshold=selected.proposed_public_threshold,
            config_value=_public_to_config(
                selected.scope.checker_type,
                selected.proposed_public_threshold,
            ),
        )
        source_id = (
            f"ok-only-{selected.scope.scope_hash}-"
            f"{selected.evidence_sha256[:16]}"
        )
        active = self.revision_store.read_active_pointer(selected.scope)
        if active is None:
            baseline = self.revision_store.commit_configuration(
                source_id,
                selected.scope,
                operator=operator,
                reason="Baseline snapshot before OK-only candidate",
                proposal_sha256=selected.evidence_sha256,
                preview_sha256=selected.evidence_sha256,
                proposed_config=_revision_config(
                    selected.scope,
                    public_threshold=current_public,
                    config_value=current_config,
                ),
                metrics=_evidence_metrics(selected, role="BASELINE"),
                parent_config_sha256=resolved.config_sha256,
            )
            self.revision_store.activate(
                baseline,
                operator=operator,
                reason="Capture rollback target before OK-only candidate",
                expected_current_sha256=resolved.config_sha256,
                event_type="COLOR_BASELINE_CAPTURED",
            )
            active = self.revision_store.read_active_pointer(selected.scope)
        if active is None:
            raise ColorCalibrationError(
                "COLOR_ACTIVE_REVISION_MISSING",
                "Could not establish a rollback baseline.",
            )
        parent = self.revision_store.load(
            selected.scope, str(active["revision_id"])
        )
        candidate_metrics = _evidence_metrics(selected, role="CANDIDATE")
        candidate = self.revision_store.commit_configuration(
            source_id,
            selected.scope,
            operator=operator,
            reason=reason,
            proposal_sha256=selected.evidence_sha256,
            preview_sha256=canonical_sha256(candidate_metrics),
            proposed_config=proposed_config,
            metrics=candidate_metrics,
            parent_revision_id=parent.revision_id,
            parent_config_sha256=parent.new_config_sha256,
        )
        return ExperimentalColorCandidate(
            scope=selected.scope,
            baseline_version=parent.display_version,
            candidate_version=candidate.display_version,
            candidate_revision_id=candidate.revision_id,
            current_public_threshold=current_public,
            proposed_public_threshold=selected.proposed_public_threshold,
            corrected_ok_count=selected.corrected_ok_count,
            sample_count=selected.sample_count,
        )

    def activate_candidate(
        self,
        candidate: ExperimentalColorCandidate,
        *,
        operator: str,
        reason: str,
    ) -> Path:
        revision = self.revision_store.load(
            candidate.scope, candidate.candidate_revision_id
        )
        if _evidence_level(revision) != "OK_ONLY":
            raise ColorCalibrationError(
                "COLOR_EXPERIMENTAL_EVIDENCE_INVALID",
                "Only an OK-only candidate can use this activation route.",
            )
        active = self.revision_store.read_active_pointer(candidate.scope)
        if active is None:
            raise ColorCalibrationError(
                "COLOR_ACTIVE_REVISION_MISSING",
                "No rollback baseline is active.",
            )
        return self.revision_store.activate(
            revision,
            operator=operator,
            reason=reason,
            expected_current_sha256=str(active["config_sha256"]),
            event_type="COLOR_OK_ONLY_CANDIDATE_ACTIVATED",
        )


def _load_feedback_rows(
    manifest_paths: Sequence[str | Path],
) -> tuple[dict[str, str], ...]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for raw_path in manifest_paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise OSError(f"Color feedback manifest is missing: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                identity = (
                    str(row.get("sample_id") or "").strip(),
                    str(row.get("item_index") or "").strip(),
                )
                if not all(identity):
                    raise ValueError(f"Color feedback identity is incomplete: {path}")
                latest[identity] = dict(row)
    return tuple(latest[key] for key in sorted(latest))


def _scope_from_row(row: Mapping[str, str]) -> ColorCalibrationScope:
    return ColorCalibrationScope(
        product=str(row.get("product") or "").strip(),
        area=str(row.get("area") or "").strip(),
        model_type=str(row.get("model_type") or "").strip().lower(),
        checker_type=str(row.get("checker_type") or "").strip().lower(),
        threshold_key=str(row.get("threshold_key") or "").strip().lower(),
    )


def _resolved_threshold_values(
    scope: ColorCalibrationScope,
    resolved_config: Mapping[str, Any],
    source_path: Path,
    *,
    fallback_public_threshold: float,
) -> tuple[float, float]:
    if resolved_config:
        public = _unit_float(
            resolved_config.get("public_threshold"), "public_threshold"
        )
        config_value = _unit_float(
            resolved_config.get("config_value"), "config_value"
        )
        return public, config_value
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ColorCalibrationError(
            "CURRENT_CONFIG_INVALID", "Color model config must be a mapping."
        )
    overrides = payload.get("color_threshold_overrides")
    raw_value = None
    if isinstance(overrides, Mapping):
        raw_value = next(
            (
                value
                for key, value in overrides.items()
                if str(key).strip().lower() == scope.threshold_key
            ),
            None,
        )
    if raw_value is None:
        raw_value = payload.get("color_score_threshold")
    if raw_value is None:
        return (
            fallback_public_threshold,
            _public_to_config(scope.checker_type, fallback_public_threshold),
        )
    config_value = _unit_float(raw_value, "config_value")
    return (
        _config_to_public(scope.checker_type, config_value),
        config_value,
    )


def _revision_config(
    scope: ColorCalibrationScope,
    *,
    public_threshold: float,
    config_value: float,
) -> dict[str, Any]:
    return {
        "schema_version": COLOR_CONFIG_SCHEMA_VERSION,
        "scope": {**asdict(scope), "scope_hash": scope.scope_hash},
        "threshold_key": scope.threshold_key,
        "checker_type": scope.checker_type,
        "public_threshold": public_threshold,
        "config_value": config_value,
    }


def _evidence_metrics(
    selected: ExperimentalColorScope, *, role: str
) -> dict[str, Any]:
    return {
        "revision_role": role,
        "corrected_ok_count": selected.corrected_ok_count,
        "unknown_escape_rate": True,
        "maximum_public_threshold_delta": (
            selected.proposed_public_threshold
            - selected.current_public_threshold
        ),
        "evidence": {
            "level": "OK_ONLY",
            "ok_count": selected.ok_count,
            "ng_count": selected.ng_count,
            "sample_count": selected.sample_count,
            "sha256": selected.evidence_sha256,
        },
    }


def _evidence_level(revision: ColorConfigurationRevision) -> str:
    evidence = revision.metrics.get("evidence")
    return (
        str(evidence.get("level") or "")
        if isinstance(evidence, Mapping)
        else ""
    )


def _mode_float(values) -> float:
    counts: dict[float, int] = {}
    for value in values:
        normalized = round(float(value), 12)
        counts[normalized] = counts.get(normalized, 0) + 1
    if not counts:
        raise ValueError("Cannot select a threshold from empty evidence.")
    return min(counts, key=lambda item: (-counts[item], item))


def _unit_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric.") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1.")
    return parsed


def _public_to_config(checker_type: str, value: float) -> float:
    return round(1.0 - value if checker_type == "stats" else value, 12)


def _config_to_public(checker_type: str, value: float) -> float:
    return round(1.0 - value if checker_type == "stats" else value, 12)
