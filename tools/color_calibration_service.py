"""Pure Phase 3C3 color-calibration proposal, preview, and gate services."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from tools.processing_pipeline import ProcessingRecord
from tools.review_routing import color_failure_items

COLOR_ALGORITHM_VERSION = "picture-tool-threshold-v1"
COLOR_CONFIG_SCHEMA_VERSION = 1
_BACKEND_LOCK = threading.Lock()


class ColorProposalStatus(str, Enum):
    PROPOSED = "PROPOSED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"
    INVALID_INPUT = "INVALID_INPUT"
    NO_CHANGE = "NO_CHANGE"
    FAILED = "FAILED"


class ColorCalibrationError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(message)


@dataclass(frozen=True, order=True)
class ColorCalibrationScope:
    product: str
    area: str
    model_type: str
    checker_type: str
    threshold_key: str

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not str(value).strip() for value in values.values()):
            raise ColorCalibrationError(
                "CALIBRATION_SCOPE_MISSING",
                f"Color calibration scope is incomplete: {values}",
            )
        for name in ("product", "area", "model_type"):
            value = str(values[name])
            if value in {".", ".."} or "/" in value or "\\" in value:
                raise ColorCalibrationError(
                    "CALIBRATION_SCOPE_INVALID", f"Unsafe {name} in color scope."
                )

    @property
    def key(self) -> str:
        return "/".join(asdict(self).values())

    @property
    def scope_hash(self) -> str:
        return hashlib.sha256(self.key.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class ColorEvidence:
    scope: ColorCalibrationScope
    sample_id: str
    item_index: str
    diff: float
    runtime_threshold: float
    actual_is_ok: bool
    failure_kind: str


@dataclass(frozen=True)
class CalibrationPolicy:
    minimum_total: int = 30
    minimum_ok: int = 5
    minimum_ng: int = 5
    false_accept_cost: float = 10.0
    false_reject_cost: float = 1.0
    maximum_false_accept_rate: float = 0.0
    maximum_regressions: int = 0


@dataclass(frozen=True)
class ColorCalibrationProposal:
    scope: ColorCalibrationScope
    status: ColorProposalStatus
    current_public_threshold: float | None
    proposed_public_threshold: float | None
    current_config_value: float | None
    proposed_config_value: float | None
    sample_count: int
    positive_count: int
    negative_count: int
    unjudgeable_count: int
    reasons: tuple[str, ...]
    current_config_path: str
    current_config_sha256: str
    algorithm_version: str = COLOR_ALGORITHM_VERSION
    config_schema_version: int = COLOR_CONFIG_SCHEMA_VERSION

    @property
    def proposal_sha256(self) -> str:
        return canonical_sha256(proposal_to_dict(self))


@dataclass(frozen=True)
class ColorCalibrationPreview:
    scope: ColorCalibrationScope
    samples: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, Any]

    @property
    def preview_sha256(self) -> str:
        return canonical_sha256(preview_to_dict(self))


@dataclass(frozen=True)
class ColorCalibrationGateResult:
    scope: ColorCalibrationScope
    passed: bool
    approval_allowed: bool
    blocking_issues: tuple[str, ...]
    warnings: tuple[str, ...]
    metric_deltas: Mapping[str, Any]
    regression_samples: tuple[str, ...]


class ColorThresholdBackend(Protocol):
    def recommend(
        self, samples: Sequence[ColorEvidence], policy: CalibrationPolicy
    ) -> Mapping[str, Any]: ...


class PictureToolThresholdBackend:
    """Adapter over the existing side-effect-free picture-tool backend API."""

    def __init__(self, module_path: str | Path | None = None) -> None:
        configured_path = module_path or os.environ.get(
            "YOLO_COLOR_CALIBRATION_BACKEND_PATH"
        )
        self._module_path = Path(configured_path).resolve() if configured_path else (
            Path(__file__).resolve().parents[2]
            / "Yolo11_auto_train"
            / "src"
            / "picture_tool"
            / "color"
            / "threshold_calibration.py"
        )

    def recommend(
        self, samples: Sequence[ColorEvidence], policy: CalibrationPolicy
    ) -> Mapping[str, Any]:
        module = self._load_module()
        backend_policy = module.CalibrationPolicy(
            minimum_total=policy.minimum_total,
            minimum_ok=policy.minimum_ok,
            minimum_ng=policy.minimum_ng,
            false_accept_cost=policy.false_accept_cost,
            false_reject_cost=policy.false_reject_cost,
            maximum_false_accept_rate=policy.maximum_false_accept_rate,
        )
        backend_samples = [
            module.ColorFeedbackSample(
                product=item.scope.product,
                area=item.scope.area,
                model_type=item.scope.model_type,
                checker_type=item.scope.checker_type,
                threshold_key=item.scope.threshold_key,
                failure_kind=item.failure_kind,
                sample_id=item.sample_id,
                item_index=item.item_index,
                diff=item.diff,
                runtime_threshold=item.runtime_threshold,
                actual_is_ok=item.actual_is_ok,
            )
            for item in samples
        ]
        results = module.recommend_color_thresholds(
            backend_samples, policy=backend_policy
        )
        if len(results) != 1:
            raise ColorCalibrationError(
                "CALIBRATION_BACKEND_INVALID",
                "Threshold backend must return exactly one recommendation per scope.",
            )
        return dict(results[0])

    def _load_module(self) -> Any:
        if not self._module_path.is_file():
            raise ColorCalibrationError(
                "CALIBRATION_BACKEND_UNAVAILABLE",
                f"Color calibration backend is unavailable: {self._module_path}",
                retryable=True,
            )
        module_name = "_yolo_phase3c3_threshold_calibration"
        with _BACKEND_LOCK:
            module = sys.modules.get(module_name)
            if module is not None:
                return module
            spec = importlib.util.spec_from_file_location(module_name, self._module_path)
            if spec is None or spec.loader is None:
                raise ColorCalibrationError(
                    "CALIBRATION_BACKEND_UNAVAILABLE", "Cannot load calibration backend."
                )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
            return module


class ColorCalibrationService:
    def __init__(
        self,
        *,
        models_root: str | Path,
        backend: ColorThresholdBackend | None = None,
        policy: CalibrationPolicy | None = None,
        current_config_resolver: Callable[[ColorCalibrationScope], tuple[Path, str]] | None = None,
    ) -> None:
        self.models_root = Path(models_root).resolve()
        self.backend = backend or PictureToolThresholdBackend()
        self.policy = policy or CalibrationPolicy()
        self.current_config_resolver = current_config_resolver

    def collect_evidence(
        self, records: Sequence[ProcessingRecord]
    ) -> tuple[tuple[ColorEvidence, ...], tuple[Mapping[str, str], ...]]:
        evidence: list[ColorEvidence] = []
        diagnostics: list[Mapping[str, str]] = []
        for record in records:
            fields = record.fields
            verdict = str(fields.get("color_verdict") or "").strip().lower()
            if verdict == "unjudgeable":
                diagnostics.append({"sample_id": record.sample_id, "code": "UNJUDGEABLE_EXCLUDED"})
                continue
            if verdict not in {"actually_ok", "confirmed_ng"}:
                diagnostics.append({"sample_id": record.sample_id, "code": "COLOR_TRUTH_MISSING"})
                continue
            items = color_failure_items(fields)
            if not items:
                diagnostics.append({"sample_id": record.sample_id, "code": "COLOR_EVIDENCE_MISSING"})
                continue
            for fallback_index, item in enumerate(items):
                try:
                    scope = self._scope(fields, item)
                    diff = _finite_unit_or_positive(item.get("diff"), "diff")
                    threshold = _finite_unit_or_positive(item.get("threshold"), "threshold")
                    evidence.append(
                        ColorEvidence(
                            scope=scope,
                            sample_id=record.sample_id,
                            item_index=str(item.get("index", fallback_index)),
                            diff=diff,
                            runtime_threshold=threshold,
                            actual_is_ok=verdict == "actually_ok",
                            failure_kind="threshold" if diff > threshold else "rule",
                        )
                    )
                except ColorCalibrationError as exc:
                    diagnostics.append({"sample_id": record.sample_id, "code": exc.code, "message": str(exc)})
        return tuple(evidence), tuple(diagnostics)

    def build(
        self, scope: ColorCalibrationScope, evidence: Sequence[ColorEvidence]
    ) -> tuple[ColorCalibrationProposal, ColorCalibrationPreview, ColorCalibrationGateResult]:
        scoped = tuple(item for item in evidence if item.scope == scope)
        if self.current_config_resolver is None:
            config_path = self._config_path(scope)
            config_sha = sha256_file(config_path)
        else:
            config_path, config_sha = self.current_config_resolver(scope)
            config_path = Path(config_path).resolve()
            if not config_path.is_file() or _configuration_sha(config_path) != config_sha:
                raise ColorCalibrationError("CURRENT_CONFIG_STALE", "Resolved current color config changed during proposal.", retryable=True)
        try:
            raw = self.backend.recommend(scoped, self.policy)
            proposal = self._proposal(scope, scoped, raw, config_path, config_sha)
            preview = build_preview(proposal, scoped, self.policy)
            gate = evaluate_gate(proposal, preview, self.policy)
            return proposal, preview, gate
        except (ColorCalibrationError, KeyError, TypeError, ValueError, OSError) as exc:
            current = _mode_threshold(scoped)
            current_config = 1.0 - current if scope.checker_type == "stats" else current
            proposal = ColorCalibrationProposal(
                scope=scope,
                status=ColorProposalStatus.FAILED,
                current_public_threshold=current,
                proposed_public_threshold=None,
                current_config_value=current_config,
                proposed_config_value=None,
                sample_count=len(scoped),
                positive_count=sum(item.actual_is_ok for item in scoped),
                negative_count=sum(not item.actual_is_ok for item in scoped),
                unjudgeable_count=0,
                reasons=(f"{getattr(exc, 'code', 'CALIBRATION_PROPOSAL_FAILED')}: {exc}",),
                current_config_path=str(config_path),
                current_config_sha256=config_sha,
            )
            preview = build_preview(proposal, scoped, self.policy)
            return proposal, preview, evaluate_gate(proposal, preview, self.policy)

    def _scope(self, fields: Mapping[str, Any], item: Mapping[str, Any]) -> ColorCalibrationScope:
        expected = str(item.get("class_name") or item.get("class") or "").strip()
        predicted = str(item.get("best_color") or "").strip()
        threshold_key = predicted or expected
        return ColorCalibrationScope(
            product=str(fields.get("product") or "").strip(),
            area=str(fields.get("area") or "").strip(),
            model_type=str(fields.get("detector") or "").strip().lower(),
            checker_type=str(fields.get("color_checker_type") or "").strip().lower(),
            threshold_key=threshold_key.lower(),
        )

    def _config_path(self, scope: ColorCalibrationScope) -> Path:
        path = (self.models_root / scope.product / scope.area / scope.model_type / "config.yaml").resolve()
        try:
            path.relative_to(self.models_root)
        except ValueError as exc:
            raise ColorCalibrationError("CALIBRATION_SCOPE_INVALID", "Config path escapes models root.") from exc
        if not path.is_file() or path.is_symlink():
            raise ColorCalibrationError("CURRENT_CONFIG_MISSING", f"Current color config is unavailable: {path}")
        return path

    @staticmethod
    def _proposal(
        scope: ColorCalibrationScope,
        evidence: Sequence[ColorEvidence],
        raw: Mapping[str, Any],
        config_path: Path,
        config_sha: str,
    ) -> ColorCalibrationProposal:
        status = {
            "ready": ColorProposalStatus.PROPOSED,
            "insufficient_data": ColorProposalStatus.INSUFFICIENT_DATA,
            "blocked_by_safety_policy": ColorProposalStatus.MANUAL_REVIEW_REQUIRED,
            "no_change": ColorProposalStatus.NO_CHANGE,
        }.get(str(raw.get("status") or ""), ColorProposalStatus.INVALID_INPUT)
        return ColorCalibrationProposal(
            scope=scope,
            status=status,
            current_public_threshold=_optional_float(raw.get("current_public_threshold")),
            proposed_public_threshold=_optional_float(raw.get("suggested_public_threshold")),
            current_config_value=_optional_float(raw.get("current_config_value")),
            proposed_config_value=_optional_float(raw.get("suggested_config_value")),
            sample_count=len(evidence),
            positive_count=sum(item.actual_is_ok for item in evidence),
            negative_count=sum(not item.actual_is_ok for item in evidence),
            unjudgeable_count=0,
            reasons=tuple(str(value) for value in raw.get("reasons", ())),
            current_config_path=str(config_path),
            current_config_sha256=config_sha,
        )


def build_preview(
    proposal: ColorCalibrationProposal,
    evidence: Sequence[ColorEvidence],
    policy: CalibrationPolicy,
) -> ColorCalibrationPreview:
    current = proposal.current_public_threshold
    proposed = proposal.proposed_public_threshold
    rows: list[Mapping[str, Any]] = []
    for item in evidence:
        before = bool(current is not None and item.diff <= current)
        after = bool(proposed is not None and item.diff <= proposed)
        before_correct = before == item.actual_is_ok
        after_correct = after == item.actual_is_ok
        rows.append({
            "sample_id": item.sample_id,
            "item_index": item.item_index,
            "truth_is_ok": item.actual_is_ok,
            "diff": item.diff,
            "current_is_ok": before,
            "proposed_is_ok": after,
            "corrected": not before_correct and after_correct,
            "regressed": before_correct and not after_correct,
        })
    metrics = _preview_metrics(rows, policy)
    return ColorCalibrationPreview(proposal.scope, tuple(rows), metrics)


def evaluate_gate(
    proposal: ColorCalibrationProposal,
    preview: ColorCalibrationPreview,
    policy: CalibrationPolicy,
) -> ColorCalibrationGateResult:
    issues: list[str] = []
    if proposal.status != ColorProposalStatus.PROPOSED:
        issues.append(f"PROPOSAL_{proposal.status.value}")
    for value in (proposal.proposed_public_threshold, proposal.proposed_config_value):
        if value is None or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            issues.append("PROPOSED_CONFIG_INVALID")
            break
    if not preview.samples:
        issues.append("NO_VALID_CALIBRATION_SAMPLES")
    regressions = tuple(
        str(item["sample_id"]) for item in preview.samples if item.get("regressed")
    )
    if len(regressions) > policy.maximum_regressions:
        issues.append("REGRESSION_POLICY_FAILED")
    before = preview.metrics.get("before", {})
    after = preview.metrics.get("after", {})
    deltas = {
        "accuracy": float(after.get("accuracy", 0.0)) - float(before.get("accuracy", 0.0)),
        "false_positive": int(after.get("false_positive", 0)) - int(before.get("false_positive", 0)),
        "false_negative": int(after.get("false_negative", 0)) - int(before.get("false_negative", 0)),
        "net_improvement": int(preview.metrics.get("corrected_count", 0)) - int(preview.metrics.get("regressed_count", 0)),
    }
    unique = tuple(dict.fromkeys(issues))
    return ColorCalibrationGateResult(
        scope=proposal.scope,
        passed=not unique,
        approval_allowed=not unique,
        blocking_issues=unique,
        warnings=(),
        metric_deltas=deltas,
        regression_samples=regressions,
    )


def proposal_to_dict(value: ColorCalibrationProposal) -> dict[str, Any]:
    payload = asdict(value)
    payload["scope"]["scope_hash"] = value.scope.scope_hash
    payload["status"] = value.status.value
    return payload


def preview_to_dict(value: ColorCalibrationPreview) -> dict[str, Any]:
    return {"scope": {**asdict(value.scope), "scope_hash": value.scope.scope_hash}, "samples": [dict(item) for item in value.samples], "metrics": dict(value.metrics)}


def gate_to_dict(value: ColorCalibrationGateResult) -> dict[str, Any]:
    return {"scope": {**asdict(value.scope), "scope_hash": value.scope.scope_hash}, "passed": value.passed, "approval_allowed": value.approval_allowed, "blocking_issues": list(value.blocking_issues), "warnings": list(value.warnings), "metric_deltas": dict(value.metric_deltas), "regression_samples": list(value.regression_samples)}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configuration_sha(path: Path) -> str:
    if path.suffix.lower() == ".json":
        try:
            return canonical_sha256(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ColorCalibrationError("CURRENT_CONFIG_INVALID", f"Current color config is unreadable: {path}") from exc
    return sha256_file(path)


def _finite_unit_or_positive(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ColorCalibrationError("COLOR_INPUT_INVALID", f"{field} must be numeric.") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ColorCalibrationError("COLOR_INPUT_INVALID", f"{field} must be finite and non-negative.")
    return parsed


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Calibration threshold must be finite")
    return result


def _mode_threshold(evidence: Sequence[ColorEvidence]) -> float:
    counts: dict[float, int] = {}
    for item in evidence:
        value = round(item.runtime_threshold, 12)
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return 0.0
    return min(counts, key=lambda value: (-counts[value], value))


def _preview_metrics(rows: Sequence[Mapping[str, Any]], policy: CalibrationPolicy) -> Mapping[str, Any]:
    def metrics(field: str) -> dict[str, Any]:
        fp = sum(bool(row[field]) and not bool(row["truth_is_ok"]) for row in rows)
        fn = sum(not bool(row[field]) and bool(row["truth_is_ok"]) for row in rows)
        correct = len(rows) - fp - fn
        return {"evaluated_count": len(rows), "accuracy": correct / len(rows) if rows else 0.0, "false_positive": fp, "false_negative": fn, "weighted_cost": fp * policy.false_accept_cost + fn * policy.false_reject_cost}
    return {"before": metrics("current_is_ok"), "after": metrics("proposed_is_ok"), "corrected_count": sum(bool(row["corrected"]) for row in rows), "regressed_count": sum(bool(row["regressed"]) for row in rows), "unchanged_count": sum(row["current_is_ok"] == row["proposed_is_ok"] for row in rows), "unjudgeable_excluded_count": 0}
