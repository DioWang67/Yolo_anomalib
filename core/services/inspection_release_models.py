"""Immutable domain objects for versioned inspection releases.

An inspection release binds every runtime component used for one product,
area, and pipeline template.  The release never executes adapter code; it only
stores allow-listed component identities and immutable artifact hashes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from core.security import safe_segment

INSPECTION_RELEASE_SCHEMA_VERSION = 1
RELEASE_VALIDATION_ATTESTATION_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


class InspectionReleaseError(ValueError):
    """Base error for invalid release data or storage."""


class InspectionReleaseConflictError(InspectionReleaseError):
    """Raised when an active pointer changed during an activation request."""


class InspectionReleasePolicyError(InspectionReleaseError):
    """Raised when validation evidence does not permit an activation."""


class ReleaseStatus(str, Enum):
    DRAFT = "DRAFT"
    TESTED = "TESTED"
    BLOCKED = "BLOCKED"


class ActivationMode(str, Enum):
    FULL = "FULL"
    LIMITED_TRIAL = "LIMITED_TRIAL"
    RISK_ACCEPTED = "RISK_ACCEPTED"


@dataclass(frozen=True)
class ComponentDescriptor:
    kind: str
    display_name: str
    adapter_id: str


BUILTIN_COMPONENTS: Mapping[str, ComponentDescriptor] = MappingProxyType(
    {
        "yolo": ComponentDescriptor("yolo", "YOLO detector", "builtin.yolo"),
        "anomalib": ComponentDescriptor(
            "anomalib", "Anomalib detector", "builtin.anomalib"
        ),
        "fusion_decision": ComponentDescriptor(
            "fusion_decision", "Fusion decision", "builtin.fusion_decision"
        ),
        "stats_color": ComponentDescriptor(
            "stats_color", "Statistical color check", "builtin.stats_color"
        ),
        "rule_decision": ComponentDescriptor(
            "rule_decision", "Rule decision", "builtin.rule_decision"
        ),
    }
)


@dataclass(frozen=True)
class PipelineTemplate:
    template_id: str
    inference_type: str
    required_roles: tuple[tuple[str, tuple[str, ...]], ...]
    optional_roles: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def validate(self, components: tuple[ComponentBinding, ...]) -> None:
        by_role = {component.role: component for component in components}
        if len(by_role) != len(components):
            raise InspectionReleaseError("Component roles must be unique.")
        for role, allowed_kinds in self.required_roles:
            component = by_role.get(role)
            if component is None:
                raise InspectionReleaseError(
                    f"Pipeline {self.template_id} requires role {role}."
                )
            if component.kind not in allowed_kinds:
                raise InspectionReleaseError(
                    f"Role {role} does not accept component kind {component.kind}."
                )
        allowed = dict(self.required_roles + self.optional_roles)
        unknown_roles = set(by_role).difference(allowed)
        if unknown_roles:
            raise InspectionReleaseError(
                f"Pipeline {self.template_id} has unsupported roles: "
                f"{sorted(unknown_roles)}"
            )
        for role, component in by_role.items():
            if component.kind not in allowed[role]:
                raise InspectionReleaseError(
                    f"Role {role} does not accept component kind {component.kind}."
                )


PIPELINE_TEMPLATES: Mapping[str, PipelineTemplate] = MappingProxyType(
    {
        "yolo_visual_v1": PipelineTemplate(
            "yolo_visual_v1",
            "yolo",
            (("primary_detector", ("yolo",)),),
            (("color_check", ("stats_color",)), ("decision", ("rule_decision",))),
        ),
        "anomalib_visual_v1": PipelineTemplate(
            "anomalib_visual_v1",
            "anomalib",
            (("anomaly_detector", ("anomalib",)),),
            (("decision", ("rule_decision",)),),
        ),
        "fusion_visual_v1": PipelineTemplate(
            "fusion_visual_v1",
            "fusion",
            (
                ("primary_detector", ("yolo",)),
                ("anomaly_detector", ("anomalib",)),
                ("decision", ("fusion_decision",)),
            ),
            (("color_check", ("stats_color",)),),
        ),
    }
)


def template_for_inference_type(inference_type: str) -> PipelineTemplate:
    normalized = safe_segment(
        inference_type.lower(), field_name="inference_type"
    )
    for template in PIPELINE_TEMPLATES.values():
        if template.inference_type == normalized:
            return template
    raise InspectionReleaseError(
        f"No inspection pipeline template supports {normalized!r}."
    )


@dataclass(frozen=True)
class InspectionScope:
    product: str
    area: str
    template_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "product", safe_segment(self.product, field_name="product"))
        object.__setattr__(self, "area", safe_segment(self.area, field_name="area"))
        if self.template_id not in PIPELINE_TEMPLATES:
            raise InspectionReleaseError(
                f"Unknown pipeline template: {self.template_id}"
            )

    @property
    def inference_type(self) -> str:
        return PIPELINE_TEMPLATES[self.template_id].inference_type

    @property
    def scope_hash(self) -> str:
        payload = json.dumps(
            {
                "product": self.product,
                "area": self.area,
                "template_id": self.template_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def to_dict(self) -> dict[str, str]:
        return {
            "product": self.product,
            "area": self.area,
            "template_id": self.template_id,
            "inference_type": self.inference_type,
            "scope_hash": self.scope_hash,
        }


@dataclass(frozen=True)
class ComponentBinding:
    component_id: str
    kind: str
    role: str
    version: str
    artifact_path: str = ""
    artifact_sha256: str = ""
    config_path: str = ""
    config_sha256: str = ""
    revision_overrides: tuple[tuple[str, str], ...] = ()
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for value, label in (
            (self.component_id, "component_id"),
            (self.kind, "kind"),
            (self.role, "role"),
        ):
            if not _IDENTIFIER_RE.fullmatch(value):
                raise InspectionReleaseError(f"Invalid {label}: {value!r}")
        if self.kind not in BUILTIN_COMPONENTS:
            raise InspectionReleaseError(
                f"Component kind {self.kind!r} is not registered."
            )
        if not self.version.strip():
            raise InspectionReleaseError("Component version is required.")
        for path_value, hash_value, label in (
            (self.artifact_path, self.artifact_sha256, "artifact"),
            (self.config_path, self.config_sha256, "config"),
        ):
            if bool(path_value) != bool(hash_value):
                raise InspectionReleaseError(
                    f"{label} path and SHA-256 must be supplied together."
                )
            if hash_value and not _SHA256_RE.fullmatch(hash_value.lower()):
                raise InspectionReleaseError(f"Invalid {label} SHA-256.")
        if self.kind in {"yolo", "anomalib"} and not self.config_path:
            raise InspectionReleaseError(
                f"{self.kind} components require an immutable config snapshot."
            )
        seen_scopes: set[str] = set()
        for scope_hash, revision_id in self.revision_overrides:
            if not re.fullmatch(r"[0-9a-f]{24}", scope_hash):
                raise InspectionReleaseError("Invalid color revision scope hash.")
            if not revision_id.strip() or scope_hash in seen_scopes:
                raise InspectionReleaseError("Invalid duplicate color revision override.")
            seen_scopes.add(scope_hash)
        object.__setattr__(
            self,
            "revision_overrides",
            tuple(sorted(self.revision_overrides)),
        )
        object.__setattr__(
            self,
            "metadata",
            tuple(sorted(self.metadata, key=lambda item: item[0])),
        )

    @property
    def adapter_id(self) -> str:
        return BUILTIN_COMPONENTS[self.kind].adapter_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "kind": self.kind,
            "role": self.role,
            "adapter_id": self.adapter_id,
            "version": self.version,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256.lower(),
            "config_path": self.config_path,
            "config_sha256": self.config_sha256.lower(),
            "revision_overrides": dict(self.revision_overrides),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ComponentBinding:
        revision_overrides = payload.get("revision_overrides") or {}
        metadata = payload.get("metadata") or {}
        if not isinstance(revision_overrides, Mapping) or not isinstance(metadata, Mapping):
            raise InspectionReleaseError("Component mappings are invalid.")
        return cls(
            component_id=str(payload.get("component_id") or ""),
            kind=str(payload.get("kind") or ""),
            role=str(payload.get("role") or ""),
            version=str(payload.get("version") or ""),
            artifact_path=str(payload.get("artifact_path") or ""),
            artifact_sha256=str(payload.get("artifact_sha256") or "").lower(),
            config_path=str(payload.get("config_path") or ""),
            config_sha256=str(payload.get("config_sha256") or "").lower(),
            revision_overrides=tuple(
                sorted((str(key), str(value)) for key, value in revision_overrides.items())
            ),
            metadata=tuple(sorted((str(key), value) for key, value in metadata.items())),
        )


@dataclass(frozen=True)
class ValidationEvidence:
    report_path: str
    report_sha256: str
    run_id: str
    sample_count: int
    metrics: tuple[tuple[str, Any], ...]
    color_metrics: tuple[tuple[str, Any], ...] = ()
    combination_id: str = ""

    def __post_init__(self) -> None:
        if bool(self.report_path) != bool(self.report_sha256):
            raise InspectionReleaseError(
                "Validation report path and SHA-256 must be supplied together."
            )
        if self.report_sha256 and not _SHA256_RE.fullmatch(
            self.report_sha256.lower()
        ):
            raise InspectionReleaseError("Validation report SHA-256 is invalid.")
        if not self.run_id.strip() or self.sample_count < 0:
            raise InspectionReleaseError(
                "Validation run ID is required and sample count cannot be negative."
            )
        if self.sample_count > 0 and not self.report_path:
            raise InspectionReleaseError(
                "Validated samples require an immutable report."
            )

    def metric(self, name: str, *, color: bool = False) -> Any:
        return dict(self.color_metrics if color else self.metrics).get(name)

    @property
    def color_escape_known(self) -> bool:
        return self.metric("escape_rate", color=True) is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_path": self.report_path,
            "report_sha256": self.report_sha256.lower(),
            "run_id": self.run_id,
            "sample_count": self.sample_count,
            "combination_id": self.combination_id,
            "metrics": dict(self.metrics),
            "color_metrics": dict(self.color_metrics),
            "color_escape_known": self.color_escape_known,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ValidationEvidence:
        metrics = payload.get("metrics") or {}
        color_metrics = payload.get("color_metrics") or {}
        if not isinstance(metrics, Mapping) or not isinstance(color_metrics, Mapping):
            raise InspectionReleaseError("Validation metrics are invalid.")
        return cls(
            report_path=str(payload.get("report_path") or ""),
            report_sha256=str(payload.get("report_sha256") or "").lower(),
            run_id=str(payload.get("run_id") or ""),
            sample_count=int(payload.get("sample_count") or 0),
            combination_id=str(payload.get("combination_id") or ""),
            metrics=tuple(sorted((str(key), value) for key, value in metrics.items())),
            color_metrics=tuple(
                sorted((str(key), value) for key, value in color_metrics.items())
            ),
        )


@dataclass(frozen=True)
class ReleaseValidationAttestation:
    """Append-only proof that one immutable draft completed validation."""

    attestation_id: str
    release_id: str
    scope_hash: str
    base_release_sha256: str
    status: ReleaseStatus
    validated_at: str
    validator: str
    reason: str
    validation: ValidationEvidence

    def __post_init__(self) -> None:
        if not _SHA256_RE.fullmatch(self.attestation_id.lower()):
            raise InspectionReleaseError("Validation attestation ID is invalid.")
        if not re.fullmatch(r"[0-9a-f-]{36}", self.release_id.lower()):
            raise InspectionReleaseError("Validation attestation release ID is invalid.")
        if not re.fullmatch(r"[0-9a-f]{24}", self.scope_hash.lower()):
            raise InspectionReleaseError("Validation attestation scope is invalid.")
        if not _SHA256_RE.fullmatch(self.base_release_sha256.lower()):
            raise InspectionReleaseError("Validation base release checksum is invalid.")
        if self.status not in {ReleaseStatus.TESTED, ReleaseStatus.BLOCKED}:
            raise InspectionReleaseError("Validation attestation must be TESTED or BLOCKED.")
        if not self.validated_at.strip() or not self.validator.strip() or not self.reason.strip():
            raise InspectionReleaseError(
                "Validation timestamp, validator, and reason are required."
            )
        if self.validation.sample_count <= 0 or not self.validation.combination_id:
            raise InspectionReleaseError(
                "Validation attestation requires samples and an exact combination."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RELEASE_VALIDATION_ATTESTATION_SCHEMA_VERSION,
            "attestation_id": self.attestation_id.lower(),
            "release_id": self.release_id,
            "scope_hash": self.scope_hash.lower(),
            "base_release_sha256": self.base_release_sha256.lower(),
            "status": self.status.value,
            "validated_at": self.validated_at,
            "validator": self.validator,
            "reason": self.reason,
            "validation": self.validation.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> ReleaseValidationAttestation:
        if payload.get("schema_version") != RELEASE_VALIDATION_ATTESTATION_SCHEMA_VERSION:
            raise InspectionReleaseError("Unsupported validation attestation schema.")
        validation_payload = payload.get("validation")
        if not isinstance(validation_payload, Mapping):
            raise InspectionReleaseError("Validation attestation evidence is invalid.")
        try:
            status = ReleaseStatus(str(payload.get("status") or ""))
        except ValueError as exc:
            raise InspectionReleaseError("Validation attestation status is invalid.") from exc
        return cls(
            attestation_id=str(payload.get("attestation_id") or "").lower(),
            release_id=str(payload.get("release_id") or ""),
            scope_hash=str(payload.get("scope_hash") or "").lower(),
            base_release_sha256=str(payload.get("base_release_sha256") or "").lower(),
            status=status,
            validated_at=str(payload.get("validated_at") or ""),
            validator=str(payload.get("validator") or ""),
            reason=str(payload.get("reason") or ""),
            validation=ValidationEvidence.from_dict(validation_payload),
        )


@dataclass(frozen=True)
class InspectionRelease:
    release_id: str
    display_version: str
    scope: InspectionScope
    components: tuple[ComponentBinding, ...]
    status: ReleaseStatus
    created_at: str
    operator: str
    reason: str
    validation: ValidationEvidence

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f-]{36}", self.release_id.lower()):
            raise InspectionReleaseError("Release ID must be a UUID.")
        if not self.display_version.strip():
            raise InspectionReleaseError("Release display version is required.")
        if not self.created_at.strip() or not self.operator.strip() or not self.reason.strip():
            raise InspectionReleaseError(
                "Release timestamp, operator, and reason are required."
            )
        PIPELINE_TEMPLATES[self.scope.template_id].validate(self.components)

    def component_for_role(self, role: str) -> ComponentBinding | None:
        return next(
            (component for component in self.components if component.role == role),
            None,
        )

    def model_config_overrides(self) -> dict[str, Path]:
        overrides: dict[str, Path] = {}
        for component in self.components:
            if component.kind in {"yolo", "anomalib"}:
                overrides[component.kind] = Path(component.config_path)
        return overrides

    def color_revision_overrides(self) -> dict[str, str]:
        color = self.component_for_role("color_check")
        return dict(color.revision_overrides) if color else {}

    def color_model_override(self) -> Path | None:
        """Return the immutable full-color baseline bound by a v2-style profile."""
        color = self.component_for_role("color_check")
        if color is None or not color.artifact_path:
            return None
        return Path(color.artifact_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": INSPECTION_RELEASE_SCHEMA_VERSION,
            "release_id": self.release_id,
            "display_version": self.display_version,
            "scope": self.scope.to_dict(),
            "components": [component.to_dict() for component in self.components],
            "status": self.status.value,
            "created_at": self.created_at,
            "operator": self.operator,
            "reason": self.reason,
            "validation": self.validation.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InspectionRelease:
        if payload.get("schema_version") != INSPECTION_RELEASE_SCHEMA_VERSION:
            raise InspectionReleaseError("Unsupported inspection release schema.")
        scope_payload = payload.get("scope")
        components_payload = payload.get("components")
        validation_payload = payload.get("validation")
        if not isinstance(scope_payload, Mapping):
            raise InspectionReleaseError("Release scope is invalid.")
        if not isinstance(components_payload, list):
            raise InspectionReleaseError("Release components are invalid.")
        if not isinstance(validation_payload, Mapping):
            raise InspectionReleaseError("Release validation is invalid.")
        scope = InspectionScope(
            str(scope_payload.get("product") or ""),
            str(scope_payload.get("area") or ""),
            str(scope_payload.get("template_id") or ""),
        )
        if scope_payload.get("scope_hash") not in {None, scope.scope_hash}:
            raise InspectionReleaseError("Release scope hash does not match.")
        try:
            status = ReleaseStatus(str(payload.get("status") or ""))
        except ValueError as exc:
            raise InspectionReleaseError("Release status is invalid.") from exc
        return cls(
            release_id=str(payload.get("release_id") or ""),
            display_version=str(payload.get("display_version") or ""),
            scope=scope,
            components=tuple(
                ComponentBinding.from_dict(item)
                for item in components_payload
                if isinstance(item, Mapping)
            ),
            status=status,
            created_at=str(payload.get("created_at") or ""),
            operator=str(payload.get("operator") or ""),
            reason=str(payload.get("reason") or ""),
            validation=ValidationEvidence.from_dict(validation_payload),
        )
