from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from core.services.inspection_release_builder import (
    build_draft_release,
    build_release_from_matrix,
    build_validated_release_from_matrix,
)
from core.services.inspection_release_models import (
    PIPELINE_TEMPLATES,
    ActivationMode,
    ComponentBinding,
    InspectionRelease,
    InspectionReleaseConflictError,
    InspectionReleaseError,
    InspectionReleasePolicyError,
    InspectionScope,
    ReleaseStatus,
    ValidationEvidence,
)
from core.services.inspection_release_store import (
    InspectionReleaseResolver,
    InspectionReleaseStore,
)
from core.services.model_version_registry import ModelVersionRecord


def _write(path: Path, content: bytes) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path.resolve()), hashlib.sha256(content).hexdigest()


def _release(
    tmp_path: Path,
    *,
    display_version: str = "inspection-v1.0.1",
    color_escape: float | None = None,
    fn: int = 0,
    errors: int = 0,
    status: ReleaseStatus = ReleaseStatus.TESTED,
) -> InspectionRelease:
    model_path, model_sha = _write(tmp_path / "model.onnx", b"model")
    config_path, config_sha = _write(tmp_path / "model.config.yaml", b"weights: model.onnx")
    color_path, color_sha = _write(tmp_path / "color.json", b'{"black": 0.6}')
    report_path, report_sha = _write(tmp_path / "report.json", b'{"run": 1}')
    return InspectionRelease(
        release_id=str(uuid4()),
        display_version=display_version,
        scope=InspectionScope("Cable1", "A", "yolo_visual_v1"),
        components=(
            ComponentBinding(
                "yolo-model",
                "yolo",
                "primary_detector",
                "1.0.6",
                model_path,
                model_sha,
                config_path,
                config_sha,
            ),
            ComponentBinding(
                "stats-color",
                "stats_color",
                "color_check",
                "color-v1.0.2",
                config_path=color_path,
                config_sha256=color_sha,
                revision_overrides=(("0123456789abcdef01234567", str(uuid4())),),
            ),
        ),
        status=status,
        created_at=datetime.now(timezone.utc).isoformat(),
        operator="tester",
        reason="test release",
        validation=ValidationEvidence(
            report_path,
            report_sha,
            "matrix-test",
            250,
            (
                ("errors", errors),
                ("fn", fn),
                ("fp", 14),
                ("overkill_rate", 0.08),
            ),
            (
                ("errors", 0),
                ("fn", 0),
                ("escape_rate", color_escape),
                ("overkill_rate", 0.069),
            ),
            "combo-1",
        ),
    )


def test_builtin_templates_support_yolo_anomalib_and_fusion():
    assert set(PIPELINE_TEMPLATES) == {
        "yolo_visual_v1",
        "anomalib_visual_v1",
        "fusion_visual_v1",
    }
    assert PIPELINE_TEMPLATES["fusion_visual_v1"].inference_type == "fusion"


def test_template_rejects_missing_required_component(tmp_path):
    report_path, report_sha = _write(tmp_path / "report.json", b"{}")
    with pytest.raises(InspectionReleaseError, match="requires role"):
        InspectionRelease(
            str(uuid4()),
            "inspection-v1",
            InspectionScope("Cable1", "A", "fusion_visual_v1"),
            (),
            ReleaseStatus.TESTED,
            datetime.now(timezone.utc).isoformat(),
            "tester",
            "invalid",
            ValidationEvidence(report_path, report_sha, "run", 1, (("errors", 0),)),
        )


def test_unknown_component_kind_is_not_executable():
    with pytest.raises(InspectionReleaseError, match="not registered"):
        ComponentBinding(
            "custom-model",
            "arbitrary_python",
            "primary_detector",
            "1.0",
            config_path="config.yaml",
            config_sha256="0" * 64,
        )


def test_store_commit_load_and_checksum_tamper_detection(tmp_path):
    release = _release(tmp_path / "artifacts")
    store = InspectionReleaseStore(tmp_path / "store")
    committed = store.commit(release)
    assert store.load(release.scope, release.release_id) == committed

    release_path = store.release_dir(release) / "release.json"
    release_path.write_text("{}", encoding="utf-8")
    with pytest.raises(InspectionReleaseError, match="checksum"):
        store.load(release.scope, release.release_id)


def test_validation_attestation_projects_same_draft_as_tested(tmp_path):
    draft = _release(
        tmp_path / "artifacts",
        status=ReleaseStatus.DRAFT,
    )
    store = InspectionReleaseStore(tmp_path / "store")
    store.commit(draft)
    report_path, report_sha = _write(
        tmp_path / "validation" / "report.json",
        b'{"run": "quick-validation"}',
    )
    validation = ValidationEvidence(
        report_path=report_path,
        report_sha256=report_sha,
        run_id="quick-validation",
        sample_count=250,
        metrics=(("errors", 0), ("fn", 0), ("fp", 9)),
        color_metrics=(("escape_rate", None), ("overkill_rate", 0.0347)),
        combination_id="exact-release-combination",
    )
    validated = replace(
        draft,
        status=ReleaseStatus.TESTED,
        validation=validation,
    )

    projected = store.commit_validation(
        validated,
        validator="reviewer",
        reason="single exact combination completed",
    )
    repeated = store.commit_validation(
        validated,
        validator="reviewer",
        reason="idempotent retry",
    )

    assert projected == repeated
    assert projected.release_id == draft.release_id
    assert projected.display_version == draft.display_version
    assert projected.components == draft.components
    assert projected.status is ReleaseStatus.TESTED
    assert projected.validation == validation
    assert store.commit(draft) == projected
    assert store.list_releases()[0] == projected
    raw_payload = json.loads(
        (store.release_dir(draft) / "release.json").read_text(encoding="utf-8")
    )
    assert raw_payload["status"] == "DRAFT"
    attestations = tuple(
        (
            store.root
            / "validations"
            / draft.scope.scope_hash
            / draft.release_id
        ).glob("*.json")
    )
    assert len(attestations) == 1


def test_validation_attestation_checksum_tamper_is_rejected(tmp_path):
    draft = _release(
        tmp_path / "artifacts",
        status=ReleaseStatus.DRAFT,
    )
    store = InspectionReleaseStore(tmp_path / "store")
    store.commit(draft)
    validated = replace(draft, status=ReleaseStatus.TESTED)
    store.commit_validation(
        validated,
        validator="reviewer",
        reason="completed",
    )
    validation_root = (
        store.root
        / "validations"
        / draft.scope.scope_hash
        / draft.release_id
    )
    attestation_path = next(validation_root.glob("*.json"))
    payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    payload["reason"] = "tampered"
    attestation_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InspectionReleaseError, match="checksum"):
        store.load(draft.scope, draft.release_id)


def test_validation_attestation_rejects_changed_draft_identity(tmp_path):
    draft = _release(
        tmp_path / "artifacts",
        status=ReleaseStatus.DRAFT,
    )
    store = InspectionReleaseStore(tmp_path / "store")
    store.commit(draft)
    changed = replace(
        draft,
        display_version="inspection-v9.9.9",
        status=ReleaseStatus.TESTED,
    )

    with pytest.raises(InspectionReleaseConflictError, match="does not match"):
        store.commit_validation(
            changed,
            validator="reviewer",
            reason="must not rebind a version",
        )


def test_unknown_color_escape_allows_trial_or_explicit_risk_acceptance(tmp_path):
    release = _release(tmp_path / "artifacts", color_escape=None)
    store = InspectionReleaseStore(tmp_path / "store")
    store.commit(release)
    assert store.policy.allowed_modes(release) == (
        ActivationMode.LIMITED_TRIAL,
        ActivationMode.RISK_ACCEPTED,
    )
    with pytest.raises(InspectionReleasePolicyError, match="complete validation"):
        store.activate(
            release,
            mode=ActivationMode.FULL,
            operator="tester",
            reason="unsafe",
            expected_release_id=None,
        )


def test_false_negative_or_draft_remains_available_with_visible_risk(tmp_path):
    failing = _release(tmp_path / "failing", fn=1)
    draft = _release(tmp_path / "draft", status=ReleaseStatus.DRAFT)
    store = InspectionReleaseStore(tmp_path / "store")
    expected = (
        ActivationMode.LIMITED_TRIAL,
        ActivationMode.RISK_ACCEPTED,
    )
    assert store.policy.allowed_modes(failing) == expected
    assert store.policy.allowed_modes(draft) == expected
    assert any("false negatives" in warning for warning in store.policy.validation_warnings(failing))
    assert any("draft" in warning.lower() for warning in store.policy.validation_warnings(draft))


def test_risk_accepted_activation_is_audited_in_pointer(tmp_path):
    release = _release(tmp_path / "artifacts", color_escape=None)
    store = InspectionReleaseStore(tmp_path / "store")
    store.commit(release)

    pointer = store.activate(
        release,
        mode=ActivationMode.RISK_ACCEPTED,
        operator="engineer",
        reason="No Black NG available; production owner accepts the risk.",
        expected_release_id=None,
    )

    assert pointer["mode"] == "RISK_ACCEPTED"
    assert pointer["operator"] == "engineer"
    assert "accepts the risk" in pointer["reason"]


def test_explicitly_blocked_release_cannot_be_activated(tmp_path):
    blocked = _release(tmp_path / "blocked", status=ReleaseStatus.BLOCKED)
    store = InspectionReleaseStore(tmp_path / "store")
    store.commit(blocked)

    assert store.policy.allowed_modes(blocked) == ()
    with pytest.raises(InspectionReleasePolicyError, match="explicitly blocked"):
        store.activate(
            blocked,
            mode=ActivationMode.RISK_ACCEPTED,
            operator="engineer",
            reason="must remain blocked",
            expected_release_id=None,
        )


def test_atomic_activation_compare_and_swap_and_rollback(tmp_path):
    store = InspectionReleaseStore(tmp_path / "store")
    baseline = store.commit(_release(tmp_path / "baseline"))
    candidate = store.commit(_release(tmp_path / "candidate", display_version="inspection-v1.0.2"))
    first = store.activate(
        baseline,
        mode=ActivationMode.LIMITED_TRIAL,
        operator="tester",
        reason="capture baseline",
        expected_release_id=None,
    )
    with pytest.raises(InspectionReleaseConflictError):
        store.activate(
            candidate,
            mode=ActivationMode.LIMITED_TRIAL,
            operator="tester",
            reason="stale",
            expected_release_id=None,
        )
    store.activate(
        candidate,
        mode=ActivationMode.LIMITED_TRIAL,
        operator="tester",
        reason="trial",
        expected_release_id=first["release_id"],
    )
    restored = store.rollback(baseline.scope, operator="tester", reason="trial rejected")
    assert restored["release_id"] == baseline.release_id
    assert restored["previous_release_id"] == candidate.release_id


def test_resolver_caches_unchanged_release_and_revalidates_changed_artifact(tmp_path):
    store = InspectionReleaseStore(tmp_path / "store")
    release = store.commit(_release(tmp_path / "artifacts", color_escape=0.0))
    store.activate(
        release,
        mode=ActivationMode.FULL,
        operator="tester",
        reason="verified",
        expected_release_id=None,
    )
    resolver = InspectionReleaseResolver(store)
    assert resolver.resolve("Cable1", "A", "yolo") == release
    assert resolver.resolve("Cable1", "A", "yolo") == release
    Path(release.components[0].artifact_path).write_bytes(b"tampered")
    with pytest.raises(InspectionReleaseError, match="checksum"):
        resolver.resolve("Cable1", "A", "yolo")


def test_builder_binds_exact_matrix_combination(tmp_path):
    models_root = tmp_path / "models"
    model_path, model_sha = _write(models_root / "model.onnx", b"model")
    config_path, config_sha = _write(models_root / "v1.config.yaml", b"weights: model.onnx")
    scope_hash = "0123456789abcdef01234567"
    revision_id = str(uuid4())
    color_path, _ = _write(
        tmp_path / ".color_revisions" / scope_hash / revision_id / "config.json",
        b'{"black": 0.6}',
    )
    report = {
        "schema_version": 1,
        "run_id": "matrix-1",
        "product": "Cable1",
        "area": "A",
        "inference_type": "yolo",
        "sample_count": 250,
        "model_variants": [
            {
                "variant_id": "model-1",
                "models_root": str(models_root),
                "config_path": config_path,
                "weight_path": model_path,
                "identity": {
                    "version": "1.0.6",
                    "sha256": model_sha,
                    "runtime_config_sha256": config_sha,
                },
            }
        ],
        "color_variants": [
            {
                "variant_id": "color-1",
                "label": "black / color-v1.0.2",
                "revision_overrides": {scope_hash: revision_id},
                "include_active_revisions": False,
            }
        ],
        "combinations": [
            {
                "combination_id": "combo-1",
                "model_variant_id": "model-1",
                "color_variant_id": "color-1",
                "metrics": {"errors": 0, "fn": 0, "fp": 14},
                "color_metrics": {"errors": 0, "fn": 0, "escape_rate": None},
            }
        ],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    release = build_release_from_matrix(
        report_path,
        combination_id="combo-1",
        display_version="inspection-v1.0.2",
        operator="tester",
        reason="matrix candidate",
    )

    assert release.components[0].artifact_sha256 == model_sha
    assert release.components[1].config_path == str(Path(color_path).resolve())
    assert release.color_revision_overrides() == {scope_hash: revision_id}
    assert release.validation.color_escape_known is False

    draft = replace(
        release,
        status=ReleaseStatus.DRAFT,
        validation=ValidationEvidence(
            report_path="",
            report_sha256="",
            run_id="UNVALIDATED",
            sample_count=0,
            metrics=(("errors", None), ("fn", None), ("fp", None)),
            color_metrics=(("escape_rate", None),),
        ),
    )
    validated = build_validated_release_from_matrix(
        draft,
        report_path,
        combination_id="combo-1",
    )

    assert validated.release_id == draft.release_id
    assert validated.display_version == draft.display_version
    assert validated.components == draft.components
    assert validated.status is ReleaseStatus.TESTED
    assert validated.validation.sample_count == 250

    mismatched_report = tmp_path / "mismatched-report.json"
    mismatched_payload = json.loads(report_path.read_text(encoding="utf-8"))
    mismatched_payload["model_variants"][0]["identity"]["version"] = "9.9.9"
    mismatched_report.write_text(json.dumps(mismatched_payload), encoding="utf-8")
    with pytest.raises(InspectionReleaseError, match="不是選取的組合版本"):
        build_validated_release_from_matrix(
            draft,
            mismatched_report,
            combination_id="combo-1",
        )


def test_builder_binds_exact_full_color_baseline_from_matrix(tmp_path):
    models_root = tmp_path / "models"
    model_path, model_sha = _write(models_root / "model.onnx", b"model")
    embedded_path = models_root / "embedded-color.json"
    embedded_path.write_text(
        json.dumps({"summary": {"Black": {"count": 6}}}),
        encoding="utf-8",
    )
    config_path = models_root / "v1.config.yaml"
    config_path.write_text(
        "weights: model.onnx\n"
        "enable_color_check: true\n"
        "color_checker_type: stats\n"
        f"color_model_path: {embedded_path.as_posix()}\n",
        encoding="utf-8",
    )
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    candidate_path = tmp_path / ".color_baselines" / "candidate" / "color_stats.json"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(
        json.dumps({"summary": {"Black": {"count": 120}}}),
        encoding="utf-8",
    )
    candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    report = {
        "schema_version": 1,
        "run_id": "matrix-2",
        "product": "Cable1",
        "area": "A",
        "inference_type": "yolo",
        "sample_count": 250,
        "model_variants": [
            {
                "variant_id": "model-1",
                "models_root": str(models_root),
                "config_path": str(config_path),
                "weight_path": model_path,
                "identity": {
                    "version": "1.0.6",
                    "sha256": model_sha,
                    "runtime_config_sha256": config_sha,
                },
            }
        ],
        "color_variants": [
            {
                "variant_id": "color-base-candidate",
                "label": "完整顏色基準 / candidate",
                "revision_overrides": {},
                "include_active_revisions": False,
                "color_model_path": str(candidate_path),
                "color_model_sha256": candidate_sha,
            }
        ],
        "combinations": [
            {
                "combination_id": "combo-2",
                "model_variant_id": "model-1",
                "color_variant_id": "color-base-candidate",
                "metrics": {"errors": 0, "fn": 0, "fp": 0},
                "color_metrics": {
                    "errors": 0,
                    "fn": 0,
                    "escape_rate": None,
                },
            }
        ],
    }
    report_path = tmp_path / "report-2.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    release = build_release_from_matrix(
        report_path,
        combination_id="combo-2",
        display_version="inspection-v1.0.3",
        operator="tester",
        reason="full baseline candidate",
    )

    assert release.components[1].artifact_sha256 == candidate_sha
    assert release.components[1].artifact_path != str(candidate_path)
    assert Path(release.components[1].artifact_path).read_bytes() == (candidate_path.read_bytes())


def test_builder_rejects_symbolic_active_color_pointer(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "matrix-1",
                "product": "Cable1",
                "area": "A",
                "inference_type": "yolo",
                "sample_count": 1,
                "model_variants": [],
                "color_variants": [],
                "combinations": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(InspectionReleaseError):
        build_release_from_matrix(
            report_path,
            combination_id="missing",
            display_version="inspection-v1",
            operator="tester",
            reason="invalid",
        )


def test_engineering_composer_builds_unvalidated_draft_for_any_model_version(
    tmp_path,
):
    weight_path = tmp_path / "models" / "Cable1_A_v1.0.5.onnx"
    config_path = tmp_path / "models" / "Cable1_A_v1.0.5.onnx.config.yaml"
    _, weight_sha = _write(weight_path, b"older-model")
    _write(
        config_path,
        f"weights: {weight_path.as_posix()}\nenable_yolo: true\n".encode(),
    )
    record = ModelVersionRecord(
        product="Cable1",
        area="A",
        model_type="yolo",
        version="1.0.5",
        weight_path=weight_path,
        is_current=False,
        trained_at=None,
        deployed_at=None,
        activated_at=None,
        training_time_inferred=False,
        weight_sha256=weight_sha,
        config_snapshot_path=config_path,
        file_size=weight_path.stat().st_size,
    )

    release = build_draft_release(
        record,
        display_version="inspection-v1.0.3",
        operator="engineer",
        reason="Compare older YOLO with embedded color settings.",
    )
    store = InspectionReleaseStore(tmp_path / "release-store")
    store.commit(release)
    pointer = store.activate(
        release,
        mode=ActivationMode.RISK_ACCEPTED,
        operator="engineer",
        reason="Run selected YOLO 1.0.5 without matrix evidence.",
        expected_release_id=None,
    )

    assert release.status is ReleaseStatus.DRAFT
    assert release.validation.sample_count == 0
    assert release.validation.report_path == ""
    assert release.component_for_role("primary_detector").version == "1.0.5"
    assert pointer["mode"] == "RISK_ACCEPTED"
    assert InspectionReleaseResolver(store).resolve("Cable1", "A", "yolo") == release
