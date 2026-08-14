from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.services import acceptance_matrix
from core.services.acceptance_matrix import (
    AcceptanceColorVariant,
    AcceptanceMatrixError,
    AcceptanceMatrixRequest,
    AcceptanceModelVariant,
    build_model_variant,
    build_registered_model_variant,
    build_release_acceptance_variants,
    discover_color_variants,
    run_acceptance_matrix,
)
from core.services.color_baseline_recalibration import (
    ColorBaselineBuild,
    ColorBaselineCandidateStore,
    ColorBaselineOutlierFilterReport,
)
from core.services.color_profile_store import ColorProfileStore
from core.services.inspection_release_builder import build_draft_release
from core.services.model_acceptance import (
    AcceptanceInferenceOutcome,
    AcceptanceRecord,
    AcceptanceRepository,
    ModelIdentity,
)
from core.services.model_version_registry import ModelVersionRecord
from tools.color_calibration_service import (
    COLOR_CONFIG_SCHEMA_VERSION,
    ColorCalibrationScope,
)
from tools.color_configuration_revisions import (
    ColorConfigurationRevisionStore,
)

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)

#: A color model the stats checker can actually load. Discovery now withholds
#: one it cannot, so a fixture carrying only ``count`` would test that a stored
#: baseline is offered using a file that could never back an inference run.
_LOADABLE_STATS_PAYLOAD = {
    "summary": {
        "Black": {
            "count": 30,
            "hsv_min": [0.0, 0.0, 0.0],
            "hsv_max": [180.0, 40.0, 60.0],
            "lab_min": [0.0, 118.0, 118.0],
            "lab_max": [40.0, 138.0, 138.0],
        }
    }
}


def _write_image(path: Path, value: int) -> None:
    image = np.full((20, 24, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _fixture_repository(tmp_path: Path) -> AcceptanceRepository:
    repository = AcceptanceRepository(tmp_path / "acceptance")
    ok_path = tmp_path / "ok.png"
    ng_path = tmp_path / "ng.png"
    _write_image(ok_path, 50)
    _write_image(ng_path, 180)
    ok_record, ng_record = repository.import_images(
        (ok_path, ng_path),
        product="Cable1",
        area="A",
    )
    repository.confirm(
        ok_record.sample_id,
        verdict="OK",
        reviewed_by="reviewer",
    )
    repository.confirm(
        ng_record.sample_id,
        verdict="NG",
        reasons=("MISSING",),
        reviewed_by="reviewer",
    )
    return repository


class FakeInferenceService:
    created: list[dict[str, object]] = []
    closed = 0

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.created.append(kwargs)

    def infer(
        self,
        record: AcceptanceRecord,
        _image_path: str | Path,
        *,
        inference_type: str,
        cancel_cb=None,
    ) -> AcceptanceInferenceOutcome:
        assert inference_type == "yolo"
        assert not cancel_cb or not cancel_cb()
        identity = self.kwargs["model_identity"]
        assert isinstance(identity, ModelIdentity)
        overrides = self.kwargs["color_revision_overrides"]
        assert isinstance(overrides, dict)
        is_ok_truth = record.expected_verdict == "OK"
        candidate_rejects_ok = bool(overrides) and is_ok_truth
        second_model_misses_ng = identity.version == "v2" and not is_ok_truth
        if candidate_rejects_ok:
            machine_status = "NG"
            reasons = ("COLOR_MISMATCH",)
            color_status = "FAIL"
        elif second_model_misses_ng:
            machine_status = "OK"
            reasons = ()
            color_status = "PASS"
        else:
            machine_status = record.expected_verdict
            reasons = () if is_ok_truth else ("MISSING",)
            color_status = "PASS"
        return AcceptanceInferenceOutcome(
            sample_id=record.sample_id,
            machine_status=machine_status,
            machine_reasons=reasons,
            model_version=identity.version,
            model_sha256=identity.sha256,
            runtime_config_sha256=identity.runtime_config_sha256,
            color_model_sha256=identity.color_model_sha256,
            inference_at=NOW.isoformat(),
            latency_ms=10.0 if identity.version == "v1" else 20.0,
            error="",
            color_check_status=color_status,
            color_details_json="[]",
        )

    def close(self) -> None:
        type(self).closed += 1


def _request(
    tmp_path: Path,
    repository: AcceptanceRepository,
) -> AcceptanceMatrixRequest:
    (tmp_path / "config.yaml").write_text("device: cpu\n", encoding="utf-8")
    model_paths: list[tuple[Path, Path, Path]] = []
    for version in ("v1", "v2"):
        root = tmp_path / f"models-{version}"
        config_path = root / "Cable1" / "A" / "yolo" / "config.yaml"
        config_path.parent.mkdir(parents=True)
        weight_path = config_path.parent / f"{version}.onnx"
        color_path = config_path.parent / f"{version}-color.json"
        weight_path.write_bytes(version.encode("utf-8"))
        color_path.write_text('{"summary": {}}', encoding="utf-8")
        config_path.write_text(
            f"enable_yolo: true\n"
            f"weights: {weight_path.as_posix()}\n"
            "enable_color_check: true\n"
            "color_checker_type: stats\n"
            f"color_model_path: {color_path.as_posix()}\n",
            encoding="utf-8",
        )
        model_paths.append((root, config_path, weight_path))
    return AcceptanceMatrixRequest(
        project_root=tmp_path,
        global_config_path=tmp_path / "config.yaml",
        color_revisions_root=tmp_path / ".color_revisions",
        dataset_root=repository.root,
        manifest_path=repository.manifest_path,
        output_root=tmp_path / "matrix-reports",
        product="Cable1",
        area="A",
        inference_type="yolo",
        model_variants=(
            AcceptanceModelVariant(
                "model-v1",
                "YOLO v1",
                model_paths[0][0],
                ModelIdentity("v1", ""),
                model_paths[0][1],
                model_paths[0][2],
            ),
            AcceptanceModelVariant(
                "model-v2",
                "YOLO v2",
                model_paths[1][0],
                ModelIdentity("v2", ""),
                model_paths[1][1],
                model_paths[1][2],
            ),
        ),
        color_variants=(
            AcceptanceColorVariant("embedded", "內建"),
            AcceptanceColorVariant(
                "candidate",
                "Black / color-v1.0.2",
                (("black-scope", "revision-2"),),
            ),
        ),
    )


def test_matrix_runs_cartesian_product_and_preserves_ground_truth(
    tmp_path: Path,
) -> None:
    repository = _fixture_repository(tmp_path)
    request = _request(tmp_path, repository)
    truth_before = hashlib.sha256(repository.manifest_path.read_bytes()).hexdigest()
    progress: list[tuple[int, int, str, str]] = []
    FakeInferenceService.created = []
    FakeInferenceService.closed = 0

    result = run_acceptance_matrix(
        request,
        service_factory=FakeInferenceService,
        progress_callback=lambda *args: progress.append(args),
        clock=lambda: NOW,
        id_generator=lambda: "run-001",
    )

    truth_after = hashlib.sha256(repository.manifest_path.read_bytes()).hexdigest()
    assert truth_after == truth_before == result.manifest_sha256
    assert result.sample_count == 2
    assert len(result.combinations) == 4
    assert len(FakeInferenceService.created) == 4
    assert all(
        created["model_weight_path_override"] is not None
        for created in FakeInferenceService.created
    )
    assert FakeInferenceService.closed == 4
    assert progress[-1][:2] == (8, 8)
    assert result.run_root.name == "matrix-20260730T120000Z-run-001"
    assert result.report_path.is_file()
    assert result.summary_csv_path.is_file()
    assert result.samples_csv_path.is_file()

    baseline = result.combinations[0]
    candidate = result.combinations[1]
    second_model = result.combinations[2]
    assert (baseline.metrics.tp, baseline.metrics.tn) == (1, 1)
    assert baseline.color_metrics.escape_rate is None
    assert baseline.color_metrics.overkill_rate == 0.0
    assert baseline.color_metrics.unknown_truth == 1
    assert candidate.metrics.fp == 1
    assert candidate.color_metrics.fp == 1
    assert candidate.changed_from_reference == 1
    assert second_model.metrics.fn == 1
    assert second_model.average_latency_ms == 20.0
    assert second_model.p95_latency_ms == 20.0

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["reference_combination_id"] == baseline.combination_id
    assert report["sample_count"] == 2
    assert len(report["samples"]) == 8
    first_bundle = report["combinations"][0]["artifact_bundle"]
    assert first_bundle["color_model_mode"] == "embedded"
    assert len(first_bundle["color_model"]["sha256"]) == 64
    with result.summary_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert len(summary_rows) == 4
    assert summary_rows[0]["color_escape_rate"] == ""


def test_matrix_reports_are_append_only(tmp_path: Path) -> None:
    repository = _fixture_repository(tmp_path)
    request = _request(tmp_path, repository)
    kwargs = {
        "service_factory": FakeInferenceService,
        "clock": lambda: NOW,
        "id_generator": lambda: "same-run",
    }

    run_acceptance_matrix(request, **kwargs)

    with pytest.raises(AcceptanceMatrixError, match="已存在"):
        run_acceptance_matrix(request, **kwargs)


def test_matrix_rejects_changed_image_evidence(tmp_path: Path) -> None:
    repository = _fixture_repository(tmp_path)
    record = repository.records()[0]
    _write_image(repository.image_file(record), 99)

    with pytest.raises(AcceptanceMatrixError, match="雜湊不符"):
        run_acceptance_matrix(
            _request(tmp_path, repository),
            service_factory=FakeInferenceService,
        )


def test_matrix_rejects_config_and_selected_weight_mismatch(
    tmp_path: Path,
) -> None:
    repository = _fixture_repository(tmp_path)
    request = _request(tmp_path, repository)
    mismatched_weight = tmp_path / "wrong.onnx"
    mismatched_weight.write_bytes(b"wrong")
    first = replace(request.model_variants[0], weight_path=mismatched_weight)

    with pytest.raises(AcceptanceMatrixError, match="do not resolve"):
        run_acceptance_matrix(
            replace(
                request,
                model_variants=(first, *request.model_variants[1:]),
            ),
            service_factory=FakeInferenceService,
        )


def test_matrix_rejects_unwritable_report_target_before_inference(
    tmp_path: Path,
) -> None:
    repository = _fixture_repository(tmp_path)
    request = _request(tmp_path, repository)
    blocked_output = tmp_path / "report-is-a-file"
    blocked_output.write_text("not a directory", encoding="utf-8")
    FakeInferenceService.created = []

    with pytest.raises(AcceptanceMatrixError, match="不可寫入"):
        run_acceptance_matrix(
            replace(request, output_root=blocked_output),
            service_factory=FakeInferenceService,
        )

    assert FakeInferenceService.created == []


def test_registered_historical_model_is_read_only_and_hashes_at_run_time(
    tmp_path: Path,
) -> None:
    weight_path = tmp_path / "Cable1_A_v1.0.5.onnx"
    weight_path.write_bytes(b"historical-weight")
    config_path = tmp_path / "Cable1_A_v1.0.5.onnx.config.yaml"
    config_path.write_text(
        f"enable_yolo: true\nweights: {weight_path.as_posix()}\n",
        encoding="utf-8",
    )
    record = ModelVersionRecord(
        product="Cable1",
        area="A",
        model_type="yolo",
        version="1.0.5",
        weight_path=weight_path,
        is_current=False,
        trained_at=NOW,
        deployed_at=NOW,
        activated_at=None,
        training_time_inferred=False,
        config_snapshot_path=config_path,
        file_size=weight_path.stat().st_size,
    )

    variant = build_registered_model_variant(
        record,
        models_root=tmp_path / "models",
    )

    assert variant.label == "YOLO 1.0.5"
    assert variant.weight_path == weight_path
    assert variant.config_path == config_path
    assert variant.identity.sha256 == ""
    assert len(variant.identity.runtime_config_sha256) == 64


def test_registered_current_model_uses_effective_station_config(
    tmp_path: Path,
) -> None:
    models_root = tmp_path / "models"
    station_root = models_root / "Cable1" / "A" / "yolo"
    station_root.mkdir(parents=True)
    weight_path = station_root / "Cable1_A_v1.0.6.onnx"
    weight_path.write_bytes(b"current-weight")
    snapshot_path = station_root / "versions" / "Cable1_A_v1.0.6.config.yaml"
    snapshot_path.parent.mkdir()
    snapshot_path.write_text(
        f"weights: {weight_path.as_posix()}\ncalibration: old\n",
        encoding="utf-8",
    )
    active_config_path = station_root / "config.yaml"
    active_config_path.write_text(
        f"weights: {weight_path.as_posix()}\ncalibration: current\n",
        encoding="utf-8",
    )
    record = ModelVersionRecord(
        product="Cable1",
        area="A",
        model_type="yolo",
        version="1.0.6",
        weight_path=weight_path,
        is_current=True,
        trained_at=NOW,
        deployed_at=NOW,
        activated_at=NOW,
        training_time_inferred=False,
        config_snapshot_path=snapshot_path,
        file_size=weight_path.stat().st_size,
    )

    variant = build_registered_model_variant(record, models_root=models_root)
    main_window_variant = build_model_variant(
        models_root,
        product="Cable1",
        area="A",
        inference_type="yolo",
    )

    assert variant.config_path == active_config_path.resolve()
    assert variant.config_path == main_window_variant.config_path
    assert variant.identity.runtime_config_sha256 == hashlib.sha256(
        active_config_path.read_bytes()
    ).hexdigest()
    assert (
        variant.identity.runtime_config_sha256
        == main_window_variant.identity.runtime_config_sha256
    )


def test_release_quick_validation_pair_uses_exact_draft_artifacts(
    tmp_path: Path,
) -> None:
    weight_path = tmp_path / "history" / "Cable1_A_v1.0.5.onnx"
    weight_path.parent.mkdir(parents=True)
    weight_path.write_bytes(b"historical-weight")
    config_path = weight_path.with_suffix(".config.yaml")
    config_path.write_text(
        f"enable_yolo: true\nweights: {weight_path.as_posix()}\n",
        encoding="utf-8",
    )
    record = ModelVersionRecord(
        product="Cable1",
        area="A",
        model_type="yolo",
        version="1.0.5",
        weight_path=weight_path,
        is_current=False,
        trained_at=NOW,
        deployed_at=NOW,
        activated_at=None,
        training_time_inferred=False,
        config_snapshot_path=config_path,
        file_size=weight_path.stat().st_size,
    )
    draft = build_draft_release(
        record,
        display_version="inspection-v1.0.4",
        operator="engineer",
        reason="quick validation target",
    )

    model, color = build_release_acceptance_variants(
        draft,
        project_root=tmp_path,
    )

    assert model.weight_path == weight_path.resolve()
    assert model.config_path == config_path.resolve()
    assert model.identity.version == "1.0.5"
    assert model.identity.sha256 == draft.components[0].artifact_sha256
    assert color.revision_overrides == ()
    assert color.color_model_path is None
    assert color.include_active_revisions is False


def test_color_discovery_lists_embedded_active_and_exact_revision(
    tmp_path: Path,
) -> None:
    scope = ColorCalibrationScope(
        "Cable1",
        "A",
        "yolo",
        "stats",
        "black",
    )
    revisions_root = tmp_path / ".color_revisions"
    store = ColorConfigurationRevisionStore(
        root=revisions_root,
        clock=lambda: NOW,
        id_generator=lambda: "revision-1",
    )
    revision = store.commit_configuration(
        "test-source",
        scope,
        operator="reviewer",
        reason="matrix candidate",
        proposal_sha256="proposal",
        preview_sha256="preview",
        proposed_config={
            "schema_version": COLOR_CONFIG_SCHEMA_VERSION,
            "scope": {
                "product": scope.product,
                "area": scope.area,
                "model_type": scope.model_type,
                "checker_type": scope.checker_type,
                "threshold_key": scope.threshold_key,
                "scope_hash": scope.scope_hash,
            },
            "threshold_key": scope.threshold_key,
            "checker_type": scope.checker_type,
            "public_threshold": 0.6,
            "config_value": 0.6,
        },
        metrics={},
        parent_config_sha256="base-sha",
    )
    before_activation = discover_color_variants(
        revisions_root,
        product="Cable1",
        area="A",
        model_type="yolo",
    ).variants
    store.activate(
        revision,
        operator="reviewer",
        reason="production approved",
        expected_current_sha256="base-sha",
    )
    after_activation = discover_color_variants(
        revisions_root,
        product="Cable1",
        area="A",
        model_type="yolo",
    ).variants

    assert [item.variant_id for item in before_activation] == [
        "color-embedded",
        "color-black-color-v1.0.1",
    ]
    assert [item.variant_id for item in after_activation] == [
        "color-embedded",
        "color-active",
        "color-black-color-v1.0.1",
    ]
    assert after_activation[-1].revision_overrides == ((scope.scope_hash, revision.revision_id),)

    color_model = tmp_path / "models" / "Cable1" / "A" / "yolo" / "color_stats.json"
    color_model.parent.mkdir(parents=True)
    color_model.write_text(
        json.dumps(_LOADABLE_STATS_PAYLOAD),
        encoding="utf-8",
    )
    config = color_model.with_name("config.yaml")
    config.write_text(
        "enable_color_check: true\n"
        "color_checker_type: stats\n"
        f"color_model_path: {color_model.as_posix()}\n",
        encoding="utf-8",
    )
    profile = ColorProfileStore(tmp_path / ".color_profiles").create(
        product="Cable1",
        area="A",
        model_type="yolo",
        model_config_path=config,
        project_root=tmp_path,
        revisions=(revision,),
    )
    assert profile is not None
    with_profiles = discover_color_variants(
        revisions_root,
        product="Cable1",
        area="A",
        model_type="yolo",
        profiles_root=tmp_path / ".color_profiles",
    ).variants
    profile_variant = next(
        item
        for item in with_profiles
        if item.variant_id == f"color-profile-{profile.package_id}"
    )
    assert profile_variant.color_model_path == profile.color_model_path
    assert profile_variant.revision_overrides == profile.revision_overrides


def test_color_discovery_ignores_revision_store_infrastructure_directories(
    tmp_path: Path,
) -> None:
    revisions_root = tmp_path / ".color_revisions"
    for directory_name in (
        "active",
        "activation_events",
        "future-infrastructure",
        "locks",
        "revocations",
    ):
        infrastructure_root = revisions_root / directory_name
        infrastructure_root.mkdir(parents=True)
        (infrastructure_root / "revision.json").write_text(
            "{}",
            encoding="utf-8",
        )

    variants = discover_color_variants(
        revisions_root,
        product="Cable1",
        area="A",
        model_type="yolo",
    ).variants

    assert [variant.variant_id for variant in variants] == ["color-embedded"]


def test_color_discovery_includes_immutable_baseline_candidate(
    tmp_path: Path,
) -> None:
    candidate = ColorBaselineCandidateStore(tmp_path / ".color_baselines").commit(
        product="Cable1",
        area="A",
        model_type="yolo",
        build=ColorBaselineBuild(
            status="INCOMPLETE",
            model_payload=_LOADABLE_STATS_PAYLOAD,
            report_payload={
                "status": "INCOMPLETE",
                "color_reports": [],
            },
            evidence_sha256="e" * 64,
            color_reports=(),
            outlier_filter=ColorBaselineOutlierFilterReport(
                status="NOT_RUN",
                total_sample_count=0,
                z_score_threshold=6.0,
                maximum_auto_exclusion_fraction=0.1,
                candidate_sample_ids=(),
                excluded_sample_ids=(),
                findings=(),
            ),
        ),
    )

    variants = discover_color_variants(
        tmp_path / ".color_revisions",
        product="Cable1",
        area="A",
        model_type="yolo",
        baselines_root=tmp_path / ".color_baselines",
    ).variants

    baseline = variants[1]
    assert baseline.variant_id == f"color-base-{candidate.candidate_id}"
    assert baseline.color_model_path == candidate.color_model_path
    assert baseline.color_model_sha256 == candidate.color_model_sha256
    assert "INCOMPLETE" in baseline.label


def _commit_baseline_candidate(root: Path):
    """Commit one minimal in-scope baseline candidate for Cable1/A/yolo."""
    return ColorBaselineCandidateStore(root).commit(
        product="Cable1",
        area="A",
        model_type="yolo",
        build=ColorBaselineBuild(
            status="INCOMPLETE",
            model_payload=_LOADABLE_STATS_PAYLOAD,
            report_payload={"status": "INCOMPLETE", "color_reports": []},
            evidence_sha256="e" * 64,
            color_reports=(),
            outlier_filter=ColorBaselineOutlierFilterReport(
                status="NOT_RUN",
                total_sample_count=0,
                z_score_threshold=6.0,
                maximum_auto_exclusion_fraction=0.1,
                candidate_sample_ids=(),
                excluded_sample_ids=(),
                findings=(),
            ),
        ),
    )


def test_incompatible_baseline_is_reported_as_an_exclusion_not_dropped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A superseded-algorithm candidate must be explained, not made invisible.

    It stays unselectable because its crop coordinate space differs, but an
    unexplained disappearance reads as data loss and invites rebuilding a
    baseline that is not actually missing.
    """
    candidate = _commit_baseline_candidate(tmp_path / ".color_baselines")
    monkeypatch.setattr(acceptance_matrix, "ALGORITHM_VERSION", "stats-robust-v99")

    discovery = discover_color_variants(
        tmp_path / ".color_revisions",
        product="Cable1",
        area="A",
        model_type="yolo",
        baselines_root=tmp_path / ".color_baselines",
    )

    assert [variant.variant_id for variant in discovery.variants] == ["color-embedded"]
    assert discovery.stored_color_models == ()
    assert len(discovery.exclusions) == 1
    exclusion = discovery.exclusions[0]
    assert candidate.display_version in exclusion.label
    assert candidate.algorithm in exclusion.reason
    assert "stats-robust-v99" in exclusion.reason


def test_unloadable_baseline_is_excluded_instead_of_offered(tmp_path: Path) -> None:
    """A stored model the checker cannot load must never reach a matrix run.

    Offering it costs an entire combination: the inference service turns each
    per-image failure into an ERROR outcome and carries on, so all samples come
    back ERROR after the whole set has been inferred. Reported as an exclusion
    the operator sees the reason before spending the run.
    """
    candidate = ColorBaselineCandidateStore(tmp_path / ".color_baselines").commit(
        product="Cable1",
        area="A",
        model_type="yolo",
        build=ColorBaselineBuild(
            status="READY",
            # Only a count: no hsv/lab ranges, so StatsColorChecker cannot load it.
            model_payload={"summary": {"Black": {"count": 30}}},
            report_payload={"status": "READY", "color_reports": []},
            evidence_sha256="e" * 64,
            color_reports=(),
            outlier_filter=ColorBaselineOutlierFilterReport(
                status="NOT_RUN",
                total_sample_count=0,
                z_score_threshold=6.0,
                maximum_auto_exclusion_fraction=0.1,
                candidate_sample_ids=(),
                excluded_sample_ids=(),
                findings=(),
            ),
        ),
    )

    discovery = discover_color_variants(
        tmp_path / ".color_revisions",
        product="Cable1",
        area="A",
        model_type="yolo",
        baselines_root=tmp_path / ".color_baselines",
    )

    assert discovery.stored_color_models == ()
    assert len(discovery.exclusions) == 1
    exclusion = discovery.exclusions[0]
    assert candidate.display_version in exclusion.label
    assert "hsv_min" in exclusion.reason


def test_incomplete_but_loadable_baseline_is_still_offered(tmp_path: Path) -> None:
    """Status alone must not withhold a baseline.

    An unfinished recalibration can still hold usable statistics for the colors
    it did finish, and the operator may legitimately want that comparison. Only
    whether the file loads decides whether it is offered.
    """
    _commit_baseline_candidate(tmp_path / ".color_baselines")

    discovery = discover_color_variants(
        tmp_path / ".color_revisions",
        product="Cable1",
        area="A",
        model_type="yolo",
        baselines_root=tmp_path / ".color_baselines",
    )

    assert discovery.exclusions == ()
    assert len(discovery.stored_color_models) == 1
    assert "INCOMPLETE" in discovery.stored_color_models[0].label


def test_compatible_baseline_produces_no_exclusion(tmp_path: Path) -> None:
    """A usable candidate must not be listed as withheld."""
    _commit_baseline_candidate(tmp_path / ".color_baselines")

    discovery = discover_color_variants(
        tmp_path / ".color_revisions",
        product="Cable1",
        area="A",
        model_type="yolo",
        baselines_root=tmp_path / ".color_baselines",
    )

    assert discovery.exclusions == ()
    assert len(discovery.stored_color_models) == 1


def test_out_of_scope_baseline_is_not_reported_as_an_exclusion(
    tmp_path: Path,
) -> None:
    """Another station's candidate was never a candidate here.

    Reporting it would bury the exclusions that actually need explaining.
    """
    _commit_baseline_candidate(tmp_path / ".color_baselines")

    discovery = discover_color_variants(
        tmp_path / ".color_revisions",
        product="LED",
        area="A",
        model_type="yolo",
        baselines_root=tmp_path / ".color_baselines",
    )

    assert discovery.exclusions == ()
    assert discovery.stored_color_models == ()
