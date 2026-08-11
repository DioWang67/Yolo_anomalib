from __future__ import annotations

import hashlib
import json
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from core.detection_system import DetectionSystem
from core.services.acceptance_gate import (
    AcceptanceGatePolicy,
    run_candidate_acceptance,
)
from core.services.model_acceptance import (
    AcceptanceDataError,
    AcceptanceInferenceOutcome,
    AcceptanceInferenceService,
    AcceptanceRecord,
    AcceptanceRepository,
    ModelIdentity,
    calculate_acceptance_metrics,
    color_evidence,
    load_model_identity,
    machine_reason_codes,
)
from core.types import DetectionItem, DetectionResult


def _write_image(path: Path, value: int = 127) -> None:
    image = np.full((24, 32, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_repository_imports_once_and_persists_confirmation(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _write_image(source)
    repository = AcceptanceRepository(tmp_path / "acceptance")

    first = repository.import_images([source], product="Cable1", area="A")
    duplicate = repository.import_images([source], product="Cable1", area="A")
    confirmed = repository.confirm(
        first[0].sample_id,
        verdict="NG",
        reasons=("POSITION_SHIFT", "MISSING"),
        reviewed_by="operator-1",
        defect_class="Black",
        notes="confirmed on fixture",
    )

    assert duplicate[0].sample_id == first[0].sample_id
    assert len(repository.records()) == 1
    assert repository.image_file(confirmed).is_file()
    assert confirmed.expected_verdict == "NG"
    assert confirmed.expected_reasons == "MISSING|POSITION_SHIFT"
    assert confirmed.review_status == "confirmed"


def test_repository_rejects_ng_without_reason_and_ok_clears_reason(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    _write_image(source)
    repository = AcceptanceRepository(tmp_path / "acceptance")
    record = repository.import_images([source], product="Cable1", area="A")[0]

    with pytest.raises(AcceptanceDataError, match="reason"):
        repository.confirm(
            record.sample_id,
            verdict="NG",
            reviewed_by="operator-1",
        )

    confirmed = repository.confirm(
        record.sample_id,
        verdict="OK",
        reasons=("MISSING",),
        reviewed_by="operator-1",
        defect_class="ignored",
    )
    assert confirmed.expected_reasons == ""
    assert confirmed.defect_class == ""


def test_metrics_exclude_pending_and_error_from_confusion_denominators() -> None:
    def record(sample_id: str, truth: str, prediction: str) -> AcceptanceRecord:
        return AcceptanceRecord(
            sample_id=sample_id,
            image_path=f"images/{sample_id}.png",
            image_sha256=sample_id,
            product="Cable1",
            area="A",
            expected_verdict=truth,
            review_status="confirmed",
            machine_status=prediction,
        )

    records = (
        record("tp", "NG", "NG"),
        record("fp", "OK", "NG"),
        record("fn", "NG", "OK"),
        record("tn", "OK", "OK"),
        record("error", "NG", "ERROR"),
        AcceptanceRecord(
            sample_id="pending",
            image_path="images/pending.png",
            image_sha256="pending",
            product="Cable1",
            area="A",
        ),
    )

    metrics = calculate_acceptance_metrics(records)

    assert (metrics.tp, metrics.fp, metrics.fn, metrics.tn) == (1, 1, 1, 1)
    assert metrics.errors == 1
    assert metrics.confirmed == 5
    assert metrics.pending == 1
    assert metrics.true_yield == pytest.approx(0.4)
    assert metrics.machine_yield == pytest.approx(0.5)
    assert metrics.escape_rate == pytest.approx(0.5)
    assert metrics.overkill_rate == pytest.approx(0.5)


def test_machine_reason_codes_normalize_pipeline_evidence() -> None:
    result = DetectionResult(
        status="DETECTION_FAIL",
        missing_items=["Black"],
        unexpected_items=["Red"],
        items=[
            DetectionItem(
                label="Green",
                confidence=0.9,
                bbox_xyxy=(1, 2, 3, 4),
                metadata={"position_status": "WRONG"},
            )
        ],
        color_check={"is_ok": False},
        sequence_check={"is_ok": False},
    )

    assert machine_reason_codes(result) == (
        "COLOR_MISMATCH",
        "MISSING",
        "POSITION_SHIFT",
        "SEQUENCE_MISMATCH",
        "UNEXPECTED_COMPONENT",
    )


def test_color_evidence_keeps_per_item_diff_and_threshold() -> None:
    result = DetectionResult(
        status="DETECTION_FAIL",
        color_check={
            "is_ok": False,
            "items": [
                {
                    "index": 2,
                    "class_name": "Red",
                    "best_color": "Orange",
                    "diff": 0.42,
                    "threshold": 0.30,
                    "is_ok": False,
                }
            ],
        },
    )

    status, items = color_evidence(result)

    assert status == "FAIL"
    assert items == (
        {
            "index": 2,
            "detector_class": "Red",
            "predicted_color": "Orange",
            "diff": 0.42,
            "threshold": 0.30,
            "is_ok": False,
        },
    )


def test_snapshot_compare_and_zip_backup_preserve_human_truth(
    tmp_path: Path,
) -> None:
    source_ok = tmp_path / "ok.png"
    source_ng = tmp_path / "ng.png"
    _write_image(source_ok, 80)
    _write_image(source_ng, 180)
    repository = AcceptanceRepository(tmp_path / "acceptance")
    ok_record, ng_record = repository.import_images([source_ok, source_ng], product="Cable1", area="A")
    repository.confirm(
        ok_record.sample_id,
        verdict="OK",
        reviewed_by="operator-1",
    )
    repository.confirm(
        ng_record.sample_id,
        verdict="NG",
        reasons=("MISSING",),
        reviewed_by="operator-1",
    )
    repository.save_inference(
        AcceptanceInferenceOutcome(
            sample_id=ok_record.sample_id,
            machine_status="NG",
            machine_reasons=("COLOR_MISMATCH",),
            model_version="1.0.6",
            model_sha256="old",
            inference_at="2026-07-30T10:00:00+08:00",
            latency_ms=10.0,
            error="",
            color_check_status="FAIL",
            color_details_json="[]",
        )
    )
    repository.save_inference(
        AcceptanceInferenceOutcome(
            sample_id=ng_record.sample_id,
            machine_status="NG",
            machine_reasons=("MISSING",),
            model_version="1.0.6",
            model_sha256="old",
            inference_at="2026-07-30T10:00:00+08:00",
            latency_ms=10.0,
            error="",
        )
    )
    snapshot = repository.create_snapshot(label="baseline-v1.0.6")

    repository.save_inference(
        AcceptanceInferenceOutcome(
            sample_id=ok_record.sample_id,
            machine_status="OK",
            machine_reasons=(),
            model_version="1.0.7",
            model_sha256="new",
            inference_at="2026-07-31T10:00:00+08:00",
            latency_ms=9.0,
            error="",
            color_check_status="PASS",
            color_details_json=json.dumps(
                [
                    {
                        "index": 0,
                        "detector_class": "Red",
                        "predicted_color": "Red",
                        "diff": 0.1,
                        "threshold": 0.3,
                        "is_ok": True,
                    }
                ]
            ),
        )
    )
    comparison = repository.compare_with_snapshot(snapshot)
    backup = repository.export_backup_zip(tmp_path / "backup.zip")

    assert comparison.improved == 1
    assert comparison.regressed == 0
    assert comparison.baseline_false_positives == 1
    assert comparison.current_false_positives == 0
    assert repository.records()[0].expected_verdict == "OK"
    assert repository.records()[0].reviewed_by == "operator-1"
    assert snapshot.manifest_path.is_file()
    with zipfile.ZipFile(backup) as archive:
        names = set(archive.namelist())
    assert "ground_truth.csv" in names
    assert any(name.startswith("images/") for name in names)
    assert any(name.startswith("snapshots/") for name in names)


def test_load_model_identity_uses_deployment_manifest(tmp_path: Path) -> None:
    model_dir = tmp_path / "Cable1" / "A" / "yolo"
    model_dir.mkdir(parents=True)
    (model_dir / "config.yaml").write_text("conf_thres: 0.4\n", encoding="utf-8")
    (model_dir / "deployment_manifest.yaml").write_text(
        "deployed_version: 1.2.3\nweight_sha256: AABBCC\ncolor_model_sha256: DDEEFF\n",
        encoding="utf-8",
    )

    identity = load_model_identity(tmp_path, "Cable1", "A", "yolo")

    assert identity.version == "1.2.3"
    assert identity.sha256 == "aabbcc"
    assert len(identity.runtime_config_sha256) == 64
    assert identity.color_model_sha256 == "ddeeff"


def test_inference_service_calls_production_core_without_persistence(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sample.png"
    _write_image(image_path)
    fake_system = Mock()
    fake_system.detect.return_value = DetectionResult(
        status="PASS",
        latency=0.012,
        color_check={"is_ok": True, "items": []},
        result_frame=np.zeros((24, 32, 3), dtype=np.uint8),
    )
    system_factory = Mock(return_value=fake_system)

    service = AcceptanceInferenceService(
        project_root=tmp_path,
        system_factory=system_factory,
    )
    record = AcceptanceRecord(
        sample_id="sample",
        image_path="images/sample.png",
        image_sha256="abc",
        product="Cable1",
        area="A",
    )

    outcome = service.infer(record, image_path, inference_type="yolo")
    service.close()

    assert outcome.machine_status == "OK"
    assert outcome.color_check_status == "PASS"
    assert system_factory.call_args.kwargs["initialize_camera"] is False
    fake_system.detect.assert_called_once()
    assert fake_system.detect.call_args.kwargs["persist"] is False
    fake_system.shutdown.assert_called_once()


def test_inference_service_stages_color_baseline_override_without_mutation(
    tmp_path: Path,
) -> None:
    config = tmp_path / "model.config.yaml"
    config.write_text(
        "enable_color_check: true\ncolor_checker_type: stats\ncolor_model_path: original.json\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "candidate.json"
    baseline.write_text(
        '{"summary":{"Black":{"count":30}}}',
        encoding="utf-8",
    )
    fake_system = Mock()
    system_factory = Mock(return_value=fake_system)

    service = AcceptanceInferenceService(
        project_root=tmp_path,
        system_factory=system_factory,
        model_identity=ModelIdentity("1.0.6", "model-sha"),
        model_config_overrides={
            ("Cable1", "A", "yolo"): config,
        },
        color_model_path_override=baseline,
    )
    staged = Path(system_factory.call_args.kwargs["model_config_overrides"][("Cable1", "A", "yolo")])

    assert staged != config
    assert str(baseline.resolve()) in staged.read_text(encoding="utf-8")
    assert "original.json" in config.read_text(encoding="utf-8")
    assert service._model_identity is not None
    assert service._model_identity.color_model_sha256 == hashlib.sha256(baseline.read_bytes()).hexdigest()
    service.close()
    assert not staged.exists()


def test_detection_system_forwards_non_persistent_mode() -> None:
    system = object.__new__(DetectionSystem)
    system._pipeline = SimpleNamespace(running=False)
    system._inference_lock = threading.RLock()
    system._detect_locked = Mock(return_value=DetectionResult(status="PASS", product="Cable1", area="A"))

    result = system.detect(
        "Cable1",
        "A",
        "yolo",
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        persist=False,
    )

    assert result.status == "PASS"
    assert system._detect_locked.call_args.kwargs["persist"] is False


def test_headless_gate_uses_snapshot_without_mutating_human_truth(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "acceptance"
    source_ok = tmp_path / "ok.png"
    source_ng = tmp_path / "ng.png"
    _write_image(source_ok, 60)
    _write_image(source_ng, 180)
    repository = AcceptanceRepository(dataset_root)
    ok_record, ng_record = repository.import_images(
        [source_ok, source_ng],
        product="Cable1",
        area="A",
    )
    repository.confirm(
        ok_record.sample_id,
        verdict="OK",
        reviewed_by="operator-1",
    )
    repository.confirm(
        ng_record.sample_id,
        verdict="NG",
        reasons=("MISSING",),
        reviewed_by="operator-1",
    )
    for record, status in ((ok_record, "OK"), (ng_record, "NG")):
        repository.save_inference(
            AcceptanceInferenceOutcome(
                sample_id=record.sample_id,
                machine_status=status,
                machine_reasons=(() if status == "OK" else ("MISSING",)),
                model_version="1.0.6",
                model_sha256="baseline",
                inference_at="2026-07-30T10:00:00+08:00",
                latency_ms=1.0,
                error="",
            )
        )
    snapshot = repository.create_snapshot(label="frozen")
    truth_before = hashlib.sha256(repository.manifest_path.read_bytes()).hexdigest()

    class FakeService:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.closed = False

        def infer(self, record, image_path, *, inference_type):
            return AcceptanceInferenceOutcome(
                sample_id=record.sample_id,
                machine_status=record.expected_verdict,
                machine_reasons=(() if record.expected_verdict == "OK" else ("MISSING",)),
                model_version="candidate",
                model_sha256="candidate-sha",
                runtime_config_sha256="config-sha",
                color_model_sha256="color-sha",
                inference_at="2026-07-31T10:00:00+08:00",
                latency_ms=2.0,
                error="",
            )

        def close(self) -> None:
            self.closed = True

    report_path = tmp_path / "run" / "model_acceptance_gate.json"
    result = run_candidate_acceptance(
        project_root=tmp_path,
        models_root=tmp_path / "candidate_models",
        global_config_path=tmp_path / "config.yaml",
        color_revisions_root=tmp_path / ".color_revisions",
        dataset_root=dataset_root,
        snapshot_manifest_path=snapshot.manifest_path,
        report_path=report_path,
        product="Cable1",
        area="A",
        inference_type="yolo",
        model_identity=ModelIdentity(
            version="candidate",
            sha256="candidate-sha",
            runtime_config_sha256="config-sha",
            color_model_sha256="color-sha",
        ),
        policy=AcceptanceGatePolicy(
            min_confirmed=2,
            max_false_positives=0,
            max_false_negatives=0,
        ),
        service_factory=FakeService,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    truth_after = hashlib.sha256(repository.manifest_path.read_bytes()).hexdigest()
    assert result.passed is True
    assert report["metrics"]["accuracy"] == 1.0
    assert report["comparison"]["regressed"] == 0
    assert truth_after == truth_before


def test_headless_gate_blocks_a_sample_regression(tmp_path: Path) -> None:
    source = tmp_path / "ok.png"
    _write_image(source)
    repository = AcceptanceRepository(tmp_path / "acceptance")
    record = repository.import_images(
        [source],
        product="Cable1",
        area="A",
    )[0]
    repository.confirm(
        record.sample_id,
        verdict="OK",
        reviewed_by="operator-1",
    )
    repository.save_inference(
        AcceptanceInferenceOutcome(
            sample_id=record.sample_id,
            machine_status="OK",
            machine_reasons=(),
            model_version="1.0.6",
            model_sha256="baseline",
            inference_at="2026-07-30T10:00:00+08:00",
            latency_ms=1.0,
            error="",
        )
    )
    snapshot = repository.create_snapshot(label="frozen")

    class RegressingService:
        def __init__(self, **kwargs) -> None:
            pass

        def infer(self, record, image_path, *, inference_type):
            return AcceptanceInferenceOutcome(
                sample_id=record.sample_id,
                machine_status="NG",
                machine_reasons=("COLOR_MISMATCH",),
                model_version="candidate",
                model_sha256="candidate",
                inference_at="2026-07-31T10:00:00+08:00",
                latency_ms=1.0,
                error="",
            )

        def close(self) -> None:
            pass

    def reject_changed_color_revisions() -> tuple[str, ...]:
        raise RuntimeError("active pointer changed during inference")

    color_revision_contract = {
        "schema_version": 1,
        "identity_sha256": "contract-sha",
    }

    result = run_candidate_acceptance(
        project_root=tmp_path,
        models_root=tmp_path / "models",
        global_config_path=tmp_path / "config.yaml",
        color_revisions_root=None,
        dataset_root=repository.root,
        snapshot_manifest_path=snapshot.manifest_path,
        report_path=tmp_path / "report.json",
        product="Cable1",
        area="A",
        inference_type="yolo",
        model_identity=ModelIdentity(version="candidate", sha256="candidate"),
        policy=AcceptanceGatePolicy(
            min_confirmed=1,
            max_false_positives=0,
            max_false_negatives=0,
            max_regressions=0,
        ),
        color_revision_contract=color_revision_contract,
        color_revision_contract_validator=reject_changed_color_revisions,
        service_factory=RegressingService,
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert result.passed is False
    assert any("false positives" in failure for failure in result.failures)
    assert any("regressed samples" in failure for failure in result.failures)
    assert any(
        "active pointer changed during inference" in failure
        for failure in result.failures
    )
    assert report["color_revisions"] == color_revision_contract
