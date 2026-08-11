from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.acceptance.headless as headless
import tools.color_configuration_revisions as revision_module
import tools.color_revision_publication_lock as publication_lock_module
from app.acceptance.color_revision_contract import main as verify_contract_main
from core.services.color_revision_contract import (
    ColorRevisionContractError,
    capture_candidate_color_revision_contract,
    color_revision_overrides,
    verify_active_color_revision_contract,
)
from tools.color_calibration_service import (
    COLOR_CONFIG_SCHEMA_VERSION,
    ColorCalibrationError,
    ColorCalibrationScope,
)
from tools.color_configuration_resolver import ColorConfigurationResolver
from tools.color_configuration_revisions import (
    ColorConfigurationRevision,
    ColorConfigurationRevisionStore,
)
from tools.color_revision_publication_lock import color_revision_publication_lock

_NOW = datetime(2026, 8, 11, tzinfo=timezone.utc)


def _commit_revision(
    store: ColorConfigurationRevisionStore,
    scope: ColorCalibrationScope,
    *,
    source_id: str,
    value: float,
    parent_revision_id: str = "",
    parent_config_sha256: str = "base-config",
) -> ColorConfigurationRevision:
    return store.commit_configuration(
        source_id,
        scope,
        operator="reviewer",
        reason="acceptance contract test",
        proposal_sha256=f"proposal-{source_id}",
        preview_sha256=f"preview-{source_id}",
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
            "public_threshold": value,
            "config_value": value,
        },
        metrics={},
        parent_revision_id=parent_revision_id,
        parent_config_sha256=parent_config_sha256,
    )


def _activate(
    store: ColorConfigurationRevisionStore,
    revision: ColorConfigurationRevision,
    *,
    expected_sha256: str,
) -> None:
    store.activate(
        revision,
        operator="reviewer",
        reason="production active",
        expected_current_sha256=expected_sha256,
    )


def _candidate_config(path: Path, *, enabled: object = True) -> Path:
    path.write_text(
        "weights: candidate.pt\n"
        f"enable_color_check: {json.dumps(enabled)}\n"
        "color_checker_type: stats\n"
        "color_model_path: color_stats.json\n",
        encoding="utf-8",
    )
    return path


def _as_legacy_schema2_pointer(
    pointer_path: Path,
    revision: ColorConfigurationRevision,
) -> tuple[dict[str, object], Path]:
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    event_path = (
        revision.root
        / "activation_events"
        / f"{pointer['event_id']}.json"
    )
    pointer.pop("event_id")
    pointer.pop("event_type")
    pointer_path.write_text(
        json.dumps(pointer, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return pointer, event_path


def test_contract_records_enabled_runtime_with_no_active_revisions(
    tmp_path: Path,
) -> None:
    contract = capture_candidate_color_revision_contract(
        revisions_root=tmp_path / ".color_revisions",
        candidate_config_path=_candidate_config(tmp_path / "config.yaml"),
        color_model_present=True,
        product="Cable1",
        area="A",
        inference_type="yolo",
    )

    assert contract["enabled"] is True
    assert contract["checker_type"] == "stats"
    assert contract["entries"] == []
    assert len(contract["identity_sha256"]) == 64
    assert color_revision_overrides(contract) == {}
    assert verify_active_color_revision_contract(
        contract,
        revisions_root=tmp_path / ".color_revisions",
    ) == contract


def test_contract_pins_multiple_matching_colors_and_ignores_other_station(
    tmp_path: Path,
) -> None:
    revision_ids = count()
    store = ColorConfigurationRevisionStore(
        root=tmp_path / ".color_revisions",
        clock=lambda: _NOW,
        id_generator=lambda: f"id-{next(revision_ids)}",
    )
    scopes = (
        ColorCalibrationScope("Cable1", "A", "yolo", "stats", "black"),
        ColorCalibrationScope("Cable1", "A", "yolo", "stats", "red"),
        ColorCalibrationScope("Cable1", "B", "yolo", "stats", "black"),
    )
    revisions = tuple(
        _commit_revision(
            store,
            scope,
            source_id=f"source-{index}",
            value=0.5 + index / 10,
        )
        for index, scope in enumerate(scopes)
    )
    for revision in revisions:
        _activate(store, revision, expected_sha256="base-config")

    contract = capture_candidate_color_revision_contract(
        revisions_root=store.root,
        candidate_config_path=_candidate_config(tmp_path / "config.yaml"),
        color_model_present=True,
        product="Cable1",
        area="A",
        inference_type="yolo",
    )

    assert {entry["threshold_key"] for entry in contract["entries"]} == {
        "black",
        "red",
    }
    assert [entry["scope_hash"] for entry in contract["entries"]] == sorted(
        entry["scope_hash"] for entry in contract["entries"]
    )
    assert color_revision_overrides(contract) == {
        scopes[0].scope_hash: revisions[0].revision_id,
        scopes[1].scope_hash: revisions[1].revision_id,
    }
    assert all(len(entry["config_sha256"]) == 64 for entry in contract["entries"])
    assert all(
        len(entry["config_file_sha256"]) == 64 for entry in contract["entries"]
    )
    assert all(len(entry["pointer_sha256"]) == 64 for entry in contract["entries"])


def test_contract_rejects_pointer_switch_and_revision_content_mutation(
    tmp_path: Path,
) -> None:
    revision_ids = count()
    store = ColorConfigurationRevisionStore(
        root=tmp_path / ".color_revisions",
        clock=lambda: _NOW,
        id_generator=lambda: f"id-{next(revision_ids)}",
    )
    scope = ColorCalibrationScope("Cable1", "A", "yolo", "stats", "black")
    first = _commit_revision(
        store,
        scope,
        source_id="source-one",
        value=0.5,
    )
    _activate(store, first, expected_sha256="base-config")
    contract = capture_candidate_color_revision_contract(
        revisions_root=store.root,
        candidate_config_path=_candidate_config(tmp_path / "config.yaml"),
        color_model_present=True,
        product="Cable1",
        area="A",
        inference_type="yolo",
    )
    second = _commit_revision(
        store,
        scope,
        source_id="source-two",
        value=0.7,
        parent_revision_id=first.revision_id,
        parent_config_sha256=first.new_config_sha256,
    )
    _activate(store, second, expected_sha256=first.new_config_sha256)

    with pytest.raises(ColorRevisionContractError, match="contract changed"):
        verify_active_color_revision_contract(contract, revisions_root=store.root)

    second.config_path.write_text('{"config_value": 0.1}', encoding="utf-8")
    with pytest.raises(ColorRevisionContractError, match="revision is invalid"):
        capture_candidate_color_revision_contract(
            revisions_root=store.root,
            candidate_config_path=tmp_path / "config.yaml",
            color_model_present=True,
            product="Cable1",
            area="A",
            inference_type="yolo",
        )


def test_contract_cli_rechecks_report_and_blocks_stale_pointer(
    tmp_path: Path,
) -> None:
    revision_ids = count()
    store = ColorConfigurationRevisionStore(
        root=tmp_path / ".color_revisions",
        clock=lambda: _NOW,
        id_generator=lambda: f"id-{next(revision_ids)}",
    )
    scope = ColorCalibrationScope("Cable1", "A", "yolo", "stats", "black")
    first = _commit_revision(
        store,
        scope,
        source_id="source-one",
        value=0.5,
    )
    _activate(store, first, expected_sha256="base-config")
    contract = capture_candidate_color_revision_contract(
        revisions_root=store.root,
        candidate_config_path=_candidate_config(tmp_path / "config.yaml"),
        color_model_present=True,
        product="Cable1",
        area="A",
        inference_type="yolo",
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({"color_revisions": contract}),
        encoding="utf-8",
    )

    assert verify_contract_main(
        ["--revisions-root", str(store.root), "--report", str(report_path)]
    ) == 0

    second = _commit_revision(
        store,
        scope,
        source_id="source-two",
        value=0.8,
        parent_revision_id=first.revision_id,
        parent_config_sha256=first.new_config_sha256,
    )
    _activate(store, second, expected_sha256=first.new_config_sha256)

    assert verify_contract_main(
        ["--revisions-root", str(store.root), "--report", str(report_path)]
    ) == 2


def test_candidate_contract_uses_formal_boolean_parsing(tmp_path: Path) -> None:
    contract = capture_candidate_color_revision_contract(
        revisions_root=tmp_path / ".color_revisions",
        candidate_config_path=_candidate_config(
            tmp_path / "config.yaml",
            enabled="false",
        ),
        color_model_present=True,
        product="Cable1",
        area="A",
        inference_type="yolo",
    )

    assert contract["enabled"] is False
    assert contract["checker_type"] == ""


def test_candidate_contract_uses_formal_merged_runtime_semantics(
    tmp_path: Path,
) -> None:
    global_config = tmp_path / "global.yaml"
    global_config.write_text(
        "weights: global.pt\n"
        "enable_color_check: true\n"
        "color_checker_type: stats\n",
        encoding="utf-8",
    )
    candidate_config = tmp_path / "candidate.yaml"
    candidate_config.write_text("weights: candidate.pt\n", encoding="utf-8")

    contract = capture_candidate_color_revision_contract(
        revisions_root=tmp_path / ".color_revisions",
        candidate_config_path=candidate_config,
        global_config_path=global_config,
        color_model_present=True,
        product="Cable1",
        area="A",
        inference_type="yolo",
    )

    assert contract["enabled"] is True
    assert contract["checker_type"] == "color_qc"


def test_headless_pins_revisions_and_detects_pointer_switch_during_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision_ids = count()
    store = ColorConfigurationRevisionStore(
        root=tmp_path / ".color_revisions",
        clock=lambda: _NOW,
        id_generator=lambda: f"id-{next(revision_ids)}",
    )
    scope = ColorCalibrationScope("Cable1", "A", "yolo", "stats", "black")
    first = _commit_revision(
        store,
        scope,
        source_id="source-one",
        value=0.5,
    )
    _activate(store, first, expected_sha256="base-config")
    second = _commit_revision(
        store,
        scope,
        source_id="source-two",
        value=0.8,
        parent_revision_id=first.revision_id,
        parent_config_sha256=first.new_config_sha256,
    )
    config_path = _candidate_config(tmp_path / "config.yaml")
    weight_path = tmp_path / "candidate.pt"
    color_path = tmp_path / "color.json"
    weight_path.write_bytes(b"weight")
    color_path.write_bytes(b"color")

    def fake_run_candidate_acceptance(**kwargs):
        assert kwargs["include_active_color_revisions"] is False
        assert kwargs["color_revision_overrides"] == {
            scope.scope_hash: first.revision_id
        }
        _activate(store, second, expected_sha256=first.new_config_sha256)
        with pytest.raises(ColorRevisionContractError, match="contract changed"):
            kwargs["color_revision_contract_validator"]()
        return SimpleNamespace(
            passed=False,
            report_path=tmp_path / "report.json",
            failures=("active color revision contract changed",),
        )

    monkeypatch.setattr(
        headless,
        "run_candidate_acceptance",
        fake_run_candidate_acceptance,
    )
    exit_code = headless.main(
        [
            "--project-root",
            str(tmp_path),
            "--models-root",
            str(tmp_path / "models"),
            "--global-config",
            str(config_path),
            "--color-revisions-root",
            str(store.root),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--snapshot-manifest",
            str(tmp_path / "snapshot.csv"),
            "--report",
            str(tmp_path / "report.json"),
            "--product",
            "Cable1",
            "--area",
            "A",
            "--inference-type",
            "yolo",
            "--candidate-weight",
            str(weight_path),
            "--candidate-config",
            str(config_path),
            "--color-model",
            str(color_path),
            "--min-confirmed",
            "1",
            "--max-false-positives",
            "0",
            "--max-false-negatives",
            "0",
        ]
    )

    assert exit_code == 2


def test_activation_event_failure_leaves_pointer_absent_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision_ids = count()
    store = ColorConfigurationRevisionStore(
        root=tmp_path / ".color_revisions",
        clock=lambda: _NOW,
        id_generator=lambda: f"id-{next(revision_ids)}",
    )
    scope = ColorCalibrationScope("Cable1", "A", "yolo", "stats", "black")
    revision = _commit_revision(
        store,
        scope,
        source_id="source-one",
        value=0.5,
    )
    real_write_json_atomic = revision_module._write_json_atomic

    def fail_event(path: Path, payload: object) -> None:
        if path.parent.name == "activation_events":
            raise PermissionError("simulated event failure")
        real_write_json_atomic(path, payload)

    monkeypatch.setattr(revision_module, "_write_json_atomic", fail_event)

    with pytest.raises(ColorCalibrationError) as exc_info:
        _activate(store, revision, expected_sha256="base-config")

    assert exc_info.value.code == "COLOR_ACTIVATION_EVENT_FAILED"
    assert exc_info.value.retryable is True
    assert store.read_active_pointer(scope) is None
    assert not list((revision.root / "activation_events").glob("*.json"))

    monkeypatch.setattr(
        revision_module,
        "_write_json_atomic",
        real_write_json_atomic,
    )
    _activate(store, revision, expected_sha256="base-config")
    assert store.read_active_pointer(scope)["revision_id"] == revision.revision_id


def test_pointer_failure_never_exposes_uncommitted_revision_to_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision_ids = count()
    store = ColorConfigurationRevisionStore(
        root=tmp_path / ".color_revisions",
        clock=lambda: _NOW,
        id_generator=lambda: f"id-{next(revision_ids)}",
    )
    scope = ColorCalibrationScope("Cable1", "A", "yolo", "stats", "black")
    first = _commit_revision(
        store,
        scope,
        source_id="source-one",
        value=0.5,
    )
    _activate(store, first, expected_sha256="base-config")
    second = _commit_revision(
        store,
        scope,
        source_id="source-two",
        value=0.8,
        parent_revision_id=first.revision_id,
        parent_config_sha256=first.new_config_sha256,
    )
    pointer_write_started = threading.Event()
    allow_pointer_failure = threading.Event()

    def fail_pointer(path: Path, payload: object) -> None:
        event_path = (
            second.root
            / "activation_events"
            / f"{payload['event_id']}.json"  # type: ignore[index]
        )
        assert event_path.is_file()
        pointer_write_started.set()
        assert allow_pointer_failure.wait(timeout=2.0)
        raise ColorCalibrationError(
            "COLOR_ACTIVATION_FAILED",
            "simulated pointer failure",
            retryable=True,
        )

    monkeypatch.setattr(store, "_atomic_pointer_write", fail_pointer)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _activate,
            store,
            second,
            expected_sha256=first.new_config_sha256,
        )
        assert pointer_write_started.wait(timeout=2.0)
        assert store.read_active_pointer(scope)["revision_id"] == first.revision_id
        allow_pointer_failure.set()
        with pytest.raises(ColorCalibrationError) as exc_info:
            future.result(timeout=2.0)

    assert exc_info.value.code == "COLOR_ACTIVATION_FAILED"
    assert store.read_active_pointer(scope)["revision_id"] == first.revision_id
    orphan_events = list((second.root / "activation_events").glob("*.json"))
    assert len(orphan_events) == 1
    assert json.loads(orphan_events[0].read_text(encoding="utf-8"))[
        "revision_id"
    ] == second.revision_id


def test_post_replace_event_sync_error_is_committed_not_rolled_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    revision_ids = count()
    store = ColorConfigurationRevisionStore(
        root=tmp_path / ".color_revisions",
        clock=lambda: _NOW,
        id_generator=lambda: f"id-{next(revision_ids)}",
    )
    scope = ColorCalibrationScope("Cable1", "A", "yolo", "stats", "black")
    revision = _commit_revision(
        store,
        scope,
        source_id="source-one",
        value=0.5,
    )
    real_write_json_atomic = revision_module._write_json_atomic

    def write_then_report_sync_error(path: Path, payload: object) -> None:
        real_write_json_atomic(path, payload)
        if path.parent.name == "activation_events":
            raise OSError("simulated directory sync failure")

    monkeypatch.setattr(
        revision_module,
        "_write_json_atomic",
        write_then_report_sync_error,
    )
    caplog.set_level(logging.WARNING, logger=revision_module.__name__)

    pointer_path = store.activate(
        revision,
        operator="reviewer",
        reason="production active",
        expected_current_sha256="base-config",
    )

    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    event_path = revision.root / "activation_events" / f"{pointer['event_id']}.json"
    assert json.loads(event_path.read_text(encoding="utf-8")) == pointer
    assert "durability sync reported an error" in caplog.text


def test_schema2_pointer_without_event_fails_runtime_and_acceptance(
    tmp_path: Path,
) -> None:
    revision_ids = count()
    store = ColorConfigurationRevisionStore(
        root=tmp_path / ".color_revisions",
        clock=lambda: _NOW,
        id_generator=lambda: f"id-{next(revision_ids)}",
    )
    scope = ColorCalibrationScope("Cable1", "A", "yolo", "stats", "black")
    revision = _commit_revision(
        store,
        scope,
        source_id="source-one",
        value=0.5,
    )
    pointer_path = store.activate(
        revision,
        operator="reviewer",
        reason="production active",
        expected_current_sha256="base-config",
    )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    (
        revision.root
        / "activation_events"
        / f"{pointer['event_id']}.json"
    ).unlink()

    resolver = ColorConfigurationResolver(
        models_root=tmp_path / "models",
        revisions_root=store.root,
    )
    with pytest.raises(ColorCalibrationError) as resolver_error:
        resolver.resolve(scope)
    assert resolver_error.value.code == "COLOR_ACTIVATION_EVIDENCE_MISSING"

    with pytest.raises(ColorRevisionContractError, match="invalid"):
        capture_candidate_color_revision_contract(
            revisions_root=store.root,
            candidate_config_path=_candidate_config(tmp_path / "config.yaml"),
            color_model_present=True,
            product="Cable1",
            area="A",
            inference_type="yolo",
        )


def test_schema1_active_pointer_remains_explicitly_compatible(
    tmp_path: Path,
) -> None:
    revision_ids = count()
    store = ColorConfigurationRevisionStore(
        root=tmp_path / ".color_revisions",
        clock=lambda: _NOW,
        id_generator=lambda: f"id-{next(revision_ids)}",
    )
    scope = ColorCalibrationScope("Cable1", "A", "yolo", "stats", "black")
    revision = _commit_revision(
        store,
        scope,
        source_id="source-one",
        value=0.5,
    )
    pointer_path = store.activate(
        revision,
        operator="reviewer",
        reason="production active",
        expected_current_sha256="base-config",
    )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    event_path = (
        revision.root
        / "activation_events"
        / f"{pointer.pop('event_id')}.json"
    )
    pointer.pop("event_type")
    pointer["schema_version"] = 1
    pointer_path.write_text(
        json.dumps(pointer, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    event_path.unlink()

    assert store.read_active_pointer(scope) == pointer
    resolver = ColorConfigurationResolver(
        models_root=tmp_path / "models",
        revisions_root=store.root,
    )
    assert resolver.resolve(scope).revision_id == revision.revision_id
    contract = capture_candidate_color_revision_contract(
        revisions_root=store.root,
        candidate_config_path=_candidate_config(tmp_path / "config.yaml"),
        color_model_present=True,
        product="Cable1",
        area="A",
        inference_type="yolo",
    )
    assert [entry["revision_id"] for entry in contract["entries"]] == [
        revision.revision_id
    ]


def test_legacy_schema2_pointer_requires_one_exact_activation_event(
    tmp_path: Path,
) -> None:
    revision_ids = count()
    store = ColorConfigurationRevisionStore(
        root=tmp_path / ".color_revisions",
        clock=lambda: _NOW,
        id_generator=lambda: f"id-{next(revision_ids)}",
    )
    scope = ColorCalibrationScope("Cable1", "A", "yolo", "stats", "black")
    revision = _commit_revision(
        store,
        scope,
        source_id="source-one",
        value=0.5,
    )
    pointer_path = store.activate(
        revision,
        operator="reviewer",
        reason="production active",
        expected_current_sha256="base-config",
    )
    legacy_pointer, _ = _as_legacy_schema2_pointer(pointer_path, revision)

    assert store.read_active_pointer(scope) == legacy_pointer
    resolver = ColorConfigurationResolver(
        models_root=tmp_path / "models",
        revisions_root=store.root,
    )
    assert resolver.resolve(scope).revision_id == revision.revision_id
    contract = capture_candidate_color_revision_contract(
        revisions_root=store.root,
        candidate_config_path=_candidate_config(tmp_path / "config.yaml"),
        color_model_present=True,
        product="Cable1",
        area="A",
        inference_type="yolo",
    )
    assert len(contract["entries"]) == 1


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "tampered", "invalid-json"],
)
def test_legacy_schema2_pointer_rejects_ambiguous_or_invalid_event(
    tmp_path: Path,
    mutation: str,
) -> None:
    revision_ids = count()
    store = ColorConfigurationRevisionStore(
        root=tmp_path / ".color_revisions",
        clock=lambda: _NOW,
        id_generator=lambda: f"id-{next(revision_ids)}",
    )
    scope = ColorCalibrationScope("Cable1", "A", "yolo", "stats", "black")
    revision = _commit_revision(
        store,
        scope,
        source_id="source-one",
        value=0.5,
    )
    pointer_path = store.activate(
        revision,
        operator="reviewer",
        reason="production active",
        expected_current_sha256="base-config",
    )
    _, event_path = _as_legacy_schema2_pointer(pointer_path, revision)
    event = json.loads(event_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        event_path.unlink()
    elif mutation == "duplicate":
        event["event_id"] = "duplicate-event"
        (event_path.parent / "duplicate-event.json").write_text(
            json.dumps(event),
            encoding="utf-8",
        )
    elif mutation == "tampered":
        event["operator"] = "attacker"
        event_path.write_text(json.dumps(event), encoding="utf-8")
    else:
        event_path.write_text("{", encoding="utf-8")

    with pytest.raises(ColorCalibrationError) as exc_info:
        store.read_active_pointer(scope)
    assert exc_info.value.code == "COLOR_ACTIVATION_EVIDENCE_MISSING"


def test_publication_lock_blocks_activation_and_revoke_without_deadlock(
    tmp_path: Path,
) -> None:
    revision_ids = count()
    store = ColorConfigurationRevisionStore(
        root=tmp_path / ".color_revisions",
        clock=lambda: _NOW,
        id_generator=lambda: f"id-{next(revision_ids)}",
        publication_lock_timeout=2.0,
    )
    scope = ColorCalibrationScope("Cable1", "A", "yolo", "stats", "black")
    first = _commit_revision(
        store,
        scope,
        source_id="source-one",
        value=0.5,
    )
    second = _commit_revision(
        store,
        scope,
        source_id="source-two",
        value=0.8,
        parent_revision_id=first.revision_id,
        parent_config_sha256=first.new_config_sha256,
    )
    activation_started = threading.Event()

    def activate_first() -> Path:
        activation_started.set()
        return store.activate(
            first,
            operator="reviewer",
            reason="production active",
            expected_current_sha256="base-config",
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        with color_revision_publication_lock(store.root, timeout=1.0):
            future = executor.submit(activate_first)
            assert activation_started.wait(timeout=1.0)
            time.sleep(0.1)
            assert not future.done()
        assert future.result(timeout=2.0).is_file()

    revoke_started = threading.Event()

    def revoke_second() -> Path:
        revoke_started.set()
        return store.revoke(
            second,
            operator="reviewer",
            reason="unsafe candidate",
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        with color_revision_publication_lock(store.root, timeout=1.0):
            future = executor.submit(revoke_second)
            assert revoke_started.wait(timeout=1.0)
            time.sleep(0.1)
            assert not future.done()
        assert future.result(timeout=2.0).is_file()

    with pytest.raises(ColorCalibrationError) as exc_info:
        store.activate(
            second,
            operator="reviewer",
            reason="must stay blocked",
            expected_current_sha256=first.new_config_sha256,
        )
    assert exc_info.value.code == "COLOR_REVISION_REVOKED"


def test_publication_lock_timeout_releases_process_lock_for_retry(
    tmp_path: Path,
) -> None:
    store = ColorConfigurationRevisionStore(
        root=tmp_path / ".color_revisions",
        clock=lambda: _NOW,
        id_generator=iter(("revision", "activation")).__next__,
        publication_lock_timeout=0.05,
    )
    scope = ColorCalibrationScope("Cable1", "A", "yolo", "stats", "black")
    revision = _commit_revision(
        store,
        scope,
        source_id="source-timeout",
        value=0.5,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        with color_revision_publication_lock(store.root, timeout=1.0):
            future = executor.submit(
                store.activate,
                revision,
                operator="reviewer",
                reason="must time out",
                expected_current_sha256="base-config",
            )
            with pytest.raises(ColorCalibrationError) as exc_info:
                future.result(timeout=1.0)

    assert exc_info.value.code == "COLOR_REVISION_PUBLICATION_BUSY"
    assert exc_info.value.retryable is True
    assert store.activate(
        revision,
        operator="reviewer",
        reason="retry after lock release",
        expected_current_sha256="base-config",
    ).is_file()


def test_publication_unlock_error_does_not_leak_process_mutex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".color_revisions"
    real_unlock = publication_lock_module._unlock_byte

    def fail_unlock(handle) -> None:
        raise OSError("simulated unlock failure")

    monkeypatch.setattr(publication_lock_module, "_unlock_byte", fail_unlock)
    with color_revision_publication_lock(root, timeout=1.0):
        pass

    monkeypatch.setattr(publication_lock_module, "_unlock_byte", real_unlock)
    with color_revision_publication_lock(root, timeout=1.0):
        pass
