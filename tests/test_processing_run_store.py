import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from tools.processing_pipeline import ProcessingPlanner
from tools.processing_run_store import (
    ProcessingPersistenceError,
    ProcessingRunStore,
)

NOW = datetime(2026, 7, 21, 1, 2, 3, tzinfo=timezone.utc)


def _plan():
    record = {
        "sample_id": "sample-1",
        "review_selected": "1",
        "review_outcome": "fail",
        "review_label": "confirmed_ng",
        "failure_category": "threshold_not_met",
        "skip_reason": "",
        "product_verdict": "ng",
        "detection_verdict": "correct",
        "color_verdict": "not_applicable",
        "action_route": "yolo",
        "training_selected": "1",
    }
    plan = ProcessingPlanner().create_plan(
        [(0, record)], operator="operator-a", created_at=NOW
    )
    return replace(plan, plan_id="plan-1")


def test_run_store_creates_required_layout(tmp_path):
    store = ProcessingRunStore(tmp_path / ".processing_runs")

    assert store.plans_dir.is_dir()
    assert store.reports_dir.is_dir()
    assert store.events_dir.is_dir()
    assert store.artifacts_dir.is_dir()


def test_plan_snapshot_is_atomic_hashed_and_idempotent(tmp_path):
    store = ProcessingRunStore(tmp_path / ".processing_runs")
    plan = _plan()

    first = store.persist_plan(plan)
    second = store.persist_plan(plan)

    assert first == second
    assert first.sha256
    payload = json.loads(first.path.read_text(encoding="utf-8"))
    assert payload["plan_id"] == "plan-1"
    assert payload["records"][0]["record_sha"]
    assert payload["attempt"] == 1


def test_same_plan_id_cannot_overwrite_different_snapshot(tmp_path):
    store = ProcessingRunStore(tmp_path / ".processing_runs")
    plan = _plan()
    store.persist_plan(plan)

    with pytest.raises(ProcessingPersistenceError, match="collision"):
        store.persist_plan(replace(plan, operator="different"))


@pytest.mark.parametrize("unsafe_id", ["../escape", "a/b", "", ".."])
def test_identifier_cannot_escape_run_root(tmp_path, unsafe_id):
    store = ProcessingRunStore(tmp_path / ".processing_runs")

    with pytest.raises(ProcessingPersistenceError, match="Unsafe"):
        store.persist_plan(replace(_plan(), plan_id=unsafe_id))


def test_event_log_is_append_only_and_report_id_cannot_be_reused(tmp_path):
    store = ProcessingRunStore(tmp_path / ".processing_runs")

    path, sink = store.prepare_event_log("report-1", fsync=True)
    assert path.is_file()
    assert sink.path == path
    with pytest.raises(ProcessingPersistenceError, match="already exists"):
        store.prepare_event_log("report-1")


def test_atomic_write_failure_preserves_original_and_cleans_temp(tmp_path, monkeypatch):
    store = ProcessingRunStore(tmp_path / ".processing_runs")
    destination = store.root / "latest.json"
    store._atomic_json(destination, {"value": "original"}, replace=True)
    original = destination.read_bytes()

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("tools.processing_run_store.os.replace", fail_replace)

    with pytest.raises(ProcessingPersistenceError, match="Atomic write failed"):
        store._atomic_json(destination, {"value": "new"}, replace=True)

    assert destination.read_bytes() == original
    assert not list(store.root.glob(".latest.json.*.tmp"))


def test_describe_file_rejects_path_outside_root(tmp_path):
    store = ProcessingRunStore(tmp_path / ".processing_runs")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    with pytest.raises(ProcessingPersistenceError, match="escapes"):
        store.describe_file(outside, kind="test")
