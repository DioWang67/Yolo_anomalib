"""Verdict regression matrix and sync/async parity for the inspection pipeline.

Why this file exists
--------------------
``finalize_status`` is the single owner of the final PASS/DETECTION_FAIL
verdict: every pipeline step may only *downgrade* to FAIL, and the final call
is recomputed from the corrected facts just before results are persisted. That
design is only safe while finalization knows about **every** failure dimension.

It once did not. An anomalib-only run leaves no detection-side signal behind
(no detections, no missing/unexpected items, no color or sequence result), so
finalization re-derived PASS from empty evidence and shipped defective boards
as good — observed in production on 2026-07-28 for PCBA1/B, where records with
``anomaly_score`` 1.0 (threshold 0.5) were saved as PASS with no fail reasons.

The matrix below pins one row per failure dimension so that adding a dimension
without wiring it into finalization fails loudly here instead of in the field.
"""

from __future__ import annotations

import logging
import queue
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from core.config import DetectionConfig
from core.detection_system import DetectionSystem
from core.exceptions import ResultPersistenceError
from core.pipeline.context import DetectionContext
from core.pipeline.finalize import finalize_status
from core.services.decision_engine import InspectionReason, collect_fail_reasons
from core.types import DetectionTask
from core.workers import InferenceWorker, StorageWorker

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Part 1 — the verdict matrix, evaluated at the decision layer
# ---------------------------------------------------------------------------


def _clean_facts() -> dict[str, Any]:
    """Facts for a board on which every dimension reports good."""
    return {
        "detections": [{"class": "Red", "confidence": 0.9, "position_status": "CORRECT"}],
        "missing_items": [],
        "unexpected_items": [],
        "slot_mismatches": [],
        "alignment_quality": {"enabled": True, "is_ok": True, "issues": []},
        "sequence_check": {"is_ok": True},
        "count_check": {"missing": [], "over": [], "strict": True, "is_ok": True},
        "is_anomaly": False,
    }


def _with(**overrides: Any) -> dict[str, Any]:
    facts = _clean_facts()
    facts.update(overrides)
    return facts


#: ``(id, incoming ctx.status, facts, color_result, expected final status)``.
#:
#: ``stale_yolo_fail_cleared_by_color`` is the row that motivated
#: ``finalize_status`` in the first place: YOLO mislabeled a part, so it
#: reported FAIL, but the color checker corrected the class and every fact is
#: now clean. That board must still come out PASS — no anomaly or count gate
#: may take that recovery away.
_MATRIX = [
    ("all_clean", "PASS", _clean_facts(), {"is_ok": True}, "PASS"),
    (
        "stale_yolo_fail_cleared_by_color",
        "DETECTION_FAIL",
        _clean_facts(),
        {"is_ok": True},
        "PASS",
    ),
    ("color_fail", "PASS", _clean_facts(), {"is_ok": False}, "DETECTION_FAIL"),
    (
        "count_fail",
        "PASS",
        _with(
            missing_items=["Green"],
            count_check={
                "missing": ["Green"],
                "over": [],
                "strict": True,
                "is_ok": False,
            },
        ),
        {"is_ok": True},
        "DETECTION_FAIL",
    ),
    (
        "position_fail",
        "PASS",
        _with(
            detections=[
                {"class": "Red", "confidence": 0.9, "position_status": "WRONG"}
            ]
        ),
        {"is_ok": True},
        "DETECTION_FAIL",
    ),
    (
        "sequence_fail",
        "PASS",
        _with(sequence_check={"is_ok": False, "reason": "order_mismatch"}),
        {"is_ok": True},
        "DETECTION_FAIL",
    ),
    (
        "anomaly_fail",
        "PASS",
        _with(is_anomaly=True, anomaly_score=0.97),
        {"is_ok": True},
        "DETECTION_FAIL",
    ),
    (
        "anomaly_explicitly_clean",
        "PASS",
        _with(is_anomaly=False, anomaly_score=0.12),
        {"is_ok": True},
        "PASS",
    ),
]


@pytest.mark.parametrize(
    ("status_in", "facts", "color", "expected"),
    [pytest.param(*row[1:], id=row[0]) for row in _MATRIX],
)
def test_verdict_matrix(status_in, facts, color, expected):
    ctx = SimpleNamespace(status=status_in, result=dict(facts), color_result=color)

    finalize_status(ctx)

    assert ctx.status == expected


@pytest.mark.parametrize(
    "terminal", ["INFERENCE_ERROR", "ERROR", "CANCELED"]
)
def test_terminal_statuses_are_never_reinterpreted(terminal):
    """A terminal abort outranks every fact, clean or failing."""
    for facts in (_clean_facts(), _with(is_anomaly=True, anomaly_score=1.0)):
        ctx = SimpleNamespace(
            status=terminal, result=dict(facts), color_result={"is_ok": False}
        )
        finalize_status(ctx)
        assert ctx.status == terminal


# ---------------------------------------------------------------------------
# Part 2 — status and machine-readable reasons must agree
# ---------------------------------------------------------------------------


def test_anomaly_fail_is_reported_as_anomaly_detected():
    """A saved anomaly FAIL must never carry an empty reason list."""
    ctx = SimpleNamespace(
        status="DETECTION_FAIL",
        result=_with(is_anomaly=True, anomaly_score=0.97),
        color_result={"is_ok": True},
    )
    finalize_status(ctx)
    assert ctx.status == "DETECTION_FAIL"

    reasons = collect_fail_reasons(
        status=ctx.status,
        decision=ctx.result["decision"],
        color_result=ctx.color_result,
        sequence_check=ctx.result.get("sequence_check"),
        detector="anomalib",
        anomaly_score=ctx.result.get("anomaly_score"),
    )

    assert reasons == [InspectionReason.ANOMALY_DETECTED.value]


def test_fusion_anomaly_only_fail_is_attributed_to_the_anomaly():
    ctx = SimpleNamespace(
        status="DETECTION_FAIL",
        result=_with(is_anomaly=True, anomaly_score=0.88),
        color_result={"is_ok": True},
    )
    finalize_status(ctx)

    reasons = collect_fail_reasons(
        status=ctx.status,
        decision=ctx.result["decision"],
        color_result=ctx.color_result,
        sequence_check=ctx.result.get("sequence_check"),
        detector="fusion",
        anomaly_score=ctx.result.get("anomaly_score"),
    )

    assert InspectionReason.ANOMALY_DETECTED.value in reasons


def test_a_passing_verdict_never_carries_fail_reasons():
    ctx = SimpleNamespace(
        status="PASS", result=_clean_facts(), color_result={"is_ok": True}
    )
    finalize_status(ctx)
    assert ctx.status == "PASS"

    assert (
        collect_fail_reasons(
            status=ctx.status,
            decision=ctx.result["decision"],
            color_result=ctx.color_result,
            detector="anomalib",
            anomaly_score=0.97,
        )
        == []
    )


@pytest.mark.parametrize(
    ("name", "facts", "color", "detector"),
    [
        ("anomaly", _with(is_anomaly=True, anomaly_score=0.97), {"is_ok": True}, "anomalib"),
        ("color", _clean_facts(), {"is_ok": False}, "yolo"),
        (
            "sequence",
            _with(sequence_check={"is_ok": False, "reason": "order_mismatch"}),
            {"is_ok": True},
            "yolo",
        ),
        ("missing", _with(missing_items=["Green"]), {"is_ok": True}, "yolo"),
    ],
)
def test_every_failing_dimension_yields_at_least_one_reason(
    name, facts, color, detector
):
    """No dimension may fail a board without explaining itself."""
    ctx = SimpleNamespace(status="PASS", result=dict(facts), color_result=color)
    finalize_status(ctx)
    assert ctx.status == "DETECTION_FAIL", name

    reasons = collect_fail_reasons(
        status=ctx.status,
        decision=ctx.result["decision"],
        color_result=ctx.color_result,
        sequence_check=ctx.result.get("sequence_check"),
        detector=detector,
        anomaly_score=ctx.result.get("anomaly_score"),
    )
    assert reasons, f"{name} failed the board without a machine-readable reason"


# ---------------------------------------------------------------------------
# Part 3 — the runtime status boundary
# ---------------------------------------------------------------------------


class _FakeSink:
    """Records what the pipeline asked to persist."""

    def __init__(self) -> None:
        self.saved: list[dict[str, Any]] = []
        self.flush_async_calls = 0

    def save(self, **kwargs: Any) -> dict[str, Any]:
        self.saved.append(kwargs)
        return {
            "status": "SUCCESS",
            "original_path": "Result/original.jpg",
            "preprocessed_path": "Result/processed.png",
            "annotated_path": "Result/annotated.jpg",
            "heatmap_path": "",
            "cropped_paths": [],
            "missing_locations": [],
        }

    def flush(self) -> None:
        pass

    def flush_async(self) -> None:
        self.flush_async_calls += 1

    def get_annotated_path(self, **_kwargs: Any) -> str:
        return "Result/TEMP/anomalib.png"


class _RecordingLogger:
    """Minimal stand-in for ``DetectionLogger``."""

    def __init__(self) -> None:
        self.logger = LOGGER
        self.detections: list[str] = []
        self.anomalies: list[str] = []

    def log_detection(self, status: str, _detections: Any) -> None:
        self.detections.append(status)

    def log_anomaly(self, status: str, _score: Any) -> None:
        self.anomalies.append(status)


class _FakeEngine:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def infer(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return dict(self.result)


class _FakeModelManager:
    def __init__(self, engines: dict[str, Any]) -> None:
        self.engines = engines

    def get_cached_engine(self, _product: str, _area: str, inference_type: str):
        return self.engines.get(inference_type)


def _make_system(
    config: DetectionConfig, *, model_manager: Any = None
) -> DetectionSystem:
    """Build a DetectionSystem exercising the real orchestration methods.

    ``__init__`` loads config files, opens a camera and creates result sinks,
    none of which this test needs. Bypassing it keeps the test honest about
    what it covers: the verdict/persistence orchestration itself, not startup.
    """
    system = object.__new__(DetectionSystem)
    system.config = config
    system.color_service = SimpleNamespace(is_ready=lambda: False)
    system.result_sink = _FakeSink()
    system.logger = _RecordingLogger()
    system.model_manager = model_manager
    system._inference_lock = threading.RLock()
    system.inference_engine = None
    return system


def _yolo_config() -> DetectionConfig:
    config = DetectionConfig(weights="dummy.pt")
    config.expected_items = {"P": {"A": ["Red", "Green"]}}
    config.pipeline = ["count_check", "save_results"]
    config.steps = {"count_check": {"strict": True}}
    config.enable_color_check = False
    config.fail_on_unexpected = True
    return config


def _anomalib_config() -> DetectionConfig:
    """Mirrors the shipped ``models/PCBA1/B/anomalib/config.yaml`` shape.

    No ``pipeline``, no color check — so ``default_pipeline`` yields
    ``save_results`` alone and **no verdict step runs at all**. That empty
    verdict stage is precisely what let finalization recompute an anomaly FAIL
    into PASS, so the anomaly rows must use this config or a count check would
    mask the regression.
    """
    config = DetectionConfig(weights="dummy.pt")
    config.expected_items = {}
    config.pipeline = None
    config.steps = {}
    config.enable_color_check = False
    config.enable_anomalib = True
    config.fail_on_unexpected = True
    return config


def test_fusion_yolo_only_fallback_normalizes_bare_fail_at_the_boundary():
    """Model layers speak ``FAIL``; everything downstream is promised
    ``DETECTION_FAIL``.

    Fusion used to return before that translation, so its YOLO-only fallback
    leaked a bare ``FAIL`` into the pipeline, GUI and storage.
    """
    config = _yolo_config()
    system = _make_system(
        config,
        model_manager=_FakeModelManager(
            {
                # No anomalib engine -> FusionInferenceRunner falls back to YOLO.
                "yolo": _FakeEngine(
                    {
                        "status": "FAIL",
                        "detections": [],
                        "missing_items": ["Red"],
                        "unexpected_items": [],
                    }
                )
            }
        ),
    )

    result = system._run_inference_locked(
        np.zeros((8, 8, 3), dtype=np.uint8), "P", "A", "fusion", LOGGER
    )

    assert result["status"] == "DETECTION_FAIL"


def test_yolo_backend_bare_fail_is_still_normalized():
    """The pre-existing non-fusion boundary must keep working."""
    config = _yolo_config()
    system = _make_system(config)
    system.inference_engine = _FakeEngine(
        {
            "status": "FAIL",
            "detections": [],
            "missing_items": ["Red"],
            "unexpected_items": [],
        }
    )

    result = system._run_inference_locked(
        np.zeros((8, 8, 3), dtype=np.uint8), "P", "A", "yolo", LOGGER
    )

    assert result["status"] == "DETECTION_FAIL"


# ---------------------------------------------------------------------------
# Part 4 — sync and async must reach the same verdict
# ---------------------------------------------------------------------------


def _frame() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def _inference_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "DETECTION_FAIL",
        "inference_type": "yolo",
        "detections": [{"class": "Red", "confidence": 0.9}],
        "missing_items": ["Green"],
        "unexpected_items": [],
        "processed_image": _frame(),
    }
    payload.update(overrides)
    return payload


def _run_sync(config: DetectionConfig, payload: dict[str, Any]):
    system = _make_system(config)
    ctx = DetectionContext(
        product="P",
        area="A",
        inference_type=payload.get("inference_type", "yolo"),
        frame=_frame(),
        processed_image=payload["processed_image"],
        result=payload,
        status=payload["status"],
        config=config,
    )
    system._execute_pipeline(ctx, LOGGER)
    return system, ctx.status, ctx.result.get("decision")


def _run_async(config: DetectionConfig, payload: dict[str, Any]):
    """Drive the real InferenceWorker/StorageWorker glue, without threads."""
    system = _make_system(config)
    task = DetectionTask(
        task_id="t0",
        timestamp=0.0,
        product="P",
        area="A",
        inference_type=payload.get("inference_type", "yolo"),
        frame=_frame(),
    )
    task.result = payload

    inference_worker = InferenceWorker(
        in_queue=queue.Queue(), out_queue=queue.Queue(), detection_system=system
    )
    inference_worker._finalize_result(task)
    verdict_seen_by_gui = task.result["status"]

    storage_worker = StorageWorker(in_queue=queue.Queue(), detection_system=system)
    storage_worker.process(task)
    assert storage_worker.saved_count == 1, "storage swallowed an exception"

    return system, verdict_seen_by_gui, task.result.get("decision")


@pytest.mark.parametrize(
    ("name", "make_config", "overrides", "expected"),
    [
        ("missing_part", _yolo_config, {}, "DETECTION_FAIL"),
        (
            "stale_yolo_fail_recovered",
            _yolo_config,
            {
                "detections": [{"class": "Red"}, {"class": "Green"}],
                "missing_items": ["Green"],
            },
            "PASS",
        ),
        (
            "anomaly",
            _anomalib_config,
            {
                "inference_type": "anomalib",
                "detections": [],
                "missing_items": [],
                "is_anomaly": True,
                "anomaly_score": 0.97,
            },
            "DETECTION_FAIL",
        ),
        (
            "clean_anomaly",
            _anomalib_config,
            {
                "status": "PASS",
                "inference_type": "anomalib",
                "detections": [],
                "missing_items": [],
                "is_anomaly": False,
                "anomaly_score": 0.12,
            },
            "PASS",
        ),
    ],
)
def test_sync_and_async_reach_the_same_verdict(name, make_config, overrides, expected):
    """Both routes share ``_run_verdict_steps``; this pins that they stay shared.

    ``stale_yolo_fail_recovered`` feeds a payload whose YOLO-side
    ``missing_items`` is stale — the detections do contain the part — so
    ``CountCheckStep`` recomputes it away and the board must recover to PASS on
    both routes.
    """
    sync_system, sync_status, sync_decision = _run_sync(
        make_config(), _inference_payload(**overrides)
    )
    async_system, async_status, async_decision = _run_async(
        make_config(), _inference_payload(**overrides)
    )

    assert sync_status == expected, name
    assert async_status == expected, name
    assert sync_decision == async_decision

    # The verdict the operator saw is also the verdict that was persisted.
    assert sync_system.result_sink.saved[0]["status"] == expected
    assert async_system.result_sink.saved[0]["status"] == expected


class _FailingSink(_FakeSink):
    def save(self, **kwargs: Any) -> dict[str, Any]:
        raise ResultPersistenceError("disk full")


def test_async_persistence_failure_is_visible_on_the_task_but_not_the_verdict():
    """Characterizes today's behaviour — it is not asserted to be desirable.

    The operator's verdict is published by ``on_task_inferred`` before storage
    runs. When storage then fails, ``SaveResultsStep`` rewrites ``ctx.status``
    to ``ERROR`` *after* finalization, and ``_attach_pipeline_outputs`` copies
    that onto the task. The GUI's ``storage_completed`` handler deliberately
    does not replay the verdict, so the operator keeps seeing PASS for an
    inspection that was never recorded.

    This test pins the two facts a future fix needs: the failure *is* carried
    on ``task.result``, so surfacing it needs no new plumbing — only a
    presentation decision (show "result not saved" alongside the verdict
    rather than overwriting the verdict with ERROR).
    """
    system = _make_system(_yolo_config())
    system.result_sink = _FailingSink()

    task = DetectionTask(
        task_id="t2",
        timestamp=0.0,
        product="P",
        area="A",
        inference_type="yolo",
        frame=_frame(),
    )
    task.result = _inference_payload(
        detections=[{"class": "Red"}, {"class": "Green"}], missing_items=[]
    )

    inference_worker = InferenceWorker(
        in_queue=queue.Queue(), out_queue=queue.Queue(), detection_system=system
    )
    inference_worker._finalize_result(task)
    verdict_shown_to_operator = task.result["status"]
    assert verdict_shown_to_operator == "PASS"

    StorageWorker(in_queue=queue.Queue(), detection_system=system).process(task)

    # The verdict itself was never wrong — persistence was.
    assert task.result["status"] == "ERROR"
    assert task.result["save_result"]["status"] == "ERROR"
    assert task.result["original_image_path"] == ""


def test_async_gui_verdict_is_not_provisional():
    """``on_task_inferred`` fires only after finalization, never before."""
    config = _anomalib_config()
    system = _make_system(config)
    task = DetectionTask(
        task_id="t1",
        timestamp=0.0,
        product="P",
        area="A",
        inference_type="anomalib",
        frame=_frame(),
    )
    task.result = _inference_payload(
        status="DETECTION_FAIL",
        inference_type="anomalib",
        detections=[],
        missing_items=[],
        is_anomaly=True,
        anomaly_score=0.97,
    )

    worker = InferenceWorker(
        in_queue=queue.Queue(), out_queue=queue.Queue(), detection_system=system
    )
    worker._finalize_result(task)

    assert task.result["status"] == "DETECTION_FAIL"
