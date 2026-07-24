import json
from datetime import datetime, timezone

import pytest

from tools.processing_events import (
    CompositeEventSink,
    InMemoryEventSink,
    JsonlEventSink,
    ProcessingEvent,
    ProcessingEventLevel,
    ProcessingEventPublisher,
    ProcessingEventSink,
    ProcessingEventSinkError,
    ProcessingEventType,
    exception_metadata,
)

NOW = datetime(2026, 7, 21, 1, 2, 3, tzinfo=timezone.utc)


def _event(sequence=1, event_id="event-1", metadata=None):
    return ProcessingEvent(
        event_id=event_id,
        plan_id="plan-1",
        report_id="report-1",
        timestamp=NOW,
        sequence=sequence,
        level=ProcessingEventLevel.INFO,
        event_type=ProcessingEventType.EXECUTION_STARTED,
        stage="execution",
        message="started",
        metadata=metadata or {},
    )


def test_event_requires_monotonic_sequence_and_utc_timestamp():
    with pytest.raises(ValueError, match="sequence"):
        _event(sequence=0)
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        ProcessingEvent(
            event_id="event-1",
            plan_id="plan-1",
            report_id="report-1",
            timestamp=datetime(2026, 7, 21),
            sequence=1,
            level=ProcessingEventLevel.INFO,
            event_type=ProcessingEventType.EXECUTION_STARTED,
            stage="execution",
            message="started",
        )


def test_event_metadata_must_be_json_safe_and_not_sensitive():
    with pytest.raises(TypeError, match="not JSON serializable"):
        _event(metadata={"object": object()})
    with pytest.raises(ValueError, match="Sensitive metadata key"):
        _event(metadata={"api_token": "do-not-store"})
    with pytest.raises(ValueError, match="Sensitive metadata key"):
        _event(metadata={"nested": {"password": "do-not-store"}})


def test_exception_metadata_redacts_credentials_and_never_keeps_exception():
    error = RuntimeError("token=secret-value failed")

    metadata = exception_metadata(error)

    assert metadata == {
        "error_type": "RuntimeError",
        "error_message": "token=[REDACTED] failed",
    }
    assert all(not isinstance(value, BaseException) for value in metadata.values())


def test_in_memory_sink_rejects_duplicate_ids_and_non_monotonic_sequence():
    sink = InMemoryEventSink()
    sink.emit(_event())

    with pytest.raises(ProcessingEventSinkError, match="Duplicate"):
        sink.emit(_event(sequence=2))
    with pytest.raises(ProcessingEventSinkError, match="strictly increasing"):
        sink.emit(_event(sequence=1, event_id="event-2"))


def test_jsonl_sink_is_append_only_and_flushes_each_event(tmp_path):
    path = tmp_path / "events.jsonl"
    sink = JsonlEventSink(path, fsync=True)

    sink.emit(_event())
    sink.emit(_event(sequence=2, event_id="event-2"))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["sequence"] for line in lines] == [1, 2]


class FailingSink(ProcessingEventSink):
    def emit(self, event):
        raise ProcessingEventSinkError(f"failed {event.event_id}")


def test_composite_sink_delivers_to_healthy_sink_before_reporting_failure():
    memory = InMemoryEventSink()
    sink = CompositeEventSink((FailingSink(), memory))

    with pytest.raises(ProcessingEventSinkError, match="FailingSink"):
        sink.emit(_event())

    assert memory.events == (_event(),)


def test_publisher_keeps_sequence_and_sink_failure_does_not_escape():
    ids = iter(("event-1", "event-2"))
    publisher = ProcessingEventPublisher(
        plan_id="plan-1",
        report_id="report-1",
        sink=FailingSink(),
        clock=lambda: NOW,
        id_generator=lambda: next(ids),
    )

    first = publisher.emit(
        ProcessingEventType.EXECUTION_STARTED,
        "started",
        stage="execution",
    )
    second = publisher.emit(
        ProcessingEventType.EXECUTION_COMPLETED,
        "completed",
        stage="execution",
    )

    assert (first.sequence, second.sequence) == (1, 2)
    assert len(publisher.warnings) == 2


def test_event_dict_has_schema_and_no_python_objects():
    payload = _event(metadata={"nested": [1, True]}).to_dict()

    assert payload["schema_version"] == 1
    json.dumps(payload)
    assert payload["metadata"] == {"nested": [1, True]}
