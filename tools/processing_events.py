"""Immutable processing events and auditable sink implementations."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

EVENT_SCHEMA_VERSION = 1
_SENSITIVE_KEY = re.compile(
    r"(?:password|passwd|secret|token|credential|api[_-]?key)", re.IGNORECASE
)
_SENSITIVE_TEXT = re.compile(
    r"(?i)\b(password|passwd|secret|token|credential|api[_-]?key)\s*[:=]\s*[^\s,;]+"
)

logger = logging.getLogger(__name__)


class ProcessingEventLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ProcessingEventType(str, Enum):
    PLAN_VALIDATION_STARTED = "PLAN_VALIDATION_STARTED"
    PLAN_VALIDATION_COMPLETED = "PLAN_VALIDATION_COMPLETED"
    EXECUTION_STARTED = "EXECUTION_STARTED"
    EXECUTION_COMPLETED = "EXECUTION_COMPLETED"
    EXECUTION_CANCEL_REQUESTED = "EXECUTION_CANCEL_REQUESTED"
    EXECUTION_CANCELLED = "EXECUTION_CANCELLED"
    STEP_STARTED = "STEP_STARTED"
    STEP_COMPLETED = "STEP_COMPLETED"
    STEP_SKIPPED = "STEP_SKIPPED"
    STEP_FAILED = "STEP_FAILED"
    SAMPLE_STARTED = "SAMPLE_STARTED"
    SAMPLE_COMPLETED = "SAMPLE_COMPLETED"
    SAMPLE_SKIPPED = "SAMPLE_SKIPPED"
    SAMPLE_FAILED = "SAMPLE_FAILED"
    ARTIFACT_CREATED = "ARTIFACT_CREATED"
    DATASET_PREPARATION_STARTED = "DATASET_PREPARATION_STARTED"
    DATASET_SAMPLE_VALIDATED = "DATASET_SAMPLE_VALIDATED"
    DATASET_SAMPLE_REJECTED = "DATASET_SAMPLE_REJECTED"
    DATASET_SPLIT_CREATED = "DATASET_SPLIT_CREATED"
    DATASET_STAGED = "DATASET_STAGED"
    DATASET_VALIDATED = "DATASET_VALIDATED"
    DATASET_COMMITTED = "DATASET_COMMITTED"
    DATASET_PREPARATION_FAILED = "DATASET_PREPARATION_FAILED"
    ANNOTATION_PACKAGE_STARTED = "ANNOTATION_PACKAGE_STARTED"
    ANNOTATION_PACKAGE_CREATED = "ANNOTATION_PACKAGE_CREATED"
    ANNOTATION_TOOL_LAUNCHED = "ANNOTATION_TOOL_LAUNCHED"
    ANNOTATION_TOOL_EXITED = "ANNOTATION_TOOL_EXITED"
    ANNOTATION_VALIDATION_STARTED = "ANNOTATION_VALIDATION_STARTED"
    ANNOTATION_ITEM_VALIDATED = "ANNOTATION_ITEM_VALIDATED"
    ANNOTATION_ITEM_REJECTED = "ANNOTATION_ITEM_REJECTED"
    ANNOTATION_REVISION_CREATED = "ANNOTATION_REVISION_CREATED"
    ANNOTATION_PACKAGE_COMPLETED = "ANNOTATION_PACKAGE_COMPLETED"
    ANNOTATION_PACKAGE_INCOMPLETE = "ANNOTATION_PACKAGE_INCOMPLETE"
    ANNOTATION_PACKAGE_FAILED = "ANNOTATION_PACKAGE_FAILED"
    ANNOTATION_REVISION_REVOKED = "ANNOTATION_REVISION_REVOKED"
    COLOR_PACKAGE_STARTED = "COLOR_PACKAGE_STARTED"
    COLOR_PACKAGE_CREATED = "COLOR_PACKAGE_CREATED"
    COLOR_PROPOSAL_CREATED = "COLOR_PROPOSAL_CREATED"
    COLOR_PROPOSAL_INSUFFICIENT_DATA = "COLOR_PROPOSAL_INSUFFICIENT_DATA"
    COLOR_PREVIEW_CREATED = "COLOR_PREVIEW_CREATED"
    COLOR_GATE_PASSED = "COLOR_GATE_PASSED"
    COLOR_GATE_FAILED = "COLOR_GATE_FAILED"
    COLOR_SCOPE_APPROVED = "COLOR_SCOPE_APPROVED"
    COLOR_SCOPE_REJECTED = "COLOR_SCOPE_REJECTED"
    COLOR_REVISION_CREATED = "COLOR_REVISION_CREATED"
    COLOR_REVISION_ACTIVATED = "COLOR_REVISION_ACTIVATED"
    COLOR_ACTIVATION_FAILED = "COLOR_ACTIVATION_FAILED"
    COLOR_PACKAGE_COMPLETED = "COLOR_PACKAGE_COMPLETED"
    COLOR_PACKAGE_PARTIAL = "COLOR_PACKAGE_PARTIAL"
    COLOR_PACKAGE_FAILED = "COLOR_PACKAGE_FAILED"
    COLOR_REVISION_REVOKED = "COLOR_REVISION_REVOKED"
    COLOR_REVISION_ROLLED_BACK = "COLOR_REVISION_ROLLED_BACK"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class ProcessingEvent:
    event_id: str
    plan_id: str
    report_id: str
    timestamp: datetime
    sequence: int
    level: ProcessingEventLevel
    event_type: ProcessingEventType
    stage: str
    message: str
    sample_id: str = ""
    step_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id.strip() or not self.plan_id.strip() or not self.report_id.strip():
            raise ValueError("event_id, plan_id, and report_id must not be empty")
        if self.sequence < 1:
            raise ValueError("event sequence must start at 1")
        if not _is_aware_utc(self.timestamp):
            raise ValueError("event timestamp must be timezone-aware UTC")
        sanitized = _validate_and_freeze_metadata(self.metadata)
        object.__setattr__(self, "metadata", sanitized)
        object.__setattr__(self, "message", redact_sensitive_text(self.message))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_id": self.event_id,
            "plan_id": self.plan_id,
            "report_id": self.report_id,
            "timestamp": self.timestamp.isoformat(),
            "sequence": self.sequence,
            "level": self.level.value,
            "event_type": self.event_type.value,
            "stage": self.stage,
            "sample_id": self.sample_id or None,
            "step_id": self.step_id or None,
            "message": self.message,
            "metadata": _thaw(self.metadata),
        }


class ProcessingEventSinkError(RuntimeError):
    """Raised when one or more sinks cannot preserve an event."""


class ProcessingEventSink(ABC):
    @abstractmethod
    def emit(self, event: ProcessingEvent) -> None:
        """Persist or publish one immutable event."""


class InMemoryEventSink(ProcessingEventSink):
    def __init__(self) -> None:
        self._events: list[ProcessingEvent] = []
        self._event_ids: set[str] = set()
        self._lock = threading.Lock()

    def emit(self, event: ProcessingEvent) -> None:
        with self._lock:
            if event.event_id in self._event_ids:
                raise ProcessingEventSinkError(
                    f"Duplicate processing event ID: {event.event_id}"
                )
            if self._events and event.sequence <= self._events[-1].sequence:
                raise ProcessingEventSinkError(
                    "Processing event sequence must be strictly increasing"
                )
            self._event_ids.add(event.event_id)
            self._events.append(event)

    @property
    def events(self) -> tuple[ProcessingEvent, ...]:
        with self._lock:
            return tuple(self._events)


class JsonlEventSink(ProcessingEventSink):
    """Append-only UTF-8 JSONL sink with per-event flush and optional fsync."""

    def __init__(self, path: str | Path, *, fsync: bool = False) -> None:
        self.path = Path(path)
        self.fsync = fsync
        self._event_ids: set[str] = set()
        self._last_sequence = 0
        self._lock = threading.Lock()

    def emit(self, event: ProcessingEvent) -> None:
        encoded = json.dumps(
            event.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._lock:
            if event.event_id in self._event_ids:
                raise ProcessingEventSinkError(
                    f"Duplicate processing event ID: {event.event_id}"
                )
            if event.sequence <= self._last_sequence:
                raise ProcessingEventSinkError(
                    "Processing event sequence must be strictly increasing"
                )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(encoded + "\n")
                    handle.flush()
                    if self.fsync:
                        os.fsync(handle.fileno())
            except OSError as exc:
                raise ProcessingEventSinkError(
                    f"Could not append processing event to {self.path}: {exc}"
                ) from exc
            self._event_ids.add(event.event_id)
            self._last_sequence = event.sequence


class CompositeEventSink(ProcessingEventSink):
    """Fan out to all sinks and report failures after healthy sinks receive data."""

    def __init__(self, sinks: Sequence[ProcessingEventSink]) -> None:
        self._sinks = tuple(sinks)

    def emit(self, event: ProcessingEvent) -> None:
        failures: list[str] = []
        for sink in self._sinks:
            try:
                sink.emit(event)
            except (ProcessingEventSinkError, OSError, ValueError) as exc:
                failures.append(f"{type(sink).__name__}: {exc}")
            except Exception as exc:  # boundary: third-party subscriber isolation
                failures.append(
                    f"{type(sink).__name__}: {type(exc).__name__}: {redact_sensitive_text(str(exc))}"
                )
        if failures:
            raise ProcessingEventSinkError("; ".join(failures))


class ProcessingEventPublisher:
    """Create ordered events and isolate subscriber failures from execution."""

    def __init__(
        self,
        *,
        plan_id: str,
        report_id: str,
        sink: ProcessingEventSink,
        clock: Callable[[], datetime],
        id_generator: Callable[[], str],
    ) -> None:
        self.plan_id = plan_id
        self.report_id = report_id
        self._sink = sink
        self._clock = clock
        self._id_generator = id_generator
        self._sequence = 0
        self._warnings: list[str] = []
        self._lock = threading.Lock()

    @property
    def warnings(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._warnings)

    def emit(
        self,
        event_type: ProcessingEventType,
        message: str,
        *,
        stage: str,
        level: ProcessingEventLevel = ProcessingEventLevel.INFO,
        sample_id: str = "",
        step_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> ProcessingEvent:
        with self._lock:
            self._sequence += 1
            event = ProcessingEvent(
                event_id=self._id_generator(),
                plan_id=self.plan_id,
                report_id=self.report_id,
                timestamp=self._clock(),
                sequence=self._sequence,
                level=level,
                event_type=event_type,
                stage=stage,
                sample_id=sample_id,
                step_id=step_id,
                message=message,
                metadata=metadata or {},
            )
            try:
                self._sink.emit(event)
            except (ProcessingEventSinkError, OSError, ValueError) as exc:
                warning = f"event_sink_failure: {redact_sensitive_text(str(exc))}"
                self._warnings.append(warning)
                logger.error(
                    "Processing event sink failed plan_id=%s report_id=%s sequence=%s error=%s",
                    self.plan_id,
                    self.report_id,
                    event.sequence,
                    warning,
                )
            except Exception as exc:  # boundary: subscriber plugins must not stop a run
                warning = (
                    "event_sink_failure: "
                    f"{type(exc).__name__}: {redact_sensitive_text(str(exc))}"
                )
                self._warnings.append(warning)
                logger.exception(
                    "Unexpected processing event sink failure plan_id=%s report_id=%s sequence=%s",
                    self.plan_id,
                    self.report_id,
                    event.sequence,
                )
            return event


def redact_sensitive_text(value: str) -> str:
    return _SENSITIVE_TEXT.sub(lambda match: f"{match.group(1)}=[REDACTED]", str(value))


def exception_metadata(error: BaseException) -> dict[str, str]:
    """Serialize an exception without retaining traceback or exception objects."""
    return {
        "error_type": type(error).__name__,
        "error_message": redact_sensitive_text(str(error)),
    }


def _validate_and_freeze_metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    _assert_no_sensitive_keys(value)
    normalized = _normalize_json(value)
    json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    return _freeze(normalized)


def _assert_no_sensitive_keys(value: Any, path: str = "metadata") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _SENSITIVE_KEY.search(str(key)):
                raise ValueError(f"Sensitive metadata key is not allowed: {path}.{key}")
            _assert_no_sensitive_keys(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_sensitive_keys(item, f"{path}[{index}]")


def _normalize_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    raise TypeError(
        f"Processing event metadata is not JSON serializable: {type(value).__name__}"
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _is_aware_utc(value: datetime) -> bool:
    return (
        isinstance(value, datetime)
        and value.tzinfo is not None
        and value.utcoffset() == timezone.utc.utcoffset(value)
    )
