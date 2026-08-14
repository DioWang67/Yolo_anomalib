"""Append-only execution evidence for interactive model acceptance runs."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.services.acceptance_artifacts import AcceptanceArtifactBundle
from core.services.model_acceptance import AcceptanceInferenceOutcome


class AcceptanceRunError(RuntimeError):
    """Raised when an acceptance run transition or evidence write is invalid."""


class AcceptanceRunState(str, Enum):
    PREPARING = "PREPARING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class AcceptanceRun:
    run_id: str
    root: Path
    sample_ids: tuple[str, ...]
    sample_id_set: frozenset[str]
    source_manifest_sha256: str
    artifact_bundle_sha256: str


class AcceptanceRunRepository:
    """Persist one result file per sample and append-only state events."""

    def __init__(self, acceptance_root: str | Path):
        self.acceptance_root = Path(acceptance_root).expanduser().resolve()
        self.root = self.acceptance_root / "runs"

    def begin(
        self,
        *,
        artifact_bundle: AcceptanceArtifactBundle,
        sample_ids: tuple[str, ...],
        source_manifest_sha256: str,
    ) -> AcceptanceRun:
        if (
            not sample_ids
            or len(set(sample_ids)) != len(sample_ids)
            or any(not _is_safe_sample_id(sample_id) for sample_id in sample_ids)
        ):
            raise AcceptanceRunError(
                "An acceptance run requires unique, non-empty sample IDs."
            )
        self.root.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        run_id = f"run-{now.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:10]}"
        destination = self.root / run_id
        staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=self.root))
        committed = False
        try:
            (staging / "results").mkdir()
            (staging / "events").mkdir()
            request = {
                "schema_version": 1,
                "run_id": run_id,
                "created_at": now.isoformat(),
                "sample_ids": list(sample_ids),
                "source_manifest_sha256": source_manifest_sha256,
                "artifact_bundle": artifact_bundle.report_payload(),
            }
            _write_json_atomic(staging / "request.json", request)
            _write_json_atomic(
                staging / "events" / "0001-preparing.json",
                _event_payload(AcceptanceRunState.PREPARING),
            )
            _write_json_atomic(
                staging / "events" / "0002-running.json",
                _event_payload(AcceptanceRunState.RUNNING),
            )
            _write_json_atomic(
                staging / "state.json",
                _state_payload(AcceptanceRunState.RUNNING),
            )
            os.replace(staging, destination)
            committed = True
        finally:
            if not committed:
                shutil.rmtree(staging, ignore_errors=True)
        return AcceptanceRun(
            run_id=run_id,
            root=destination,
            sample_ids=sample_ids,
            sample_id_set=frozenset(sample_ids),
            source_manifest_sha256=source_manifest_sha256,
            artifact_bundle_sha256=artifact_bundle.bundle_sha256,
        )

    def append_outcome(
        self,
        run: AcceptanceRun,
        outcome: AcceptanceInferenceOutcome,
    ) -> Path:
        if outcome.sample_id not in run.sample_id_set:
            raise AcceptanceRunError(
                f"Outcome is outside the acceptance run: {outcome.sample_id}"
            )
        if self.state(run) is not AcceptanceRunState.RUNNING:
            raise AcceptanceRunError("Acceptance outcomes require a RUNNING run.")
        destination = run.root / "results" / f"{outcome.sample_id}.json"
        if destination.exists() or destination.is_symlink():
            raise AcceptanceRunError(
                f"Acceptance outcome already exists: {outcome.sample_id}"
            )
        payload = asdict(outcome)
        payload.pop("annotated_frame", None)
        _write_json_atomic(destination, payload)
        return destination

    def complete(
        self,
        run: AcceptanceRun,
        *,
        committed_manifest_sha256: str,
    ) -> None:
        result_ids = {
            path.stem
            for path in (run.root / "results").glob("*.json")
            if path.is_file() and not path.is_symlink()
        }
        if result_ids != set(run.sample_ids):
            raise AcceptanceRunError(
                "A completed acceptance run must contain every requested result."
            )
        self._transition(
            run,
            AcceptanceRunState.COMPLETED,
            details={"committed_manifest_sha256": committed_manifest_sha256},
        )

    def fail(self, run: AcceptanceRun, *, reason: str) -> None:
        self._transition(
            run,
            AcceptanceRunState.FAILED,
            details={"reason": reason.strip() or "acceptance inference failed"},
        )

    def cancel(self, run: AcceptanceRun) -> None:
        self._transition(run, AcceptanceRunState.CANCELLED)

    def state(self, run: AcceptanceRun) -> AcceptanceRunState:
        try:
            payload = json.loads((run.root / "state.json").read_text(encoding="utf-8"))
            return AcceptanceRunState(str(payload["state"]))
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise AcceptanceRunError(f"Acceptance run state is invalid: {run.run_id}") from exc

    def _transition(
        self,
        run: AcceptanceRun,
        target: AcceptanceRunState,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        if self.state(run) is not AcceptanceRunState.RUNNING:
            raise AcceptanceRunError(
                "Only a RUNNING acceptance run can reach a terminal state."
            )
        event_index = len(tuple((run.root / "events").glob("*.json"))) + 1
        event_payload = _event_payload(target, details=details)
        _write_json_atomic(
            run.root / "events" / f"{event_index:04d}-{target.value.lower()}.json",
            event_payload,
        )
        _write_json_atomic(
            run.root / "state.json",
            _state_payload(target, details=details),
        )


def _event_payload(
    state: AcceptanceRunState,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "state": state.value,
        "at": datetime.now(timezone.utc).isoformat(),
        "details": dict(details or {}),
    }


def _state_payload(
    state: AcceptanceRunState,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        **_event_payload(state, details=details),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _is_safe_sample_id(value: str) -> bool:
    return bool(value) and value not in {".", ".."} and all(
        character.isalnum() or character in {"-", "_"}
        for character in value
    )
