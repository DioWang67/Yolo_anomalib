from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from app.gui.model_update_status_dialog import load_model_update_jobs
from tools.operator_job_control import request_operator_job_cancel


def _load_training_operator_job_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "Yolo11_auto_train"
        / "src"
        / "picture_tool"
        / "operator_job.py"
    )
    module_name = "_offline_training_operator_job_contract"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load training job contract: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_cross_project_heartbeat_cancel_and_lock_release_contract(
    tmp_path: Path,
) -> None:
    training_jobs = _load_training_operator_job_module()
    data_root = tmp_path / "data"
    job_id = "offline-job-1"
    job_dir = data_root / ".operator_handoff" / "jobs" / job_id
    job_dir.mkdir(parents=True)
    handoff_path = job_dir / "handoff.json"
    status_path = job_dir / "status.json"
    handoff_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "job_id": job_id,
                "created_at": "2026-07-22T00:00:00+00:00",
                "targets": [{"product": "Cable1", "area": "A"}],
                "training_options": {
                    "epochs": 20,
                    "augmentations_per_image": 20,
                    "batch": 8,
                    "imgsz": 640,
                },
            }
        ),
        encoding="utf-8",
    )
    status_path.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "state": "queued",
                "created_at": "2026-07-22T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    lock = training_jobs.acquire_target_training_lock(
        data_root,
        product="Cable1",
        area="A",
        job_id=job_id,
        timeout_seconds=0,
    )
    training_jobs.update_job_status(
        status_path,
        state="training",
        message="模型訓練中：Epoch 3/20",
        progress=45,
        current_task="yolo_train",
    )
    assert training_jobs.refresh_operator_job_lease(
        status_path,
        job_id=job_id,
        lock=lock,
    ) is None

    visible = load_model_update_jobs(data_root)
    assert len(visible) == 1
    assert visible[0].state == "training"
    assert visible[0].progress == 45
    assert visible[0].heartbeat_at is not None

    first_request = request_operator_job_cancel(status_path, job_id=job_id)
    repeated_request = request_operator_job_cancel(status_path, job_id=job_id)
    assert repeated_request.request_id == first_request.request_id
    assert repeated_request.reused_existing is True

    control = training_jobs.refresh_operator_job_lease(
        status_path,
        job_id=job_id,
        lock=lock,
    )
    assert control is not None
    assert control.request_id == first_request.request_id
    training_jobs.update_job_status(
        status_path,
        state="cancelling",
        message="正在安全停止模型更新",
        handled_control_request_id=control.request_id,
    )
    assert training_jobs.refresh_operator_job_lease(
        status_path,
        job_id=job_id,
        lock=lock,
    ) is None

    training_jobs.update_job_status(
        status_path,
        state="cancelled",
        message="模型更新已安全停止",
        handled_control_request_id=control.request_id,
        error="",
    )
    training_jobs.release_target_training_lock(lock)

    final_jobs = load_model_update_jobs(data_root)
    assert final_jobs[0].state == "cancelled"
    assert final_jobs[0].cancel_request_pending is False
    assert not lock.path.exists()
