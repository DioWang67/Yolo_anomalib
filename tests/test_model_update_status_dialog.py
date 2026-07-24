from __future__ import annotations

import json
import time
from pathlib import Path

from PyQt5.QtWidgets import QMessageBox

from app.gui.model_update_status_dialog import (
    ModelUpdateStatusDialog,
    load_model_update_jobs,
    workflow_step_index,
)


def _write_job(
    data_root: Path,
    job_id: str,
    *,
    state: str,
    product: str = "Cable1",
    area: str = "A",
    training_options: dict[str, int] | None = None,
    status_values: dict[str, object] | None = None,
) -> None:
    job_dir = data_root / ".operator_handoff" / "jobs" / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "handoff.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "created_at": "2026-07-15T10:00:00+08:00",
                "targets": [{"product": product, "area": area}],
                "training_options": training_options or {},
            }
        ),
        encoding="utf-8",
    )
    status = {
        "job_id": job_id,
        "state": state,
        "updated_at": "2026-07-15T10:05:00+08:00",
        "progress": 35,
        "ready_count": 20,
        "pending_count": 0,
        "message": "模型訓練中",
    }
    status.update(status_values or {})
    (job_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")


def test_load_model_update_jobs_keeps_corrupt_status_visible(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_job(
        data_root,
        "job-good",
        state="training",
        training_options={
            "epochs": 80,
            "augmentations_per_image": 7,
            "batch": 4,
            "imgsz": 960,
        },
    )
    bad_dir = data_root / ".operator_handoff" / "jobs" / "job-bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "handoff.json").write_text(json.dumps({"job_id": "job-bad", "targets": []}), encoding="utf-8")
    (bad_dir / "status.json").write_text("not-json", encoding="utf-8")

    jobs = load_model_update_jobs(data_root)

    assert {job.job_id for job in jobs} == {"job-good", "job-bad"}
    good_job = next(job for job in jobs if job.job_id == "job-good")
    assert good_job.state == "training"
    assert (
        good_job.epochs,
        good_job.augmentations_per_image,
        good_job.batch,
        good_job.imgsz,
    ) == (80, 7, 4, 960)
    assert next(job for job in jobs if job.job_id == "job-bad").state == "invalid"


def test_status_dialog_filters_by_state(tmp_path: Path, qtbot) -> None:
    data_root = tmp_path / "data"
    _write_job(data_root, "job-training", state="training")
    _write_job(data_root, "job-done", state="deployed")
    dialog = ModelUpdateStatusDialog(data_root=data_root, language="zh_TW")
    qtbot.addWidget(dialog)

    dialog.state_filter.setCurrentIndex(dialog.state_filter.findData("training"))

    assert dialog.table.rowCount() == 1
    assert dialog.table.item(0, 9).text() == "job-training"


def test_operator_workflow_uses_the_same_five_step_contract() -> None:
    assert workflow_step_index("waiting_annotation") == 1
    assert workflow_step_index("training") == 2
    assert workflow_step_index("evaluating") == 3
    assert workflow_step_index("deploying") == 4
    assert workflow_step_index("deployed") == 4


def test_failed_job_keeps_the_step_of_its_last_task() -> None:
    assert workflow_step_index("failed", "yolo_evaluation") == 3
    assert workflow_step_index("failed", "deploy") == 4


def test_waiting_job_exposes_resume_when_previous_window_is_gone(
    tmp_path: Path,
    qtbot,
) -> None:
    data_root = tmp_path / "data"
    _write_job(data_root, "job-waiting", state="waiting_annotation")
    dialog = ModelUpdateStatusDialog(data_root=data_root, language="zh_TW")
    qtbot.addWidget(dialog)

    assert dialog.resume_button.isEnabled() is True


def test_completed_job_cannot_be_resumed(tmp_path: Path, qtbot) -> None:
    data_root = tmp_path / "data"
    _write_job(data_root, "job-done", state="deployed")
    dialog = ModelUpdateStatusDialog(data_root=data_root, language="zh_TW")
    qtbot.addWidget(dialog)

    assert dialog.resume_button.isEnabled() is False


def test_failed_job_can_be_retried_after_correction(tmp_path: Path, qtbot) -> None:
    data_root = tmp_path / "data"
    _write_job(data_root, "job-failed", state="failed")
    dialog = ModelUpdateStatusDialog(data_root=data_root, language="zh_TW")
    qtbot.addWidget(dialog)

    assert dialog.resume_button.isEnabled() is True
    assert dialog.resume_button.text() == "修正後重新嘗試"


def test_dead_training_process_is_projected_as_retryable_failure(
    tmp_path: Path, qtbot, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    _write_job(data_root, "job-interrupted", state="training")
    status_path = (
        data_root
        / ".operator_handoff"
        / "jobs"
        / "job-interrupted"
        / "status.json"
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "training_process_id": 4321,
            "training_process_host": "test-host",
            "current_task": "yolo_train",
        }
    )
    status_path.write_text(json.dumps(status), encoding="utf-8")
    monkeypatch.setattr(
        "app.gui.model_update_status_dialog.is_process_active",
        lambda _pid, _host: False,
    )

    dialog = ModelUpdateStatusDialog(data_root=data_root, language="zh_TW")
    qtbot.addWidget(dialog)

    assert dialog.jobs[0].state == "failed"
    assert dialog.jobs[0].message == "補訓程序已中斷，可從此工作重新嘗試"
    assert dialog.resume_button.isEnabled() is True


def test_retry_starts_retraining_in_background(tmp_path: Path, qtbot, monkeypatch) -> None:
    data_root = tmp_path / "data"
    _write_job(data_root, "job-failed", state="failed")
    launcher = tmp_path / "open_operator_training.bat"
    launcher.write_text("@echo off\n", encoding="utf-8")
    launches = []

    class DetachedProcess:
        @staticmethod
        def startDetached(program, arguments, working_directory):
            launches.append((program, arguments, working_directory))
            return True, 9876

    monkeypatch.setattr(
        "app.gui.model_update_status_dialog.QProcess",
        DetachedProcess,
    )
    dialog = ModelUpdateStatusDialog(data_root=data_root, language="zh_TW")
    qtbot.addWidget(dialog)

    dialog._resume_selected_job()

    job = dialog.jobs[0]
    assert launches == [
        (
            "cmd.exe",
            ["/c", str(launcher), str(job.handoff_path), "--background"],
            str(tmp_path),
        )
    ]
    assert "背景" in dialog.details_label.text()


def test_confirmation_feedback_waits_without_counting_as_active(tmp_path: Path, qtbot) -> None:
    data_root = tmp_path / "data"
    _write_job(data_root, "job-feedback", state="waiting_feedback")
    dialog = ModelUpdateStatusDialog(data_root=data_root, language="zh_TW")
    qtbot.addWidget(dialog)

    assert dialog.table.item(0, 0).text() == "累積改善案例"
    assert "進行中 0 筆" in dialog.summary_label.text()
    assert "累積案例 1 筆" in dialog.summary_label.text()


def test_completed_job_record_can_be_cleared_without_deleting_job_data(
    tmp_path: Path, qtbot, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    _write_job(data_root, "job-done", state="deployed")
    job_dir = data_root / ".operator_handoff" / "jobs" / "job-done"
    monkeypatch.setattr(
        "app.gui.model_update_status_dialog.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Yes,
    )
    dialog = ModelUpdateStatusDialog(data_root=data_root, language="zh_TW")
    qtbot.addWidget(dialog)

    assert dialog.clear_record_button.isEnabled() is True
    dialog._clear_selected_record()

    assert dialog.table.rowCount() == 0
    assert job_dir.is_dir()


def test_active_job_record_cannot_be_cleared(tmp_path: Path, qtbot) -> None:
    data_root = tmp_path / "data"
    _write_job(data_root, "job-training", state="training")
    dialog = ModelUpdateStatusDialog(data_root=data_root, language="zh_TW")
    qtbot.addWidget(dialog)

    assert dialog.clear_record_button.isEnabled() is False


def test_expired_heartbeat_is_visible_but_not_retryable(
    tmp_path: Path, qtbot, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    _write_job(
        data_root,
        "job-unresponsive",
        state="training",
        status_values={
            "training_process_id": 4321,
            "training_process_host": "test-host",
            "heartbeat_at": "2026-07-22T00:00:00+00:00",
            "heartbeat_timeout_seconds": 45,
            "current_task": "yolo_train",
        },
    )
    monkeypatch.setattr(
        "app.gui.model_update_status_dialog.is_process_active",
        lambda _pid, _host: True,
    )

    dialog = ModelUpdateStatusDialog(data_root=data_root, language="zh_TW")
    qtbot.addWidget(dialog)

    assert dialog.jobs[0].state == "unresponsive"
    assert dialog.resume_button.isEnabled() is False
    assert dialog.cancel_button.isEnabled() is True
    assert workflow_step_index("unresponsive", "yolo_train") == 2


def test_safe_stop_button_writes_control_request_without_changing_status(
    tmp_path: Path, qtbot, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    _write_job(data_root, "job-training", state="training")
    monkeypatch.setattr(
        "app.gui.model_update_status_dialog.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Yes,
    )
    dialog = ModelUpdateStatusDialog(data_root=data_root, language="zh_TW")
    qtbot.addWidget(dialog)

    dialog._cancel_selected_job()

    job_dir = data_root / ".operator_handoff" / "jobs" / "job-training"
    control = json.loads((job_dir / "control.json").read_text(encoding="utf-8"))
    status = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
    assert control["action"] == "cancel"
    assert control["job_id"] == "job-training"
    assert status["state"] == "training"
    assert dialog.cancel_button.isEnabled() is False


def test_auto_refresh_preserves_selected_job(tmp_path: Path, qtbot) -> None:
    data_root = tmp_path / "data"
    _write_job(data_root, "job-1", state="training")
    _write_job(data_root, "job-2", state="training")
    dialog = ModelUpdateStatusDialog(data_root=data_root, language="zh_TW")
    qtbot.addWidget(dialog)
    dialog.table.selectRow(1)
    selected_job_id = dialog._selected_job().job_id

    dialog.refresh_jobs()

    assert dialog._selected_job().job_id == selected_job_id


def test_large_job_history_refresh_stays_bounded(tmp_path: Path, qtbot) -> None:
    data_root = tmp_path / "data"
    for index in range(500):
        _write_job(data_root, f"job-{index:04d}", state="deployed")

    load_started = time.perf_counter()
    jobs = load_model_update_jobs(data_root)
    load_seconds = time.perf_counter() - load_started

    ui_started = time.perf_counter()
    dialog = ModelUpdateStatusDialog(data_root=data_root, language="zh_TW")
    qtbot.addWidget(dialog)
    for _iteration in range(3):
        dialog.refresh_jobs()
    ui_seconds = time.perf_counter() - ui_started

    assert len(jobs) == 500
    assert dialog.table.rowCount() == 100
    assert dialog.load_more_button.isHidden() is False
    dialog.load_more_button.click()
    assert dialog.table.rowCount() == 200
    assert load_seconds < 5.0
    assert ui_seconds < 10.0


def test_background_history_refresh_does_not_block_dialog_creation(
    tmp_path: Path, qtbot, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    _write_job(data_root, "job-background", state="training")
    from app.gui import model_update_status_dialog as status_module

    original_load = status_module.load_model_update_jobs

    def slow_load(root):
        time.sleep(0.4)
        return original_load(root)

    monkeypatch.setattr(status_module, "load_model_update_jobs", slow_load)
    started = time.perf_counter()
    dialog = ModelUpdateStatusDialog(
        data_root=data_root,
        language="zh_TW",
        background_refresh=True,
    )
    qtbot.addWidget(dialog)
    creation_seconds = time.perf_counter() - started

    assert creation_seconds < 0.2
    assert dialog.jobs == []
    qtbot.waitUntil(lambda: len(dialog.jobs) == 1, timeout=3000)
