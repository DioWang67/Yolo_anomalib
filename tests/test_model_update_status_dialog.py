from __future__ import annotations

import json
from pathlib import Path

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
) -> None:
    job_dir = data_root / ".operator_handoff" / "jobs" / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "handoff.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "created_at": "2026-07-15T10:00:00+08:00",
                "targets": [{"product": product, "area": area}],
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "status.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "state": state,
                "updated_at": "2026-07-15T10:05:00+08:00",
                "progress": 35,
                "ready_count": 20,
                "pending_count": 0,
                "message": "模型訓練中",
            }
        ),
        encoding="utf-8",
    )


def test_load_model_update_jobs_keeps_corrupt_status_visible(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_job(data_root, "job-good", state="training")
    bad_dir = data_root / ".operator_handoff" / "jobs" / "job-bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "handoff.json").write_text(json.dumps({"job_id": "job-bad", "targets": []}), encoding="utf-8")
    (bad_dir / "status.json").write_text("not-json", encoding="utf-8")

    jobs = load_model_update_jobs(data_root)

    assert {job.job_id for job in jobs} == {"job-good", "job-bad"}
    assert next(job for job in jobs if job.job_id == "job-good").state == "training"
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


def test_confirmation_feedback_waits_without_counting_as_active(tmp_path: Path, qtbot) -> None:
    data_root = tmp_path / "data"
    _write_job(data_root, "job-feedback", state="waiting_feedback")
    dialog = ModelUpdateStatusDialog(data_root=data_root, language="zh_TW")
    qtbot.addWidget(dialog)

    assert dialog.table.item(0, 0).text() == "累積改善案例"
    assert "進行中 0 筆" in dialog.summary_label.text()
    assert "累積案例 1 筆" in dialog.summary_label.text()
