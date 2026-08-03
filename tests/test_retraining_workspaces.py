from __future__ import annotations

import json

import pytest

from tools.retraining_workspaces import (
    RetrainingWorkspaceError,
    create_retraining_workspace,
    list_retraining_workspaces,
    load_retraining_workspace,
    mark_retraining_workspace_submitted,
)


def test_retraining_workspaces_are_created_as_independent_job_folders(
    tmp_path,
) -> None:
    data_root = tmp_path / "data"

    first = create_retraining_workspace(
        data_root,
        product="Cable1",
        area="A",
        batch_version="Cable1_A_v0.0.1",
    )
    second = create_retraining_workspace(
        data_root,
        product="Cable1",
        area="A",
        batch_version="Cable1_A_v0.0.2",
    )
    first.manifest_path.write_text("sample_id\nfirst\n", encoding="utf-8")
    second.manifest_path.write_text("sample_id\nsecond\n", encoding="utf-8")

    assert first.root == (
        data_root / ".operator_handoff" / "jobs" / "Cable1_A_v0.0.1"
    ).resolve()
    assert first.root != second.root
    assert "first" in first.manifest_path.read_text(encoding="utf-8")
    assert "second" not in first.manifest_path.read_text(encoding="utf-8")
    assert [
        workspace.batch_version
        for workspace in list_retraining_workspaces(
            data_root,
            product="Cable1",
            area="A",
        )
    ] == ["Cable1_A_v0.0.2", "Cable1_A_v0.0.1"]


def test_retraining_workspace_version_cannot_be_created_twice(tmp_path) -> None:
    data_root = tmp_path / "data"
    create_retraining_workspace(
        data_root,
        product="Cable1",
        area="A",
        batch_version="Cable1_A_v0.0.1",
    )

    with pytest.raises(ValueError, match="已存在"):
        create_retraining_workspace(
            data_root,
            product="Cable1",
            area="A",
            batch_version="Cable1_A_v0.0.1",
        )


def test_retraining_workspace_is_marked_submitted_in_the_same_folder(
    tmp_path,
) -> None:
    workspace = create_retraining_workspace(
        tmp_path / "data",
        product="Cable1",
        area="A",
        batch_version="Cable1_A_v0.0.1",
    )
    handoff_path = workspace.root / "handoff.json"
    handoff_path.write_text(json.dumps({"job_id": workspace.batch_version}))

    submitted = mark_retraining_workspace_submitted(
        workspace.root,
        job_id=workspace.batch_version,
        handoff_path=handoff_path,
    )

    assert submitted.state == "submitted"
    assert submitted.handoff_path == handoff_path.resolve()
    assert load_retraining_workspace(workspace.root).state == "submitted"


def test_retraining_workspace_rejects_handoff_from_another_folder(tmp_path) -> None:
    workspace = create_retraining_workspace(
        tmp_path / "data",
        product="Cable1",
        area="A",
        batch_version="Cable1_A_v0.0.1",
    )
    foreign_handoff = tmp_path / "handoff.json"
    foreign_handoff.write_text("{}", encoding="utf-8")

    with pytest.raises(RetrainingWorkspaceError, match="不在原始批次資料夾"):
        mark_retraining_workspace_submitted(
            workspace.root,
            job_id=workspace.batch_version,
            handoff_path=foreign_handoff,
        )
