from __future__ import annotations

from app.gui.retraining_workspace_host import RetrainingWorkspaceHost


def test_retraining_host_requires_folder_before_selecting_images(tmp_path, qtbot):
    host = RetrainingWorkspaceHost(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "legacy-review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        product="Cable1",
        area="A",
        available_targets=[("Cable1", "A")],
    )
    qtbot.addWidget(host)

    assert host.batch_filter.count() == 0
    assert "請先建立補訓資料夾" in host.status_label.text()
    assert host.workspace is None


def test_retraining_host_creates_and_opens_independent_folder(
    tmp_path,
    qtbot,
    monkeypatch,
):
    host = RetrainingWorkspaceHost(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "legacy-review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        product="Cable1",
        area="A",
        available_targets=[("Cable1", "A")],
    )
    qtbot.addWidget(host)
    monkeypatch.setattr(
        "app.gui.retraining_workspace_host.QInputDialog.getText",
        lambda *_args, **_kwargs: ("Cable1_A_v0.0.1", True),
    )

    with qtbot.waitSignal(host.workspace_ready, timeout=10_000):
        host._create_batch_workspace()

    assert host.batch_filter.count() == 1
    assert host.active_workspace is not None
    assert host.active_workspace.batch_version == "Cable1_A_v0.0.1"
    assert host.workspace is not None
    assert host.workspace.batch_version == "Cable1_A_v0.0.1"
    assert host.active_workspace.root.is_dir()
