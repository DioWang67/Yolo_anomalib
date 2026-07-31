from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock
from uuid import uuid4

import pytest
from PyQt5.QtTest import QSignalSpy
from PyQt5.QtWidgets import QApplication

from app.gui.inspection_components_dialog import InspectionComponentsDialog
from app.gui.inspection_release_composer_dialog import (
    InspectionReleaseComposerDialog,
)
from app.gui.inspection_release_presentation import format_local_timestamp
from app.gui.inspection_releases_dialog import InspectionReleasesDialog
from app.gui.inspection_version_workspace import InspectionVersionWorkspace
from app.gui.model_versions_dialog import ModelVersionsDialog
from app.gui.panels.control_panel import ControlPanel
from core.services.inspection_component_catalog import (
    InspectionComponentRecord,
)
from core.services.inspection_release_models import (
    ActivationMode,
    ComponentBinding,
    InspectionRelease,
    InspectionScope,
    ReleaseStatus,
    ValidationEvidence,
)
from core.services.inspection_release_store import InspectionReleaseStore
from core.services.model_version_registry import ModelVersionRecord

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_release_timestamp_is_rendered_in_station_timezone() -> None:
    taipei = timezone(timedelta(hours=8))

    assert (
        format_local_timestamp(
            "2026-07-31T08:11:13.014762+00:00",
            local_timezone=taipei,
        )
        == "2026-07-31 16:11:13"
    )
    assert (
        format_local_timestamp(
            "2026-07-31T16:11:13+08:00",
            local_timezone=taipei,
        )
        == "2026-07-31 16:11:13"
    )


def test_release_timestamp_preserves_legacy_naive_wall_clock() -> None:
    taipei = timezone(timedelta(hours=8))

    assert (
        format_local_timestamp(
            "2026-07-31T16:11:13",
            local_timezone=taipei,
        )
        == "2026-07-31 16:11:13"
    )
    assert format_local_timestamp("") == "—"


@pytest.fixture(scope="module")
def qapp():
    application = QApplication.instance() or QApplication([])
    yield application


def _write(path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path.resolve()), hashlib.sha256(content).hexdigest()


def _commit_release(tmp_path):
    model, model_sha = _write(tmp_path / "artifacts" / "model.onnx", b"model")
    config, config_sha = _write(
        tmp_path / "artifacts" / "config.yaml", b"weights: model.onnx"
    )
    color, color_sha = _write(
        tmp_path / "artifacts" / "color.json", b'{"black": 0.6}'
    )
    report, report_sha = _write(tmp_path / "artifacts" / "report.json", b"{}")
    release = InspectionRelease(
        str(uuid4()),
        "inspection-v1.0.2",
        InspectionScope("Cable1", "A", "yolo_visual_v1"),
        (
            ComponentBinding(
                "yolo-model",
                "yolo",
                "primary_detector",
                "1.0.6",
                model,
                model_sha,
                config,
                config_sha,
            ),
            ComponentBinding(
                "stats-color",
                "stats_color",
                "color_check",
                "color-v1.0.2",
                config_path=color,
                config_sha256=color_sha,
                revision_overrides=(
                    ("0123456789abcdef01234567", str(uuid4())),
                ),
            ),
        ),
        ReleaseStatus.TESTED,
        datetime.now(timezone.utc).isoformat(),
        "tester",
        "candidate",
        ValidationEvidence(
            report,
            report_sha,
            "matrix-1",
            250,
            (
                ("confirmed", 250),
                ("tp", 77),
                ("tn", 159),
                ("fp", 14),
                ("fn", 0),
                ("errors", 0),
            ),
            (
                ("overkill_rate", 0.069),
                ("escape_rate", None),
            ),
        ),
    )
    store = InspectionReleaseStore(tmp_path / "store")
    return store, store.commit(release)


def test_dialog_displays_combination_and_enforces_limited_trial(tmp_path, qapp):
    store, release = _commit_release(tmp_path)
    dialog = InspectionReleasesDialog(
        store=store, product="Cable1", area="A"
    )
    try:
        assert dialog.table.rowCount() == 1
        assert dialog.table.item(0, 1).text() == "inspection-v1.0.2"
        assert "yolo 1.0.6" in dialog.table.item(0, 3).text()
        assert dialog.table.item(0, 4).text() == "94.40%"
        assert dialog.activate_btn.text() == "選擇啟用模式"
        assert store.policy.allowed_modes(release) == (
            ActivationMode.LIMITED_TRIAL,
            ActivationMode.RISK_ACCEPTED,
        )
    finally:
        dialog.close()


def test_control_panel_emits_release_manager_request(qapp, tmp_path):
    from PyQt5.QtCore import QSettings

    panel = ControlPanel(
        settings=QSettings(
            str(tmp_path / "settings.ini"), QSettings.IniFormat
        )
    )
    try:
        panel._engineering_access_granted = True
        panel.engineering_panel.setEnabled(True)
        panel.set_language("zh_TW")
        spy = QSignalSpy(panel.inspection_releases_requested)
        acceptance_spy = QSignalSpy(panel.acceptance_requested)

        panel.inspection_releases_btn.click()
        panel.acceptance_btn.click()

        assert len(spy) == 1
        assert len(acceptance_spy) == 1
        assert [
            panel.engineering_tabs.tabText(index)
            for index in range(panel.engineering_tabs.count())
        ] == ["檢測版本", "資料改善", "設備與系統"]
    finally:
        panel.close()


def test_composer_freely_selects_historical_yolo_version(tmp_path, qapp):
    records = []
    for version, current in (("1.0.6", True), ("1.0.5", False)):
        weight = tmp_path / f"model-{version}.onnx"
        config = tmp_path / f"model-{version}.config.yaml"
        weight.write_bytes(version.encode())
        config.write_text(f"weights: {weight.as_posix()}\n", encoding="utf-8")
        records.append(
            ModelVersionRecord(
                product="Cable1",
                area="A",
                model_type="yolo",
                version=version,
                weight_path=weight,
                is_current=current,
                trained_at=None,
                deployed_at=None,
                activated_at=None,
                training_time_inferred=False,
                config_snapshot_path=config,
                file_size=weight.stat().st_size,
            )
        )
    dialog = InspectionReleaseComposerDialog(
        models=tuple(records),
        color_revisions=(),
        suggested_version="inspection-v1.0.3",
    )
    try:
        dialog.model_combo.setCurrentIndex(1)
        assert dialog.selected_model() is records[1]
        assert dialog.selected_model().version == "1.0.5"
        assert dialog.selected_color_revision() is None
        assert dialog.display_version() == "inspection-v1.0.3"
    finally:
        dialog.close()


def test_managed_model_version_routes_to_combination_builder(tmp_path, qapp):
    weight = tmp_path / "model-1.0.5.onnx"
    config = tmp_path / "model-1.0.5.config.yaml"
    weight.write_bytes(b"model")
    config.write_text(f"weights: {weight.as_posix()}\n", encoding="utf-8")
    record = ModelVersionRecord(
        product="Cable1",
        area="A",
        model_type="yolo",
        version="1.0.5",
        weight_path=weight,
        is_current=False,
        trained_at=None,
        deployed_at=None,
        activated_at=None,
        training_time_inferred=False,
        config_snapshot_path=config,
        file_size=weight.stat().st_size,
    )
    registry = Mock()
    registry.list_versions.return_value = [record]
    use_in_combination = Mock()
    dialog = ModelVersionsDialog(
        registry,
        language="zh_TW",
        is_combination_managed=lambda _record: True,
        on_use_in_combination=use_in_combination,
    )
    try:
        assert dialog.activate_button.text() == "建立檢測組合"

        dialog._activate_selected()

        use_in_combination.assert_called_once_with(record)
        registry.activate.assert_not_called()
    finally:
        dialog.close()


def test_component_inventory_filters_models_and_colors(tmp_path, qapp):
    records = (
        InspectionComponentRecord(
            component_id="model:Cable1:A:yolo:1.0.6",
            category="AI_MODEL",
            component_type="yolo",
            product="Cable1",
            area="A",
            inference_type="yolo",
            version="1.0.6",
            status="DEPLOYED",
            created_at="2026-07-27T12:00:00+08:00",
            integrity="VERIFIED",
            source_path=tmp_path / "model.onnx",
            detail="model",
        ),
        InspectionComponentRecord(
            component_id="color:revision-2",
            category="COLOR_REVISION",
            component_type="Black 門檻",
            product="Cable1",
            area="A",
            inference_type="yolo",
            version="color-v1.0.2",
            status="HISTORY",
            created_at="2026-07-30T12:00:00+08:00",
            integrity="VERIFIED",
            source_path=tmp_path / "config.json",
            detail="color",
        ),
    )
    catalog = Mock()
    catalog.list_components.return_value = records
    create_combination = Mock()
    dialog = InspectionComponentsDialog(
        catalog=catalog,
        selected_product="Cable1",
        selected_area="A",
        selected_inference_type="yolo",
        on_create_combination=create_combination,
    )
    try:
        assert dialog.table.rowCount() == 2
        assert {
            dialog.table.item(row, 1).text()
            for row in range(dialog.table.rowCount())
        } == {"AI 模型", "校正修訂"}

        color_index = dialog.category_filter.findData("COLOR_REVISION")
        dialog.category_filter.setCurrentIndex(color_index)
        assert dialog.table.rowCount() == 1
        assert dialog.table.item(0, 6).text() == "color-v1.0.2"

        dialog._create_combination()
        create_combination.assert_called_once_with(records[1])
    finally:
        dialog.close()


def test_embedded_version_workspace_replaces_legacy_launchers(tmp_path, qapp):
    from PyQt5.QtCore import QSettings

    panel = ControlPanel(
        settings=QSettings(
            str(tmp_path / "settings.ini"), QSettings.IniFormat
        )
    )
    workspace = InspectionVersionWorkspace(
        project_root=tmp_path,
        parent=panel.version_workspace,
    )
    try:
        panel.install_version_workspace(workspace)

        assert panel.debug_group.isHidden()
        assert panel.version_workspace.isAncestorOf(workspace)
        assert [button.text() for button in workspace.stage_buttons] == [
            "1  元件版本",
            "2  候選組合",
            "3  組合驗證",
            "4  上線紀錄",
        ]
    finally:
        panel.close()


def test_embedded_workspace_adds_any_component_to_candidate(
    tmp_path, qapp
):
    records = (
        InspectionComponentRecord(
            component_id="model:Cable1:A:yolo:1.0.5",
            category="AI_MODEL",
            component_type="yolo",
            product="Cable1",
            area="A",
            inference_type="yolo",
            version="1.0.5",
            status="HISTORY",
            created_at="2026-07-27T12:00:00+08:00",
            integrity="VERIFIED",
            source_path=tmp_path / "model.onnx",
            detail="historical model",
        ),
        InspectionComponentRecord(
            component_id="color-base:Cable1:A:yolo:abcdef",
            category="COLOR_BASE",
            component_type="stats_color",
            product="Cable1",
            area="A",
            inference_type="yolo",
            version="base-abcdef12",
            status="DEPLOYED",
            created_at="2026-07-30T12:00:00+08:00",
            integrity="VERIFIED",
            source_path=tmp_path / "config.json",
            detail="color settings",
        ),
    )
    workspace = InspectionVersionWorkspace(project_root=tmp_path)
    workspace.catalog = Mock()
    workspace.catalog.list_components.return_value = records
    try:
        workspace.set_scope("Cable1", "A", "yolo")

        assert workspace.component_table.rowCount() == 2
        assert workspace.candidate_model_combo.findData(records[0].component_id) >= 0
        assert (
            workspace.candidate_color_combo.findData(
                records[1].component_id
            )
            >= 0
        )

        model_row = next(
            row
            for row in range(workspace.component_table.rowCount())
            if workspace.component_table.item(row, 2).text() == "yolo"
        )
        workspace.component_table.selectRow(model_row)
        workspace._add_selected_component()

        assert workspace.pages.currentIndex() == 1
        assert (
            workspace.candidate_model_combo.currentData()
            == records[0].component_id
        )
    finally:
        workspace.close()
