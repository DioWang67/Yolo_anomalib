from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock
from uuid import uuid4

import pytest
from PyQt5.QtTest import QSignalSpy
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton

from app.gui.color_baseline_exclusions_dialog import (
    ColorBaselineExclusionsDialog,
)
from app.gui.color_baseline_rebuild_dialog import ColorBaselineRebuildDialog
from app.gui.inspection_components_dialog import InspectionComponentsDialog
from app.gui.inspection_release_composer_dialog import (
    InspectionReleaseComposerDialog,
)
from app.gui.inspection_release_presentation import (
    format_color_baseline_summary,
    format_local_timestamp,
)
from app.gui.inspection_releases_dialog import InspectionReleasesDialog
from app.gui.inspection_version_workspace import InspectionVersionWorkspace
from app.gui.model_versions_dialog import ModelVersionsDialog
from app.gui.panels.control_panel import ControlPanel
from core.services.color_baseline_evidence import (
    ColorBaselineEvidenceExclusion,
)
from core.services.color_baseline_recalibration import ColorBaselineColorReport
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


def test_color_baseline_summary_uses_human_status_labels() -> None:
    summary = format_color_baseline_summary(
        color_count=5,
        created_at="2026-08-03T05:12:13+00:00",
        lifecycle_status="CANDIDATE",
        quality_status="REVIEW_REQUIRED",
    )

    assert "完整 5 色基準" in summary
    assert "基準版本" in summary
    assert "候選基準" in summary
    assert "需要人工複核" in summary


def test_color_baseline_completion_hides_internal_id_from_primary_text():
    dialog = Mock()
    dialog.progress.maximum.return_value = 215
    candidate = Mock(
        colors=("Black", "Green", "Orange", "Red", "Yellow"),
        created_at="2026-08-04T05:12:13+00:00",
        display_version="color-base-e213e58d",
        status="REVIEW_REQUIRED",
    )

    ColorBaselineRebuildDialog._completed(dialog, candidate, ())

    phase_text = dialog.phase_label.setText.call_args.args[0]
    next_step_text = dialog.next_step_label.setText.call_args.args[0]
    assert "完整 5 色基準" in phase_text
    assert "需要人工複核" in phase_text
    assert "color-base-e213e58d" not in phase_text
    assert "color-base-e213e58d" not in next_step_text
    assert "color-base-e213e58d" in dialog.phase_label.setToolTip.call_args.args[0]


def test_color_baseline_completion_shows_automatic_preservation_without_metrics():
    dialog = Mock()
    dialog.progress.maximum.return_value = 215
    candidate = Mock(
        colors=("Black", "Green", "Orange", "Red", "Yellow"),
        created_at="2026-08-04T05:12:13+00:00",
        display_version="color-base-safe",
        status="READY",
    )
    report = ColorBaselineColorReport(
        color="Yellow",
        state="PRESERVED_SAFETY_REJECTED",
        total_crops=215,
        training_crops=172,
        holdout_crops=43,
        previous_holdout_correct=43,
        candidate_holdout_correct=43,
        hue_drift=0.0,
        lab_drift=0.0,
        note="自動安全檢查未通過，已沿用舊基準。",
        rejected_proposal_holdout_correct=42,
        rejected_proposal_hue_drift=17.4,
        rejected_proposal_lab_drift=75.8,
        rejection_reasons=("LAB_DRIFT_LIMIT_EXCEEDED",),
    )

    ColorBaselineRebuildDialog._completed(dialog, candidate, (report,))

    table_items = {
        (call.args[0], call.args[1]): call.args[2].text()
        for call in dialog.result_table.setItem.call_args_list
    }
    assert table_items[(0, 1)] == "未更新，沿用舊基準"
    assert table_items[(0, 7)] == "自動安全檢查未通過，已沿用舊基準。"
    assert "Hue" not in table_items[(0, 7)]
    assert "Lab" not in table_items[(0, 7)]


def test_color_baseline_exclusion_summary_marks_action_required():
    exclusion = ColorBaselineEvidenceExclusion(
        sample_id="case-1",
        source_kind="color_review",
        source_manifest="feedback.csv",
        image_path="case-1.jpg",
        image_sha256="a" * 64,
        reason_code="IMAGE_NOT_FOUND",
        reason="影像檔不存在。",
    )
    snapshot = Mock(
        excluded_samples=(exclusion,),
        samples=(),
        conflict_count=0,
        invalid_count=1,
        selected_count=10,
        selected_acceptance_count=8,
        selected_feedback_count=2,
        duplicate_count=0,
        confirmed_ng_count=3,
    )
    dialog = Mock()

    ColorBaselineRebuildDialog._show_evidence_summary(dialog, snapshot)

    assert dialog._excluded_evidence == (exclusion,)
    dialog.excluded_evidence_button.setText.assert_called_once_with(
        "查看排除照片（1）"
    )
    dialog.excluded_evidence_button.setVisible.assert_called_once_with(True)
    assert "已排除 1 張" in dialog.evidence_label.setText.call_args.args[0]


def test_mixed_product_ng_color_ok_is_listed_as_excluded(qapp):
    exclusion = ColorBaselineEvidenceExclusion(
        sample_id="case-1",
        source_kind="color_review",
        source_manifest="feedback.csv",
        image_path="case-1.jpg",
        image_sha256="a" * 64,
        reason_code="MIXED_PRODUCT_NG_COLOR_OK",
        reason="顏色 OK，但整體產品 NG。",
    )

    snapshot = Mock(
        excluded_samples=(exclusion,),
        samples=(),
        selected_count=215,
        selected_acceptance_count=173,
        selected_feedback_count=42,
        duplicate_count=0,
        confirmed_ng_count=78,
    )
    rebuild_dialog = Mock()
    ColorBaselineRebuildDialog._show_evidence_summary(
        rebuild_dialog,
        snapshot,
    )
    summary = rebuild_dialog.evidence_label.setText.call_args.args[0]
    assert "已排除 1 張" in summary

    details_dialog = ColorBaselineExclusionsDialog((exclusion,))
    try:
        assert "人工真值均未修改" in details_dialog.description_label.text()
        assert details_dialog.table.item(0, 1).text() == "case-1"
        assert details_dialog.table.item(0, 2).text() == exclusion.reason
    finally:
        details_dialog.close()


def test_statistical_outlier_is_added_to_excluded_photo_list(tmp_path, qapp):
    image_path = tmp_path / "outlier.jpg"
    image_path.write_bytes(b"image")
    sample = Mock(
        sample_id="ACC-OUTLIER",
        source_kind="acceptance",
        source_manifest="ground_truth.csv",
        image_path=image_path,
        image_sha256="a" * 64,
    )
    dialog = Mock()
    dialog._excluded_evidence = ()
    dialog._evidence_samples_by_id = {sample.sample_id: sample}
    dialog.excluded_evidence_button = QPushButton()
    dialog.evidence_label = QLabel("本次選用 215 張")
    report = Mock(
        excluded_sample_ids=(sample.sample_id,),
        excluded_count=1,
        status="AUTO_EXCLUDED",
    )

    ColorBaselineRebuildDialog._show_outlier_summary(dialog, report)

    assert len(dialog._excluded_evidence) == 1
    exclusion = dialog._excluded_evidence[0]
    assert exclusion.sample_id == "ACC-OUTLIER"
    assert exclusion.reason_code == "STATISTICAL_COLOR_OUTLIER"
    assert "重建前排除離群照片 1 張" in dialog.evidence_label.text()
    assert dialog.excluded_evidence_button.text() == "查看排除照片（1）"


@pytest.fixture(scope="module")
def qapp():
    application = QApplication.instance() or QApplication([])
    yield application


def test_color_baseline_exclusions_dialog_lists_and_opens_existing_images(
    tmp_path, qapp
):
    image_path = tmp_path / "case-1.jpg"
    image_path.write_bytes(b"image")
    exclusion = ColorBaselineEvidenceExclusion(
        sample_id="case-1",
        source_kind="color_review",
        source_manifest="feedback.csv",
        image_path=str(image_path),
        image_sha256="a" * 64,
        reason_code="IMAGE_SHA256_MISMATCH",
        reason="影像內容與 manifest 不一致。",
    )
    dialog = ColorBaselineExclusionsDialog((exclusion,))
    try:
        assert dialog.table.rowCount() == 1
        assert dialog.table.item(0, 1).text() == "case-1"
        assert dialog.table.item(0, 2).text() == "影像內容與 manifest 不一致。"
        assert "IMAGE_SHA256_MISMATCH" in dialog.table.item(0, 2).toolTip()
        assert dialog.open_image_button.isEnabled()
    finally:
        dialog.close()


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
        assert dialog.activate_btn.text() == "選擇上線模式"
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
        ] == ["版本與上線", "資料與補訓", "設備與系統"]
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
            "1  模型與顏色版本",
            "2  候選組合",
            "3  組合驗收",
            "4  上線與回退",
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


def test_candidate_color_configuration_hides_internal_ids_until_details(
    tmp_path, qapp
):
    model = InspectionComponentRecord(
        component_id="model:Cable1:A:yolo:1.0.6",
        category="AI_MODEL",
        component_type="yolo",
        product="Cable1",
        area="A",
        inference_type="yolo",
        version="1.0.6",
        status="DEPLOYED",
        created_at="2026-07-30T12:00:00+08:00",
        integrity="VERIFIED",
        source_path=tmp_path / "model.onnx",
        detail="model",
    )
    baseline = InspectionComponentRecord(
        component_id="color-base-candidate:860aed5f3a3535eda1b2ee94",
        category="COLOR_BASE",
        component_type="stats_color",
        product="Cable1",
        area="A",
        inference_type="yolo",
        version="color-base-860aed5f",
        status="HISTORY",
        created_at="2026-08-03T05:12:13+00:00",
        integrity="WARNING",
        source_path=tmp_path / "color_stats.json",
        detail=json.dumps(
            {
                "colors": ["Black", "Green", "Orange", "Red", "Yellow"],
                "color_count": 5,
                "candidate_status": "REVIEW_REQUIRED",
                "role": "BASELINE_CANDIDATE",
            }
        ),
    )
    black_revision = InspectionComponentRecord(
        component_id="color:black-revision",
        category="COLOR_REVISION",
        component_type="stats_color",
        product="Cable1",
        area="A",
        inference_type="yolo",
        version="color-v1.0.2",
        status="DEPLOYED",
        created_at="2026-07-31T02:00:00+00:00",
        integrity="VERIFIED",
        source_path=tmp_path / "black.json",
        detail="revision",
    )
    revision = Mock()
    revision.scope.threshold_key = "Black"
    revision.display_version = "color-v1.0.2"

    workspace = InspectionVersionWorkspace(project_root=tmp_path)
    workspace.catalog = Mock()
    workspace.catalog.list_components.return_value = (
        model,
        baseline,
        black_revision,
    )

    def index_revision() -> None:
        workspace._color_revisions = {black_revision.component_id: revision}

    workspace._index_color_revisions = index_revision
    try:
        workspace.set_scope("Cable1", "A", "yolo")

        summary = workspace.candidate_color_summary.text()
        assert "顏色設定" in summary
        assert "完整 5 色基準" in summary
        assert "候選基準" in summary
        assert "需要人工複核" in summary
        assert "Black：color-v1.0.2" in summary
        assert "860aed5f" not in summary
        assert "860aed5f" not in workspace.candidate_color_combo.currentText()
        assert "color-base-860aed5f" in workspace.candidate_color_summary.toolTip()
        assert workspace.candidate_color_advanced_panel.isHidden()

        workspace.candidate_color_details_button.click()

        assert not workspace.candidate_color_advanced_panel.isHidden()
        black_combo = workspace._override_combos["black"]
        assert black_combo.itemText(0) == "沿用上方完整基準"

        workspace.candidate_color_combo.setCurrentIndex(0)

        assert workspace.candidate_color_summary.text() == "不套用顏色檢查"
        assert not black_combo.isEnabled()
    finally:
        workspace.close()
