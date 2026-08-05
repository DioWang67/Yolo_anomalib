"""Embedded engineering workflow for inspection version management."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.gui.color_baseline_rebuild_dialog import (
    ColorBaselineRebuildDialog,
)
from app.gui.inspection_release_presentation import (
    format_color_baseline_summary,
    format_component_summary,
    format_local_timestamp,
)
from core.services.color_profile_store import ColorProfileStore
from core.services.inspection_component_catalog import (
    InspectionComponentCatalog,
    InspectionComponentRecord,
)
from core.services.inspection_release_builder import build_draft_release
from core.services.inspection_release_models import (
    ActivationMode,
    InspectionRelease,
    InspectionReleaseError,
    ReleaseStatus,
)
from core.services.inspection_release_store import InspectionReleaseStore
from core.services.model_acceptance import AcceptanceRepository
from core.services.model_version_registry import ModelVersionRegistry
from core.station_data import load_station_data_paths
from core.workspace import load_workspace_paths
from tools.color_configuration_revisions import (
    ColorConfigurationRevision,
    ColorConfigurationRevisionStore,
)

_RECORD_ROLE = Qt.UserRole
_RELEASE_ROLE = Qt.UserRole + 1
_CATEGORY_LABELS = {
    "AI_MODEL": "AI 模型",
    "COLOR_BASE": "顏色基準",
    "COLOR_PROFILE": "顏色設定",
    "COLOR_REVISION": "校正修訂",
}
_STATUS_LABELS = {
    "DEPLOYED": "正式組合使用中",
    "DEFAULT": "子系統預設",
    "HISTORY": "歷史版本",
    "REVOKED": "已撤銷",
}


def _detail_value(
    record: InspectionComponentRecord,
    key: str,
) -> str:
    try:
        payload = json.loads(record.detail)
    except (TypeError, json.JSONDecodeError):
        return ""
    return str(payload.get(key) or "")


class InspectionVersionWorkspace(QWidget):
    """Manage components, candidates, validation and deployment in one page."""

    validation_requested = pyqtSignal()
    quick_validation_requested = pyqtSignal(object)
    advanced_settings_requested = pyqtSignal()
    release_activated = pyqtSignal(object)

    def __init__(
        self,
        *,
        project_root: str | Path,
        is_inspection_running: Callable[[], bool] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project_root = Path(project_root).resolve()
        self.data_paths = load_station_data_paths(self.project_root)
        self.is_inspection_running = is_inspection_running or (lambda: False)
        self.release_store = InspectionReleaseStore(self.data_paths.inspection_releases)
        self.catalog = InspectionComponentCatalog(
            models_root=self.data_paths.models,
            color_revisions_root=self.data_paths.color_revisions,
            color_profiles_root=self.data_paths.color_profiles,
            color_baselines_root=self.data_paths.color_baselines,
            inspection_releases_root=self.data_paths.inspection_releases,
        )
        self.color_profile_store = ColorProfileStore(self.data_paths.color_profiles)
        self.color_store = ColorConfigurationRevisionStore(root=self.data_paths.color_revisions)
        self.product = ""
        self.area = ""
        self.inference_type = ""
        self._components: tuple[InspectionComponentRecord, ...] = ()
        self._releases: tuple[InspectionRelease, ...] = ()
        self._color_revisions: dict[str, ColorConfigurationRevision] = {}
        self._override_combos: dict[str, QComboBox] = {}
        self._candidate_initialized = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.production_summary = QLabel("目前正式組合：—")
        self.production_summary.setWordWrap(True)
        self.production_summary.setStyleSheet(
            "background:#edf6ed;color:#246b36;border:1px solid #bad8bf;border-radius:6px;padding:10px;font-weight:600;"
        )
        layout.addWidget(self.production_summary)

        navigation = QHBoxLayout()
        self.stage_group = QButtonGroup(self)
        self.stage_group.setExclusive(True)
        self.stage_buttons: list[QPushButton] = []
        for index, text in enumerate(
            ("1  模型與顏色版本", "2  候選組合", "3  組合驗收", "4  上線與回退")
        ):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setMinimumHeight(38)
            button.clicked.connect(lambda _checked, page=index: self._show_stage(page))
            self.stage_group.addButton(button, index)
            self.stage_buttons.append(button)
            navigation.addWidget(button)
        layout.addLayout(navigation)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_components_page())
        self.pages.addWidget(self._build_candidates_page())
        self.pages.addWidget(self._build_validation_page())
        self.pages.addWidget(self._build_deployment_page())
        layout.addWidget(self.pages, 1)
        self.stage_buttons[0].setChecked(True)

    def _build_components_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        filters = QHBoxLayout()
        filters.addWidget(QLabel("版本類別"))
        self.component_category_filter = QComboBox()
        self.component_category_filter.addItem("全部", "")
        self.component_category_filter.addItem("AI 模型", "AI_MODEL")
        self.component_category_filter.addItem("顏色基準", "COLOR_BASE")
        self.component_category_filter.addItem("顏色設定", "COLOR_PROFILE")
        self.component_category_filter.addItem("校正修訂", "COLOR_REVISION")
        self.component_category_filter.currentIndexChanged.connect(self._render_components)
        filters.addWidget(self.component_category_filter)
        filters.addWidget(QLabel("檢測類型"))
        self.component_type_filter = QComboBox()
        self.component_type_filter.currentIndexChanged.connect(self._render_components)
        filters.addWidget(self.component_type_filter)
        filters.addStretch(1)
        refresh_button = QPushButton("重新整理")
        refresh_button.clicked.connect(self.refresh)
        filters.addWidget(refresh_button)
        layout.addLayout(filters)

        self.component_table = self._table(
            ("狀態", "類別", "檢測項目", "版本", "完整性", "建立時間", "來源")
        )
        self.component_table.itemSelectionChanged.connect(self._update_component_details)
        self.component_table.itemDoubleClicked.connect(lambda _item: self._add_selected_component())
        layout.addWidget(self.component_table, 1)

        self.component_details = QLabel("請選取模型或顏色版本。")
        self.component_details.setWordWrap(True)
        self.component_details.setMaximumHeight(90)
        self.component_details.setStyleSheet("background:#eef2f6;border:1px solid #c8d1dc;padding:8px;")
        layout.addWidget(self.component_details)
        actions = QHBoxLayout()
        actions.addStretch(1)
        self.add_component_button = QPushButton("加入候選組合")
        self.add_component_button.clicked.connect(self._add_selected_component)
        actions.addWidget(self.add_component_button)
        self.component_settings_button = QPushButton("編輯目前檢測參數")
        self.component_settings_button.clicked.connect(self.advanced_settings_requested.emit)
        actions.addWidget(self.component_settings_button)
        layout.addLayout(actions)
        return page

    def _build_candidates_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.candidate_model_combo = QComboBox()
        self.candidate_color_combo = QComboBox()
        self.candidate_color_combo.currentIndexChanged.connect(
            self._update_color_configuration_summary
        )
        self.candidate_color_summary = QLabel("不套用顏色檢查")
        self.candidate_color_summary.setObjectName("CandidateColorConfigurationSummary")
        self.candidate_color_summary.setWordWrap(True)
        self.candidate_color_summary.setStyleSheet(
            "QLabel { background: #eef5ff; border: 1px solid #b8cce8; "
            "border-radius: 4px; padding: 8px; }"
        )
        self.candidate_version_edit = QLineEdit()
        self.candidate_operator_edit = QLineEdit()
        self.candidate_reason_edit = QTextEdit()
        self.candidate_reason_edit.setMaximumHeight(70)
        form.addRow("AI 模型版本", self.candidate_model_combo)
        form.addRow("顏色設定", self.candidate_color_summary)
        form.addRow("組合版本", self.candidate_version_edit)
        form.addRow("建立人員", self.candidate_operator_edit)
        form.addRow("建立原因", self.candidate_reason_edit)
        layout.addLayout(form)

        self.candidate_color_details_button = QPushButton("顯示進階組成設定")
        self.candidate_color_details_button.setObjectName(
            "CandidateColorConfigurationDetailsButton"
        )
        self.candidate_color_details_button.setCheckable(True)
        self.candidate_color_details_button.toggled.connect(
            self._toggle_color_configuration_details
        )
        layout.addWidget(self.candidate_color_details_button)

        self.candidate_color_advanced_panel = QWidget()
        self.candidate_color_advanced_panel.setObjectName(
            "CandidateColorConfigurationAdvancedPanel"
        )
        advanced_layout = QVBoxLayout(self.candidate_color_advanced_panel)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_form = QFormLayout()
        advanced_form.addRow("完整顏色基準", self.candidate_color_combo)
        advanced_layout.addLayout(advanced_form)
        advanced_layout.addWidget(QLabel("單色校正（選填；僅覆寫指定顏色）"))
        self.candidate_color_overrides_table = self._table(("色別", "校正版本"))
        self.candidate_color_overrides_table.setMaximumHeight(180)
        advanced_layout.addWidget(self.candidate_color_overrides_table)
        rebuild_row = QHBoxLayout()
        rebuild_button = QPushButton("重建完整顏色基準")
        rebuild_button.clicked.connect(self._open_color_baseline_rebuild)
        rebuild_row.addWidget(rebuild_button)
        rebuild_row.addStretch(1)
        advanced_layout.addLayout(rebuild_row)
        self.candidate_color_advanced_panel.setVisible(False)
        layout.addWidget(self.candidate_color_advanced_panel)

        create_row = QHBoxLayout()
        create_row.addStretch(1)
        create_button = QPushButton("建立候選組合")
        create_button.clicked.connect(self._create_candidate)
        create_row.addWidget(create_button)
        layout.addLayout(create_row)
        layout.addWidget(QLabel("候選與已驗收組合"))
        self.candidate_table = self._table(
            ("狀態", "組合版本", "模型與顏色", "樣本數", "建立時間")
        )
        layout.addWidget(self.candidate_table, 1)
        return page

    def _toggle_color_configuration_details(self, checked: bool) -> None:
        self.candidate_color_advanced_panel.setVisible(checked)
        self.candidate_color_details_button.setText(
            "隱藏進階組成設定" if checked else "顯示進階組成設定"
        )

    def _build_validation_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.acceptance_summary = QLabel("驗收資料：—")
        self.acceptance_summary.setWordWrap(True)
        self.acceptance_summary.setStyleSheet(
            "background:#fff7e8;color:#8a5a00;border:1px solid #ead3a3;border-radius:6px;padding:10px;"
        )
        layout.addWidget(self.acceptance_summary)
        validation_actions = QHBoxLayout()
        validation_actions.addStretch(1)
        open_validation = QPushButton("開啟完整驗收工具")
        open_validation.clicked.connect(self.validation_requested.emit)
        validation_actions.addWidget(open_validation)
        self.quick_validation_button = QPushButton("驗收選取版本")
        self.quick_validation_button.setEnabled(False)
        self.quick_validation_button.setStyleSheet(
            "QPushButton { background:#006f5f;color:white;padding:7px 16px; }"
            "QPushButton:disabled { background:#9ba8a5;color:#eef2f1; }"
        )
        self.quick_validation_button.clicked.connect(
            self._request_quick_validation
        )
        validation_actions.addWidget(self.quick_validation_button)
        layout.addLayout(validation_actions)
        self.validation_table = self._table(
            (
                "組合版本",
                "狀態",
                "樣本數",
                "誤殺",
                "漏檢",
                "顏色誤殺率",
                "顏色逃逸率",
            )
        )
        self.validation_table.itemSelectionChanged.connect(
            self._update_validation_actions
        )
        self.validation_table.itemDoubleClicked.connect(
            lambda _item: self._request_quick_validation()
        )
        layout.addWidget(self.validation_table, 1)
        return page

    def _build_deployment_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.deployment_table = self._table(
            ("使用中", "狀態", "組合版本", "模型與顏色", "啟用模式", "建立時間")
        )
        self.deployment_table.itemSelectionChanged.connect(self._update_deployment_actions)
        layout.addWidget(self.deployment_table, 1)
        actions = QHBoxLayout()
        actions.addStretch(1)
        self.activate_button = QPushButton("發布並啟用選取組合")
        self.activate_button.clicked.connect(self._activate_selected)
        actions.addWidget(self.activate_button)
        self.rollback_button = QPushButton("回退至前一正式組合")
        self.rollback_button.clicked.connect(self._rollback)
        actions.addWidget(self.rollback_button)
        layout.addLayout(actions)
        return page

    @staticmethod
    def _table(headers: tuple[str, ...]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def set_scope(self, product: str, area: str, inference_type: str) -> None:
        normalized = (
            str(product).strip(),
            str(area).strip(),
            str(inference_type).strip().lower(),
        )
        if normalized != (self.product, self.area, self.inference_type):
            self._candidate_initialized = False
            self._override_combos = {}
        self.product, self.area, self.inference_type = normalized
        self.refresh()

    def refresh(self) -> None:
        if not all((self.product, self.area, self.inference_type)):
            self.production_summary.setText("目前正式組合：尚未選擇完整範圍")
            return
        try:
            self._components = tuple(
                record
                for record in self.catalog.list_components()
                if (
                    record.product,
                    record.area,
                    record.inference_type,
                )
                == (self.product, self.area, self.inference_type)
            )
            self._releases = tuple(
                release
                for release in self.release_store.list_releases(
                    product=self.product,
                    area=self.area,
                )
                if release.scope.inference_type == self.inference_type
            )
            self._index_color_revisions()
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.critical(self, "版本與上線", str(exc))
            return
        self._refresh_component_filters()
        self._refresh_candidate_selectors()
        self._render_candidates()
        self._render_validation()
        self._render_deployment()
        self._refresh_acceptance_summary()
        self._refresh_production_summary()

    def _show_stage(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        self.stage_buttons[index].setChecked(True)
        if index in {2, 3}:
            self.refresh()

    def _refresh_component_filters(self) -> None:
        selected = str(self.component_type_filter.currentData() or "")
        self.component_type_filter.blockSignals(True)
        self.component_type_filter.clear()
        self.component_type_filter.addItem("全部", "")
        for value in sorted({record.component_type for record in self._components}):
            self.component_type_filter.addItem(value, value)
        index = self.component_type_filter.findData(selected)
        self.component_type_filter.setCurrentIndex(index if index >= 0 else 0)
        self.component_type_filter.blockSignals(False)
        self._render_components()

    def _render_components(self, _index: int | None = None) -> None:
        category = str(self.component_category_filter.currentData() or "")
        component_type = str(self.component_type_filter.currentData() or "")
        visible = tuple(
            record
            for record in self._components
            if (not category or record.category == category)
            and (not component_type or record.component_type == component_type)
        )
        self.component_table.setRowCount(len(visible))
        for row, record in enumerate(visible):
            values = (
                _STATUS_LABELS.get(record.status, record.status),
                _CATEGORY_LABELS.get(record.category, record.category),
                record.component_type,
                record.version,
                record.integrity,
                format_local_timestamp(record.created_at),
                str(record.source_path),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(_RECORD_ROLE, record)
                self.component_table.setItem(row, column, item)
        if visible:
            self.component_table.selectRow(0)
        else:
            self.add_component_button.setEnabled(False)
            self.component_settings_button.setEnabled(False)
            self.component_details.setText("此篩選條件沒有可用的模型或顏色版本。")

    def _selected_component(self) -> InspectionComponentRecord | None:
        row = self.component_table.currentRow()
        item = self.component_table.item(row, 0) if row >= 0 else None
        value = item.data(_RECORD_ROLE) if item else None
        return value if isinstance(value, InspectionComponentRecord) else None

    def _update_component_details(self) -> None:
        record = self._selected_component()
        self.add_component_button.setEnabled(bool(record and record.can_compose))
        self.component_settings_button.setEnabled(bool(record and record.category == "AI_MODEL"))
        self.component_details.setText(
            self._component_detail_text(record)
            if record
            else "請選取模型或顏色版本。"
        )

    @staticmethod
    def _component_detail_text(
        record: InspectionComponentRecord,
    ) -> str:
        if record.category == "COLOR_REVISION":
            try:
                details = json.loads(record.detail)
            except (TypeError, json.JSONDecodeError):
                details = {}
            return (
                f"{record.component_type}修訂｜{record.version}｜"
                f"狀態：{_STATUS_LABELS.get(record.status, record.status)}｜"
                f"操作人員：{details.get('operator') or '—'}｜"
                f"原因：{details.get('reason') or '—'}"
            )
        if record.category in {"COLOR_BASE", "COLOR_PROFILE"}:
            try:
                details = json.loads(record.detail)
            except (TypeError, json.JSONDecodeError):
                details = {}
            colors = "、".join(str(value) for value in details.get("colors") or ())
            overrides = "、".join(f"{key}: {value}" for key, value in (details.get("overrides") or {}).items())
            suffix = f"｜逐色修訂：{overrides}" if overrides else ""
            return (
                f"Stats Color 完整顏色"
                f"{'方案' if record.category == 'COLOR_PROFILE' else '基準'}｜"
                f"{details.get('color_count') or 0} 色：{colors}{suffix}"
            )
        return record.detail

    def _add_selected_component(self) -> None:
        record = self._selected_component()
        if record is None or not record.can_compose:
            return
        if record.category == "AI_MODEL":
            index = self.candidate_model_combo.findData(record.component_id)
            if index >= 0:
                self.candidate_model_combo.setCurrentIndex(index)
        elif record.category == "COLOR_REVISION":
            revision = self._color_revisions.get(record.component_id)
            if revision is not None:
                combo = self._override_combos.get(revision.scope.threshold_key.casefold())
                if combo is not None:
                    index = combo.findData(record.component_id)
                    if index >= 0:
                        combo.setCurrentIndex(index)
            self._select_first_color_base()
            self.candidate_color_details_button.setChecked(True)
        elif record.category == "COLOR_PROFILE":
            self._select_profile_overrides(record)
            self.candidate_color_details_button.setChecked(True)
        elif record.category == "COLOR_BASE":
            index = self.candidate_color_combo.findData(record.component_id)
            if index >= 0:
                self.candidate_color_combo.setCurrentIndex(index)
            for combo in self._override_combos.values():
                combo.setCurrentIndex(0)
            self.candidate_color_details_button.setChecked(True)
        self._show_stage(1)

    def _refresh_candidate_selectors(self) -> None:
        current_model = str(self.candidate_model_combo.currentData() or "")
        current_color_mode = str(self.candidate_color_combo.currentData() or "")
        current_overrides = {key: str(combo.currentData() or "") for key, combo in self._override_combos.items()}
        self.candidate_model_combo.clear()
        self.candidate_color_combo.clear()
        self.candidate_color_combo.addItem("不套用顏色檢查", "")
        color_bases = [record for record in self._components if record.category == "COLOR_BASE" and record.can_compose]
        for record in color_bases:
            label, tooltip = self._color_base_presentation(record)
            self.candidate_color_combo.addItem(
                label,
                record.component_id,
            )
            self.candidate_color_combo.setItemData(
                self.candidate_color_combo.count() - 1,
                tooltip,
                Qt.ToolTipRole,
            )
        models = [record for record in self._components if record.category == "AI_MODEL" and record.can_compose]
        for record in models:
            self.candidate_model_combo.addItem(
                f"{record.component_type} {record.version}｜{_STATUS_LABELS.get(record.status, record.status)}",
                record.component_id,
            )
        self._restore_combo_selection(
            self.candidate_model_combo,
            current_model,
            preferred_status="DEPLOYED",
        )
        if not self._candidate_initialized:
            preferred = next(
                (record.component_id for record in color_bases if record.status == "DEPLOYED"),
                color_bases[0].component_id if color_bases else "",
            )
            color_index = self.candidate_color_combo.findData(preferred)
        else:
            color_index = self.candidate_color_combo.findData(current_color_mode)
            if color_index < 0:
                color_index = 0
        self.candidate_color_combo.setCurrentIndex(color_index)
        self._render_color_override_selectors(
            current_overrides,
            prefer_deployed=not self._candidate_initialized,
        )
        self._candidate_initialized = True
        if not self.candidate_version_edit.text().strip():
            self.candidate_version_edit.setText(self._next_release_version())

    def _render_color_override_selectors(
        self,
        selections: dict[str, str],
        *,
        prefer_deployed: bool,
    ) -> None:
        grouped: dict[str, list[InspectionComponentRecord]] = {}
        display_names: dict[str, str] = {}
        for record in self._components:
            if record.category != "COLOR_BASE":
                continue
            try:
                details = json.loads(record.detail)
            except (TypeError, json.JSONDecodeError):
                continue
            for color in details.get("colors") or ():
                display_names[str(color).casefold()] = str(color)
        for normalized_key in display_names:
            grouped.setdefault(normalized_key, [])
        for record in self._components:
            if record.category != "COLOR_REVISION" or not record.can_compose:
                continue
            revision = self._color_revisions.get(record.component_id)
            if revision is None:
                continue
            grouped.setdefault(
                revision.scope.threshold_key.casefold(),
                [],
            ).append(record)
            display_names.setdefault(
                revision.scope.threshold_key.casefold(),
                revision.scope.threshold_key.title(),
            )
        self._override_combos = {}
        self.candidate_color_overrides_table.setRowCount(len(grouped))
        for row, (normalized_key, records) in enumerate(sorted(grouped.items())):
            threshold_key = display_names.get(
                normalized_key,
                normalized_key.title(),
            )
            self.candidate_color_overrides_table.setItem(
                row,
                0,
                QTableWidgetItem(threshold_key.title()),
            )
            combo = QComboBox()
            combo.addItem("使用五色基準值", "")
            for record in sorted(
                records,
                key=lambda item: (item.created_at, item.version),
            ):
                combo.addItem(
                    f"{record.version}｜{_STATUS_LABELS.get(record.status, record.status)}",
                    record.component_id,
                )
            selected = selections.get(normalized_key, "")
            if not selected and prefer_deployed:
                deployed = next(
                    (record.component_id for record in records if record.status == "DEPLOYED"),
                    "",
                )
                selected = deployed
            index = combo.findData(selected)
            combo.setCurrentIndex(index if index >= 0 else 0)
            self.candidate_color_overrides_table.setCellWidget(
                row,
                1,
                combo,
            )
            combo.currentIndexChanged.connect(
                self._update_color_configuration_summary
            )
            self._override_combos[normalized_key] = combo
        self._update_color_configuration_summary()

    def _update_color_configuration_summary(
        self,
        _index: int | None = None,
    ) -> None:
        base = self._component_by_id(
            str(self.candidate_color_combo.currentData() or "")
        )
        if base is None or base.category != "COLOR_BASE":
            self.candidate_color_summary.setText("不套用顏色檢查")
            self.candidate_color_summary.setToolTip(
                "此候選組合不會執行 Stats Color 顏色檢查。"
            )
            for combo in self._override_combos.values():
                combo.setEnabled(False)
                combo.setItemText(0, "請先選擇完整顏色基準")
            return

        baseline_label, baseline_tooltip = self._color_base_presentation(base)

        override_labels: list[str] = []
        for normalized_key, combo in sorted(self._override_combos.items()):
            combo.setEnabled(True)
            combo.setItemText(0, "沿用上方完整基準")
            component_id = str(combo.currentData() or "")
            revision = self._color_revisions.get(component_id)
            if revision is not None:
                override_labels.append(
                    f"{revision.scope.threshold_key.title()}：{revision.display_version}"
                )
                continue
            record = self._component_by_id(component_id)
            if record is not None:
                override_labels.append(
                    f"{normalized_key.title()}：{record.version}"
                )

        summary_parts = ["顏色設定", baseline_label]
        summary_parts.append(
            "單色校正 " + "、".join(override_labels)
            if override_labels
            else "未套用單色校正"
        )
        self.candidate_color_summary.setText("｜".join(summary_parts))

        tooltip_parts = [baseline_tooltip]
        if override_labels:
            tooltip_parts.append("單色校正：" + "、".join(override_labels))
        self.candidate_color_summary.setToolTip("\n".join(tooltip_parts))
        for combo in self._override_combos.values():
            combo.setToolTip(f"目前沿用完整基準：{base.version}")

    @staticmethod
    def _color_base_presentation(
        record: InspectionComponentRecord,
    ) -> tuple[str, str]:
        try:
            details = json.loads(record.detail)
        except (TypeError, json.JSONDecodeError):
            details = {}
        if not isinstance(details, dict):
            details = {}
        colors = details.get("colors")
        colors = colors if isinstance(colors, list) else []
        configured_count = details.get("color_count")
        color_count = (
            configured_count
            if isinstance(configured_count, int)
            and not isinstance(configured_count, bool)
            and configured_count >= 0
            else len(colors)
        )
        if record.status == "DEPLOYED":
            lifecycle_status = "DEPLOYED"
        elif details.get("role") == "BASELINE_CANDIDATE":
            lifecycle_status = "CANDIDATE"
        else:
            lifecycle_status = record.status

        quality_status = str(details.get("candidate_status") or "").strip()
        summary = format_color_baseline_summary(
            color_count=color_count,
            created_at=record.created_at,
            lifecycle_status=lifecycle_status,
            quality_status=quality_status,
        )

        tooltip_parts = [
            f"完整基準內部 ID：{record.version}",
            f"來源：{record.source_path}",
        ]
        if quality_status:
            tooltip_parts.append(f"品質狀態：{quality_status}")
        return summary, "\n".join(tooltip_parts)

    def _select_first_color_base(self) -> None:
        if self.candidate_color_combo.count() > 1:
            self.candidate_color_combo.setCurrentIndex(1)

    def _open_color_baseline_rebuild(self) -> None:
        model_component = self._component_by_id(str(self.candidate_model_combo.currentData() or ""))
        if model_component is None:
            QMessageBox.warning(
                self,
                "重建完整顏色基準",
                "請先選擇要用來重新偵測元件框的 AI 模型版本。",
            )
            return
        try:
            model = self._resolve_model(model_component)
        except (InspectionReleaseError, OSError, RuntimeError, ValueError) as exc:
            QMessageBox.critical(
                self,
                "重建完整顏色基準",
                str(exc),
            )
            return
        if not model.has_config_snapshot:
            QMessageBox.warning(
                self,
                "重建完整顏色基準",
                "選取的模型版本缺少 config 快照，不能安全重建。",
            )
            return
        acceptance_manifest = (
            self.data_paths.acceptance
            / self.product
            / self.area
            / "ground_truth.csv"
        )
        try:
            color_feedback_manifest = (
                load_workspace_paths(self.project_root).training_data
                / self.product
                / self.area
                / "color_review"
                / "feedback.csv"
            )
        except ValueError as exc:
            QMessageBox.critical(self, "重建完整顏色基準", str(exc))
            return
        if not acceptance_manifest.is_file() and not color_feedback_manifest.is_file():
            QMessageBox.warning(
                self,
                "重建完整顏色基準",
                "尚未建立這個產品與區域的驗收資料或顏色覆核資料。",
            )
            return
        dialog = ColorBaselineRebuildDialog(
            project_root=self.project_root,
            product=self.product,
            area=self.area,
            inference_type=self.inference_type,
            model=model,
            parent=self,
        )
        dialog.candidate_created.connect(self._on_color_baseline_created)
        dialog.exec_()

    def _on_color_baseline_created(self, candidate) -> None:
        self.refresh()
        component_id = f"color-base-candidate:{candidate.candidate_id}"
        index = self.candidate_color_combo.findData(component_id)
        if index >= 0:
            self.candidate_color_combo.setCurrentIndex(index)
        self.candidate_color_details_button.setChecked(True)
        self._show_stage(1)

    def _select_profile_overrides(
        self,
        record: InspectionComponentRecord,
    ) -> None:
        try:
            profile = self.color_profile_store.load(record.source_path)
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "顏色設定", str(exc))
            return
        matching_base = next(
            (
                component.component_id
                for component in self._components
                if component.category == "COLOR_BASE"
                and _detail_value(component, "sha256") == profile.color_model_sha256
            ),
            "",
        )
        index = self.candidate_color_combo.findData(matching_base)
        if index >= 0:
            self.candidate_color_combo.setCurrentIndex(index)
        for combo in self._override_combos.values():
            combo.setCurrentIndex(0)
        for binding in profile.revisions:
            combo = self._override_combos.get(binding.threshold_key.casefold())
            if combo is None:
                continue
            index = combo.findData(f"color:{binding.revision_id}")
            if index >= 0:
                combo.setCurrentIndex(index)

    def _restore_combo_selection(
        self,
        combo: QComboBox,
        component_id: str,
        *,
        preferred_status: str,
    ) -> None:
        index = combo.findData(component_id)
        if index >= 0:
            combo.setCurrentIndex(index)
            return
        preferred = next(
            (
                record
                for record in self._components
                if record.status == preferred_status and combo.findData(record.component_id) >= 0
            ),
            None,
        )
        if preferred is not None:
            combo.setCurrentIndex(combo.findData(preferred.component_id))

    def _component_by_id(self, component_id: str) -> InspectionComponentRecord | None:
        return next(
            (record for record in self._components if record.component_id == component_id),
            None,
        )

    def _create_candidate(self) -> None:
        model_component = self._component_by_id(str(self.candidate_model_combo.currentData() or ""))
        color_mode = str(self.candidate_color_combo.currentData() or "")
        version = self.candidate_version_edit.text().strip()
        operator = self.candidate_operator_edit.text().strip()
        reason = self.candidate_reason_edit.toPlainText().strip()
        if model_component is None:
            QMessageBox.warning(self, "建立候選組合", "請選擇 AI 模型版本。")
            return
        if not all((version, operator, reason)):
            QMessageBox.warning(
                self,
                "建立候選組合",
                "組合版本、建立人員與建立原因皆為必填。",
            )
            return
        if any(release.display_version == version for release in self._releases):
            QMessageBox.warning(self, "建立候選組合", "此組合版本已經存在。")
            return
        try:
            model = self._resolve_model(model_component)
            color_profile = None
            if color_mode:
                color_base = self._component_by_id(color_mode)
                if color_base is None or color_base.category != "COLOR_BASE":
                    raise InspectionReleaseError("選取的完整顏色基準已不存在，請重新整理。")
                color_profile = self.color_profile_store.create(
                    product=model.product,
                    area=model.area,
                    model_type=model.model_type,
                    model_config_path=model.config_snapshot_path,
                    project_root=self.project_root,
                    revisions=self._selected_color_revisions(),
                    color_model_override=color_base.source_path,
                )
                if color_profile is None:
                    raise InspectionReleaseError("選取的模型版本未啟用顏色檢查。")
            release = build_draft_release(
                model,
                display_version=version,
                operator=operator,
                reason=reason,
                color_profile=color_profile,
            )
            self.release_store.commit(release)
        except (InspectionReleaseError, OSError, RuntimeError, ValueError) as exc:
            QMessageBox.critical(self, "建立候選組合失敗", str(exc))
            return
        self.candidate_reason_edit.clear()
        self.candidate_version_edit.clear()
        self.refresh()
        QMessageBox.information(
            self,
            "候選組合已建立",
            f"{release.display_version} 已建立為 DRAFT，尚未影響正式檢測。",
        )

    def _selected_color_revisions(
        self,
    ) -> tuple[ColorConfigurationRevision, ...]:
        revisions: list[ColorConfigurationRevision] = []
        for combo in self._override_combos.values():
            component_id = str(combo.currentData() or "")
            if not component_id:
                continue
            revision = self._color_revisions.get(component_id)
            if revision is None:
                raise InspectionReleaseError("選取的校正修訂已不存在，請重新整理。")
            revisions.append(revision)
        return tuple(revisions)

    def _resolve_model(self, component: InspectionComponentRecord):
        records = ModelVersionRegistry(self.data_paths.models).list_versions(
            product=component.product,
            area=component.area,
            model_type=component.inference_type,
        )
        for record in records:
            if record.version == component.version and record.weight_path.resolve() == component.source_path.resolve():
                return record
        raise InspectionReleaseError("選取的模型版本已不存在或無法解析。")

    def _index_color_revisions(self) -> None:
        self._color_revisions = {}
        root = self.color_store.root
        if not root.is_dir():
            return
        for scope_root in root.iterdir():
            if not scope_root.is_dir() or scope_root.name == "active" or scope_root.name.startswith("."):
                continue
            scope = self.color_store.scope_for_hash(scope_root.name)
            if (
                scope.product,
                scope.area,
                scope.model_type,
            ) != (self.product, self.area, self.inference_type):
                continue
            for revision in self.color_store.list_revisions(scope):
                self._color_revisions[f"color:{revision.revision_id}"] = revision

    def _next_release_version(self) -> str:
        patches: list[int] = []
        prefix = "inspection-v1.0."
        for release in self._releases:
            if release.display_version.startswith(prefix):
                suffix = release.display_version[len(prefix) :]
                if suffix.isdigit():
                    patches.append(int(suffix))
        return f"{prefix}{max(patches, default=0) + 1}"

    def _render_candidates(self) -> None:
        self.candidate_table.setRowCount(len(self._releases))
        for row, release in enumerate(self._releases):
            values = (
                release.status.value,
                release.display_version,
                self._component_summary(release),
                str(release.validation.sample_count),
                format_local_timestamp(release.created_at),
            )
            self._set_release_row(self.candidate_table, row, release, values)

    def _render_validation(self) -> None:
        self.validation_table.setRowCount(len(self._releases))
        first_draft_row = -1
        for row, release in enumerate(self._releases):
            if first_draft_row < 0 and release.status is ReleaseStatus.DRAFT:
                first_draft_row = row
            metrics = dict(release.validation.metrics)
            color = dict(release.validation.color_metrics)
            values = (
                release.display_version,
                release.status.value,
                str(release.validation.sample_count),
                str(metrics.get("fp", "—")),
                str(metrics.get("fn", "—")),
                self._percent(color.get("overkill_rate")),
                self._percent(color.get("escape_rate")),
            )
            self._set_release_row(self.validation_table, row, release, values)
        if self._releases:
            self.validation_table.selectRow(
                first_draft_row if first_draft_row >= 0 else 0
            )
        self._update_validation_actions()

    def _selected_validation_release(self) -> InspectionRelease | None:
        row = self.validation_table.currentRow()
        item = self.validation_table.item(row, 0) if row >= 0 else None
        value = item.data(_RELEASE_ROLE) if item else None
        return value if isinstance(value, InspectionRelease) else None

    def _update_validation_actions(self) -> None:
        release = self._selected_validation_release()
        enabled = bool(
            release is not None
            and release.status is ReleaseStatus.DRAFT
            and not self.is_inspection_running()
        )
        self.quick_validation_button.setEnabled(enabled)
        if release is None:
            self.quick_validation_button.setToolTip("請先選取一個候選組合。")
        elif release.status is not ReleaseStatus.DRAFT:
            self.quick_validation_button.setToolTip("此組合已有驗收結果。")
        elif self.is_inspection_running():
            self.quick_validation_button.setToolTip("請先停止目前檢測。")
        else:
            self.quick_validation_button.setToolTip(
                f"只驗收 {release.display_version} 綁定的模型與顏色版本。"
            )

    def _request_quick_validation(self) -> None:
        release = self._selected_validation_release()
        if release is None or release.status is not ReleaseStatus.DRAFT:
            return
        if self.is_inspection_running():
            QMessageBox.warning(self, "快速驗收", "請先停止目前檢測。")
            return
        self.quick_validation_requested.emit(release)

    def _render_deployment(self) -> None:
        self.deployment_table.setRowCount(len(self._releases))
        for row, release in enumerate(self._releases):
            pointer = self.release_store.active_pointer(release.scope)
            active = bool(pointer and pointer.get("release_id") == release.release_id)
            values = (
                "目前使用" if active else "",
                release.status.value,
                release.display_version,
                self._component_summary(release),
                str(pointer.get("mode") or "") if active and pointer else "",
                format_local_timestamp(release.created_at),
            )
            self._set_release_row(self.deployment_table, row, release, values)
        if self._releases:
            self.deployment_table.selectRow(0)
        self._update_deployment_actions()

    @staticmethod
    def _set_release_row(
        table: QTableWidget,
        row: int,
        release: InspectionRelease,
        values: tuple[str, ...],
    ) -> None:
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setData(_RELEASE_ROLE, release)
            table.setItem(row, column, item)

    @staticmethod
    def _component_summary(release: InspectionRelease) -> str:
        return format_component_summary(release)

    @staticmethod
    def _percent(value: object) -> str:
        if value is None:
            return "UNKNOWN"
        try:
            return f"{float(value):.2%}"
        except (TypeError, ValueError):
            return "無效資料"

    def _refresh_acceptance_summary(self) -> None:
        root = self.data_paths.acceptance / self.product / self.area
        manifest = root / "ground_truth.csv"
        if not manifest.is_file():
            self.acceptance_summary.setText("驗收資料：尚未建立人工確認資料。")
            return
        try:
            records = AcceptanceRepository(root).records()
        except (OSError, RuntimeError, ValueError) as exc:
            self.acceptance_summary.setText(f"驗收資料無法讀取：{exc}")
            return
        confirmed = [record for record in records if record.review_status == "confirmed"]
        ok_count = sum(record.expected_verdict == "OK" for record in confirmed)
        ng_count = sum(record.expected_verdict == "NG" for record in confirmed)
        color_ng = sum("COLOR_MISMATCH" in record.expected_reasons.split("|") for record in confirmed)
        self.acceptance_summary.setText(
            f"驗收資料：共 {len(records)} 張；已人工確認 {len(confirmed)} 張｜"
            f"OK {ok_count}｜NG {ng_count}｜顏色 NG {color_ng}。"
            "組合驗收會讀取同一份資料，不需重新標註。"
        )

    def _active_release(self) -> tuple[InspectionRelease, dict] | None:
        for release in self._releases:
            pointer = self.release_store.active_pointer(release.scope)
            if pointer and pointer.get("release_id") == release.release_id:
                return release, dict(pointer)
        return None

    def _refresh_production_summary(self) -> None:
        active = self._active_release()
        scope = f"{self.product} / {self.area} / {self.inference_type}"
        if active is None:
            self.production_summary.setText(f"{scope}｜目前正式組合：尚未建立")
            return
        release, pointer = active
        self.production_summary.setText(
            f"{scope}｜目前正式組合：{release.display_version}｜"
            f"{self._component_summary(release)}｜{pointer.get('mode')}"
        )

    def _selected_release(self) -> InspectionRelease | None:
        row = self.deployment_table.currentRow()
        item = self.deployment_table.item(row, 0) if row >= 0 else None
        value = item.data(_RELEASE_ROLE) if item else None
        return value if isinstance(value, InspectionRelease) else None

    def _update_deployment_actions(self) -> None:
        release = self._selected_release()
        allowed = self.release_store.policy.allowed_modes(release) if release else ()
        running = self.is_inspection_running()
        self.activate_button.setEnabled(bool(release and allowed and not running))
        if release is None:
            self.rollback_button.setEnabled(False)
            return
        pointer = self.release_store.active_pointer(release.scope)
        self.rollback_button.setEnabled(bool(pointer and pointer.get("previous_release_id") and not running))

    def _activate_selected(self) -> None:
        release = self._selected_release()
        if release is None:
            return
        if self.is_inspection_running():
            QMessageBox.warning(self, "上線檢測組合", "請先停止目前檢測。")
            return
        allowed = self.release_store.policy.allowed_modes(release)
        if not allowed:
            QMessageBox.warning(self, "上線檢測組合", "此組合目前不符合任何上線條件。")
            return
        labels = {
            ActivationMode.FULL: "正式上線",
            ActivationMode.LIMITED_TRIAL: "限定試用",
            ActivationMode.RISK_ACCEPTED: "風險接受",
        }
        selected, ok = QInputDialog.getItem(
            self,
            "上線檢測組合",
            "上線模式",
            [labels[mode] for mode in allowed],
            editable=False,
        )
        if not ok:
            return
        mode = next(mode for mode in allowed if labels[mode] == selected)
        if mode is ActivationMode.RISK_ACCEPTED:
            warnings = "\n".join(self.release_store.policy.validation_warnings(release))
            confirmation = f"{warnings or '驗收資料尚未達正式門檻。'}\n\n仍要承擔風險並上線嗎？"
            if (
                QMessageBox.question(
                    self,
                    "風險接受確認",
                    confirmation,
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                != QMessageBox.Yes
            ):
                return
        identity = self._operator_reason("上線檢測組合")
        if identity is None:
            return
        pointer = self.release_store.active_pointer(release.scope)
        expected = str(pointer.get("release_id")) if pointer else None
        try:
            self.release_store.activate(
                release,
                mode=mode,
                operator=identity[0],
                reason=identity[1],
                expected_release_id=expected,
            )
        except InspectionReleaseError as exc:
            QMessageBox.critical(self, "上線失敗", str(exc))
            return
        self.release_activated.emit(release)
        self.refresh()

    def _rollback(self) -> None:
        release = self._selected_release()
        if release is None:
            return
        if self.is_inspection_running():
            QMessageBox.warning(self, "回退正式組合", "請先停止目前檢測。")
            return
        identity = self._operator_reason("回退正式組合")
        if identity is None:
            return
        try:
            pointer = self.release_store.rollback(
                release.scope,
                operator=identity[0],
                reason=identity[1],
            )
            restored = self.release_store.load(
                release.scope,
                str(pointer["release_id"]),
            )
        except InspectionReleaseError as exc:
            QMessageBox.critical(self, "退回失敗", str(exc))
            return
        self.release_activated.emit(restored)
        self.refresh()

    def _operator_reason(self, title: str) -> tuple[str, str] | None:
        operator, ok = QInputDialog.getText(self, title, "操作人員")
        if not ok or not operator.strip():
            return None
        reason, ok = QInputDialog.getMultiLineText(self, title, "操作原因")
        if not ok or not reason.strip():
            return None
        return operator.strip(), reason.strip()
