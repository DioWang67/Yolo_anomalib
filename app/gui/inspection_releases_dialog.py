"""Engineering UI for atomic inspection-release activation and rollback."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.gui.inspection_release_presentation import (
    format_component_summary,
    format_local_timestamp,
)
from core.services.inspection_release_models import (
    ActivationMode,
    InspectionRelease,
    InspectionReleaseError,
)
from core.services.inspection_release_store import InspectionReleaseStore


class InspectionReleasesDialog(QDialog):
    """Show immutable combinations; mutate only the active release pointer."""

    HEADERS = (
        "狀態",
        "發布版本",
        "流程",
        "元件組合",
        "準確率",
        "誤殺",
        "漏檢",
        "顏色誤殺",
        "顏色逃逸",
        "建立時間",
    )

    def __init__(
        self,
        *,
        store: InspectionReleaseStore,
        product: str | None = None,
        area: str | None = None,
        inference_type: str | None = None,
        preferred_model_version: str = "",
        preferred_color_version: str = "",
        is_inspection_running: Callable[[], bool] | None = None,
        on_activated: Callable[[InspectionRelease], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.product = product
        self.area = area
        self.inference_type = inference_type
        self.preferred_model_version = preferred_model_version
        self.preferred_color_version = preferred_color_version
        self.is_inspection_running = is_inspection_running or (lambda: False)
        self.on_activated = on_activated
        self._releases: tuple[InspectionRelease, ...] = ()
        self.setWindowTitle("檢測組合與上線")
        self.resize(1180, 620)
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "一個發布版本會綁定完整模型組合與顏色版本。切換只改一個原子指標；"
            "已開始的檢測不會在中途換版本。"
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self.active_label = QLabel("目前套用：讀取中…")
        self.active_label.setStyleSheet(
            "font-weight: 600; color: #0f766e; padding: 6px;"
        )
        layout.addWidget(self.active_label)

        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._update_details)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        self.details = QLabel("選取組合版本以查看驗收規則。")
        self.details.setWordWrap(True)
        self.details.setStyleSheet("background: #f3f4f6; padding: 8px;")
        layout.addWidget(self.details)

        actions = QHBoxLayout()
        self.refresh_btn = QPushButton("重新整理")
        self.refresh_btn.clicked.connect(self.refresh)
        self.compose_btn = QPushButton("建立檢測組合")
        self.compose_btn.clicked.connect(self._create_combination)
        self.report_btn = QPushButton("開啟驗收報告資料夾")
        self.report_btn.clicked.connect(self._open_report)
        self.activate_btn = QPushButton("發布並啟用選取組合")
        self.activate_btn.clicked.connect(self._activate_selected)
        self.rollback_btn = QPushButton("回退至前一正式組合")
        self.rollback_btn.clicked.connect(self._rollback)
        actions.addWidget(self.refresh_btn)
        actions.addWidget(self.compose_btn)
        actions.addWidget(self.report_btn)
        actions.addStretch(1)
        actions.addWidget(self.rollback_btn)
        actions.addWidget(self.activate_btn)
        layout.addLayout(actions)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def refresh(self) -> None:
        try:
            self._releases = self.store.list_releases(
                product=self.product, area=self.area
            )
        except InspectionReleaseError as exc:
            QMessageBox.critical(self, "發布版本", str(exc))
            self._releases = ()
        self.table.setRowCount(len(self._releases))
        active_versions: list[str] = []
        for row, release in enumerate(self._releases):
            pointer = self.store.active_pointer(release.scope)
            is_active = bool(
                pointer and pointer.get("release_id") == release.release_id
            )
            if is_active:
                active_versions.append(
                    f"{release.scope.product}/{release.scope.area}/"
                    f"{release.scope.inference_type}: {release.display_version} "
                    f"({pointer.get('mode')})"
                )
            metrics = dict(release.validation.metrics)
            color_metrics = dict(release.validation.color_metrics)
            confirmed = int(metrics.get("confirmed") or release.validation.sample_count)
            correct = (
                int(metrics.get("tp") or 0) + int(metrics.get("tn") or 0)
            )
            accuracy = correct / confirmed if confirmed else None
            values = (
                "目前套用" if is_active else release.status.value,
                release.display_version,
                release.scope.template_id,
                self._component_summary(release),
                self._percent(accuracy),
                str(metrics.get("fp", "—")),
                str(metrics.get("fn", "—")),
                self._percent(color_metrics.get("overkill_rate")),
                self._percent(color_metrics.get("escape_rate")),
                format_local_timestamp(release.created_at),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(256, release.release_id)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        self.active_label.setText(
            "目前套用：" + ("；".join(active_versions) if active_versions else "尚未建立發布指標")
        )
        if self._releases:
            self.table.selectRow(0)
        else:
            self.activate_btn.setEnabled(False)
            self.report_btn.setEnabled(False)
            self.rollback_btn.setEnabled(False)

    @staticmethod
    def _component_summary(release: InspectionRelease) -> str:
        return format_component_summary(release)

    @staticmethod
    def _percent(value: object) -> str:
        if value is None:
            return "UNKNOWN"
        try:
            return f"{float(value) * 100:.2f}%"
        except (TypeError, ValueError):
            return "—"

    def _selected_release(self) -> InspectionRelease | None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._releases):
            return None
        return self._releases[row]

    def _update_details(self) -> None:
        release = self._selected_release()
        if release is None:
            return
        allowed = self.store.policy.allowed_modes(release)
        warnings = self.store.policy.validation_warnings(release)
        if ActivationMode.FULL in allowed:
            policy = "驗收完整；可正式上線，也可先有限試跑。"
            self.activate_btn.setText("選擇上線模式")
        elif ActivationMode.RISK_ACCEPTED in allowed:
            policy = (
                "驗收資料有風險；你仍可選擇有限試跑，或記錄風險接受後套用。"
            )
            self.activate_btn.setText("選擇上線模式")
        else:
            policy = "此版本已被明確封鎖，不能啟用。"
            self.activate_btn.setText("不可啟用")
        self.activate_btn.setEnabled(bool(allowed))
        self.report_btn.setEnabled(bool(release.validation.report_path))
        pointer = self.store.active_pointer(release.scope)
        self.rollback_btn.setEnabled(bool(pointer and pointer.get("previous_release_id")))
        metrics = dict(release.validation.metrics)
        self.details.setText(
            f"{policy}\n樣本 {release.validation.sample_count} 張；"
            f"誤殺={metrics.get('fp', '—')}、漏檢={metrics.get('fn', '—')}、"
            f"errors={metrics.get('errors', '—')}。\n"
            f"風險：{self._warning_summary(warnings)}\n"
            f"原因：{release.reason}"
        )

    @staticmethod
    def _warning_summary(warnings: tuple[str, ...]) -> str:
        if not warnings:
            return "無"
        translated: list[str] = []
        for warning in warnings:
            if warning.startswith("Color NG truth"):
                translated.append("缺少顏色 NG 真值，顏色逃逸率未知")
            elif warning.startswith("Validation false negatives"):
                translated.append(
                    f"驗收有漏檢（{warning.split(':', 1)[-1].strip()}）"
                )
            elif warning.startswith("Validation inference errors"):
                translated.append(f"驗收有推論錯誤（{warning.split(':', 1)[-1].strip()}）")
            elif warning.startswith("Release is a draft"):
                translated.append("仍是草稿，尚未完成驗收")
            else:
                translated.append(warning)
        return "；".join(translated)

    def _operator_reason(self, action: str) -> tuple[str, str] | None:
        operator, ok = QInputDialog.getText(self, action, "操作人員：")
        if not ok or not operator.strip():
            return None
        reason, ok = QInputDialog.getMultiLineText(self, action, "操作原因：")
        if not ok or not reason.strip():
            return None
        return operator.strip(), reason.strip()

    def _create_combination(self) -> None:
        """Create a DRAFT from any registered model and exact color revision."""
        if not self.product or not self.area or not self.inference_type:
            QMessageBox.warning(
                self,
                "建立檢測組合",
                "請先在主畫面選擇產品、區域與推論類型。",
            )
            return
        if self.inference_type.lower() == "fusion":
            QMessageBox.information(
                self,
                "建立檢測組合",
                "Fusion 需要同時選 YOLO、Anomalib 與融合規則；"
                "目前這個按鈕先支援單一 YOLO 或 Anomalib。",
            )
            return
        try:
            from app.gui.inspection_release_composer_dialog import (
                InspectionReleaseComposerDialog,
            )
            from core.services.inspection_release_builder import (
                build_draft_release,
            )
            from core.services.model_version_registry import (
                ModelVersionRegistry,
            )

            project_root = self.store.root.parent
            registry = ModelVersionRegistry(project_root / "models")
            models = tuple(
                record
                for record in registry.list_versions(
                    product=self.product,
                    area=self.area,
                    model_type=self.inference_type.lower(),
                )
                if record.exists and record.has_config_snapshot
            )
            if not models:
                QMessageBox.warning(
                    self,
                    "建立檢測組合",
                    "沒有同時具備模型檔與 config 快照的可用版本。",
                )
                return
            color_revisions = self._discover_color_revisions(
                project_root / ".color_revisions"
            )
            selected = self._selected_release()
            selected_model = None
            selected_color = None
            if selected is not None:
                selected_model = (
                    selected.component_for_role("primary_detector")
                    or selected.component_for_role("anomaly_detector")
                )
                selected_color = selected.component_for_role("color_check")
            preselected_model_version = self.preferred_model_version or (
                selected_model.version if selected_model else ""
            )
            preselected_color_version = self.preferred_color_version or (
                selected_color.version if selected_color else ""
            )
            self.preferred_model_version = ""
            self.preferred_color_version = ""
            dialog = InspectionReleaseComposerDialog(
                models=models,
                color_revisions=color_revisions,
                suggested_version=self._next_release_version(),
                preselected_model_version=preselected_model_version,
                preselected_color_version=preselected_color_version,
                parent=self,
            )
            if dialog.exec_() != QDialog.Accepted:
                return
            if any(
                release.display_version == dialog.display_version()
                for release in self._releases
            ):
                QMessageBox.warning(
                    self,
                    "建立檢測組合",
                    "發布版本名稱已存在，請使用新的版本號。",
                )
                return
            model = dialog.selected_model()
            if model is None:
                return
            release = build_draft_release(
                model,
                display_version=dialog.display_version(),
                operator=dialog.operator(),
                reason=dialog.reason(),
                color_revision=dialog.selected_color_revision(),
            )
            self.store.commit(release)
            self.refresh()
            QMessageBox.information(
                self,
                "DRAFT 已建立",
                f"{release.display_version} 已建立："
                f"{self._component_summary(release)}。\n"
                "現在可選擇有限試跑或風險接受後套用。",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.critical(self, "建立檢測組合失敗", str(exc))

    def begin_combination_creation(self) -> None:
        """Open the combination editor, optionally with a preferred model."""
        self._create_combination()

    def _discover_color_revisions(
        self, root: Path
    ) -> tuple:
        from tools.color_configuration_revisions import (
            ColorConfigurationRevisionStore,
        )

        store = ColorConfigurationRevisionStore(root=root)
        if not store.root.is_dir():
            return ()
        revisions = []
        for scope_root in store.root.iterdir():
            if (
                not scope_root.is_dir()
                or scope_root.name == "active"
                or scope_root.name.startswith(".")
            ):
                continue
            try:
                scope = store.scope_for_hash(scope_root.name)
                if (
                    scope.product,
                    scope.area,
                    scope.model_type,
                    scope.checker_type,
                ) != (
                    self.product,
                    self.area,
                    self.inference_type.lower(),
                    "stats",
                ):
                    continue
                revisions.extend(
                    revision
                    for revision in store.list_revisions(scope)
                    if not store.is_revoked(revision)
                )
            except (OSError, RuntimeError, ValueError):
                continue
        return tuple(
            sorted(
                revisions,
                key=lambda revision: (
                    revision.created_at,
                    revision.display_version,
                ),
                reverse=True,
            )
        )

    def _next_release_version(self) -> str:
        patches = []
        for release in self._releases:
            match = re.fullmatch(
                r"inspection-v1\.0\.(\d+)",
                release.display_version.strip(),
            )
            if match:
                patches.append(int(match.group(1)))
        return f"inspection-v1.0.{max(patches, default=0) + 1}"

    def _guard_idle(self) -> bool:
        if not self.is_inspection_running():
            return True
        QMessageBox.warning(self, "檢測發布版本", "請先停止檢測，再切換發布版本。")
        return False

    def _activate_selected(self) -> None:
        release = self._selected_release()
        if release is None or not self._guard_idle():
            return
        allowed = self.store.policy.allowed_modes(release)
        if not allowed:
            return
        labels = {
            ActivationMode.FULL: "正式上線（驗收完整）",
            ActivationMode.LIMITED_TRIAL: "有限試跑",
            ActivationMode.RISK_ACCEPTED: "風險接受後套用",
        }
        label_to_mode = {labels[mode]: mode for mode in allowed}
        selected_label, ok = QInputDialog.getItem(
            self,
            "選擇上線模式",
            "上線方式：",
            list(label_to_mode),
            editable=False,
        )
        if not ok:
            return
        mode = label_to_mode[selected_label]
        warnings = self.store.policy.validation_warnings(release)
        if mode is ActivationMode.RISK_ACCEPTED:
            risk_text = self._warning_summary(warnings)
            if (
                QMessageBox.question(
                    self,
                    "接受驗收風險",
                    f"此操作會套用尚未完整驗收的組合。\n"
                    f"已知風險：{risk_text}\n\n仍要繼續嗎？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                != QMessageBox.Yes
            ):
                return
        if mode is not ActivationMode.RISK_ACCEPTED:
            confirmation = (
                f"確定將 {release.display_version} 設為"
                f"{labels[mode]}？"
            )
            if QMessageBox.question(self, "確認切換", confirmation) != QMessageBox.Yes:
                return
        identity = self._operator_reason("啟用檢測發布版本")
        if identity is None:
            return
        pointer = self.store.active_pointer(release.scope)
        expected_id = str(pointer.get("release_id")) if pointer else None
        try:
            self.store.activate(
                release,
                mode=mode,
                operator=identity[0],
                reason=identity[1],
                expected_release_id=expected_id,
            )
            if self.on_activated:
                self.on_activated(release)
            self.refresh()
        except InspectionReleaseError as exc:
            QMessageBox.critical(self, "切換失敗", str(exc))

    def _rollback(self) -> None:
        release = self._selected_release()
        if release is None or not self._guard_idle():
            return
        identity = self._operator_reason("退回檢測發布版本")
        if identity is None:
            return
        try:
            pointer = self.store.rollback(
                release.scope,
                operator=identity[0],
                reason=identity[1],
            )
            restored = self.store.load(
                release.scope, str(pointer["release_id"])
            )
            if self.on_activated:
                self.on_activated(restored)
            self.refresh()
        except InspectionReleaseError as exc:
            QMessageBox.critical(self, "退回失敗", str(exc))

    def _open_report(self) -> None:
        release = self._selected_release()
        if release is None or not release.validation.report_path:
            return
        report_dir = Path(release.validation.report_path).parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(report_dir)))
