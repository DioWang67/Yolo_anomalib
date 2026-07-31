"""YOLO × color-revision matrix dialog for the independent acceptance tool."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from core.services.acceptance_matrix import (
    AcceptanceColorVariant,
    AcceptanceMatrixCancelled,
    AcceptanceMatrixRequest,
    AcceptanceMatrixResult,
    AcceptanceModelVariant,
    build_model_variant,
    build_registered_model_variant,
    discover_color_variants,
    run_acceptance_matrix,
)
from core.services.model_acceptance import AcceptanceRepository
from core.services.model_version_registry import (
    ModelVersionRegistry,
    ModelVersionRegistryError,
)

VARIANT_ROLE = Qt.UserRole


class AcceptanceMatrixWorker(QThread):
    """Run GPU/CPU inference outside the UI thread, sequentially per matrix."""

    progress_changed = pyqtSignal(int, int, str, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        request: AcceptanceMatrixRequest,
        *,
        runner: Callable[..., AcceptanceMatrixResult] = run_acceptance_matrix,
    ) -> None:
        super().__init__()
        self._request = request
        self._runner = runner

    def run(self) -> None:
        try:
            result = self._runner(
                self._request,
                progress_callback=self._emit_progress,
                cancel_callback=self.isInterruptionRequested,
            )
            self.completed.emit(result)
        except AcceptanceMatrixCancelled:
            self.cancelled.emit()
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            self.failed.emit(str(exc))

    def _emit_progress(
        self,
        current: int,
        total: int,
        combination_label: str,
        sample_id: str,
    ) -> None:
        self.progress_changed.emit(
            current,
            total,
            combination_label,
            sample_id,
        )


class AcceptanceMatrixDialog(QDialog):
    """Select model/config columns and inspect the append-only matrix result."""

    def __init__(
        self,
        *,
        project_root: Path,
        repository: AcceptanceRepository,
        product: str,
        area: str,
        inference_type: str,
        runner: Callable[..., AcceptanceMatrixResult] = run_acceptance_matrix,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.project_root = project_root.resolve()
        self.repository = repository
        self.product = product.strip()
        self.area = area.strip()
        self.inference_type = inference_type.strip()
        self._runner = runner
        self._worker: AcceptanceMatrixWorker | None = None
        self._result: AcceptanceMatrixResult | None = None
        self._close_when_finished = False
        self.setWindowTitle("YOLO × 顏色版本｜驗收矩陣")
        self.resize(1220, 760)
        self._build_ui()
        self._load_initial_variants()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        description = QLabel(
            "每個勾選的 YOLO 會和每個勾選的顏色設定配對，使用相同且已確認的"
            "驗收照片重新推論。此功能不會切換正式模型、不會改 Active 顏色版本，"
            "也不會覆寫人工標註。"
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        layout.addWidget(QLabel("1. 選擇 YOLO 模型 Bundle"))
        self.model_table = QTableWidget(0, 4)
        self.model_table.setHorizontalHeaderLabels(("測試", "顯示名稱", "模型版本", "models 根目錄"))
        self.model_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.model_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.model_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.model_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.model_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        layout.addWidget(self.model_table)

        model_actions = QHBoxLayout()
        self.add_model_button = QPushButton("加入其他 models 目錄")
        self.remove_model_button = QPushButton("移除選取列")
        model_hint = QLabel(
            "可加入訓練工作目錄中的 acceptance_candidate/models，"
            "或任何具有 <產品>/<區域>/yolo/config.yaml 的完整 Bundle。"
        )
        model_hint.setWordWrap(True)
        model_actions.addWidget(self.add_model_button)
        model_actions.addWidget(self.remove_model_button)
        model_actions.addWidget(model_hint, 1)
        layout.addLayout(model_actions)

        layout.addWidget(QLabel("2. 選擇顏色設定"))
        self.color_table = QTableWidget(0, 3)
        self.color_table.setHorizontalHeaderLabels(("測試", "顏色設定", "套用方式"))
        self.color_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.color_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.color_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.color_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.color_table.setMaximumHeight(210)
        layout.addWidget(self.color_table)

        controls = QHBoxLayout()
        self.workload_label = QLabel()
        self.run_button = QPushButton("開始組合測試")
        self.run_button.setStyleSheet("QPushButton { background: #006f5f; color: white; padding: 8px 18px; }")
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.open_report_button = QPushButton("開啟報告資料夾")
        self.open_report_button.setEnabled(False)
        self.create_release_button = QPushButton("建立檢測發布版本")
        self.create_release_button.setEnabled(False)
        self.close_button = QPushButton("關閉")
        controls.addWidget(self.workload_label, 1)
        controls.addWidget(self.run_button)
        controls.addWidget(self.cancel_button)
        controls.addWidget(self.open_report_button)
        controls.addWidget(self.create_release_button)
        controls.addWidget(self.close_button)
        layout.addLayout(controls)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress_detail = QLabel()
        self.progress_detail.setWordWrap(True)
        layout.addWidget(self.progress)
        layout.addWidget(self.progress_detail)

        self.result_table = QTableWidget(0, 12)
        self.result_table.setHorizontalHeaderLabels(
            (
                "YOLO",
                "顏色設定",
                "準確率",
                "誤殺",
                "漏檢",
                "顏色誤殺率",
                "顏色逃逸率",
                "平均 ms",
                "P95 ms",
                "相較首組變動",
                "錯誤",
                "狀態",
            )
        )
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.result_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.result_table, 1)

        warning = QLabel(
            "注意：如果驗收集中沒有任何人工標為 COLOR_MISMATCH 的真 NG，"
            "顏色逃逸率會顯示 UNKNOWN；這不是 0%，而是缺少可驗證逃逸的樣本。"
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #9a5b00;")
        layout.addWidget(warning)

        self.add_model_button.clicked.connect(self._add_model_root)
        self.remove_model_button.clicked.connect(self._remove_selected_model)
        self.run_button.clicked.connect(self._start)
        self.cancel_button.clicked.connect(self._cancel)
        self.open_report_button.clicked.connect(self._open_report_folder)
        self.create_release_button.clicked.connect(self._create_release)
        self.close_button.clicked.connect(self.close)
        self.model_table.itemChanged.connect(self._update_workload)
        self.color_table.itemChanged.connect(self._update_workload)
        self.result_table.itemSelectionChanged.connect(self._update_release_button)

    def _load_initial_variants(self) -> None:
        try:
            models_root = self.project_root / "models"
            registry = ModelVersionRegistry(models_root)
            records = registry.list_versions(
                product=self.product,
                area=self.area,
                model_type="yolo",
            )
            skipped: list[str] = []
            for record in records:
                if not record.exists or not record.has_config_snapshot:
                    skipped.append(f"{record.version}（{record.warning or '缺少完整檔案'}）")
                    continue
                try:
                    self._append_model_variant(
                        build_registered_model_variant(
                            record,
                            models_root=models_root,
                        ),
                        checked=record.is_current,
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    skipped.append(f"{record.version}（{exc}）")
            if self.model_table.rowCount() == 0:
                self._append_model_variant(
                    build_model_variant(
                        models_root,
                        product=self.product,
                        area=self.area,
                        inference_type=self.inference_type,
                        label="目前推論專案模型",
                    )
                )
            if skipped:
                self.progress_detail.setText("未列入不完整／不可信的歷史模型：" + "、".join(skipped))
            variants = discover_color_variants(
                self.project_root / ".color_revisions",
                product=self.product,
                area=self.area,
                model_type=("yolo" if self.inference_type.lower() == "fusion" else self.inference_type),
                baselines_root=self.project_root / ".color_baselines",
                profiles_root=self.project_root / ".color_profiles",
            )
            for variant in variants:
                self._append_color_variant(variant)
        except (
            ModelVersionRegistryError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            QMessageBox.critical(self, "無法載入矩陣選項", str(exc))
        self._update_workload()

    def _append_model_variant(
        self,
        variant: AcceptanceModelVariant,
        *,
        checked: bool = True,
    ) -> None:
        if self._has_variant(self.model_table, variant.variant_id):
            return
        row = self.model_table.rowCount()
        self.model_table.insertRow(row)
        check_item = _check_item(variant, checked=checked)
        self.model_table.setItem(row, 0, check_item)
        self.model_table.setItem(row, 1, QTableWidgetItem(variant.label))
        self.model_table.setItem(row, 2, QTableWidgetItem(variant.identity.version))
        self.model_table.setItem(row, 3, QTableWidgetItem(str(variant.models_root)))

    def _append_color_variant(self, variant: AcceptanceColorVariant) -> None:
        if self._has_variant(self.color_table, variant.variant_id):
            return
        row = self.color_table.rowCount()
        self.color_table.insertRow(row)
        self.color_table.setItem(row, 0, _check_item(variant))
        self.color_table.setItem(row, 1, QTableWidgetItem(variant.label))
        if variant.color_model_path is not None and variant.revision_overrides:
            source = "鎖定完整顏色方案（基準＋逐色修訂，不啟用）"
        elif variant.color_model_path is not None:
            source = "鎖定 immutable 完整顏色基準（不啟用）"
        elif variant.include_active_revisions:
            source = "執行當下讀取正式 Active（不修改）"
        elif variant.revision_overrides:
            source = "鎖定 immutable revision（不啟用）"
        else:
            source = "只用所選 YOLO Bundle 內建 config"
        self.color_table.setItem(row, 2, QTableWidgetItem(source))

    @staticmethod
    def _has_variant(table: QTableWidget, variant_id: str) -> bool:
        return any(
            getattr(table.item(row, 0).data(VARIANT_ROLE), "variant_id", "") == variant_id
            for row in range(table.rowCount())
            if table.item(row, 0) is not None
        )

    def _add_model_root(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "選擇完整 models 根目錄",
            str(self.project_root.parent),
        )
        if not selected:
            return
        try:
            variant = build_model_variant(
                selected,
                product=self.product,
                area=self.area,
                inference_type=self.inference_type,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "無法加入模型", str(exc))
            return
        self._append_model_variant(variant)
        self._update_workload()

    def _remove_selected_model(self) -> None:
        rows = sorted(
            {index.row() for index in self.model_table.selectedIndexes()},
            reverse=True,
        )
        for row in rows:
            self.model_table.removeRow(row)
        self._update_workload()

    def _selected_models(self) -> tuple[AcceptanceModelVariant, ...]:
        return tuple(
            item
            for row in range(self.model_table.rowCount())
            if (check := self.model_table.item(row, 0)) is not None
            and check.checkState() == Qt.Checked
            and isinstance(
                (item := check.data(VARIANT_ROLE)),
                AcceptanceModelVariant,
            )
        )

    def _selected_colors(self) -> tuple[AcceptanceColorVariant, ...]:
        return tuple(
            item
            for row in range(self.color_table.rowCount())
            if (check := self.color_table.item(row, 0)) is not None
            and check.checkState() == Qt.Checked
            and isinstance(
                (item := check.data(VARIANT_ROLE)),
                AcceptanceColorVariant,
            )
        )

    def _update_workload(self, _item=None) -> None:
        samples = sum(record.review_status == "confirmed" for record in self.repository.records())
        combinations = len(self._selected_models()) * len(self._selected_colors())
        self.workload_label.setText(
            f"{samples} 張已確認照片 × {combinations} 組 = {samples * combinations} 次推論（循序執行）"
        )

    def _start(self) -> None:
        if self._worker is not None:
            return
        models = self._selected_models()
        colors = self._selected_colors()
        if not models or not colors:
            QMessageBox.warning(
                self,
                "缺少組合",
                "至少勾選一個 YOLO 與一個顏色設定。",
            )
            return
        request = AcceptanceMatrixRequest(
            project_root=self.project_root,
            global_config_path=self.project_root / "config.yaml",
            color_revisions_root=self.project_root / ".color_revisions",
            dataset_root=self.repository.root,
            manifest_path=self.repository.manifest_path,
            output_root=(self.project_root / "acceptance_reports" / self.product / self.area / "matrix_runs"),
            product=self.product,
            area=self.area,
            inference_type=self.inference_type,
            model_variants=models,
            color_variants=colors,
        )
        worker = AcceptanceMatrixWorker(request, runner=self._runner)
        worker.progress_changed.connect(self._progress_changed)
        worker.completed.connect(self._completed)
        worker.failed.connect(self._failed)
        worker.cancelled.connect(self._cancelled)
        worker.finished.connect(self._finished)
        self._worker = worker
        total = (
            sum(record.review_status == "confirmed" for record in self.repository.records()) * len(models) * len(colors)
        )
        self.progress.setRange(0, total)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.result_table.setRowCount(0)
        self._set_running(True)
        worker.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.progress_detail.setText("正在安全停止；不會留下半份報告…")

    def _progress_changed(
        self,
        current: int,
        total: int,
        combination_label: str,
        sample_id: str,
    ) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(current)
        self.progress_detail.setText(f"{current}/{total}｜{combination_label}｜{sample_id}")

    def _completed(self, raw_result: object) -> None:
        if not isinstance(raw_result, AcceptanceMatrixResult):
            self._failed("矩陣執行器回傳了無效結果。")
            return
        self._result = raw_result
        self.open_report_button.setEnabled(True)
        self._render_results(raw_result)
        if self.result_table.rowCount():
            self.result_table.selectRow(0)
        self.progress_detail.setText(
            f"完成：{raw_result.sample_count} 張、{len(raw_result.combinations)} 組。報告：{raw_result.run_root}"
        )

    def _render_results(self, result: AcceptanceMatrixResult) -> None:
        self.result_table.setRowCount(0)
        for combination in result.combinations:
            metrics = combination.metrics
            color = combination.color_metrics
            denominator = metrics.tp + metrics.fp + metrics.fn + metrics.tn
            accuracy = (metrics.tp + metrics.tn) / denominator if denominator else None
            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            values = (
                combination.model_label,
                combination.color_label,
                _format_rate(accuracy),
                metrics.fp,
                metrics.fn,
                _format_rate(color.overkill_rate),
                _format_color_escape(color.escape_rate),
                _format_number(combination.average_latency_ms),
                _format_number(combination.p95_latency_ms),
                combination.changed_from_reference,
                metrics.errors,
                combination.error or "完成",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(VARIANT_ROLE, combination.combination_id)
                if (
                    (column == 3 and metrics.fp > 0)
                    or (column == 4 and metrics.fn > 0)
                    or (column == 10 and metrics.errors > 0)
                    or (column == 11 and combination.error)
                ):
                    item.setForeground(Qt.red)
                self.result_table.setItem(row, column, item)
        self.result_table.resizeColumnsToContents()
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self._update_release_button()

    def _failed(self, message: str) -> None:
        self.progress_detail.setText(f"失敗：{message}")
        QMessageBox.critical(self, "驗收矩陣失敗", message)

    def _cancelled(self) -> None:
        self.progress_detail.setText("已取消；未建立半份報告。")

    def _finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_running(False)
        if self._close_when_finished:
            self._close_when_finished = False
            self.close()

    def _set_running(self, running: bool) -> None:
        self.model_table.setEnabled(not running)
        self.color_table.setEnabled(not running)
        self.add_model_button.setEnabled(not running)
        self.remove_model_button.setEnabled(not running)
        self.run_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.close_button.setEnabled(not running)
        self._update_release_button()

    def _update_release_button(self) -> None:
        self.create_release_button.setEnabled(
            self._worker is None and self._result is not None and self.result_table.currentRow() >= 0
        )

    def _create_release(self) -> None:
        """Create an immutable candidate; activation stays in Engineering."""
        if self._result is None or self.result_table.currentRow() < 0:
            return
        row = self.result_table.currentRow()
        item = self.result_table.item(row, 0)
        combination_id = str(item.data(VARIANT_ROLE) or "") if item else ""
        if not combination_id:
            QMessageBox.warning(self, "建立發布版本", "找不到選取的組合識別碼。")
            return
        try:
            from core.services.inspection_release_builder import (
                build_release_from_matrix,
            )
            from core.services.inspection_release_models import ActivationMode
            from core.services.inspection_release_store import (
                InspectionReleaseStore,
            )

            store = InspectionReleaseStore(self.project_root / ".inspection_releases")
            existing = store.list_releases(product=self.product, area=self.area)
            suggested = _next_release_version(existing)
            version, ok = QInputDialog.getText(
                self,
                "建立檢測發布版本",
                "發布版本：",
                text=suggested,
            )
            if not ok or not version.strip():
                return
            operator, ok = QInputDialog.getText(self, "建立檢測發布版本", "建立人員：")
            if not ok or not operator.strip():
                return
            reason, ok = QInputDialog.getMultiLineText(self, "建立檢測發布版本", "建立原因：")
            if not ok or not reason.strip():
                return
            release = build_release_from_matrix(
                self._result.report_path,
                combination_id=combination_id,
                display_version=version.strip(),
                operator=operator.strip(),
                reason=reason.strip(),
            )
            store.commit(release)
            allowed = store.policy.allowed_modes(release)
            if ActivationMode.FULL in allowed:
                policy = "驗證完整，可正式啟用或先有限試跑。"
            else:
                policy = "驗證資料仍有風險；工程設定中可選有限試跑，也可記錄風險接受後套用。"
            QMessageBox.information(
                self,
                "發布版本已建立",
                f"{release.display_version} 已建立，但尚未套用。\n{policy}",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.critical(self, "建立發布版本失敗", str(exc))

    def _open_report_folder(self) -> None:
        if self._result is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._result.run_root)))

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._close_when_finished = True
            self.progress_detail.setText("正在安全停止；停止後會自動關閉。")
            event.ignore()
            return
        event.accept()


def _check_item(variant: object, *, checked: bool = True) -> QTableWidgetItem:
    item = QTableWidgetItem()
    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
    item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
    item.setData(VARIANT_ROLE, variant)
    return item


def _next_release_version(releases) -> str:
    """Return a collision-free human version for one product/area."""
    patches = []
    for release in releases:
        match = re.fullmatch(r"inspection-v1\.0\.(\d+)", release.display_version.strip())
        if match:
            patches.append(int(match.group(1)))
    return f"inspection-v1.0.{max(patches, default=0) + 1}"


def _format_rate(value: float | None) -> str:
    return "UNKNOWN" if value is None else f"{value:.2%}"


def _format_color_escape(value: float | None) -> str:
    return "UNKNOWN（無真顏色 NG）" if value is None else f"{value:.2%}"


def _format_number(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"
