"""Background workflow for rebuilding a complete Stats Color baseline."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from app.gui.color_baseline_exclusions_dialog import (
    ColorBaselineExclusionsDialog,
)
from app.gui.inspection_release_presentation import (
    format_color_baseline_summary,
)
from core.services.color_baseline_evidence import (
    ColorBaselineEvidenceExclusion,
    ColorBaselineEvidenceProvider,
)
from core.services.color_baseline_recalibration import (
    ColorBaselineCancelled,
    ColorBaselineCandidateStore,
    ColorBaselineError,
    StatsColorBaselineRebuilder,
    collect_color_baseline_evidence,
)
from core.services.model_acceptance import (
    AcceptanceInferenceService,
    AcceptanceRepository,
    ModelIdentity,
)
from core.services.model_version_registry import ModelVersionRecord
from core.station_data import load_station_data_paths
from core.workspace import load_workspace_paths


class ColorBaselineRebuildWorker(QThread):
    """Run detector inference and robust statistics outside the UI thread."""

    progress_changed = pyqtSignal(int, int, str)
    phase_changed = pyqtSignal(str)
    evidence_ready = pyqtSignal(object)
    outlier_filter_ready = pyqtSignal(object)
    completed = pyqtSignal(object, object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        *,
        project_root: Path,
        product: str,
        area: str,
        inference_type: str,
        model: ModelVersionRecord,
    ) -> None:
        super().__init__()
        self.project_root = project_root.resolve()
        self.data_paths = load_station_data_paths(self.project_root)
        self.product = product
        self.area = area
        self.inference_type = inference_type
        self.model = model

    def run(self) -> None:
        service: AcceptanceInferenceService | None = None
        try:
            if self.model.config_snapshot_path is None:
                raise ColorBaselineError("選取的模型版本缺少 config 快照，無法安全重建。")
            config_path = self.model.config_snapshot_path.resolve()
            base_model_path = _resolve_color_model(
                self.project_root,
                config_path,
            )
            selected_type = "yolo" if self.inference_type.casefold() == "fusion" else self.inference_type.casefold()
            evidence_provider = ColorBaselineEvidenceProvider(
                product=self.product,
                area=self.area,
                model_type=selected_type,
            )
            repository = AcceptanceRepository(
                self.data_paths.acceptance
                / evidence_provider.product
                / evidence_provider.area
            )
            feedback_manifest = (
                load_workspace_paths(self.project_root).training_data
                / evidence_provider.product
                / evidence_provider.area
                / "color_review"
                / "feedback.csv"
            )
            self.phase_changed.emit("正在整理驗收與顏色覆核的人工 OK 樣本…")
            evidence_snapshot = evidence_provider.collect(
                acceptance_repository=repository,
                feedback_manifest=feedback_manifest,
            )
            self.evidence_ready.emit(evidence_snapshot)
            identity = ModelIdentity(
                version=self.model.version,
                sha256=self.model.weight_sha256,
                runtime_config_sha256=_sha256_file(config_path),
            )
            self.phase_changed.emit("正在使用選定模型重新偵測人工確認 OK 照片…")
            service = AcceptanceInferenceService(
                project_root=self.project_root,
                models_root=self.data_paths.models,
                model_identity=identity,
                include_active_color_revisions=False,
                model_config_overrides={
                    (
                        self.product,
                        self.area,
                        selected_type,
                    ): config_path
                },
            )
            evidence = collect_color_baseline_evidence(
                inference_service=service,
                samples=evidence_snapshot.samples,
                inference_type=self.inference_type,
                progress_callback=self.progress_changed.emit,
                cancel_callback=self.isInterruptionRequested,
            )
            self.phase_changed.emit("正在分割建模與保留樣本，重算顏色統計…")
            build = StatsColorBaselineRebuilder().build(
                base_model_path=base_model_path,
                evidence=evidence,
                evidence_metadata=evidence_snapshot.to_report_dict(),
                cancel_callback=self.isInterruptionRequested,
            )
            self.outlier_filter_ready.emit(build.outlier_filter)
            candidate = ColorBaselineCandidateStore(self.data_paths.color_baselines).commit(
                product=self.product,
                area=self.area,
                model_type=selected_type,
                build=build,
            )
            self.completed.emit(candidate, build.color_reports)
        except ColorBaselineCancelled:
            self.cancelled.emit()
        except (
            ImportError,
            OSError,
            RuntimeError,
            ValueError,
            yaml.YAMLError,
        ) as exc:
            self.failed.emit(str(exc))
        finally:
            if service is not None:
                service.close()


class ColorBaselineRebuildDialog(QDialog):
    """Show evidence progress and the per-color rebuild safety result."""

    candidate_created = pyqtSignal(object)

    def __init__(
        self,
        *,
        project_root: Path,
        product: str,
        area: str,
        inference_type: str,
        model: ModelVersionRecord,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("重建完整顏色基準")
        self.resize(920, 560)
        self._worker = ColorBaselineRebuildWorker(
            project_root=project_root,
            product=product,
            area=area,
            inference_type=inference_type,
            model=model,
        )
        self._finished = False
        self._excluded_evidence = ()
        self._evidence_samples_by_id = {}
        self._build_ui(model)
        self._connect_worker()

    def _build_ui(self, model: ModelVersionRecord) -> None:
        layout = QVBoxLayout(self)
        description = QLabel(
            f"模型：{model.model_type.upper()} {model.version}\n"
            "來源只使用人工確認為 OK 的驗收照片；系統會重新偵測元件框，"
            "每張裁切等量取樣，並保留一部分照片比較新舊基準。"
            "此操作不會修改人工標註、模型設定或目前正式組合。"
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self.phase_label = QLabel("準備中…")
        self.phase_label.setWordWrap(True)
        layout.addWidget(self.phase_label)
        self.evidence_label = QLabel("證據來源會在開始重建後顯示。")
        self.evidence_label.setWordWrap(True)
        self.evidence_label.setStyleSheet(
            "background:#eef6ff;color:#174f78;border:1px solid #b8d4ea;"
            "border-radius:5px;padding:8px;"
        )
        layout.addWidget(self.evidence_label)
        self.excluded_evidence_button = QPushButton("查看排除照片")
        self.excluded_evidence_button.setVisible(False)
        self.excluded_evidence_button.clicked.connect(
            self._show_excluded_evidence
        )
        layout.addWidget(self.excluded_evidence_button)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        layout.addWidget(self.progress)

        self.result_table = QTableWidget(0, 8)
        self.result_table.setHorizontalHeaderLabels(
            (
                "色別",
                "結果",
                "全部裁切",
                "建模",
                "保留驗證",
                "舊基準",
                "新基準",
                "說明",
            )
        )
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        header = self.result_table.horizontalHeader()
        for column in range(7):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.Stretch)
        layout.addWidget(self.result_table, 1)

        self.next_step_label = QLabel()
        self.next_step_label.setWordWrap(True)
        self.next_step_label.setVisible(False)
        self.next_step_label.setStyleSheet(
            "background:#eaf4ff;color:#174f78;border:1px solid #a8cbe5;"
            "border-radius:5px;padding:8px;font-weight:600;"
        )
        layout.addWidget(self.next_step_label)

        warning = QLabel(
            "限制：人工 OK 可建立正常顏色分布，但無法證明真實顏色缺陷的漏檢率。"
            "候選建立後仍要在「組合驗收」使用既有驗收資料確認。"
        )
        warning.setWordWrap(True)
        warning.setStyleSheet(
            "background:#fff7e8;color:#8a5a00;border:1px solid #ead3a3;border-radius:5px;padding:8px;"
        )
        layout.addWidget(warning)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self._cancel_or_close)
        actions.addWidget(self.cancel_button)
        self.start_button = QPushButton("開始重建")
        self.start_button.clicked.connect(self._start)
        actions.addWidget(self.start_button)
        layout.addLayout(actions)

    def _connect_worker(self) -> None:
        self._worker.phase_changed.connect(self.phase_label.setText)
        self._worker.evidence_ready.connect(self._show_evidence_summary)
        self._worker.outlier_filter_ready.connect(self._show_outlier_summary)
        self._worker.progress_changed.connect(self._update_progress)
        self._worker.completed.connect(self._completed)
        self._worker.failed.connect(self._failed)
        self._worker.cancelled.connect(self._cancelled)

    def _start(self) -> None:
        self.start_button.setEnabled(False)
        self.cancel_button.setText("取消")
        self._worker.start()

    def _update_progress(
        self,
        current: int,
        total: int,
        sample_id: str,
    ) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(current)
        self.progress.setFormat(f"{current}/{total}｜{sample_id}" if total else sample_id)

    def _show_evidence_summary(self, snapshot) -> None:
        self._excluded_evidence = tuple(snapshot.excluded_samples)
        self._evidence_samples_by_id = {
            sample.sample_id: sample for sample in snapshot.samples
        }
        excluded_count = len(self._excluded_evidence)
        self.excluded_evidence_button.setText(
            f"查看排除照片（{excluded_count}）"
        )
        self.excluded_evidence_button.setVisible(excluded_count > 0)
        exclusion_text = f"｜已排除 {excluded_count} 張" if excluded_count else ""
        self.evidence_label.setText(
            f"本次選用 {snapshot.selected_count} 張：驗收 OK "
            f"{snapshot.selected_acceptance_count} 張＋顏色覆核 OK "
            f"{snapshot.selected_feedback_count} 張｜去重 {snapshot.duplicate_count} 張｜"
            f"NG 不納入基準統計 {snapshot.confirmed_ng_count} 張"
            f"{exclusion_text}"
        )

    def _show_outlier_summary(self, report) -> None:
        statistical_exclusions: list[ColorBaselineEvidenceExclusion] = []
        existing_sample_ids = {
            exclusion.sample_id for exclusion in self._excluded_evidence
        }
        for sample_id in report.excluded_sample_ids:
            if sample_id in existing_sample_ids:
                continue
            sample = self._evidence_samples_by_id.get(sample_id)
            statistical_exclusions.append(
                ColorBaselineEvidenceExclusion(
                    sample_id=sample_id,
                    source_kind=(sample.source_kind if sample is not None else ""),
                    source_manifest=(
                        sample.source_manifest if sample is not None else ""
                    ),
                    image_path=(
                        str(sample.image_path) if sample is not None else ""
                    ),
                    image_sha256=(
                        sample.image_sha256 if sample is not None else ""
                    ),
                    reason_code="STATISTICAL_COLOR_OUTLIER",
                    reason=(
                        "照片顏色與同批多數正常照片差異過大，已從本次基準建模排除；"
                        "原始照片與人工真值未修改。"
                    ),
                )
            )
        if statistical_exclusions:
            self._excluded_evidence = (
                *self._excluded_evidence,
                *statistical_exclusions,
            )
            self.excluded_evidence_button.setText(
                f"查看排除照片（{len(self._excluded_evidence)}）"
            )
            self.excluded_evidence_button.setVisible(True)
            self.evidence_label.setText(
                self.evidence_label.text()
                + f"｜重建前排除離群照片 {report.excluded_count} 張"
            )

    def _show_excluded_evidence(self) -> None:
        if not self._excluded_evidence:
            return
        ColorBaselineExclusionsDialog(
            self._excluded_evidence,
            parent=self,
        ).exec_()

    def _completed(self, candidate, color_reports) -> None:
        self._finished = True
        self.progress.setValue(self.progress.maximum())
        candidate_summary = format_color_baseline_summary(
            color_count=len(candidate.colors),
            created_at=candidate.created_at,
            lifecycle_status="CANDIDATE",
            quality_status=candidate.status,
        )
        internal_id_tooltip = f"完整基準內部 ID：{candidate.display_version}"
        self.phase_label.setText(f"候選已建立：{candidate_summary}")
        self.phase_label.setToolTip(internal_id_tooltip)
        result_text = (
            "數值檢查通過"
            if candidate.status == "READY"
            else "需先比較新舊組合，不代表已核准"
        )
        self.next_step_label.setText(
            "下一步：此完整顏色基準候選已自動選回「候選組合」"
            f"（{result_text}）。關閉本視窗後，填寫組合版本、建立人員與原因，"
            "建立候選組合；再到「組合驗收」使用同一批驗收照片比較，"
            "通過後才可上線。"
        )
        self.next_step_label.setToolTip(internal_id_tooltip)
        self.next_step_label.setVisible(True)
        self.result_table.setRowCount(len(color_reports))
        for row, report in enumerate(color_reports):
            values = (
                report.color,
                _state_label(report.state),
                str(report.total_crops),
                str(report.training_crops),
                str(report.holdout_crops),
                _percent(report.previous_accuracy),
                _percent(report.candidate_accuracy),
                report.note or "統計已重建",
            )
            for column, value in enumerate(values):
                self.result_table.setItem(
                    row,
                    column,
                    QTableWidgetItem(value),
                )
        self.cancel_button.setText("關閉")
        self.start_button.setText("重建完成")
        self.candidate_created.emit(candidate)

    def _failed(self, message: str) -> None:
        self._finished = True
        self.phase_label.setText("重建失敗")
        self.cancel_button.setText("關閉")
        QMessageBox.critical(self, "顏色基準重建失敗", message)

    def _cancelled(self) -> None:
        self._finished = True
        self.phase_label.setText("已取消，未建立候選基準。")
        self.cancel_button.setText("關閉")

    def _cancel_or_close(self) -> None:
        if self._worker.isRunning():
            self._worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.phase_label.setText("正在安全停止…")
            return
        self.close()

    def closeEvent(self, event) -> None:
        if self._worker.isRunning():
            self._worker.requestInterruption()
            event.ignore()
            return
        super().closeEvent(event)


def _resolve_color_model(project_root: Path, config_path: Path) -> Path:
    if config_path.is_symlink() or not config_path.is_file():
        raise ColorBaselineError("選取的模型 config 快照不存在。")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict) or not payload.get("enable_color_check"):
        raise ColorBaselineError("選取的模型版本未啟用 Stats Color。")
    if str(payload.get("color_checker_type") or "stats").casefold() != "stats":
        raise ColorBaselineError("目前只支援重建 Stats Color 完整基準。")
    raw_value = str(payload.get("color_model_path") or "").strip()
    if not raw_value:
        raise ColorBaselineError("模型 config 缺少 color_model_path。")
    raw = Path(raw_value).expanduser()
    candidates = (raw,) if raw.is_absolute() else (project_root / raw, config_path.parent / raw)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and not resolved.is_symlink():
            return resolved
    raise ColorBaselineError(f"找不到舊顏色基準：{raw_value}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _state_label(value: str) -> str:
    return {
        "REBUILT": "已重建",
        "PRESERVED_INSUFFICIENT": "證據不足，沿用舊值",
        "PRESERVED_SAFETY_REJECTED": "未更新，沿用舊基準",
        "REVIEW_REQUIRED": "需人工檢查",
    }.get(value, value)
