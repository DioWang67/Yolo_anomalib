"""PyQt user interface for independent, production-equivalent acceptance."""

from __future__ import annotations

import getpass
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.acceptance.matrix_dialog import AcceptanceMatrixDialog
from core.services.model_acceptance import (
    ACCEPTANCE_REASON_CODES,
    SUPPORTED_IMAGE_SUFFIXES,
    AcceptanceDataError,
    AcceptanceInferenceOutcome,
    AcceptanceInferenceService,
    AcceptanceRecord,
    AcceptanceRepository,
    calculate_acceptance_metrics,
)
from core.services.model_catalog import ModelCatalog

SAMPLE_ID_ROLE = Qt.UserRole
REASON_LABELS = {
    "MISSING": "缺件",
    "WRONG_COMPONENT": "元件錯誤",
    "UNEXPECTED_COMPONENT": "多件／非預期元件",
    "POSITION_SHIFT": "位置偏移",
    "COLOR_MISMATCH": "顏色錯誤",
    "COUNT_MISMATCH": "數量錯誤",
    "SEQUENCE_MISMATCH": "順序錯誤",
    "OTHER": "其他",
}
FILTER_OPTIONS = (
    ("全部", "all"),
    ("人工／模型不一致", "mismatch"),
    ("誤殺", "false_positive"),
    ("漏檢", "false_negative"),
    ("模型顏色 NG", "color_mismatch"),
    ("待覆核", "pending"),
    ("推論錯誤", "error"),
)


class ScaledImageLabel(QLabel):
    """Image label that preserves the source pixmap while resizing."""

    def __init__(self, empty_text: str):
        super().__init__(empty_text)
        self._source: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(360, 300)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("background: #161a20; color: #aeb6c2;")

    def set_source(self, pixmap: QPixmap | None) -> None:
        self._source = pixmap
        self._refresh()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._source is None or self._source.isNull():
            self.setPixmap(QPixmap())
            return
        self.setPixmap(
            self._source.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


class InferenceBatchWorker(QThread):
    """Sequential inference worker; one backend instance, no shared-state races."""

    outcome_ready = pyqtSignal(object)
    progress_changed = pyqtSignal(int, int)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        project_root: Path,
        repository: AcceptanceRepository,
        records: tuple[AcceptanceRecord, ...],
        inference_type: str,
    ):
        super().__init__()
        self._project_root = project_root
        self._repository = repository
        self._records = records
        self._inference_type = inference_type

    def run(self) -> None:
        service: AcceptanceInferenceService | None = None
        try:
            service = AcceptanceInferenceService(project_root=self._project_root)
            total = len(self._records)
            for index, record in enumerate(self._records, start=1):
                if self.isInterruptionRequested():
                    break
                outcome = service.infer(
                    record,
                    self._repository.image_file(record),
                    inference_type=self._inference_type,
                    cancel_cb=self.isInterruptionRequested,
                )
                self.outcome_ready.emit(outcome)
                self.progress_changed.emit(index, total)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            self.failed.emit(str(exc))
        finally:
            if service is not None:
                service.close()


class BackupWorker(QThread):
    """Background ZIP exporter for large image evidence sets."""

    completed = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, repository: AcceptanceRepository, destination: Path):
        super().__init__()
        self._repository = repository
        self._destination = destination

    def run(self) -> None:
        try:
            exported = self._repository.export_backup_zip(self._destination)
            self.completed.emit(str(exported))
        except (OSError, ValueError) as exc:
            self.failed.emit(str(exc))


class ModelAcceptanceWindow(QMainWindow):
    """Independent acceptance UI backed by the production inference core."""

    def __init__(self, *, project_root: Path):
        super().__init__()
        self.project_root = project_root.resolve()
        self.catalog = ModelCatalog(self.project_root / "models")
        self.repository: AcceptanceRepository | None = None
        self._records: tuple[AcceptanceRecord, ...] = ()
        self._visible_records: tuple[AcceptanceRecord, ...] = ()
        self._annotated_pixmaps: dict[str, QPixmap] = {}
        self._worker: InferenceBatchWorker | None = None
        self._backup_worker: BackupWorker | None = None
        self._close_when_finished = False
        self._reason_checks: dict[str, QCheckBox] = {}
        self.setWindowTitle("模型驗收工具｜Model Acceptance")
        self.resize(1480, 900)
        self._build_ui()
        self._load_catalog()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)

        scope_row = QHBoxLayout()
        self.product_combo = QComboBox()
        self.area_combo = QComboBox()
        self.type_combo = QComboBox()
        self.batch_edit = QLineEdit()
        self.batch_edit.setPlaceholderText("例如 LOT-20260730")
        self.reviewer_edit = QLineEdit(getpass.getuser())
        for label, widget in (
            ("產品", self.product_combo),
            ("區域", self.area_combo),
            ("模型", self.type_combo),
            ("批次", self.batch_edit),
            ("覆核者", self.reviewer_edit),
        ):
            scope_row.addWidget(QLabel(label))
            scope_row.addWidget(widget)
        root_layout.addLayout(scope_row)

        action_row = QHBoxLayout()
        self.add_files_button = QPushButton("加入圖片")
        self.add_folder_button = QPushButton("加入資料夾")
        self.run_selected_button = QPushButton("推論目前圖片")
        self.run_pending_button = QPushButton("推論未完成圖片")
        self.run_all_button = QPushButton("全部重新推論")
        self.snapshot_button = QPushButton("建立基準快照")
        self.compare_button = QPushButton("比較最新快照")
        self.backup_button = QPushButton("匯出備份 ZIP")
        self.matrix_button = QPushButton("YOLO × 顏色組合測試")
        self.matrix_button.setStyleSheet(
            "QPushButton { background: #006f5f; color: white; }"
        )
        for button in (
            self.add_files_button,
            self.add_folder_button,
            self.run_selected_button,
            self.run_pending_button,
            self.run_all_button,
            self.matrix_button,
            self.snapshot_button,
            self.compare_button,
            self.backup_button,
        ):
            action_row.addWidget(button)
        action_row.addStretch(1)
        root_layout.addLayout(action_row)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_sample_panel())
        splitter.addWidget(self._build_image_panel())
        splitter.addWidget(self._build_review_panel())
        splitter.setSizes([300, 800, 360])
        root_layout.addWidget(splitter, 1)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.summary_label = QLabel()
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root_layout.addWidget(self.progress)
        root_layout.addWidget(self.summary_label)

        self.setCentralWidget(root)
        self.statusBar().showMessage("驗收圖片不會自動進入訓練資料。")

        self.product_combo.currentTextChanged.connect(self._product_changed)
        self.area_combo.currentTextChanged.connect(self._area_changed)
        self.type_combo.currentTextChanged.connect(self._scope_changed)
        self.sample_list.currentItemChanged.connect(self._sample_changed)
        self.add_files_button.clicked.connect(self._add_files)
        self.add_folder_button.clicked.connect(self._add_folder)
        self.run_selected_button.clicked.connect(self._run_selected)
        self.run_pending_button.clicked.connect(self._run_pending)
        self.run_all_button.clicked.connect(self._run_all)
        self.matrix_button.clicked.connect(self._open_matrix)
        self.snapshot_button.clicked.connect(self._create_snapshot)
        self.compare_button.clicked.connect(self._compare_latest_snapshot)
        self.backup_button.clicked.connect(self._export_backup)
        self.confirm_ok_button.clicked.connect(self._confirm_ok)
        self.confirm_ng_button.clicked.connect(self._confirm_ng)

    def _build_sample_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("驗收圖片"))
        self.filter_combo = QComboBox()
        for label, value in FILTER_OPTIONS:
            self.filter_combo.addItem(label, value)
        self.filter_combo.currentIndexChanged.connect(lambda _index: self._render_records())
        layout.addWidget(self.filter_combo)
        self.sample_list = QListWidget()
        self.sample_list.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.sample_list, 1)
        self.sample_count_label = QLabel("0 張")
        layout.addWidget(self.sample_count_label)
        return panel

    def _build_image_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        image_splitter = QSplitter(Qt.Vertical)
        original_group = QGroupBox("原始圖片")
        original_layout = QVBoxLayout(original_group)
        self.original_image = ScaledImageLabel("尚未選擇圖片")
        original_scroll = QScrollArea()
        original_scroll.setWidgetResizable(True)
        original_scroll.setWidget(self.original_image)
        original_layout.addWidget(original_scroll)
        prediction_group = QGroupBox("模型結果")
        prediction_layout = QVBoxLayout(prediction_group)
        self.prediction_image = ScaledImageLabel("尚未執行推論")
        prediction_scroll = QScrollArea()
        prediction_scroll.setWidgetResizable(True)
        prediction_scroll.setWidget(self.prediction_image)
        prediction_layout.addWidget(prediction_scroll)
        image_splitter.addWidget(original_group)
        image_splitter.addWidget(prediction_group)
        layout.addWidget(image_splitter)
        return panel

    def _build_review_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.detail_label = QLabel("請先加入圖片")
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.detail_label)

        color_group = QGroupBox("顏色判定明細")
        color_layout = QVBoxLayout(color_group)
        self.color_table = QTableWidget(0, 6)
        self.color_table.setHorizontalHeaderLabels(("索引", "YOLO 類別", "判定顏色", "差異值", "門檻", "結果"))
        self.color_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.color_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.color_table.setMaximumHeight(210)
        self.color_table.horizontalHeader().setStretchLastSection(True)
        color_layout.addWidget(self.color_table)
        layout.addWidget(color_group)

        reason_group = QGroupBox("實際 NG 原因（可複選）")
        reason_layout = QVBoxLayout(reason_group)
        for code in ACCEPTANCE_REASON_CODES:
            checkbox = QCheckBox(f"{REASON_LABELS[code]}  ({code})")
            self._reason_checks[code] = checkbox
            reason_layout.addWidget(checkbox)
        layout.addWidget(reason_group)

        form = QFormLayout()
        self.defect_class_edit = QLineEdit()
        self.defect_class_edit.setPlaceholderText("選填，例如 Black")
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setMaximumHeight(100)
        form.addRow("缺陷類別", self.defect_class_edit)
        form.addRow("備註", self.notes_edit)
        layout.addLayout(form)

        decision_row = QHBoxLayout()
        self.confirm_ok_button = QPushButton("確認實際 OK")
        self.confirm_ok_button.setStyleSheet("QPushButton { background: #197a45; color: white; padding: 10px; }")
        self.confirm_ng_button = QPushButton("確認實際 NG")
        self.confirm_ng_button.setStyleSheet("QPushButton { background: #a83b3b; color: white; padding: 10px; }")
        decision_row.addWidget(self.confirm_ok_button)
        decision_row.addWidget(self.confirm_ng_button)
        layout.addLayout(decision_row)
        layout.addStretch(1)
        return panel

    def _load_catalog(self) -> None:
        self.product_combo.blockSignals(True)
        self.product_combo.clear()
        self.product_combo.addItems(self.catalog.products())
        preferred = self.product_combo.findText("Cable1")
        if preferred >= 0:
            self.product_combo.setCurrentIndex(preferred)
        self.product_combo.blockSignals(False)
        self._product_changed(self.product_combo.currentText())

    def _product_changed(self, product: str) -> None:
        self.area_combo.blockSignals(True)
        self.area_combo.clear()
        self.area_combo.addItems(self.catalog.areas(product) if product else [])
        preferred = self.area_combo.findText("A")
        if preferred >= 0:
            self.area_combo.setCurrentIndex(preferred)
        self.area_combo.blockSignals(False)
        self._area_changed(self.area_combo.currentText())

    def _area_changed(self, area: str) -> None:
        product = self.product_combo.currentText()
        self.type_combo.blockSignals(True)
        self.type_combo.clear()
        available = self.catalog.inference_types(product, area) if product and area else []
        self.type_combo.addItems(available)
        preferred = self.type_combo.findText("yolo")
        if preferred >= 0:
            self.type_combo.setCurrentIndex(preferred)
        self.type_combo.blockSignals(False)
        self._scope_changed()

    def _scope_changed(self) -> None:
        product = self.product_combo.currentText().strip()
        area = self.area_combo.currentText().strip()
        if not product or not area:
            self.repository = None
            self._records = ()
            self._render_records()
            return
        try:
            self.repository = AcceptanceRepository(self.project_root / "acceptance" / product / area)
            self._reload_records()
        except (OSError, AcceptanceDataError) as exc:
            QMessageBox.critical(self, "驗收資料錯誤", str(exc))

    def _reload_records(self, *, select_id: str = "") -> None:
        if self.repository is None:
            return
        self._records = self.repository.records()
        self._render_records(select_id=select_id)
        self._update_summary()

    def _render_records(self, *, select_id: str = "") -> None:
        previous_id = select_id or self._selected_sample_id()
        filter_value = str(self.filter_combo.currentData() or "all") if hasattr(self, "filter_combo") else "all"
        self._visible_records = tuple(
            record for record in self._records if _record_matches_filter(record, filter_value)
        )
        self.sample_list.blockSignals(True)
        self.sample_list.clear()
        selected_row = -1
        for index, record in enumerate(self._visible_records):
            truth = record.expected_verdict or "待覆核"
            machine = record.machine_status or "未推論"
            item = QListWidgetItem(f"{record.sample_id}\n人工 {truth}｜模型 {machine}")
            item.setData(SAMPLE_ID_ROLE, record.sample_id)
            if record.machine_status == "ERROR":
                item.setForeground(Qt.darkRed)
            elif (
                record.review_status == "confirmed"
                and record.machine_status
                and record.expected_verdict != record.machine_status
            ):
                item.setForeground(Qt.red)
            elif record.review_status == "confirmed":
                item.setForeground(Qt.darkGreen)
            self.sample_list.addItem(item)
            if record.sample_id == previous_id:
                selected_row = index
        self.sample_list.blockSignals(False)
        self.sample_count_label.setText(f"顯示 {len(self._visible_records)}／{len(self._records)} 張")
        if self._visible_records:
            self.sample_list.setCurrentRow(selected_row if selected_row >= 0 else 0)
        else:
            self.original_image.set_source(None)
            self.prediction_image.set_source(None)
            self.detail_label.setText("請先加入圖片")

    def _selected_sample_id(self) -> str:
        item = self.sample_list.currentItem()
        return str(item.data(SAMPLE_ID_ROLE) or "") if item else ""

    def _selected_record(self) -> AcceptanceRecord | None:
        sample_id = self._selected_sample_id()
        return next(
            (record for record in self._records if record.sample_id == sample_id),
            None,
        )

    def _sample_changed(self, current: QListWidgetItem | None, _previous) -> None:
        if current is None or self.repository is None:
            return
        record = self._selected_record()
        if record is None:
            return
        pixmap = QPixmap(str(self.repository.image_file(record)))
        self.original_image.set_source(pixmap if not pixmap.isNull() else None)
        self.prediction_image.set_source(self._annotated_pixmaps.get(record.sample_id))
        self._render_record_detail(record)

    def _render_record_detail(self, record: AcceptanceRecord) -> None:
        reasons = record.expected_reasons.split("|") if record.expected_reasons else []
        for code, checkbox in self._reason_checks.items():
            checkbox.setChecked(code in reasons)
        self.defect_class_edit.setText(record.defect_class)
        self.notes_edit.setPlainText(record.notes)
        lines = [
            f"樣本：{record.sample_id}",
            f"人工真值：{record.expected_verdict or '尚未確認'}",
            f"模型判定：{record.machine_status or '尚未推論'}",
            f"模型原因：{record.machine_reasons or '—'}",
            f"模型版本：{record.model_version or '—'}",
            f"Runtime config：{record.runtime_config_sha256[:12] or '—'}",
            f"Color model：{record.color_model_sha256[:12] or '—'}",
            f"推論時間：{record.latency_ms + ' ms' if record.latency_ms else '—'}",
        ]
        if record.error:
            lines.append(f"錯誤：{record.error}")
        self.detail_label.setText("\n".join(lines))
        self._render_color_details(record)

    def _render_color_details(self, record: AcceptanceRecord) -> None:
        self.color_table.setRowCount(0)
        try:
            details = json.loads(record.color_details_json) if record.color_details_json else []
        except json.JSONDecodeError:
            details = []
        if not isinstance(details, list):
            details = []
        for raw_item in details:
            if not isinstance(raw_item, dict):
                continue
            row = self.color_table.rowCount()
            self.color_table.insertRow(row)
            values = (
                raw_item.get("index", ""),
                raw_item.get("detector_class", ""),
                raw_item.get("predicted_color", ""),
                _format_number(raw_item.get("diff")),
                _format_number(raw_item.get("threshold")),
                "PASS" if raw_item.get("is_ok") else "FAIL",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 5 and value == "FAIL":
                    item.setForeground(Qt.red)
                self.color_table.setItem(row, column, item)
        self.color_table.resizeColumnsToContents()

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "加入驗收圖片",
            str(self.project_root),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        self._import_paths(Path(path) for path in paths)

    def _add_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "加入圖片資料夾", str(self.project_root))
        if not selected:
            return
        folder = Path(selected)
        paths = (
            path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        self._import_paths(paths)

    def _import_paths(self, paths: Iterable[Path]) -> None:
        if self.repository is None:
            return
        try:
            imported = self.repository.import_images(
                paths,
                product=self.product_combo.currentText(),
                area=self.area_combo.currentText(),
                source_batch=self.batch_edit.text(),
            )
        except (OSError, AcceptanceDataError) as exc:
            QMessageBox.critical(self, "圖片匯入失敗", str(exc))
            return
        self._reload_records(select_id=imported[0].sample_id if imported else "")
        self.statusBar().showMessage(f"已加入或找到 {len(imported)} 張圖片。", 5000)

    def _run_selected(self) -> None:
        record = self._selected_record()
        self._start_inference((record,) if record else ())

    def _run_pending(self) -> None:
        self._start_inference(tuple(record for record in self._records if not record.machine_status))

    def _run_all(self) -> None:
        self._start_inference(self._records)

    def _open_matrix(self) -> None:
        if self.repository is None:
            return
        confirmed = sum(
            record.review_status == "confirmed" for record in self._records
        )
        if confirmed == 0:
            QMessageBox.information(
                self,
                "沒有已確認真值",
                "請先把驗收照片確認為實際 OK 或 NG，再執行組合測試。",
            )
            return
        inference_type = self.type_combo.currentText().strip()
        if not inference_type:
            QMessageBox.warning(
                self,
                "缺少模型",
                "目前產品／區域沒有可用模型。",
            )
            return
        dialog = AcceptanceMatrixDialog(
            project_root=self.project_root,
            repository=self.repository,
            product=self.product_combo.currentText(),
            area=self.area_combo.currentText(),
            inference_type=inference_type,
            parent=self,
        )
        dialog.exec_()

    def _create_snapshot(self) -> None:
        if self.repository is None or not self._records:
            return
        versions = sorted({record.model_version for record in self._records if record.model_version})
        label = f"baseline-v{versions[0]}" if len(versions) == 1 else "baseline-mixed"
        try:
            snapshot = self.repository.create_snapshot(label=label)
        except (OSError, AcceptanceDataError) as exc:
            QMessageBox.critical(self, "建立快照失敗", str(exc))
            return
        QMessageBox.information(
            self,
            "基準快照已建立",
            f"人工真值與目前模型結果已凍結：\n{snapshot.root}",
        )

    def _compare_latest_snapshot(self) -> None:
        if self.repository is None:
            return
        snapshots = self.repository.snapshots()
        if not snapshots:
            QMessageBox.information(self, "沒有基準快照", "請先建立基準快照。")
            return
        comparison = self.repository.compare_with_snapshot(snapshots[-1])
        QMessageBox.information(
            self,
            "模型比較",
            "\n".join(
                (
                    f"基準：{comparison.snapshot_id}",
                    f"模型：{comparison.baseline_version} → {comparison.current_version}",
                    f"共同樣本：{comparison.common_samples}",
                    f"改善：{comparison.improved}",
                    f"退步：{comparison.regressed}",
                    f"誤殺：{comparison.baseline_false_positives} → {comparison.current_false_positives}",
                    f"漏檢：{comparison.baseline_false_negatives} → {comparison.current_false_negatives}",
                    f"機器判定／原因有變化：{len(comparison.changed_sample_ids)}",
                )
            ),
        )

    def _export_backup(self) -> None:
        if self.repository is None or self._backup_worker is not None or not self._records:
            return
        default_name = f"{self.product_combo.currentText()}_{self.area_combo.currentText()}_acceptance_backup.zip"
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "匯出驗收備份",
            str(self.project_root / default_name),
            "ZIP archive (*.zip)",
        )
        if not selected:
            return
        worker = BackupWorker(self.repository, Path(selected))
        worker.completed.connect(self._backup_completed)
        worker.failed.connect(self._backup_failed)
        worker.finished.connect(self._backup_finished)
        self._backup_worker = worker
        self._set_busy(True)
        self.statusBar().showMessage("正在背景建立驗收備份…")
        worker.start()

    def _backup_completed(self, path: str) -> None:
        QMessageBox.information(self, "備份完成", f"驗收備份已建立：\n{path}")

    def _backup_failed(self, message: str) -> None:
        QMessageBox.critical(self, "備份失敗", message)

    def _backup_finished(self) -> None:
        worker = self._backup_worker
        self._backup_worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_busy(False)
        self.statusBar().showMessage("備份作業完成。", 5000)
        if self._close_when_finished:
            self._close_when_finished = False
            self.close()

    def _start_inference(self, records: tuple[AcceptanceRecord, ...]) -> None:
        if not records or self.repository is None or self._worker is not None:
            return
        inference_type = self.type_combo.currentText().strip()
        if not inference_type:
            QMessageBox.warning(self, "缺少模型", "目前產品／區域沒有可用模型。")
            return
        worker = InferenceBatchWorker(
            project_root=self.project_root,
            repository=self.repository,
            records=records,
            inference_type=inference_type,
        )
        worker.outcome_ready.connect(self._inference_ready)
        worker.progress_changed.connect(self._inference_progress)
        worker.failed.connect(self._inference_failed)
        worker.finished.connect(self._inference_finished)
        self._worker = worker
        self.progress.setRange(0, len(records))
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self._set_busy(True)
        worker.start()

    def _inference_ready(self, raw_outcome: object) -> None:
        if not isinstance(raw_outcome, AcceptanceInferenceOutcome):
            return
        if self.repository is None:
            return
        try:
            self.repository.save_inference(raw_outcome)
        except (OSError, AcceptanceDataError) as exc:
            self._inference_failed(str(exc))
            return
        if raw_outcome.annotated_frame is not None:
            self._annotated_pixmaps[raw_outcome.sample_id] = _frame_to_pixmap(raw_outcome.annotated_frame)
        self._reload_records(select_id=raw_outcome.sample_id)

    def _inference_progress(self, current: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(current)
        self.statusBar().showMessage(f"推論進度 {current}/{total}")

    def _inference_failed(self, message: str) -> None:
        QMessageBox.critical(self, "推論失敗", message)

    def _inference_finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self.progress.setVisible(False)
        self._set_busy(False)
        self._reload_records()
        self.statusBar().showMessage("推論完成。", 5000)
        if self._close_when_finished:
            self._close_when_finished = False
            self.close()

    def _set_busy(self, busy: bool) -> None:
        for widget in (
            self.product_combo,
            self.area_combo,
            self.type_combo,
            self.add_files_button,
            self.add_folder_button,
            self.run_selected_button,
            self.run_pending_button,
            self.run_all_button,
            self.matrix_button,
            self.snapshot_button,
            self.compare_button,
            self.backup_button,
            self.confirm_ok_button,
            self.confirm_ng_button,
        ):
            widget.setEnabled(not busy)

    def _confirm_ok(self) -> None:
        self._confirm("OK")

    def _confirm_ng(self) -> None:
        self._confirm("NG")

    def _confirm(self, verdict: str) -> None:
        record = self._selected_record()
        if record is None or self.repository is None:
            return
        reasons = tuple(code for code, checkbox in self._reason_checks.items() if checkbox.isChecked())
        try:
            self.repository.confirm(
                record.sample_id,
                verdict=verdict,
                reasons=reasons,
                reviewed_by=self.reviewer_edit.text(),
                defect_class=self.defect_class_edit.text(),
                notes=self.notes_edit.toPlainText(),
            )
        except (OSError, AcceptanceDataError) as exc:
            QMessageBox.warning(self, "無法確認", str(exc))
            return
        current_row = self.sample_list.currentRow()
        self._reload_records(select_id=record.sample_id)
        if current_row + 1 < self.sample_list.count():
            self.sample_list.setCurrentRow(current_row + 1)

    def _update_summary(self) -> None:
        metrics = calculate_acceptance_metrics(self._records)
        self.summary_label.setText(
            "｜".join(
                (
                    f"總數 {len(self._records)}",
                    f"已確認 {metrics.confirmed}",
                    f"待確認 {metrics.pending}",
                    f"已推論 {metrics.inferred}",
                    f"TP {metrics.tp}",
                    f"誤殺 {metrics.fp}",
                    f"漏檢 {metrics.fn}",
                    f"TN {metrics.tn}",
                    f"真實良率 {_format_rate(metrics.true_yield)}",
                    f"模型判定良率 {_format_rate(metrics.machine_yield)}",
                    f"漏檢率 {_format_rate(metrics.escape_rate)}",
                    f"誤殺率 {_format_rate(metrics.overkill_rate)}",
                )
            )
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._backup_worker is not None and self._backup_worker.isRunning():
            self._close_when_finished = True
            self.statusBar().showMessage("正在完成備份，完成後將關閉。")
            event.ignore()
            return
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._close_when_finished = True
            self.statusBar().showMessage("正在安全停止推論，完成後將關閉。")
            event.ignore()
            return
        event.accept()


def _frame_to_pixmap(frame: np.ndarray) -> QPixmap:
    if frame.ndim == 2:
        contiguous = np.ascontiguousarray(frame)
        image = QImage(
            contiguous.data,
            contiguous.shape[1],
            contiguous.shape[0],
            contiguous.strides[0],
            QImage.Format_Grayscale8,
        )
    else:
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        image = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format_RGB888,
        )
    return QPixmap.fromImage(image.copy())


def _format_rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def _format_number(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "—"


def _record_matches_filter(record: AcceptanceRecord, filter_value: str) -> bool:
    if filter_value == "all":
        return True
    if filter_value == "mismatch":
        return (
            record.review_status == "confirmed"
            and record.machine_status in {"OK", "NG"}
            and record.expected_verdict != record.machine_status
        )
    if filter_value == "false_positive":
        return record.expected_verdict == "OK" and record.machine_status == "NG"
    if filter_value == "false_negative":
        return record.expected_verdict == "NG" and record.machine_status == "OK"
    if filter_value == "color_mismatch":
        return "COLOR_MISMATCH" in record.machine_reasons.split("|")
    if filter_value == "pending":
        return record.review_status != "confirmed"
    if filter_value == "error":
        return record.machine_status == "ERROR"
    return True


def main() -> int:
    """Convenience entrypoint for direct module execution."""
    from core.path_utils import project_root

    app = QApplication.instance() or QApplication([])
    window = ModelAcceptanceWindow(project_root=Path(project_root()))
    window.show()
    return int(app.exec_())
