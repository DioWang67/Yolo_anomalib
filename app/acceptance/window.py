"""PyQt user interface for independent, production-equivalent acceptance."""

from __future__ import annotations

import getpass
import json
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QBrush, QImage, QPixmap, QStandardItemModel
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
from core.services.acceptance_artifacts import (
    AcceptanceArtifactBundle,
    AcceptanceArtifactError,
    build_acceptance_artifact_bundle,
    color_scope_model_type,
    resolve_effective_color_model,
    verify_acceptance_artifact_bundle,
)
from core.services.acceptance_matrix import (
    AcceptanceColorVariant,
    AcceptanceMatrixError,
    build_model_variant,
    discover_color_variants,
)
from core.services.acceptance_runs import (
    AcceptanceRun,
    AcceptanceRunError,
    AcceptanceRunRepository,
)
from core.services.color_revision_contract import (
    capture_candidate_color_revision_contract,
    color_revision_overrides,
)
from core.services.model_acceptance import (
    ACCEPTANCE_REASON_CODES,
    SUPPORTED_IMAGE_SUFFIXES,
    AcceptanceDataError,
    AcceptanceInferenceOutcome,
    AcceptanceInferenceService,
    AcceptanceRecord,
    AcceptanceRepository,
    ModelIdentity,
    calculate_acceptance_metrics,
)
from core.services.model_catalog import ModelCatalog
from core.station_data import load_station_data_paths

SAMPLE_ID_ROLE = Qt.UserRole

#: Annotated frames are display-only and never persisted, so this cache is
#: deliberately bounded rather than complete. A full batch of station images
#: (2048x3072) holds roughly 24 MB per QPixmap at source resolution; keeping
#: every one of a few hundred samples exhausted the graphics heap and killed
#: the process outright, with no Python traceback to show for it. The preview
#: is only ever drawn scaled into a panel, so storing it at source resolution
#: bought nothing.
ANNOTATED_PREVIEW_MAX_EDGE = 1600
ANNOTATED_PREVIEW_CACHE_SIZE = 24

#: Every failure mode that can surface while pinning one run's artifacts. They
#: are all reported to the operator the same way, so they are named once.
RUN_SETUP_ERRORS = (
    AcceptanceArtifactError,
    AcceptanceMatrixError,
    AcceptanceRunError,
    OSError,
    RuntimeError,
    ValueError,
)
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
        self._default_empty_text = empty_text
        self._empty_text = empty_text
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(360, 300)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("background: #161a20; color: #aeb6c2;")

    def set_source(self, pixmap: QPixmap | None, *, empty_text: str = "") -> None:
        """Show ``pixmap``, or ``empty_text`` explaining why there is none."""
        self._source = pixmap
        self._empty_text = empty_text or self._default_empty_text
        self._refresh()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._source is None or self._source.isNull():
            self.setPixmap(QPixmap())
            self.setText(self._empty_text)
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
        models_root: Path,
        repository: AcceptanceRepository,
        records: tuple[AcceptanceRecord, ...],
        inference_type: str,
        artifact_bundle: AcceptanceArtifactBundle,
    ):
        super().__init__()
        self._project_root = project_root
        self._models_root = models_root
        self._repository = repository
        self._records = records
        self._inference_type = inference_type
        self._artifact_bundle = artifact_bundle
        self.failed_message = ""
        self.cancelled = False

    def run(self) -> None:
        service: AcceptanceInferenceService | None = None
        self.failed_message = ""
        self.cancelled = False
        try:
            verify_acceptance_artifact_bundle(
                self._artifact_bundle,
                models_root=self._models_root,
            )
            identity = ModelIdentity(
                version=self._artifact_bundle.version,
                sha256=self._artifact_bundle.model_weight.sha256,
                runtime_config_sha256=self._artifact_bundle.model_config.sha256,
                color_model_sha256=(
                    self._artifact_bundle.color_model.sha256
                    if self._artifact_bundle.color_model is not None
                    else ""
                ),
            )
            service = AcceptanceInferenceService(
                project_root=self._project_root,
                models_root=self._models_root,
                global_config_path=self._artifact_bundle.global_config.path,
                color_model_path_override=(
                    self._artifact_bundle.color_model.path
                    if self._artifact_bundle.color_model is not None
                    and self._artifact_bundle.color_model_mode == "override"
                    else None
                ),
                model_config_overrides={
                    (
                        self._artifact_bundle.product,
                        self._artifact_bundle.area,
                        color_scope_model_type(self._artifact_bundle.inference_type),
                    ): self._artifact_bundle.model_config.path
                },
                model_weight_path_override=self._artifact_bundle.model_weight.path,
                model_identity=identity,
                color_revision_overrides=dict(
                    self._artifact_bundle.color_revision_overrides
                ),
                include_active_color_revisions=(
                    self._artifact_bundle.include_active_color_revisions
                ),
            )
            total = len(self._records)
            for index, record in enumerate(self._records, start=1):
                if self.isInterruptionRequested():
                    self.cancelled = True
                    break
                outcome = service.infer(
                    record,
                    self._repository.verified_image_file(record),
                    inference_type=self._inference_type,
                    cancel_cb=self.isInterruptionRequested,
                )
                if self.isInterruptionRequested():
                    self.cancelled = True
                    break
                if (
                    outcome.model_version,
                    outcome.model_sha256.lower(),
                    outcome.runtime_config_sha256.lower(),
                    outcome.color_model_sha256.lower(),
                ) != (
                    identity.version,
                    identity.sha256.lower(),
                    identity.runtime_config_sha256.lower(),
                    identity.color_model_sha256.lower(),
                ):
                    raise AcceptanceDataError(
                        "Inference identity does not match the pinned artifact bundle."
                    )
                self.outcome_ready.emit(outcome)
                self.progress_changed.emit(index, total)
            if not self.cancelled:
                verify_acceptance_artifact_bundle(
                    self._artifact_bundle,
                    models_root=self._models_root,
                )
        except (
            AcceptanceArtifactError,
            ImportError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            self.failed_message = str(exc)
            self.failed.emit(self.failed_message)
        finally:
            if service is not None:
                try:
                    service.close()
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    if not self.failed_message and not self.cancelled:
                        self.failed_message = f"Inference cleanup failed: {exc}"
                        self.failed.emit(self.failed_message)


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
        self.data_paths = load_station_data_paths(self.project_root)
        self.catalog = ModelCatalog(self.data_paths.models)
        self.repository: AcceptanceRepository | None = None
        self._run_repository: AcceptanceRunRepository | None = None
        self._active_run: AcceptanceRun | None = None
        self._active_bundle: AcceptanceArtifactBundle | None = None
        self._pending_outcomes: dict[str, AcceptanceInferenceOutcome] = {}
        self._inference_failure_message = ""
        self._records: tuple[AcceptanceRecord, ...] = ()
        self._visible_records: tuple[AcceptanceRecord, ...] = ()
        self._annotated_pixmaps: OrderedDict[str, QPixmap] = OrderedDict()
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
        self.color_model_combo = QComboBox()
        self.color_model_combo.setToolTip(
            "選擇本次推論要使用的顏色模型。預設沿用工位目前正式生效的顏色設定。"
        )
        self.batch_edit = QLineEdit()
        self.batch_edit.setPlaceholderText("例如 LOT-20260730")
        self.reviewer_edit = QLineEdit(getpass.getuser())
        for label, widget in (
            ("產品", self.product_combo),
            ("區域", self.area_combo),
            ("模型", self.type_combo),
            ("顏色模型", self.color_model_combo),
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
        self._reload_color_models()
        if not product or not area:
            self.repository = None
            self._run_repository = None
            self._records = ()
            self._render_records()
            return
        try:
            self.repository = AcceptanceRepository(
                self.data_paths.acceptance / product / area
            )
            self._run_repository = AcceptanceRunRepository(self.repository.root)
            self._reload_records()
        except (OSError, AcceptanceDataError) as exc:
            QMessageBox.critical(self, "驗收資料錯誤", str(exc))

    def _reload_color_models(self) -> None:
        """List the color models this scope can be inferred with.

        The first entry keeps the previous behavior -- whatever the station has
        active -- so opening the tool and pressing 推論 does the same thing it
        always did. Anything else pins one stored color model for this run only;
        nothing here activates or edits a color version.
        """
        self.color_model_combo.blockSignals(True)
        self.color_model_combo.clear()
        self.color_model_combo.addItem("目前正式設定（不覆寫）", None)
        product = self.product_combo.currentText().strip()
        area = self.area_combo.currentText().strip()
        inference_type = self.type_combo.currentText().strip()
        if not product or not area or not inference_type:
            self.color_model_combo.blockSignals(False)
            return
        try:
            discovery = discover_color_variants(
                self.data_paths.color_revisions,
                product=product,
                area=area,
                model_type=color_scope_model_type(inference_type),
                baselines_root=self.data_paths.color_baselines,
                profiles_root=self.data_paths.color_profiles,
            )
        except (AcceptanceMatrixError, OSError, RuntimeError, ValueError) as exc:
            # A scope whose color store cannot be read must not silently look
            # like a scope that simply has no color model.
            self.color_model_combo.addItem(f"（無法讀取顏色模型：{exc}）", None)
            self._disable_last_color_item()
            self.color_model_combo.blockSignals(False)
            return
        for variant in discovery.stored_color_models:
            self.color_model_combo.addItem(variant.label, variant)
        for exclusion in discovery.exclusions:
            self.color_model_combo.addItem(
                f"{exclusion.label}（無法使用：{exclusion.reason}）", None
            )
            self._disable_last_color_item()
        self.color_model_combo.blockSignals(False)

    def _disable_last_color_item(self) -> None:
        """Make the item just added visible but unselectable.

        Only a standard item model exposes per-item enabling. If Qt ever hands
        back another model the entry stays selectable, which is safe rather than
        merely tolerable: an excluded entry carries no variant, so selecting it
        resolves to the same no-override run as the default entry.
        """
        model = self.color_model_combo.model()
        if not isinstance(model, QStandardItemModel):
            return
        item = model.item(self.color_model_combo.count() - 1)
        if item is not None:
            item.setEnabled(False)

    def _selected_color_variant(self) -> AcceptanceColorVariant | None:
        variant = self.color_model_combo.currentData()
        return variant if isinstance(variant, AcceptanceColorVariant) else None

    def _artifact_bundle_for_run(
        self,
        inference_type: str,
    ) -> AcceptanceArtifactBundle:
        product = self.product_combo.currentText().strip()
        area = self.area_combo.currentText().strip()
        model_variant = build_model_variant(
            self.data_paths.models,
            product=product,
            area=area,
            inference_type=inference_type,
        )
        if model_variant.config_path is None or model_variant.weight_path is None:
            raise AcceptanceMatrixError(
                "目前模型組合缺少 config 或 weight，無法建立固定驗收組合。"
            )
        global_config_path = self.project_root / "config.yaml"
        selected_color = self._selected_color_variant()
        selected_color_path = (
            selected_color.color_model_path
            if selected_color is not None
            else None
        )
        if selected_color is not None:
            revision_overrides = selected_color.override_mapping()
            include_active_revisions = selected_color.include_active_revisions
            color_contract: Mapping[str, object] = {}
        else:
            effective_color = resolve_effective_color_model(
                model_config_path=model_variant.config_path,
                global_config_path=global_config_path,
                models_root=self.data_paths.models,
            )
            color_contract = capture_candidate_color_revision_contract(
                revisions_root=self.data_paths.color_revisions,
                candidate_config_path=model_variant.config_path,
                global_config_path=global_config_path,
                color_model_present=effective_color is not None,
                product=product,
                area=area,
                inference_type=color_scope_model_type(inference_type),
            )
            revision_overrides = color_revision_overrides(color_contract)
            # Resolve Active once, then use only its exact immutable revisions.
            include_active_revisions = False
        return build_acceptance_artifact_bundle(
            product=product,
            area=area,
            inference_type=inference_type,
            version=model_variant.identity.version,
            global_config_path=global_config_path,
            model_config_path=model_variant.config_path,
            models_root=self.data_paths.models,
            model_weight_path=model_variant.weight_path,
            color_model_path=selected_color_path,
            color_model_is_override=selected_color_path is not None,
            color_revision_overrides=revision_overrides,
            include_active_color_revisions=include_active_revisions,
            color_revision_contract=color_contract,
        )

    def _reload_records(self, *, select_id: str = "") -> None:
        if self.repository is None:
            return
        self._records = self.repository.records()
        self._render_records(select_id=select_id)
        self._update_summary()

    def _active_filter(self) -> str:
        return (
            str(self.filter_combo.currentData() or "all")
            if hasattr(self, "filter_combo")
            else "all"
        )

    def _render_records(self, *, select_id: str = "") -> None:
        previous_id = select_id or self._selected_sample_id()
        filter_value = self._active_filter()
        self._visible_records = tuple(
            record for record in self._records if _record_matches_filter(record, filter_value)
        )
        self.sample_list.blockSignals(True)
        self.sample_list.clear()
        selected_row = -1
        for index, record in enumerate(self._visible_records):
            item = QListWidgetItem()
            item.setData(SAMPLE_ID_ROLE, record.sample_id)
            _paint_sample_item(item, record)
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

    def _refresh_record_in_place(self, record: AcceptanceRecord) -> bool:
        """Repaint one row instead of rebuilding the whole list.

        A batch emits one outcome per sample and each one used to clear and
        refill the entire list widget, so the cost of watching a run grew with
        the square of its size -- a few hundred station images meant tens of
        thousands of discarded rows. An outcome can only change its own row and
        never the order, because the visible order follows the manifest, so a
        rebuild is needed only when the new verdict moves the record in or out
        of the active filter.

        Returns:
            ``False`` when the caller must fall back to a full rebuild.
        """
        was_visible = any(
            visible.sample_id == record.sample_id
            for visible in self._visible_records
        )
        if _record_matches_filter(record, self._active_filter()) != was_visible:
            return False
        self._visible_records = tuple(
            record if visible.sample_id == record.sample_id else visible
            for visible in self._visible_records
        )
        if not was_visible:
            return True
        row = next(
            (
                index
                for index, visible in enumerate(self._visible_records)
                if visible.sample_id == record.sample_id
            ),
            -1,
        )
        item = self.sample_list.item(row)
        if item is None:
            return False
        _paint_sample_item(item, record)
        # The full rebuild always re-selected the sample, which is what drives
        # the preview panel. Re-selecting an already-current row emits nothing,
        # so that case is refreshed explicitly rather than left stale.
        if self.sample_list.currentRow() == row:
            self._sample_changed(item, None)
        else:
            self.sample_list.setCurrentRow(row)
        return True

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
        preview = self._annotated_preview(record.sample_id)
        # A record can be inferred yet have no preview, because previews are
        # bounded and never persisted. Saying "尚未執行推論" there would report a
        # verdict that does exist as one that was never produced.
        self.prediction_image.set_source(
            preview,
            empty_text=(
                "標註圖預覽已釋出，重新推論此張即可再次檢視"
                if preview is None and record.machine_status
                else ""
            ),
        )
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
        self._start_inference(
            tuple(
                record
                for record in self._records
                if not record.machine_status or record.machine_status == "ERROR"
            )
        )

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
            snapshot = self.repository.create_snapshot(
                label=label,
                require_completed_run=True,
            )
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
            str(self.data_paths.acceptance / default_name),
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

    def _confirm_discarded_results(
        self,
        records: tuple[AcceptanceRecord, ...],
        artifact_bundle: AcceptanceArtifactBundle,
    ) -> bool:
        """Ask before a partial run discards results from another combination.

        A snapshot may only mix results produced by one artifact combination,
        so starting a run under a different bundle necessarily invalidates the
        earlier ones. That is the operator's decision to make, not something
        they should have to infer from a table that quietly emptied itself.

        Re-running samples under the *same* bundle discards nothing, so the
        common case of retrying a few errored images asks nothing at all.
        """
        selected = {record.sample_id for record in records}
        discarded = sum(
            1
            for record in self._records
            if record.sample_id not in selected
            and record.machine_status
            and record.artifact_bundle_sha256 != artifact_bundle.bundle_sha256
        )
        if not discarded:
            return True
        answer = QMessageBox.question(
            self,
            "將清除既有推論結果",
            f"本次只推論 {len(records)} 張，但另有 {discarded} 張的既有結果"
            "來自不同的模型／顏色組合，提交時會被清除（人工真值不受影響）。"
            "\n\n要繼續嗎？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _start_inference(self, records: tuple[AcceptanceRecord, ...]) -> None:
        if (
            not records
            or self.repository is None
            or self._run_repository is None
            or self._worker is not None
        ):
            return
        inference_type = self.type_combo.currentText().strip()
        if not inference_type:
            QMessageBox.warning(self, "缺少模型", "目前產品／區域沒有可用模型。")
            return
        try:
            artifact_bundle = self._artifact_bundle_for_run(inference_type)
        except RUN_SETUP_ERRORS as exc:
            QMessageBox.critical(self, "無法建立固定驗收組合", str(exc))
            return
        # Asked before the run is opened, so declining leaves no run behind.
        if not self._confirm_discarded_results(records, artifact_bundle):
            return
        try:
            source_manifest_sha256 = self.repository.manifest_sha256()
            active_run = self._run_repository.begin(
                artifact_bundle=artifact_bundle,
                sample_ids=tuple(record.sample_id for record in records),
                source_manifest_sha256=source_manifest_sha256,
            )
        except RUN_SETUP_ERRORS as exc:
            QMessageBox.critical(self, "無法建立固定驗收組合", str(exc))
            return
        worker = InferenceBatchWorker(
            project_root=self.project_root,
            models_root=self.data_paths.models,
            repository=self.repository,
            records=records,
            inference_type=inference_type,
            artifact_bundle=artifact_bundle,
        )
        worker.outcome_ready.connect(self._inference_ready)
        worker.progress_changed.connect(self._inference_progress)
        worker.failed.connect(self._inference_failed)
        worker.finished.connect(self._inference_finished)
        self._active_run = active_run
        self._active_bundle = artifact_bundle
        self._pending_outcomes = {}
        self._inference_failure_message = ""
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
            if self._active_run is None or self._run_repository is None:
                raise AcceptanceRunError("No active acceptance run owns this outcome.")
            self._run_repository.append_outcome(self._active_run, raw_outcome)
            self._pending_outcomes[raw_outcome.sample_id] = raw_outcome
        except (OSError, AcceptanceDataError, AcceptanceRunError) as exc:
            if self._worker is not None:
                self._worker.requestInterruption()
            self._inference_failed(str(exc))
            return
        if raw_outcome.annotated_frame is not None:
            self._cache_annotated_preview(
                raw_outcome.sample_id, raw_outcome.annotated_frame
            )
        updated = next(
            (
                _record_with_outcome(record, raw_outcome)
                for record in self._records
                if record.sample_id == raw_outcome.sample_id
            ),
            None,
        )
        if updated is None:
            return
        self._records = tuple(
            updated if record.sample_id == updated.sample_id else record
            for record in self._records
        )
        if not self._refresh_record_in_place(updated):
            self._render_records(select_id=updated.sample_id)

    def _cache_annotated_preview(self, sample_id: str, frame: np.ndarray) -> None:
        """Store one display-sized preview, dropping the least recently viewed.

        Both halves matter: downscaling keeps a single entry small, and the cap
        keeps a long batch from growing without limit. Either alone still ends
        in the graphics heap being exhausted on a large enough run.
        """
        pixmap = _frame_to_pixmap(frame)
        if max(pixmap.width(), pixmap.height()) > ANNOTATED_PREVIEW_MAX_EDGE:
            pixmap = pixmap.scaled(
                ANNOTATED_PREVIEW_MAX_EDGE,
                ANNOTATED_PREVIEW_MAX_EDGE,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        self._annotated_pixmaps.pop(sample_id, None)
        self._annotated_pixmaps[sample_id] = pixmap
        while len(self._annotated_pixmaps) > ANNOTATED_PREVIEW_CACHE_SIZE:
            self._annotated_pixmaps.popitem(last=False)

    def _annotated_preview(self, sample_id: str) -> QPixmap | None:
        """Return a cached preview, counting this read as a use."""
        pixmap = self._annotated_pixmaps.get(sample_id)
        if pixmap is not None:
            self._annotated_pixmaps.move_to_end(sample_id)
        return pixmap

    def _inference_progress(self, current: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(current)
        self.statusBar().showMessage(f"推論進度 {current}/{total}")

    def _inference_failed(self, message: str) -> None:
        self._inference_failure_message = message
        QMessageBox.critical(self, "推論失敗", message)

    def _inference_finished(self) -> None:
        worker = self._worker
        run = self._active_run
        artifact_bundle = self._active_bundle
        run_repository = self._run_repository
        repository = self.repository
        final_message = "推論失敗。"
        try:
            failure_message = self._inference_failure_message or str(
                getattr(worker, "failed_message", "") or ""
            )
            cancelled = bool(getattr(worker, "cancelled", False))
            if (
                run is None
                or artifact_bundle is None
                or run_repository is None
                or repository is None
            ):
                final_message = "推論工作缺少 run context，結果未提交。"
            elif failure_message:
                run_repository.fail(run, reason=failure_message)
                final_message = "推論失敗，原正式結果未變更。"
            elif cancelled or len(self._pending_outcomes) != len(run.sample_ids):
                run_repository.cancel(run)
                final_message = "推論已取消，原正式結果未變更。"
            else:
                _committed, committed_sha256 = repository.save_inference_batch(
                    tuple(
                        self._pending_outcomes[sample_id]
                        for sample_id in run.sample_ids
                    ),
                    run_id=run.run_id,
                    artifact_bundle=artifact_bundle,
                    expected_manifest_sha256=run.source_manifest_sha256,
                )
                run_repository.complete(
                    run,
                    committed_manifest_sha256=committed_sha256,
                )
                final_message = f"推論完成並原子提交：{run.run_id}"
        except (OSError, AcceptanceDataError, AcceptanceRunError) as exc:
            final_message = f"推論結果提交失敗：{exc}"
            if run is not None and run_repository is not None:
                try:
                    run_repository.fail(run, reason=str(exc))
                except AcceptanceRunError:
                    pass
            QMessageBox.critical(self, "推論結果未提交", str(exc))
        self._worker = None
        self._active_run = None
        self._active_bundle = None
        self._pending_outcomes = {}
        self._inference_failure_message = ""
        if worker is not None:
            worker.deleteLater()
        self.progress.setVisible(False)
        self._set_busy(False)
        self._reload_records()
        self.statusBar().showMessage(final_message, 8000)
        if self._close_when_finished:
            self._close_when_finished = False
            self.close()

    def _set_busy(self, busy: bool) -> None:
        for widget in (
            self.product_combo,
            self.area_combo,
            self.type_combo,
            self.color_model_combo,
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
        parts = [
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
        ]
        if metrics.malformed:
            # Shown only when present, and never folded into 待確認: these rows
            # claim to be reviewed but carry no verdict, so they block a formal
            # snapshot and the operator has to repair them by hand.
            parts.append(f"真值異常 {metrics.malformed}（無法建立正式快照）")
        self.summary_label.setText("｜".join(parts))

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


def _paint_sample_item(item: QListWidgetItem, record: AcceptanceRecord) -> None:
    """Write one record's label and status colour onto its row.

    Shared by the full rebuild and the single-row refresh so the two paths
    cannot drift into showing the same record differently.
    """
    truth = record.expected_verdict or "待覆核"
    machine = record.machine_status or "未推論"
    item.setText(f"{record.sample_id}\n人工 {truth}｜模型 {machine}")
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
    else:
        # Reset explicitly: a reused row keeps the brush from its previous
        # verdict, so an ERROR that reruns to OK would stay dark red.
        item.setForeground(QBrush())


def _record_with_outcome(
    record: AcceptanceRecord,
    outcome: AcceptanceInferenceOutcome,
) -> AcceptanceRecord:
    return record.with_changes(
        machine_status=outcome.machine_status,
        machine_reasons="|".join(outcome.machine_reasons),
        model_version=outcome.model_version,
        model_sha256=outcome.model_sha256,
        runtime_config_sha256=outcome.runtime_config_sha256,
        color_model_sha256=outcome.color_model_sha256,
        inference_at=outcome.inference_at,
        latency_ms=f"{outcome.latency_ms:.3f}",
        error=outcome.error,
        color_check_status=outcome.color_check_status,
        color_details_json=outcome.color_details_json,
    )


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
