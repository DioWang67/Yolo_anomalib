"""Engineering dialog for freely composing model and color components."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.services.model_version_registry import ModelVersionRecord
from tools.color_configuration_revisions import ColorConfigurationRevision


class InspectionReleaseComposerDialog(QDialog):
    """Select exact immutable components without requiring prior validation."""

    def __init__(
        self,
        *,
        models: tuple[ModelVersionRecord, ...],
        color_revisions: tuple[ColorConfigurationRevision, ...],
        suggested_version: str,
        preselected_model_version: str = "",
        preselected_color_version: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.models = models
        self.color_revisions = color_revisions
        self.setWindowTitle("建立檢測組合")
        self.setMinimumWidth(620)
        layout = QVBoxLayout(self)
        description = QLabel(
            "可直接選擇任意已登錄模型與顏色版本。建立後狀態為 DRAFT，"
            "不會偽造準確率；可回發布版本畫面選擇有限試跑或風險接受。"
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form = QFormLayout()
        self.model_combo = QComboBox()
        for record in models:
            suffix = "（目前模型指標）" if record.is_current else ""
            label = (
                f"{record.model_type.upper()} {record.version}{suffix} — "
                f"{record.weight_path.name}"
            )
            self.model_combo.addItem(label, record)
        self.color_combo = QComboBox()
        self.color_combo.addItem("使用模型快照內嵌顏色設定", None)
        for revision in color_revisions:
            label = (
                f"{revision.scope.threshold_key} / {revision.display_version} — "
                f"{revision.revision_id[:8]}"
            )
            self.color_combo.addItem(label, revision)
        self.version_edit = QLineEdit(suggested_version)
        self.operator_edit = QLineEdit()
        self.reason_edit = QTextEdit()
        self.reason_edit.setPlaceholderText(
            "例如：比較 YOLO 1.0.5 與目前 color-v1.0.2"
        )
        self.reason_edit.setMaximumHeight(90)
        form.addRow("YOLO／模型版本：", self.model_combo)
        form.addRow("顏色版本：", self.color_combo)
        form.addRow("發布版本：", self.version_edit)
        form.addRow("建立人員：", self.operator_edit)
        form.addRow("建立原因：", self.reason_edit)
        layout.addLayout(form)

        warning = QLabel(
            "注意：這裡只建立可重現的版本組合，不會改動舊模型指標，"
            "也不會立即套用。"
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #9a5b00;")
        layout.addWidget(warning)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.button(QDialogButtonBox.Save).setText("建立 DRAFT")
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._preselect(
            preselected_model_version, preselected_color_version
        )

    def _preselect(
        self, model_version: str, color_version: str
    ) -> None:
        for index in range(self.model_combo.count()):
            record = self.model_combo.itemData(index, Qt.UserRole)
            if (
                isinstance(record, ModelVersionRecord)
                and record.version == model_version
            ):
                self.model_combo.setCurrentIndex(index)
                break
        for index in range(self.color_combo.count()):
            revision = self.color_combo.itemData(index, Qt.UserRole)
            if (
                isinstance(revision, ColorConfigurationRevision)
                and revision.display_version == color_version
            ):
                self.color_combo.setCurrentIndex(index)
                break

    def _validate_and_accept(self) -> None:
        if self.selected_model() is None:
            QMessageBox.warning(self, "建立檢測組合", "沒有可用的模型版本。")
            return
        if not self.display_version().strip():
            QMessageBox.warning(self, "建立檢測組合", "請輸入發布版本。")
            return
        if not self.operator().strip() or not self.reason().strip():
            QMessageBox.warning(
                self, "建立檢測組合", "建立人員與原因不得空白。"
            )
            return
        self.accept()

    def selected_model(self) -> ModelVersionRecord | None:
        value = self.model_combo.currentData(Qt.UserRole)
        return value if isinstance(value, ModelVersionRecord) else None

    def selected_color_revision(self) -> ColorConfigurationRevision | None:
        value = self.color_combo.currentData(Qt.UserRole)
        return value if isinstance(value, ColorConfigurationRevision) else None

    def display_version(self) -> str:
        return self.version_edit.text().strip()

    def operator(self) -> str:
        return self.operator_edit.text().strip()

    def reason(self) -> str:
        return self.reason_edit.toPlainText().strip()
