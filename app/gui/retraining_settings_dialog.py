"""Operator settings dialog shown before a retraining job is exported."""

from __future__ import annotations

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.gui.dialog_geometry import configure_responsive_dialog
from core.retraining_options import (
    MAX_AUGMENTATIONS_PER_IMAGE,
    MAX_BATCH_SIZE,
    MAX_EPOCHS,
    MIN_AUGMENTATIONS_PER_IMAGE,
    MIN_BATCH_SIZE,
    MIN_EPOCHS,
    RetrainingOptions,
)


class RetrainingSettingsDialog(QDialog):
    """Collect validated, job-scoped training options from an operator."""

    SETTINGS_GROUP = "operator_retraining"

    def __init__(
        self,
        source_image_count: int,
        *,
        initial: RetrainingOptions | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.source_image_count = max(int(source_image_count), 0)
        self.setWindowTitle("補訓設定")
        self.setModal(True)
        configure_responsive_dialog(
            self,
            preferred=(600, 620),
            minimum=(420, 420),
            parent=parent,
        )
        selected = initial or self.load_saved_options()

        title = QLabel("開始前確認訓練參數")
        title.setStyleSheet("font-size: 16pt; font-weight: 700; color: #1f3550;")
        description = QLabel(
            "系統已準備好安全的建議值，通常直接確認即可。完成後會先比較新舊模型，"
            "只有通過品質檢查才會部署。"
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #5b6673;")

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(MIN_EPOCHS, MAX_EPOCHS)
        self.epochs_spin.setValue(selected.epochs)
        self.epochs_spin.setSuffix(" 次")

        self.augmentation_spin = QSpinBox()
        self.augmentation_spin.setRange(
            MIN_AUGMENTATIONS_PER_IMAGE, MAX_AUGMENTATIONS_PER_IMAGE
        )
        self.augmentation_spin.setValue(selected.augmentations_per_image)
        self.augmentation_spin.setSuffix(" 張／原圖")
        self.augmentation_spin.setSpecialValueText("0（只使用原圖）")

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(MIN_BATCH_SIZE, MAX_BATCH_SIZE)
        self.batch_spin.setValue(selected.batch)

        self.imgsz_combo = QComboBox()
        for size in (320, 416, 512, 640, 768, 960, 1024, 1280):
            self.imgsz_combo.addItem(f"{size} × {size}", size)
        selected_index = self.imgsz_combo.findData(selected.imgsz)
        self.imgsz_combo.setCurrentIndex(max(selected_index, 0))

        form = QFormLayout()
        form.setSpacing(12)
        form.addRow("訓練輪數（Epochs）", self.epochs_spin)
        form.addRow("影像增強張數", self.augmentation_spin)
        form.addRow("Batch size", self.batch_spin)
        form.addRow("訓練圖片尺寸", self.imgsz_combo)

        self.settings_card = QFrame()
        self.settings_card.setFrameShape(QFrame.StyledPanel)
        self.settings_card.setStyleSheet(
            "QFrame { background: #f7f9fb; border: 1px solid #d9e0e7; "
            "border-radius: 8px; padding: 10px; }"
        )
        self.settings_card.setLayout(form)

        reset_button = QPushButton("恢復系統建議值")
        reset_button.setToolTip("恢復 20 Epochs、每張增強 20 張、Batch 8、640×640。")
        reset_button.clicked.connect(self._restore_recommended_options)
        form.addRow("", reset_button)

        self.advanced_toggle = QPushButton()
        self.advanced_toggle.setCheckable(True)
        # Epochs and augmentation count directly change the resulting job.  Keep
        # them visible when the dialog opens so an operator cannot unknowingly
        # accept previously persisted values.
        self.advanced_toggle.setChecked(True)
        self.advanced_toggle.setStyleSheet(
            "QPushButton { text-align:left; background:#f5f7fa; color:#344054; "
            "border:1px solid #d8dee6; border-radius:6px; padding:9px 12px; "
            "font-weight:bold; } QPushButton:checked { background:#eef6ff; "
            "border-color:#8bb8e8; }"
        )
        self.advanced_toggle.toggled.connect(self._toggle_advanced_settings)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(
            "background: #eaf5ee; color: #245c36; padding: 10px; border-radius: 6px;"
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.button(QDialogButtonBox.Ok).setText("使用此設定開始補訓")
        buttons.button(QDialogButtonBox.Ok).setMinimumHeight(40)
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.advanced_toggle)
        layout.addWidget(self.settings_card)
        button_row = QHBoxLayout()
        button_row.addStretch()
        button_row.addWidget(buttons)
        layout.addLayout(button_row)

        self.epochs_spin.valueChanged.connect(self._update_summary)
        self.augmentation_spin.valueChanged.connect(self._update_summary)
        self.batch_spin.valueChanged.connect(self._update_summary)
        self.imgsz_combo.currentIndexChanged.connect(self._update_summary)
        self._toggle_advanced_settings(True)
        self._update_summary()

    def options(self) -> RetrainingOptions:
        return RetrainingOptions(
            epochs=self.epochs_spin.value(),
            augmentations_per_image=self.augmentation_spin.value(),
            batch=self.batch_spin.value(),
            imgsz=int(self.imgsz_combo.currentData()),
        )

    def accept(self) -> None:
        options = self.options()
        settings = self._settings()
        settings.beginGroup(self.SETTINGS_GROUP)
        for key, value in options.to_dict().items():
            settings.setValue(key, value)
        settings.endGroup()
        settings.sync()
        super().accept()

    @classmethod
    def load_saved_options(cls) -> RetrainingOptions:
        settings = cls._settings()
        settings.beginGroup(cls.SETTINGS_GROUP)
        values = {
            key: settings.value(key, type=int)
            for key in ("epochs", "augmentations_per_image", "batch", "imgsz")
            if settings.contains(key)
        }
        settings.endGroup()
        try:
            return RetrainingOptions.from_mapping(values)
        except ValueError:
            return RetrainingOptions()

    @staticmethod
    def _settings() -> QSettings:
        return QSettings("RobotLearning", "Yolo11Inference")

    def _update_summary(self) -> None:
        options = self.options()
        maximum = options.estimated_maximum_images(self.source_image_count)
        self.summary_label.setText(
            f"準備建立補訓工作｜本次 {self.source_image_count} 張新原圖\n"
            f"單計本批最多形成約 {maximum} 張；歷史樣本也會一併納入。"
        )
        marker = "收合" if self.advanced_toggle.isChecked() else "展開"
        self.advanced_toggle.setText(
            f"{marker}進階設定　{options.epochs} Epochs／增強 {options.augmentations_per_image}／"
            f"Batch {options.batch}／{options.imgsz}px"
        )

    def _toggle_advanced_settings(self, expanded: bool) -> None:
        """Keep technical controls out of the operator's primary path."""
        self.settings_card.setVisible(expanded)
        self._update_summary()

    def _restore_recommended_options(self) -> None:
        """Restore the validated defaults shared with the training pipeline."""
        recommended = RetrainingOptions()
        self.epochs_spin.setValue(recommended.epochs)
        self.augmentation_spin.setValue(recommended.augmentations_per_image)
        self.batch_spin.setValue(recommended.batch)
        selected_index = self.imgsz_combo.findData(recommended.imgsz)
        self.imgsz_combo.setCurrentIndex(max(selected_index, 0))
