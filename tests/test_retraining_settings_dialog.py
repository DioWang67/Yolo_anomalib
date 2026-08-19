from PyQt5.QtCore import QSettings, Qt

from app.gui.retraining_settings_dialog import RetrainingSettingsDialog
from core.retraining_options import RetrainingOptions


def test_retraining_settings_dialog_exposes_job_options(qtbot):
    dialog = RetrainingSettingsDialog(
        10,
        initial=RetrainingOptions(
            epochs=60,
            augmentations_per_image=5,
            batch=4,
            imgsz=960,
        ),
    )
    qtbot.addWidget(dialog)

    assert dialog.options() == RetrainingOptions(
        epochs=60,
        augmentations_per_image=5,
        batch=4,
        imgsz=960,
    )
    assert "單計本批最多形成約 60 張" in dialog.summary_label.text()


def test_retraining_settings_dialog_identifies_target_scoped_batch(qtbot):
    dialog = RetrainingSettingsDialog(
        10,
        product="Cable1",
        area="A",
        suggested_batch_version="Cable1_A_v0.0.2",
    )
    qtbot.addWidget(dialog)

    assert dialog.batch_version() == "Cable1_A_v0.0.2"
    assert "補訓批次：Cable1_A_v0.0.2" in dialog.summary_label.text()


def test_retraining_settings_dialog_does_not_rename_precreated_folder(qtbot):
    dialog = RetrainingSettingsDialog(
        10,
        product="Cable1",
        area="A",
        batch_version="Cable1_A_v0.0.2",
    )
    qtbot.addWidget(dialog)

    assert dialog.batch_version_edit.isReadOnly() is True
    assert dialog.batch_version() == "Cable1_A_v0.0.2"


def test_retraining_settings_dialog_blocks_version_for_another_target(
    qtbot,
    monkeypatch,
):
    warnings = []
    monkeypatch.setattr(
        "app.gui.retraining_settings_dialog.QMessageBox.warning",
        lambda *_args: warnings.append(_args[-1]),
    )
    dialog = RetrainingSettingsDialog(1, product="Cable1", area="A")
    qtbot.addWidget(dialog)
    dialog.batch_version_edit.setText("Cable1_B_v0.0.1")

    dialog.accept()

    assert warnings
    assert dialog.result() != dialog.Accepted


def test_retraining_settings_dialog_supports_originals_only(qtbot):
    dialog = RetrainingSettingsDialog(3, initial=RetrainingOptions())
    qtbot.addWidget(dialog)

    dialog.augmentation_spin.setValue(0)

    assert dialog.options().augmentations_per_image == 0
    assert "單計本批最多形成約 3 張" in dialog.summary_label.text()


def test_retraining_settings_dialog_shows_job_controls_by_default(qtbot):
    dialog = RetrainingSettingsDialog(3, initial=RetrainingOptions())
    qtbot.addWidget(dialog)
    dialog.show()

    assert dialog.advanced_toggle.isChecked() is True
    assert dialog.settings_card.isHidden() is False

    qtbot.mouseClick(dialog.advanced_toggle, Qt.LeftButton)

    assert dialog.settings_card.isHidden() is True


def test_retraining_settings_dialog_can_restore_recommended_options(qtbot):
    dialog = RetrainingSettingsDialog(
        3,
        initial=RetrainingOptions(
            epochs=80,
            augmentations_per_image=7,
            batch=4,
            imgsz=960,
        ),
    )
    qtbot.addWidget(dialog)

    dialog._restore_recommended_options()

    assert dialog.options() == RetrainingOptions()


def test_retraining_settings_dialog_persists_integer_options(
    tmp_path, qtbot, monkeypatch
):
    settings = QSettings(str(tmp_path / "retraining.ini"), QSettings.IniFormat)
    monkeypatch.setattr(
        RetrainingSettingsDialog,
        "_settings",
        staticmethod(lambda: settings),
    )
    dialog = RetrainingSettingsDialog(
        2,
        initial=RetrainingOptions(
            epochs=80,
            augmentations_per_image=7,
            batch=4,
            imgsz=960,
        ),
    )
    qtbot.addWidget(dialog)

    dialog.accept()

    assert RetrainingSettingsDialog.load_saved_options() == RetrainingOptions(
        epochs=80,
        augmentations_per_image=7,
        batch=4,
        imgsz=960,
    )


def test_position_training_requires_explicit_checkbox_opt_in(qtbot):
    dialog = RetrainingSettingsDialog(3, initial=RetrainingOptions())
    qtbot.addWidget(dialog)
    dialog.show()

    assert dialog.position_training_checkbox.isChecked() is False
    assert dialog.position_activation_checkbox.isEnabled() is False
    assert dialog.options().position_training_mode == "yolo_only"

    dialog.position_training_checkbox.click()
    dialog.position_activation_checkbox.click()

    assert dialog.options().position_training_mode == "calibrate_validate"
    assert dialog.options().position_activation == "enable_after_gate"
    assert "位置檢測補訓：已啟用" in dialog.summary_label.text()

    dialog.position_training_checkbox.click()

    assert dialog.options().position_training_mode == "yolo_only"
    assert dialog.options().position_activation == "preserve"
    assert dialog.position_activation_checkbox.isEnabled() is False


def test_position_activation_is_not_persisted_between_jobs(
    tmp_path, qtbot, monkeypatch
):
    settings = QSettings(str(tmp_path / "retraining.ini"), QSettings.IniFormat)
    monkeypatch.setattr(
        RetrainingSettingsDialog,
        "_settings",
        staticmethod(lambda: settings),
    )
    dialog = RetrainingSettingsDialog(
        2,
        initial=RetrainingOptions(
            position_training_mode="calibrate_validate",
            position_activation="enable_after_gate",
        ),
    )
    qtbot.addWidget(dialog)

    dialog.accept()

    loaded = RetrainingSettingsDialog.load_saved_options()
    assert loaded.position_training_mode == "yolo_only"
    assert loaded.position_activation == "preserve"
