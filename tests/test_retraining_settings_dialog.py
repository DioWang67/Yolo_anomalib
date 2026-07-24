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
