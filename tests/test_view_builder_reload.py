from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from app.gui.view_builder import (
    _open_model_update_status,
    _open_model_versions,
    _open_training_review,
    _reload_models,
)


def _combo(value: str) -> SimpleNamespace:
    return SimpleNamespace(currentText=Mock(return_value=value))


def test_reload_models_invalidates_runtime_and_refreshes_catalog() -> None:
    """Reloading must update both the live engine and model catalog."""
    gui = SimpleNamespace(
        is_detection_running=Mock(return_value=False),
        product_combo=_combo("Cable1"),
        area_combo=_combo("A"),
        inference_combo=_combo("extreme"),
        controller=SimpleNamespace(reload_model_settings=Mock()),
        load_available_models=Mock(),
        log_message=Mock(),
    )

    _reload_models(gui)

    gui.controller.reload_model_settings.assert_called_once_with(
        "Cable1", "A", "extreme"
    )
    gui.load_available_models.assert_called_once_with()


def test_reload_models_refuses_while_detection_is_running(monkeypatch) -> None:
    """A running inspection must not swap its model underneath a frame."""
    warning = Mock()
    monkeypatch.setattr("app.gui.view_builder.QMessageBox.warning", warning)
    gui = SimpleNamespace(
        is_detection_running=Mock(return_value=True),
        controller=SimpleNamespace(reload_model_settings=Mock()),
        log_message=Mock(),
    )

    _reload_models(gui)

    gui.controller.reload_model_settings.assert_not_called()
    gui.log_message.assert_called_once()
    warning.assert_called_once()


def test_training_review_opens_project_scoped_dialog(tmp_path, monkeypatch) -> None:
    run_dialog = Mock()
    monkeypatch.setattr("app.gui.view_builder._run_review_dialog", run_dialog)
    gui = SimpleNamespace(
        is_detection_running=Mock(return_value=False),
        _project_root=tmp_path / "yolo11_inference",
        current_language="zh_TW",
        product_combo=_combo("Cable1"),
        area_combo=_combo("A"),
        log_message=Mock(),
    )

    _open_training_review(gui)

    run_dialog.assert_called_once_with(
        result_root=gui._project_root / "Result",
        manifest_path=gui._project_root / "review_manifest.csv",
        training_data_dir=tmp_path / "Yolo11_auto_train" / "data",
        language="zh_TW",
        product="Cable1",
        area="A",
        parent=gui,
    )


def test_training_review_refuses_while_detection_is_running(monkeypatch) -> None:
    warning = Mock()
    monkeypatch.setattr("app.gui.view_builder.QMessageBox.warning", warning)
    gui = SimpleNamespace(
        is_detection_running=Mock(return_value=True),
        current_language="zh_TW",
        log_message=Mock(),
    )

    _open_training_review(gui)

    gui.log_message.assert_called_once()
    warning.assert_called_once()


def test_model_versions_dialog_is_scoped_to_inference_models(
    tmp_path, monkeypatch
) -> None:
    run_dialog = Mock()
    monkeypatch.setattr(
        "app.gui.view_builder._run_model_versions_dialog", run_dialog
    )
    models_root = tmp_path / "models"
    models_root.mkdir()
    gui = SimpleNamespace(
        _models_base=models_root,
        current_language="zh_TW",
        product_combo=_combo("PCBA1"),
        area_combo=_combo("A"),
        inference_combo=_combo("yolo"),
        is_detection_running=Mock(return_value=False),
        controller=SimpleNamespace(reload_model_settings=Mock()),
        _catalog=SimpleNamespace(refresh=Mock()),
        load_available_models=Mock(),
        log_message=Mock(),
    )

    _open_model_versions(gui)

    run_dialog.assert_called_once()
    kwargs = run_dialog.call_args.kwargs
    assert kwargs["registry"].models_root == models_root.resolve()
    assert kwargs["selected_product"] == "PCBA1"
    assert kwargs["selected_area"] == "A"
    assert kwargs["selected_model_type"] == "yolo"


def test_model_activation_callback_clears_runtime_cache(tmp_path, monkeypatch) -> None:
    captured = {}

    def run_dialog(**kwargs):
        captured.update(kwargs)
        kwargs["on_activated"](
            SimpleNamespace(
                product="PCBA1", area="A", model_type="yolo", version="1.2.3"
            )
        )
        return 1

    monkeypatch.setattr(
        "app.gui.view_builder._run_model_versions_dialog", run_dialog
    )
    models_root = tmp_path / "models"
    models_root.mkdir()
    gui = SimpleNamespace(
        _models_base=models_root,
        current_language="zh_TW",
        product_combo=_combo("PCBA1"),
        area_combo=_combo("A"),
        inference_combo=_combo("yolo"),
        is_detection_running=Mock(return_value=False),
        controller=SimpleNamespace(reload_model_settings=Mock()),
        _catalog=SimpleNamespace(refresh=Mock()),
        load_available_models=Mock(),
        log_message=Mock(),
    )

    _open_model_versions(gui)

    gui.controller.reload_model_settings.assert_called_once_with(
        "PCBA1", "A", "yolo"
    )
    gui._catalog.refresh.assert_called_once_with()
    gui.load_available_models.assert_called_once_with()


def test_model_update_status_uses_training_data_directory(
    tmp_path, monkeypatch
) -> None:
    run_dialog = Mock()
    monkeypatch.setattr(
        "app.gui.view_builder._run_model_update_status_dialog", run_dialog
    )
    project_root = tmp_path / "yolo11_inference"
    gui = SimpleNamespace(
        _project_root=project_root,
        current_language="zh_TW",
        product_combo=_combo("Cable1"),
        area_combo=_combo("A"),
        log_message=Mock(),
    )

    _open_model_update_status(gui)

    run_dialog.assert_called_once_with(
        data_root=tmp_path / "Yolo11_auto_train" / "data",
        language="zh_TW",
        selected_product="Cable1",
        selected_area="A",
        parent=gui,
    )
