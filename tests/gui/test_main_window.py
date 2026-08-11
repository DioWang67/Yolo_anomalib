import time

import numpy as np
import pytest
import yaml

pytest.importorskip("PyQt5", reason="PyQt5 is required for GUI tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for GUI tests")
pytestmark = pytest.mark.gui

qt_gui = pytest.importorskip("PyQt5.QtGui")
qt_core = pytest.importorskip("PyQt5.QtCore")
qt_widgets = pytest.importorskip("PyQt5.QtWidgets")
Qt = qt_core.Qt
QKeySequence = qt_gui.QKeySequence
QCloseEvent = qt_gui.QCloseEvent
QShortcut = qt_widgets.QShortcut

from app.gui.i18n import tr  # noqa: E402
from app.gui.main_window import DetectionSystemGUI  # noqa: E402
from core._version import __version__ as SYSTEM_VERSION  # noqa: E402
from core.types import DetectionResult, DetectionTask  # noqa: E402
from tools.retraining_workspaces import create_retraining_workspace  # noqa: E402


@pytest.fixture
def gui(qtbot):
    """Fixture to create and show the GUI window."""
    window = DetectionSystemGUI()
    qtbot.addWidget(window)
    yield window
    # Ensure clean shutdown to prevent logging errors
    window.close()

def test_window_title(gui):
    """Verify window title indicates correct system."""
    assert gui.windowTitle() == tr(gui.current_language, "window_title")


def test_system_version_is_visible_and_survives_language_change(gui):
    assert gui.system_version_label is not None
    assert gui.system_version_label.text() == (
        f"{tr(gui.current_language, 'system_version')}: v{SYSTEM_VERSION}"
    )

    gui.apply_language("en")

    assert gui.system_version_label.text() == f"System version: v{SYSTEM_VERSION}"
    assert "model" in gui.system_version_label.toolTip().lower()


def test_about_dialog_uses_authoritative_system_version(gui, monkeypatch):
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        qt_widgets.QMessageBox,
        "about",
        lambda _parent, title, body: calls.append((title, body)),
    )

    gui.show_about()

    assert calls == [
        (
            tr(gui.current_language, "about_title"),
            tr(gui.current_language, "about_body").format(
                version=SYSTEM_VERSION
            ),
        )
    ]


def test_panels_present(gui):
    """Verify all major panels are instantiated."""
    assert gui.control_panel is not None
    assert gui.image_panel is not None
    assert gui.info_panel is not None
    assert gui.engineering_settings_page is not None
    assert gui.inspection_history_page is not None
    assert gui.camera_status_indicator is not None


def test_camera_indicator_requires_a_valid_frame_before_ready(gui):
    class ConnectedSystem:
        pipeline_running = False

        @staticmethod
        def is_camera_connected():
            raise AssertionError(
                "A delivered frame must not trigger another SDK status query."
            )

        @staticmethod
        def shutdown():
            return None

    gui.controller._system = ConnectedSystem()
    gui.use_camera_chk.blockSignals(True)
    gui.use_camera_chk.setChecked(True)
    gui.use_camera_chk.blockSignals(False)
    gui._set_camera_status("connected")

    gui.on_image_ready(np.zeros((8, 8, 3), dtype=np.uint8))

    assert gui.camera_status_indicator.state == "ready"


def test_camera_controls_show_unavailable_and_disable_only_camera_auto_mode(
    gui,
):
    class DisconnectedSystem:
        pipeline_running = False

        @staticmethod
        def is_camera_connected():
            return False

        @staticmethod
        def shutdown():
            return None

    gui.controller._system = DisconnectedSystem()
    gui._camera_check_ts = 0
    gui._set_camera_status("ready")
    gui.auto_mode_chk.setChecked(False)

    gui.update_camera_controls()

    assert gui.camera_status_indicator.state == "unavailable"
    assert not gui.camera_status_indicator.reconnect_button.isHidden()
    assert not gui.auto_mode_chk.isEnabled()
    assert gui.pick_image_btn.isEnabledTo(gui.control_panel.camera_group)


def test_manual_camera_inspection_reenables_auto_mode_after_pipeline_stops(
    gui,
    qtbot,
):
    class CompletingManualCameraSystem:
        def __init__(self) -> None:
            self.pipeline_running = True

        @staticmethod
        def is_camera_connected() -> bool:
            return True

        @staticmethod
        def shutdown() -> None:
            return None

    system = CompletingManualCameraSystem()
    gui.controller._system = system
    gui._camera_check_ts = 0
    for combo, value in (
        (gui.product_combo, "Cable1"),
        (gui.area_combo, "A"),
        (gui.inference_combo, "yolo"),
    ):
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(value)
        combo.blockSignals(False)
    gui._single_shot_running = True
    gui.start_btn.setEnabled(False)
    gui.stop_btn.setEnabled(True)
    gui.update_camera_controls()

    task = DetectionTask(
        task_id="manual-camera-1",
        timestamp=time.time(),
        product="Cable1",
        area="A",
        inference_type="yolo",
        frame=np.zeros((8, 8, 3), dtype=np.uint8),
        result={"status": "PASS", "detections": []},
    )
    gui.on_pipeline_result(task)

    assert gui._single_shot_running is True
    assert gui.start_btn.isEnabled() is False
    assert gui.auto_mode_chk.isEnabled() is False

    system.pipeline_running = False
    qtbot.waitUntil(gui.auto_mode_chk.isEnabled, timeout=1000)

    assert gui._single_shot_running is False
    assert gui.start_btn.isEnabled() is True
    assert gui.stop_btn.isEnabled() is False


def test_manual_pipeline_finalization_timeout_restores_safe_stop_action(
    gui,
    monkeypatch,
):
    class StuckManualCameraSystem:
        pipeline_running = True

        @staticmethod
        def is_camera_connected() -> bool:
            return True

        @staticmethod
        def shutdown() -> None:
            return None

    monkeypatch.setattr(gui.controller, "_system", StuckManualCameraSystem())
    try:
        gui._single_shot_running = True
        gui.start_btn.setEnabled(False)
        gui.stop_btn.setEnabled(False)
        gui._schedule_manual_pipeline_release(gui._run_generation)
        gui._manual_pipeline_release_deadline = time.monotonic() - 1.0

        gui._restore_manual_pipeline_controls_when_idle()

        assert gui._manual_pipeline_release_generation is None
        assert gui._manual_pipeline_release_deadline is None
        assert gui._manual_pipeline_release_timer.isActive() is False
        assert gui._single_shot_running is False
        assert gui.start_btn.isEnabled() is False
        assert gui.stop_btn.isEnabled() is True

        gui._shutdown_in_progress = True
        gui._pipeline_shutdown_deadline = time.monotonic() - 1.0
        gui.stop_btn.setEnabled(False)
        gui._on_pipeline_stopped()

        assert gui._pipeline_shutdown_deadline is None
        assert gui._shutdown_in_progress is False
        assert gui.start_btn.isEnabled() is False
        assert gui.stop_btn.isEnabled() is True
    finally:
        # pytest-qt closes widgets before fixture finalizers restore monkeypatches.
        # Remove the deliberately stuck system so closeEvent cannot open a modal.
        gui.controller._system = None


def test_manual_pipeline_state_error_does_not_lock_both_controls(
    gui,
    monkeypatch,
):
    class UnreadableManualCameraSystem:
        @property
        def pipeline_running(self):
            raise RuntimeError("state unavailable")

        @staticmethod
        def is_camera_connected() -> bool:
            return True

        @staticmethod
        def shutdown() -> None:
            return None

    messages: list[str] = []
    shutdown_attempts: list[bool] = []
    monkeypatch.setattr(gui, "log_message", messages.append)
    monkeypatch.setattr(
        gui,
        "_begin_pipeline_shutdown",
        lambda: shutdown_attempts.append(True),
    )
    monkeypatch.setattr(gui.controller, "_system", UnreadableManualCameraSystem())
    try:
        gui._single_shot_running = True
        gui.start_btn.setEnabled(False)
        gui.stop_btn.setEnabled(False)
        gui._schedule_manual_pipeline_release(gui._run_generation)
        gui._manual_pipeline_release_deadline = time.monotonic() - 1.0

        gui._restore_manual_pipeline_controls_when_idle()

        assert gui._manual_pipeline_release_generation is None
        assert gui._manual_pipeline_release_deadline is None
        assert gui._single_shot_running is False
        assert gui.start_btn.isEnabled() is False
        assert gui.stop_btn.isEnabled() is True
        assert any("state unavailable" in message for message in messages)

        gui.stop_btn.click()

        assert shutdown_attempts == [True]
        assert gui._shutdown_in_progress is True
        assert gui.stop_btn.isEnabled() is False

        gui._on_pipeline_stopped()

        assert gui._shutdown_in_progress is False
        assert gui.start_btn.isEnabled() is False
        assert gui.stop_btn.isEnabled() is True
    finally:
        gui.controller._system = None


def test_failed_manual_disconnect_does_not_claim_camera_was_disconnected(
    gui,
    monkeypatch,
):
    class ConnectedSystem:
        pipeline_running = False

        @staticmethod
        def is_camera_connected():
            return True

        @staticmethod
        def disconnect_camera():
            raise RuntimeError("device busy")

        @staticmethod
        def shutdown():
            return None

    gui.controller._system = ConnectedSystem()
    gui._camera_connected_cache = True
    gui._camera_check_ts = time.monotonic()
    gui._set_camera_status("ready")
    monkeypatch.setattr(
        "app.gui.camera_handler.QMessageBox.critical",
        lambda *args, **kwargs: None,
    )

    gui.handle_disconnect_camera()

    assert gui.camera_status_indicator.state == "ready"


def test_auto_preview_camera_error_is_visible_without_terminal(
    gui,
    monkeypatch,
):
    stop_calls = []
    gui.auto_mode_chk.blockSignals(True)
    gui.auto_mode_chk.setChecked(True)
    gui.auto_mode_chk.blockSignals(False)
    gui._camera_connected_cache = True
    gui._camera_check_ts = time.monotonic()
    gui._set_camera_status("ready")
    monkeypatch.setattr(
        gui,
        "_stop_auto_mode",
        lambda: stop_calls.append(True),
    )

    gui._on_auto_error("Camera returned None for 10 consecutive frames")

    assert gui.camera_status_indicator.state == "lost"
    assert not gui.auto_mode_chk.isChecked()
    assert gui._camera_connected_cache is False
    assert gui._camera_check_ts == 0
    assert stop_calls == [True]


def test_inspection_history_navigates_in_main_workspace(
    gui,
    monkeypatch,
):
    shown_targets = []
    monkeypatch.setattr(
        gui.inspection_history_page,
        "show_for_target",
        lambda product, area: shown_targets.append((product, area)),
    )
    gui.product_combo.addItem("Cable1")
    gui.area_combo.addItem("A")

    gui.control_panel.inspection_history_btn.click()

    assert (
        gui.workspace_stack.currentWidget()
        is gui.inspection_history_page
    )
    assert shown_targets == [("Cable1", "A")]
    assert gui.control_panel.engineering_access_granted is False

    gui.inspection_history_page.back_button.click()

    assert gui.workspace_stack.currentWidget() is gui.inspection_workspace


def test_engineering_settings_navigates_as_persistent_full_page(
    gui, monkeypatch
):
    stable_widget_count = gui.workspace_stack.count()
    monkeypatch.setattr(gui.control_panel, "_verify_pin", lambda: True)

    gui.control_panel.engineering_toggle_btn.click()

    assert (
        gui.workspace_stack.currentWidget()
        is gui.engineering_settings_page
    )
    assert (
        gui.engineering_settings_page.engineering_controls
        is gui.control_panel.engineering_panel
    )
    assert gui.engineering_settings_page.parent() is gui.workspace_stack
    assert gui.control_panel.layout().indexOf(
        gui.control_panel.engineering_panel
    ) == -1

    gui.engineering_settings_page.back_button.click()
    assert gui.workspace_stack.currentWidget() is gui.inspection_workspace

    gui.control_panel.engineering_toggle_btn.click()
    assert gui.workspace_stack.count() == stable_widget_count
    gui.control_panel._lock_btn.click()
    assert gui.workspace_stack.currentWidget() is gui.inspection_workspace


def test_engineering_navigation_requires_pin_and_keeps_failed_entry_locked(
    gui, monkeypatch
):
    """A failed PIN must neither reveal nor enable privileged controls."""
    pin_results = iter((False, True))
    verification_calls: list[None] = []

    def verify_pin() -> bool:
        verification_calls.append(None)
        return next(pin_results)

    monkeypatch.setattr(gui.control_panel, "_verify_pin", verify_pin)

    gui.control_panel.engineering_toggle_btn.click()

    assert gui.workspace_stack.currentWidget() is gui.inspection_workspace
    assert gui.control_panel.engineering_panel.isEnabled() is False

    gui.control_panel.engineering_toggle_btn.click()

    assert gui.workspace_stack.currentWidget() is gui.engineering_settings_page
    assert gui.control_panel.engineering_panel.isEnabled() is True
    assert len(verification_calls) == 2


@pytest.mark.parametrize("exit_action", ["back", "lock"])
def test_engineering_exit_locks_controls_and_reentry_reauthenticates(
    gui, monkeypatch, exit_action
):
    """Back and Lock both end the current authenticated engineering session."""
    verification_calls: list[None] = []

    def verify_pin() -> bool:
        verification_calls.append(None)
        return True

    monkeypatch.setattr(gui.control_panel, "_verify_pin", verify_pin)
    gui.control_panel.engineering_toggle_btn.click()

    assert gui.workspace_stack.currentWidget() is gui.engineering_settings_page
    assert gui.control_panel.engineering_panel.isEnabled() is True

    if exit_action == "back":
        gui.engineering_settings_page.back_button.click()
    else:
        gui.control_panel._lock_btn.click()

    assert gui.workspace_stack.currentWidget() is gui.inspection_workspace
    assert gui.control_panel.engineering_panel.isEnabled() is False

    gui.control_panel.engineering_toggle_btn.click()

    assert gui.workspace_stack.currentWidget() is gui.engineering_settings_page
    assert gui.control_panel.engineering_panel.isEnabled() is True
    assert len(verification_calls) == 2


def test_start_shortcuts_are_ignored_outside_inspection_workspace(
    gui, monkeypatch
):
    """Space/Return must not start a hidden inspection from engineering settings."""
    monkeypatch.setattr(gui.control_panel, "_verify_pin", lambda: True)
    start_calls: list[None] = []
    monkeypatch.setattr(gui, "start_detection", lambda: start_calls.append(None))
    gui.start_btn.setEnabled(True)

    shortcuts = gui.findChildren(QShortcut)
    start_shortcuts = [
        shortcut
        for key_name in ("Space", "Return")
        for shortcut in shortcuts
        if shortcut.key() == QKeySequence(key_name)
    ]
    assert len(start_shortcuts) == 2

    gui.control_panel.engineering_toggle_btn.click()
    for shortcut in start_shortcuts:
        shortcut.activated.emit()

    assert gui.workspace_stack.currentWidget() is gui.engineering_settings_page
    assert start_calls == []

    gui.show_inspection_workspace()
    for shortcut in start_shortcuts:
        shortcut.activated.emit()

    assert start_calls == [None, None]


def test_space_activates_engineering_button_without_starting_inspection(
    gui,
    monkeypatch,
    qtbot,
):
    monkeypatch.setattr(gui.control_panel, "_verify_pin", lambda: True)
    start_calls: list[None] = []
    monkeypatch.setattr(gui, "start_detection", lambda: start_calls.append(None))
    gui.start_btn.setEnabled(True)
    gui.control_panel.engineering_toggle_btn.click()
    gui.engineering_settings_page.back_button.setFocus()

    qtbot.keyClick(gui.engineering_settings_page.back_button, Qt.Key_Space)

    assert gui.workspace_stack.currentWidget() is gui.inspection_workspace
    assert start_calls == []


def _configure_calibration_targets(gui) -> None:
    """Install a deterministic two-product catalogue without loading models."""
    gui.available_products = ["Cable1", "Cable2"]
    gui.available_areas = {
        "Cable1": ["A", "B"],
        "Cable2": ["TOP"],
    }
    for combo in (gui.product_combo, gui.area_combo, gui.inference_combo):
        combo.blockSignals(True)
        combo.clear()
    gui.product_combo.addItems(gui.available_products)
    gui.product_combo.setCurrentText("Cable1")
    gui.area_combo.addItems(gui.available_areas["Cable1"])
    gui.area_combo.setCurrentText("A")
    for combo in (gui.product_combo, gui.area_combo, gui.inference_combo):
        combo.blockSignals(False)


@pytest.mark.parametrize("changed_scope", ["product", "area"])
def test_calibration_samples_are_invalidated_when_target_changes(
    gui, monkeypatch, changed_scope
):
    """Samples captured for one target must never be applicable to another."""
    _configure_calibration_targets(gui)
    panel = gui.control_panel
    monkeypatch.setattr(panel, "_verify_pin", lambda: True)
    panel.engineering_toggle_btn.click()
    panel.set_calib_empty(100.0, target=("Cable1", "A"))
    panel.set_calib_product(300.0, target=("Cable1", "A"))
    assert panel._calib_apply_btn.isEnabled() is True

    if changed_scope == "product":
        gui.product_combo.setCurrentText("Cable2")
        assert (gui.product_combo.currentText(), gui.area_combo.currentText()) == (
            "Cable2",
            "TOP",
        )
    else:
        gui.area_combo.setCurrentText("B")
        assert (gui.product_combo.currentText(), gui.area_combo.currentText()) == (
            "Cable1",
            "B",
        )

    assert panel._calib_apply_btn.isEnabled() is False


def test_lock_clears_calibration_samples_instead_of_reusing_them(
    gui, monkeypatch
):
    """After Lock, one new sample cannot combine with an old hidden sample."""
    _configure_calibration_targets(gui)
    panel = gui.control_panel
    monkeypatch.setattr(panel, "_verify_pin", lambda: True)
    gui.control_panel.engineering_toggle_btn.click()

    panel.set_calib_empty(100.0, target=("Cable1", "A"))
    panel.set_calib_product(300.0, target=("Cable1", "A"))
    assert panel._calib_apply_btn.isEnabled() is True

    panel._lock_btn.click()
    gui.control_panel.engineering_toggle_btn.click()
    panel.set_calib_empty(120.0, target=("Cable1", "A"))

    assert panel._calib_apply_btn.isEnabled() is False

    panel.set_calib_product(320.0, target=("Cable1", "A"))
    assert panel._calib_apply_btn.isEnabled() is True


def test_calibration_apply_writes_only_the_bound_target(
    gui,
    monkeypatch,
    tmp_path,
):
    _configure_calibration_targets(gui)
    panel = gui.control_panel
    gui._models_base = tmp_path / "models"
    monkeypatch.setattr(panel, "_verify_pin", lambda: True)
    monkeypatch.setattr(
        qt_widgets.QMessageBox,
        "information",
        lambda *_args, **_kwargs: qt_widgets.QMessageBox.Ok,
    )
    panel.engineering_toggle_btn.click()
    panel.set_calib_empty(100.0, target=("Cable1", "A"))
    panel.set_calib_product(300.0, target=("Cable1", "A"))

    panel._calib_apply_btn.click()

    saved_path = (
        tmp_path / "models" / "Cable1" / "A" / "auto_trigger.yaml"
    )
    assert yaml.safe_load(saved_path.read_text(encoding="utf-8"))[
        "product_area_threshold"
    ] == 200
    assert not (
        tmp_path / "models" / "Cable2" / "TOP" / "auto_trigger.yaml"
    ).exists()
    assert panel._calib_apply_btn.isEnabled() is False


def test_engineering_equipment_tab_exposes_visible_live_preview(gui, monkeypatch):
    """Equipment calibration must provide visual context on its owning tab."""
    monkeypatch.setattr(gui.control_panel, "_verify_pin", lambda: True)
    gui.control_panel.engineering_toggle_btn.click()

    preview = gui.engineering_settings_page.preview_viewer
    assert preview.objectName() == "engineeringPreview"
    assert gui.engineering_settings_page.isAncestorOf(preview)
    assert preview.isHidden() is False
    assert preview.isVisibleTo(gui.engineering_settings_page) is False

    gui.control_panel.engineering_tabs.setCurrentIndex(2)

    assert preview.isVisibleTo(gui.engineering_settings_page) is True


class _FakeAutoPreviewSource:
    def __init__(self) -> None:
        self.running = True
        self.stop_calls = 0
        self.active_generation = 1

    def is_running(self) -> bool:
        return self.running

    def stop(self) -> bool:
        self.stop_calls += 1
        self.running = False
        self.active_generation = None
        return True


def _viewer_has_pixmap(viewer) -> bool:
    pixmap = viewer.pixmap()
    return pixmap is not None and not pixmap.isNull()


def test_active_source_routes_frame_to_independent_engineering_preview(
    gui, monkeypatch
):
    """The preview reuses the GUI frame path without reparenting inspection tabs."""
    monkeypatch.setattr(gui.control_panel, "_verify_pin", lambda: True)
    source = _FakeAutoPreviewSource()
    gui._auto_controller = source
    inspection_viewers = (
        gui.original_image,
        gui.processed_image,
        gui.result_image,
    )
    original_parent = gui.original_image.parent()
    original_tab_index = gui.image_panel.image_tabs.indexOf(gui.original_image)
    gui.control_panel.engineering_toggle_btn.click()

    frame = np.full((24, 32, 3), 127, dtype=np.uint8)
    gui.on_image_ready(frame)

    preview = gui.engineering_settings_page.preview_viewer
    assert _viewer_has_pixmap(preview) is True
    assert all(preview is not viewer for viewer in inspection_viewers)
    assert gui.original_image.parent() is original_parent
    assert gui.image_panel.image_tabs.indexOf(gui.original_image) == original_tab_index
    source.running = False


@pytest.mark.parametrize(
    "finish_route",
    ["auto_stop", "worker_finished", "pipeline_reset"],
)
def test_engineering_preview_is_cleared_when_live_source_finishes(
    gui, monkeypatch, finish_route
):
    """A stopped source must not leave a stale frame presented as live."""
    monkeypatch.setattr(gui.control_panel, "_verify_pin", lambda: True)
    source = _FakeAutoPreviewSource()
    gui._auto_controller = source
    gui.control_panel.engineering_toggle_btn.click()
    gui.on_image_ready(np.full((16, 16, 3), 64, dtype=np.uint8))
    preview = gui.engineering_settings_page.preview_viewer
    assert _viewer_has_pixmap(preview) is True

    if finish_route == "auto_stop":
        gui._stop_auto_mode()
        assert source.stop_calls == 1
    elif finish_route == "worker_finished":
        source.running = False
        gui._on_worker_finished()
    else:
        source.running = False
        gui._reset_ui_state()

    assert _viewer_has_pixmap(preview) is False


class _SlowStoppingAutoPreviewSource(_FakeAutoPreviewSource):
    def __init__(self) -> None:
        super().__init__()
        self.active_generation = 7

    def stop(self) -> bool:
        self.stop_calls += 1
        if self.active_generation is None:
            self.running = False
            return True
        return False


def _arm_auto_checkbox_without_starting(gui) -> None:
    gui.auto_mode_chk.blockSignals(True)
    gui.auto_mode_chk.setChecked(True)
    gui.auto_mode_chk.blockSignals(False)


def test_calibration_restart_waits_for_matching_fully_stopped_generation(
    gui,
    monkeypatch,
):
    source = _SlowStoppingAutoPreviewSource()
    gui._auto_controller = source
    _arm_auto_checkbox_without_starting(gui)
    restart_calls = []
    monkeypatch.setattr(
        gui,
        "_start_auto_mode",
        lambda: restart_calls.append("started"),
    )

    assert gui._stop_auto_mode(restart_after_stop=True) is False
    assert restart_calls == []

    gui._on_auto_controller_stopped(6)
    assert restart_calls == []

    source.active_generation = None
    source.running = False
    gui._on_auto_controller_stopped(7)

    assert restart_calls == ["started"]
    assert gui._pending_auto_restart is None


def test_manual_stop_cancels_pending_calibration_restart(gui, monkeypatch):
    source = _SlowStoppingAutoPreviewSource()
    gui._auto_controller = source
    _arm_auto_checkbox_without_starting(gui)
    restart_calls = []
    monkeypatch.setattr(
        gui,
        "_start_auto_mode",
        lambda: restart_calls.append("started"),
    )

    assert gui._stop_auto_mode(restart_after_stop=True) is False
    assert gui._pending_auto_restart is not None

    assert gui._stop_auto_mode() is False
    assert gui._pending_auto_restart is None
    source.active_generation = None
    source.running = False
    gui._on_auto_controller_stopped(7)

    assert restart_calls == []


def test_close_waits_for_auto_work_to_finish(
    gui,
    monkeypatch,
    qtbot,
):
    source = _SlowStoppingAutoPreviewSource()
    gui._auto_controller = source
    close_requests = []
    monkeypatch.setattr(
        qt_widgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: qt_widgets.QMessageBox.Yes,
    )
    monkeypatch.setattr(
        gui,
        "close",
        lambda: close_requests.append("close"),
    )

    event = QCloseEvent()
    gui.closeEvent(event)

    assert event.isAccepted() is False
    assert gui._close_after_auto_stop is True
    assert close_requests == []

    source.active_generation = None
    source.running = False
    gui._on_auto_controller_stopped(7)
    qtbot.waitUntil(lambda: close_requests == ["close"], timeout=1000)


def test_engineer_retraining_button_routes_to_existing_workspace_entry(
    gui, monkeypatch
):
    """Engineer navigation must reuse the established retraining entry point."""
    from app.gui import main_window

    opened_for = []
    monkeypatch.setattr(
        main_window,
        "_open_training_review",
        lambda window: opened_for.append(window),
    )
    monkeypatch.setattr(gui.control_panel, "_verify_pin", lambda: True)

    gui.control_panel.engineering_toggle_btn.click()
    gui.control_panel.retraining_workspace_btn.click()

    assert opened_for == [gui]


def test_legacy_file_menu_retraining_entry_is_removed(gui):
    file_menu = gui.menuBar().actions()[0].menu()
    action_texts = {
        action.text()
        for action in file_menu.actions()
        if not action.isSeparator()
    }

    assert action_texts.isdisjoint(
        {"Production Retraining", "產線模型補訓"}
    )


def test_retraining_workspace_is_reused_and_can_return_to_inspection(
    gui, tmp_path, qtbot
):
    """Leaving retraining must switch pages without destroying its state."""
    create_retraining_workspace(
        tmp_path / "training-data",
        product="Cable1",
        area="A",
        batch_version="Cable1_A_v0.0.1",
    )
    arguments = {
        "result_root": tmp_path / "Result",
        "manifest_path": tmp_path / "review.csv",
        "training_data_dir": tmp_path / "training-data",
        "language": "zh_TW",
        "product": "Cable1",
        "area": "A",
    }

    first = gui.show_retraining_workspace(**arguments)
    gui.show_inspection_workspace()
    second = gui.show_retraining_workspace(**arguments)

    assert first is second
    assert gui.workspace_stack.currentWidget() is first
    qtbot.waitUntil(lambda: first.workspace is not None, timeout=5000)
    first.back_to_inspection_requested.emit()
    assert gui.workspace_stack.currentWidget() is gui.inspection_workspace


def test_color_revision_signal_invalidates_active_model(
    gui, tmp_path, qtbot, monkeypatch
):
    create_retraining_workspace(
        tmp_path / "training-data",
        product="Cable1",
        area="A",
        batch_version="Cable1_A_v0.0.1",
    )
    calls = []
    monkeypatch.setattr(
        gui.controller,
        "reload_model_settings",
        lambda product, area, inference_type: calls.append(
            (product, area, inference_type)
        ),
    )
    host = gui.show_retraining_workspace(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        product="Cable1",
        area="A",
    )
    qtbot.waitUntil(lambda: host.workspace is not None, timeout=5000)

    host.color_configuration_changed.emit("Cable1", "A", "yolo")

    assert calls == [("Cable1", "A", "yolo")]


def test_retraining_manifest_scan_does_not_block_page_switch(
    gui, tmp_path, qtbot, monkeypatch
):
    """A slow disk scan must remain outside the Qt main thread."""
    from app.gui import review_cases_dialog

    original_prepare = review_cases_dialog.prepare_review_manifest
    create_retraining_workspace(
        tmp_path / "training-data",
        product="Cable1",
        area="A",
        batch_version="Cable1_A_v0.0.1",
    )

    def slow_prepare(**kwargs):
        time.sleep(0.4)
        return original_prepare(**kwargs)

    monkeypatch.setattr(review_cases_dialog, "prepare_review_manifest", slow_prepare)
    started = time.perf_counter()
    host = gui.show_retraining_workspace(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        product="Cable1",
        area="A",
    )
    call_seconds = time.perf_counter() - started

    assert call_seconds < 0.2
    assert gui.workspace_stack.currentWidget() is host
    assert host.workspace is None
    qtbot.waitUntil(lambda: host.workspace is not None, timeout=5000)


def test_retraining_workspace_survives_repeated_page_switches(
    gui, tmp_path, qtbot
):
    """Repeated navigation must reuse one workspace and remain responsive."""
    create_retraining_workspace(
        tmp_path / "training-data",
        product="Cable1",
        area="A",
        batch_version="Cable1_A_v0.0.1",
    )
    arguments = {
        "result_root": tmp_path / "Result",
        "manifest_path": tmp_path / "review.csv",
        "training_data_dir": tmp_path / "training-data",
        "language": "zh_TW",
        "product": "Cable1",
        "area": "A",
    }
    host = gui.show_retraining_workspace(**arguments)
    qtbot.waitUntil(lambda: host.workspace is not None, timeout=5000)
    stable_widget_count = gui.workspace_stack.count()

    iterations = 250
    started = time.perf_counter()
    for _iteration in range(iterations):
        gui.show_inspection_workspace()
        assert gui.show_retraining_workspace(**arguments) is host
    elapsed_seconds = time.perf_counter() - started

    assert gui.workspace_stack.currentWidget() is host
    assert gui.workspace_stack.count() == stable_widget_count
    assert elapsed_seconds / iterations < 0.02


def test_retraining_photo_stage_switches_product_and_area_without_mixing(
    gui, tmp_path, qtbot
):
    create_retraining_workspace(
        tmp_path / "training-data",
        product="Cable1",
        area="A",
        batch_version="Cable1_A_v0.0.1",
    )
    create_retraining_workspace(
        tmp_path / "training-data",
        product="PCBA1",
        area="TOP",
        batch_version="PCBA1_TOP_v0.0.1",
    )
    host = gui.show_retraining_workspace(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        product="Cable1",
        area="A",
        available_targets=(
            ("Cable1", "A"),
            ("Cable1", "B"),
            ("PCBA1", "TOP"),
        ),
    )
    qtbot.waitUntil(lambda: host.workspace is not None, timeout=5000)
    first_workspace = host.workspace

    assert [
        host.product_filter.itemText(index)
        for index in range(host.product_filter.count())
    ] == ["Cable1", "PCBA1"]
    assert "全部" not in {
        host.product_filter.itemText(index)
        for index in range(host.product_filter.count())
    }

    host.product_filter.setCurrentText("PCBA1")

    qtbot.waitUntil(
        lambda: host.workspace is not None
        and host.workspace is not first_workspace
        and host.workspace.product == "PCBA1",
        timeout=5000,
    )
    assert host.area_filter.currentText() == "TOP"
    assert host.workspace.area == "TOP"
    assert host.target_scope_label.text() == "ⓘ 提示"
    assert "PCBA1/TOP" in host.target_scope_label.toolTip()


def test_retraining_target_switch_discards_stale_background_result(
    gui, tmp_path, qtbot, monkeypatch
):
    from app.gui import review_cases_dialog

    original_prepare = review_cases_dialog.prepare_review_manifest
    create_retraining_workspace(
        tmp_path / "training-data",
        product="Cable1",
        area="A",
        batch_version="Cable1_A_v0.0.1",
    )
    create_retraining_workspace(
        tmp_path / "training-data",
        product="PCBA1",
        area="TOP",
        batch_version="PCBA1_TOP_v0.0.1",
    )
    prepared_targets: list[tuple[str | None, str | None]] = []

    def slow_prepare(**kwargs):
        target = (kwargs.get("product"), kwargs.get("area"))
        prepared_targets.append(target)
        if target == ("Cable1", "A"):
            time.sleep(0.3)
        return original_prepare(**kwargs)

    monkeypatch.setattr(review_cases_dialog, "prepare_review_manifest", slow_prepare)
    host = gui.show_retraining_workspace(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        product="Cable1",
        area="A",
        available_targets=(("Cable1", "A"), ("PCBA1", "TOP")),
    )

    host.product_filter.setCurrentText("PCBA1")

    qtbot.waitUntil(
        lambda: host.workspace is not None
        and host.workspace.product == "PCBA1"
        and host.workspace.area == "TOP",
        timeout=5000,
    )
    assert prepared_targets[-1] == ("PCBA1", "TOP")
    assert "PCBA1/TOP" in host.target_scope_label.toolTip()


def test_initial_state(gui):
    """Verify initial button states."""
    assert gui.start_btn.isEnabled() is False  # Should be disabled until configs loaded/selected
    assert gui.stop_btn.isEnabled() is False
    assert gui.save_btn.isEnabled() is False
    assert gui.show_detection_boxes_chk is not None
    assert gui.show_original_tab_chk is not None
    assert gui.show_processed_tab_chk is not None

def test_model_loading_async(gui, qtbot):
    """Verify that model loading triggers signals and updates combos."""
    # Since load_available_models is async, we wait for the log message or combo update
    # But checking combos is easier.
    # Note: real model loading depends on file system.
    # If this test env has no models, combos remain empty.

    # Trigger refresh manually
    # We call it once to ensure the attribute is created since we skip auto-load in __init__
    gui.load_available_models()
    with qtbot.waitSignal(gui.model_loader.models_ready, timeout=5000, raising=False):
        # Thread already started by the call above
        pass

    # Even if timeout (no models found or error), we check that GUI didn't crash
    # and combos are objects (not None)
    assert gui.product_combo is not None

def test_interaction_flow(gui, qtbot):
    """Test a simple interaction flow."""
    # Simulate selecting a product if available
    if gui.product_combo.count() > 0:
        gui.product_combo.setCurrentIndex(0)
        # Check area update
        assert gui.area_combo.count() >= 0


def test_result_image_uses_preprocessed_path_when_boxes_hidden(gui, monkeypatch):
    """YOLO result view should switch to the clean image when boxes are hidden."""
    calls: list[str] = []

    def fake_load_image_with_retry(widget, image_path, **kwargs):
        calls.append(image_path)

    monkeypatch.setattr(
        "app.gui.main_window.load_image_with_retry",
        fake_load_image_with_retry,
    )

    gui.current_result = DetectionResult(
        status="PASS",
        product="P",
        area="A",
        inference_type="yolo",
        original_image_path="original.jpg",
        preprocessed_image_path="processed.jpg",
        annotated_path="annotated.jpg",
    )

    gui.show_detection_boxes_chk.setChecked(True)
    gui.show_detection_boxes_chk.setChecked(False)

    assert calls[-1] == "processed.jpg"


def test_pipeline_storage_completion_refreshes_all_three_artifact_tabs(
    gui, monkeypatch
):
    """Pipeline UI must wait for persistence and then use its exact paths."""
    loaded: list[tuple[object, str]] = []
    live_result_frames: list[np.ndarray] = []

    def fake_load_image_with_retry(widget, image_path, **_kwargs):
        loaded.append((widget, image_path))

    monkeypatch.setattr(
        "app.gui.main_window.load_image_with_retry",
        fake_load_image_with_retry,
    )
    monkeypatch.setattr(
        gui.result_image,
        "display_image",
        live_result_frames.append,
    )
    history_refreshes = []
    monkeypatch.setattr(
        gui.inspection_history_page,
        "mark_dirty",
        lambda: history_refreshes.append(True),
    )

    task = DetectionTask(
        task_id="inspection-1",
        timestamp=time.time(),
        product="Cable1",
        area="A",
        inference_type="yolo",
        frame=np.zeros((8, 8, 3), dtype=np.uint8),
        result={
            "status": "PASS",
            "detections": [],
            "result_frame": np.ones((8, 8, 3), dtype=np.uint8),
        },
    )

    gui.on_pipeline_result(task)

    assert gui.current_result is not None
    assert gui.current_result.metadata["storage_completed"] is False
    assert live_result_frames == []

    loaded.clear()
    task.result.update(
        {
            "original_image_path": "persisted-original.jpg",
            "preprocessed_image_path": "persisted-processed.jpg",
            "annotated_path": "persisted-annotated.jpg",
        }
    )
    gui.on_pipeline_storage_completed(task)

    assert gui.current_result.metadata["storage_completed"] is True
    assert history_refreshes == [True]
    assert loaded == [
        (gui.original_image, "persisted-original.jpg"),
        (gui.processed_image, "persisted-processed.jpg"),
        (gui.result_image, "persisted-annotated.jpg"),
    ]
    assert live_result_frames == []


def test_engineer_image_tab_toggles_hide_optional_tabs(gui):
    """Engineer settings should control original/processed tab visibility."""
    gui.show_original_tab_chk.setChecked(True)
    gui.show_processed_tab_chk.setChecked(True)

    assert gui.image_panel.image_tabs.indexOf(gui.original_image) >= 0
    assert gui.image_panel.image_tabs.indexOf(gui.processed_image) >= 0
    assert gui.image_panel.image_tabs.indexOf(gui.result_image) >= 0

    gui.show_original_tab_chk.setChecked(False)
    assert gui.image_panel.image_tabs.indexOf(gui.original_image) == -1
    assert gui.image_panel.image_tabs.indexOf(gui.processed_image) >= 0
    assert gui.image_panel.image_tabs.indexOf(gui.result_image) >= 0

    gui.show_processed_tab_chk.setChecked(False)
    assert gui.image_panel.image_tabs.indexOf(gui.original_image) == -1
    assert gui.image_panel.image_tabs.indexOf(gui.processed_image) == -1
    assert gui.image_panel.image_tabs.indexOf(gui.result_image) >= 0
    assert gui.image_panel.image_tabs.count() == 1

    gui.show_original_tab_chk.setChecked(True)
    gui.show_processed_tab_chk.setChecked(True)
