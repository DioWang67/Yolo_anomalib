"""Mixin that handles all camera-related UI interactions."""

from __future__ import annotations

import time

from PyQt5.QtWidgets import QMessageBox

from app.gui.i18n import tr


class CameraHandlerMixin:
    """Camera control methods extracted from DetectionSystemGUI."""

    def _t(self, key: str, **kwargs: object) -> str:
        text = tr(getattr(self, "current_language", "en"), key)
        return text.format(**kwargs) if kwargs else text

    def update_camera_controls(self) -> None:
        """Sync camera-related widgets with the current connection state."""
        try:
            running = self.is_detection_running()
            camera_connected = False
            if self.controller.has_system():
                now = time.monotonic()
                if now - getattr(self, "_camera_check_ts", 0) > 2.0:
                    self._camera_connected_cache = self.controller.is_camera_connected()
                    self._camera_check_ts = now
                camera_connected = getattr(self, "_camera_connected_cache", False)

            if getattr(self, "reconnect_camera_btn", None):
                self.reconnect_camera_btn.setEnabled(not running)
            if getattr(self, "disconnect_camera_btn", None):
                self.disconnect_camera_btn.setEnabled(camera_connected and not running)
            if getattr(self, "use_camera_chk", None):
                if not camera_connected and self.use_camera_chk.isChecked():
                    self.use_camera_chk.blockSignals(True)
                    self.use_camera_chk.setChecked(False)
                    self.use_camera_chk.blockSignals(False)
                self.use_camera_chk.setEnabled(camera_connected and not running)
            if not camera_connected:
                if getattr(self, "pick_image_btn", None):
                    self.pick_image_btn.setEnabled(True)
                if getattr(self, "clear_image_btn", None):
                    self.clear_image_btn.setEnabled(
                        bool(getattr(self, "selected_image_path", None))
                    )
        except Exception:
            pass

    def handle_reconnect_camera(self) -> None:
        """Manually reconnect the camera via the detection system."""
        if self.is_detection_running():
            QMessageBox.warning(
                self,
                self._t("camera_action_title"),
                self._t("camera_stop_before_reconnect"),
            )
            return
        if not self.controller.has_system():
            self.init_system()
            if not self.controller.has_system():
                return
        self.log_message(self._t("camera_reconnecting"))
        try:
            success = self.controller.reconnect_camera()
        except Exception as exc:
            self.log_message(self._t("camera_reconnect_failed_log", error=exc))
            QMessageBox.critical(
                self,
                self._t("camera_error_title"),
                self._t("camera_reconnect_failed", error=exc),
            )
        else:
            if success:
                self.log_message(self._t("camera_reconnect_success_log"))
                QMessageBox.information(
                    self,
                    self._t("camera_status_title"),
                    self._t("camera_reconnect_success"),
                )
                if getattr(self, "use_camera_chk", None):
                    self.use_camera_chk.blockSignals(True)
                    self.use_camera_chk.setChecked(True)
                    self.use_camera_chk.blockSignals(False)
                    self.on_use_camera_toggled(True)
            else:
                self.log_message(self._t("camera_reconnect_failed_log", error=""))
                QMessageBox.critical(
                    self,
                    self._t("camera_error_title"),
                    self._t("camera_reconnect_check"),
                )
        self._camera_check_ts = 0
        self.update_camera_controls()

    def handle_disconnect_camera(self) -> None:
        """Allow the operator to disconnect the camera manually."""
        if self.is_detection_running():
            QMessageBox.warning(
                self,
                self._t("camera_action_title"),
                self._t("camera_stop_before_disconnect"),
            )
            return
        if not self.controller.has_system():
            QMessageBox.information(
                self,
                self._t("camera_status_title"),
                self._t("system_uninitialized"),
            )
            return
        self.log_message(self._t("camera_disconnecting"))
        try:
            self.controller.disconnect_camera()
            self.log_message(self._t("camera_disconnected_log"))
            QMessageBox.information(
                self,
                self._t("camera_status_title"),
                self._t("camera_disconnected"),
            )
        except Exception as exc:
            self.log_message(self._t("camera_disconnect_failed_log", error=exc))
            QMessageBox.critical(
                self,
                self._t("camera_error_title"),
                self._t("camera_disconnect_failed", error=exc),
            )
        if getattr(self, "use_camera_chk", None):
            self.use_camera_chk.blockSignals(True)
            self.use_camera_chk.setChecked(False)
            self.use_camera_chk.blockSignals(False)
            self.on_use_camera_toggled(False)
        self._camera_check_ts = 0
        self.update_camera_controls()

    def on_use_camera_toggled(self, checked: bool) -> None:
        """Switch between camera mode and static-image mode."""
        try:
            if self.is_detection_running():
                return
            camera_ready = (
                self.controller.is_camera_connected()
                if self.controller.has_system()
                else False
            )
            if checked:
                if not camera_ready:
                    QMessageBox.warning(
                        self,
                        self._t("camera_unavailable_title"),
                        self._t("camera_unavailable"),
                    )
                    if getattr(self, "use_camera_chk", None):
                        self.use_camera_chk.blockSignals(True)
                        self.use_camera_chk.setChecked(False)
                        self.use_camera_chk.blockSignals(False)
                    self.log_message(self._t("camera_mode_failed"))
                    return
                self.selected_image_path = None
                if getattr(self, "pick_image_btn", None):
                    self.pick_image_btn.setEnabled(False)
                if getattr(self, "clear_image_btn", None):
                    self.clear_image_btn.setEnabled(False)
                if getattr(self, "image_path_label", None):
                    self.image_path_label.setText(self._t("camera_input"))
            else:
                if getattr(self, "pick_image_btn", None):
                    self.pick_image_btn.setEnabled(True)
                if getattr(self, "clear_image_btn", None):
                    self.clear_image_btn.setEnabled(bool(self.selected_image_path))
                if not self.selected_image_path and getattr(self, "image_path_label", None):
                    self.image_path_label.setText(self._t("select_image_or_camera"))
        except Exception:
            pass
        finally:
            try:
                self.update_camera_controls()
            except Exception:
                pass

    def _on_camera_disconnected(self) -> None:
        """Camera disconnected mid-pipeline and starts recovery."""
        try:
            self.controller.bridge.end_run(getattr(self, "_run_generation", None))
            self._run_generation += 1
            self._shutdown_in_progress = True
        except Exception:
            pass
        self.log_message(self._t("camera_lost_log"))
        self.stats_timer.stop()
        self._shutdown_worker = self.controller.build_shutdown_worker()
        self._shutdown_worker.shutdown_complete.connect(self._on_camera_lost_pipeline_stopped)
        self._shutdown_worker.start()

    def _on_camera_lost_pipeline_stopped(self) -> None:
        """Pipeline fully stopped after camera loss, then show reconnect dialog."""
        self._reset_ui_state()
        reply = QMessageBox.question(
            self,
            self._t("camera_lost_title"),
            self._t("camera_lost_prompt"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            self.handle_reconnect_camera()
            if self.controller.has_system() and self.controller.is_camera_connected():
                self.log_message(self._t("camera_restarted"))
                self.start_detection()
            else:
                self.log_message(self._t("camera_manual_check"))
        else:
            self.log_message(self._t("camera_reconnect_canceled"))
