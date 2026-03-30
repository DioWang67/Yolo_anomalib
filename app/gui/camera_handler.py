"""Mixin that handles all camera-related UI interactions.

``DetectionSystemGUI`` inherits this alongside ``QMainWindow``.  Every method
here accesses ``self`` as if it were ``DetectionSystemGUI``, keeping the API
and all existing call-sites unchanged while reducing ``main_window.py`` size.
"""

from __future__ import annotations

import time

from PyQt5.QtWidgets import QMessageBox


class CameraHandlerMixin:
    """Camera control methods extracted from DetectionSystemGUI."""

    # ------------------------------------------------------------------
    # Camera status helpers
    # ------------------------------------------------------------------

    def update_camera_controls(self) -> None:
        """Sync camera-related widgets with the current connection state.

        Uses a 2-second cache to avoid repeated blocking SDK calls during
        rapid signal cascading.
        """
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

    # ------------------------------------------------------------------
    # Manual reconnect / disconnect
    # ------------------------------------------------------------------

    def handle_reconnect_camera(self) -> None:
        """Manually reconnect the camera via the detection system."""
        if self.is_detection_running():
            QMessageBox.warning(self, "相機操作", "檢測進行中，請先停止後再重新連線。")
            return
        if not self.controller.has_system():
            self.init_system()
            if not self.controller.has_system():
                return
        self.log_message("正在嘗試重新連線相機...")
        try:
            success = self.controller.reconnect_camera()
        except Exception as exc:
            self.log_message(f"相機重連失敗：{exc}")
            QMessageBox.critical(self, "相機錯誤", f"重連失敗：\n{exc}")
        else:
            if success:
                self.log_message("相機重連成功")
                QMessageBox.information(self, "相機狀態", "相機已重新連線。")
                if getattr(self, "use_camera_chk", None):
                    self.use_camera_chk.blockSignals(True)
                    self.use_camera_chk.setChecked(True)
                    self.use_camera_chk.blockSignals(False)
                    self.on_use_camera_toggled(True)
            else:
                self.log_message("相機重連失敗")
                QMessageBox.critical(self, "相機錯誤", "重連失敗，請檢查硬體或設定。")
        # Invalidate cache so the next update is live
        self._camera_check_ts = 0
        self.update_camera_controls()

    def handle_disconnect_camera(self) -> None:
        """Allow the operator to disconnect the camera manually."""
        if self.is_detection_running():
            QMessageBox.warning(self, "相機操作", "檢測進行中，請先停止後再中斷連線。")
            return
        if not self.controller.has_system():
            QMessageBox.information(self, "相機狀態", "檢測系統尚未初始化。")
            return
        self.log_message("正在中斷相機連線...")
        try:
            self.controller.disconnect_camera()
            self.log_message("相機已中斷連線")
            QMessageBox.information(self, "相機狀態", "相機已中斷連線。")
        except Exception as exc:
            self.log_message(f"相機中斷失敗：{exc}")
            QMessageBox.critical(self, "相機錯誤", f"中斷連線失敗：\n{exc}")
        if getattr(self, "use_camera_chk", None):
            self.use_camera_chk.blockSignals(True)
            self.use_camera_chk.setChecked(False)
            self.use_camera_chk.blockSignals(False)
            self.on_use_camera_toggled(False)
        self._camera_check_ts = 0
        self.update_camera_controls()

    # ------------------------------------------------------------------
    # Camera / image mode toggle
    # ------------------------------------------------------------------

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
                    QMessageBox.warning(self, "相機不可用", "目前相機尚未連線，請先重新連線。")
                    if getattr(self, "use_camera_chk", None):
                        self.use_camera_chk.blockSignals(True)
                        self.use_camera_chk.setChecked(False)
                        self.use_camera_chk.blockSignals(False)
                    self.log_message("切換相機模式失敗：相機未連線")
                    return
                self.selected_image_path = None
                if getattr(self, "pick_image_btn", None):
                    self.pick_image_btn.setEnabled(False)
                if getattr(self, "clear_image_btn", None):
                    self.clear_image_btn.setEnabled(False)
                if getattr(self, "image_path_label", None):
                    self.image_path_label.setText("使用相機輸入（檢測時自動拍攝）")
            else:
                if getattr(self, "pick_image_btn", None):
                    self.pick_image_btn.setEnabled(True)
                if getattr(self, "clear_image_btn", None):
                    self.clear_image_btn.setEnabled(bool(self.selected_image_path))
                if not self.selected_image_path and getattr(self, "image_path_label", None):
                    self.image_path_label.setText("請選擇影像或重新連線相機")
        except Exception:
            pass
        finally:
            try:
                self.update_camera_controls()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Automatic camera-loss recovery (pipeline signal → UI)
    # ------------------------------------------------------------------

    def _on_camera_disconnected(self) -> None:
        """Camera disconnected mid-pipeline — shut down gracefully and offer reconnect."""
        self.log_message("相機連線中斷（連續擷取失敗）")
        self.stats_timer.stop()
        self._shutdown_worker = self.controller.build_shutdown_worker()
        self._shutdown_worker.shutdown_complete.connect(self._on_camera_lost_pipeline_stopped)
        self._shutdown_worker.start()

    def _on_camera_lost_pipeline_stopped(self) -> None:
        """Pipeline fully stopped after camera loss — show reconnect dialog."""
        self._reset_ui_state()
        reply = QMessageBox.question(
            self,
            "相機連線中斷",
            "檢測過程中相機連線已中斷。\n是否嘗試重新連線並繼續檢測？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            self.handle_reconnect_camera()
            if self.controller.has_system() and self.controller.is_camera_connected():
                self.log_message("相機重連成功，自動重啟檢測管線...")
                self.start_detection()
            else:
                self.log_message("相機重連失敗，請手動檢查後再試。")
        else:
            self.log_message("使用者取消重連，檢測已停止。")
