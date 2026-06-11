import logging
import os
import time

import cv2
import numpy as np

import MvImport.MvCameraControl_class as _mvs_binding
from MvImport.MvCameraControl_class import *

from core.exceptions import CameraConnectionError

_camera_logger = logging.getLogger("camera.mvs")


class MVSCamera:
    def __init__(self, config):
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        self.cam = MvCamera()
        self.nPayloadSize = None
        self.supported_features = {}
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0
        self.save_image = False
        self.auto_exposure = False
        self.save_path = "captured_images"
        self.config = config
        # SDK lifecycle stage flags; close() releases only the stages that
        # were actually reached, making teardown safe and idempotent.
        self._handle_created = False
        self._device_opened = False
        self._grabbing = False

        # if not os.path.exists(self.save_path):
        #     os.makedirs(self.save_path)

    def _check_feature_support(self, feature_name):
        try:
            # MV_CC_GetEnumValue writes an MVCC_ENUMVALUE; passing the smaller
            # MVCC_INTVALUE here lets the SDK write past the buffer.
            stParam = MVCC_ENUMVALUE()
            ret = self.cam.MV_CC_GetEnumValue(feature_name, stParam)
            return ret == 0
        except Exception:
            return False

    def set_resolution(self, width, height):
        """設定解析度"""
        ret = self.cam.MV_CC_SetIntValue("Width", width)
        if ret != 0:
            _camera_logger.error("設定寬度失敗! ret[0x%x]", ret)
            return False

        ret = self.cam.MV_CC_SetIntValue("Height", height)
        if ret != 0:
            _camera_logger.error("設定高度失敗! ret[0x%x]", ret)
            return False

        _camera_logger.info("解析度已設置為 %dx%d", width, height)
        return True

    def _initialize_supported_features(self):
        features_to_check = {"ExposureAuto": "自動曝光", "TriggerMode": "觸發模式"}

        self.supported_features = {}
        for feature, description in features_to_check.items():
            supported = self._check_feature_support(feature)
            self.supported_features[feature] = supported
            _camera_logger.info(
                "%s: %s", description, "支援" if supported else "不支援"
            )

    def check_trigger_mode(self):
        """檢查觸發模式是否為關閉狀態"""
        stParam = MVCC_ENUMVALUE()
        ret = self.cam.MV_CC_GetEnumValue("TriggerMode", stParam)
        if ret != 0:
            _camera_logger.error("獲取觸發模式狀態失敗! ret[0x%x]", ret)
            return None
        mode = stParam.nCurValue
        _camera_logger.info(
            "當前觸發模式為: %s", "關閉" if mode == MV_TRIGGER_MODE_OFF else "開啟"
        )
        return mode

    def disable_trigger_mode(self):
        """關閉觸發模式"""
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            _camera_logger.error("關閉觸發模式失敗! ret[0x%x]", ret)
            return False
        _camera_logger.info("觸發模式已關閉")
        return True

    def connect_to_camera(self, device_index=0):

        if not self._basic_connect(device_index):
            return False

        self._initialize_supported_features()
        try:
            if self.supported_features.get("ExposureAuto", False):
                self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)
                _camera_logger.info("已關閉自動曝光")

            self.set_exposure_time(self.config.exposure_time)
            self.set_gain(self.config.gain)

        except Exception as e:
            _camera_logger.error("初始化相機參數時發生錯誤: %s", e)
            return False

        _camera_logger.info("相機連接成功且參數初始化完成!")
        try:
            # 確認並關閉觸發模式
            trigger_mode = self.check_trigger_mode()
            if trigger_mode != MV_TRIGGER_MODE_OFF:
                _camera_logger.warning("觸發模式未關閉，正在關閉...")
                if not self.disable_trigger_mode():
                    _camera_logger.error("無法關閉觸發模式，請檢查設定!")
                    self.close()  # 釋放資源
                    return False
        except Exception as e:
            _camera_logger.error("設定觸發模式時發生錯誤: %s", e)
            self.close()  # 錯誤時釋放資源
            return False

        self.cam.MV_CC_StopGrabbing()
        self._grabbing = False
        self.set_resolution(self.config.width, self.config.height)
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            _camera_logger.error("重新開始取流失敗! ret[0x%x]", ret)
            self.close()
            return False
        self._grabbing = True
        return True

    def toggle_auto_exposure(self):
        if not self.supported_features.get("ExposureAuto", False):
            _camera_logger.warning("此相機不支援自動曝光功能")
            return False

        self.auto_exposure = not self.auto_exposure
        value = 2 if self.auto_exposure else 0
        ret = self.cam.MV_CC_SetEnumValue("ExposureAuto", value)
        if ret != 0:
            _camera_logger.error("切換自動曝光模式失敗! ret[0x%x]", ret)
            return False
        _camera_logger.info(
            "自動曝光模式: %s", "開啟" if self.auto_exposure else "關閉"
        )
        return True

    def get_frame(self):
        try:
            self.frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self.start_time

            if elapsed_time >= 1.0:
                self.current_fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = current_time

            frame = self._get_frame_internal()
            if frame is not None:
                # NOTE: the frame is consumed by inference and color checks.
                # Never draw overlays (FPS text, status, etc.) on it here —
                # display decoration belongs to the preview layer, on a copy.
                # Validated on 2026-06-08 PCBA1 field images via
                # tools/validate_overlay_impact.py (26/26 outcomes unchanged).
                if self.save_image:
                    try:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(
                            self.save_path, f"captured_image_{timestamp}.jpg"
                        )
                        success = cv2.imwrite(filename, frame)
                        if success:
                            _camera_logger.info("影像已成功保存: %s", filename)
                        else:
                            _camera_logger.error("保存影像失敗: %s", filename)
                    except Exception as e:
                        _camera_logger.error("保存影像時發生錯誤: %s", e)
                    finally:
                        self.save_image = False

                return frame
            else:
                _camera_logger.warning("Capture failed; no frame returned.")
                return None
        except Exception as e:
            _camera_logger.error("獲取影像時發生錯誤: %s", e, exc_info=True)
            return None

    def _get_frame_internal(self):
        try:
            stOutFrame = MV_FRAME_OUT()

            ret = self.cam.MV_CC_GetImageBuffer(
                stOutFrame, self.config.MV_CC_GetImageBuffer_nMsec
            )
            if ret == 0:
                try:
                    frame_len = int(stOutFrame.stFrameInfo.nFrameLen)
                    height = int(stOutFrame.stFrameInfo.nHeight)
                    width = int(stOutFrame.stFrameInfo.nWidth)
                    if frame_len < height * width:
                        _camera_logger.error(
                            "Frame buffer smaller than expected: len=%d for %dx%d",
                            frame_len, width, height,
                        )
                        return None
                    # Zero-copy view into the SDK buffer; cvtColor below copies
                    # the pixels out before FreeImageBuffer returns the buffer
                    # to the SDK (saves a 6MB memcpy per frame at full res).
                    buf = np.ctypeslib.as_array(
                        stOutFrame.pBufAddr, shape=(frame_len,)
                    )
                    bayer_img = buf[: height * width].reshape((height, width))
                    rgb_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerRG2RGB)
                    # CHANNEL CONTRACT: this frame gets channel-swapped once
                    # more in CameraController.capture_frame, so the pipeline
                    # actually consumes RGB-as-BGR. All deployed models and
                    # color stats are trained/calibrated on that order —
                    # changing it flips outcomes (verified 2026-06-11 on the
                    # 2026-06-08 PCBA1/A field set: 12/22 results changed,
                    # 4 PASS became FAIL). Do NOT "fix" either conversion
                    # without retraining and revalidating every model.
                    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                    return bgr_img
                except Exception as e:
                    _camera_logger.error("處理影像時發生錯誤: %s", e, exc_info=True)
                    return None
                finally:
                    self.cam.MV_CC_FreeImageBuffer(stOutFrame)
            else:
                if ret == MV_E_NODATA:
                    _camera_logger.warning(
                        "No image data received from camera buffer "
                        "(ret[0x%x], timeout=%dms).",
                        ret,
                        self.config.MV_CC_GetImageBuffer_nMsec,
                    )
                    return None
                _camera_logger.error("獲取影像緩衝區失敗! ret[0x%x]", ret)
                return None
        except Exception as e:
            _camera_logger.error("獲取影像幀時發生錯誤: %s", e, exc_info=True)
            return None

    def process_key(self, key):
        """處理鍵盤輸入"""
        if key == ord("s"):  # 按's'保存圖片
            self.save_image = True
            print("準備保存下一幀影像...")
        elif key == ord("a"):  # 按'a'切換自動曝光
            self.toggle_auto_exposure()
        elif key == ord("q"):  # 按'q'退出
            return False
        return True

    def _basic_connect(self, device_index):
        # 將原始的連接代碼移到這個內部方法
        device_count = int(self.deviceList.nDeviceNum)
        if device_index < 0 or device_index >= device_count:
            # Casting pDeviceInfo[i] beyond nDeviceNum dereferences garbage
            # and crashes the whole process, not just this call.
            _camera_logger.error(
                "Invalid device index %d (devices enumerated: %d); "
                "call enum_devices() before connect_to_camera().",
                device_index,
                device_count,
            )
            return False

        stDeviceList = cast(
            self.deviceList.pDeviceInfo[device_index], POINTER(
                MV_CC_DEVICE_INFO)
        ).contents
        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            _camera_logger.error("創建相機句柄失敗! ret[0x%x]", ret)
            return False
        self._handle_created = True

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            _camera_logger.error("打開設備失敗! ret[0x%x]", ret)
            self.close()  # release the orphaned handle
            return False
        self._device_opened = True

        if not self._setup_initial_parameters():
            self.close()
            return False
        return True

    def _setup_initial_parameters(self):
        # 設定像素格式為 Bayer RG 8 (PixelType_Gvsp_BayerRG8)
        ret = self.cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_BayerRG8)
        if ret != 0:
            _camera_logger.error("設置像素格式失敗! ret[0x%x]", ret)
            return False

        # 獲取數據包大小
        stParam = MVCC_INTVALUE()
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            _camera_logger.error("獲取數據包大小失敗! ret[0x%x]", ret)
            return False
        self.nPayloadSize = stParam.nCurValue

        # 設定觸發模式為關閉
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            _camera_logger.error("設置觸發模式失敗! ret[0x%x]", ret)
            return False

        # 開始取流
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            _camera_logger.error("開始取流失敗! ret[0x%x]", ret)
            return False
        self._grabbing = True

        _camera_logger.info("相機連接成功!")
        return True

    def set_exposure_time(self, exposure_time):
        """設定曝光時間（微秒）"""
        ret = self.cam.MV_CC_SetFloatValue(
            "ExposureTime", float(exposure_time))
        if ret != 0:
            _camera_logger.error("設定曝光時間失敗! ret[0x%x]", ret)
            return False
        _camera_logger.info("已設定曝光時間為 %s 微秒", exposure_time)
        return True

    def get_exposure_time(self):
        """獲取當前曝光時間"""
        stParam = MVCC_FLOATVALUE()
        ret = self.cam.MV_CC_GetFloatValue("ExposureTime", stParam)
        if ret != 0:
            _camera_logger.error("獲取曝光時間失敗! ret[0x%x]", ret)
            return None
        return stParam.fCurValue

    def set_gain(self, gain):
        """設定增益"""
        ret = self.cam.MV_CC_SetFloatValue("Gain", float(gain))
        if ret != 0:
            _camera_logger.error("設定增益失敗! ret[0x%x]", ret)
            return False
        _camera_logger.info("已設定增益為 %s", gain)
        return True

    def get_gain(self):
        """獲取當前增益"""
        stParam = MVCC_FLOATVALUE()
        ret = self.cam.MV_CC_GetFloatValue("Gain", stParam)
        if ret != 0:
            _camera_logger.error("獲取增益失敗! ret[0x%x]", ret)
            return None
        return stParam.fCurValue

    def get_parameter_range(self, param_name):
        """獲取參數的有效範圍"""
        stParam = MVCC_FLOATVALUE()
        ret = self.cam.MV_CC_GetFloatValue(param_name, stParam)
        if ret != 0:
            _camera_logger.error("獲取參數範圍失敗! ret[0x%x]", ret)
            return None
        return {"current": stParam.fCurValue, "max": stParam.fMax, "min": stParam.fMin}

    def enum_devices(self):
        # The mock DLL answers MV_OK to every call, so a load failure would
        # otherwise surface as the misleading "no devices found". Report the
        # real root cause (missing/broken Runtime DLLs) instead.
        if _mvs_binding.MVCAM_DLL_LOAD_ERROR:
            _camera_logger.error(
                "Cannot enumerate cameras: %s", _mvs_binding.MVCAM_DLL_LOAD_ERROR
            )
            raise CameraConnectionError(
                "MVS SDK DLL 載入失敗，無法使用相機: "
                f"{_mvs_binding.MVCAM_DLL_LOAD_ERROR}"
            )
        transport_mask = MV_GIGE_DEVICE | MV_USB_DEVICE
        _camera_logger.info(
            "Enumerating Hikrobot devices (transport_mask=0x%x)", transport_mask
        )
        ret = MvCamera.MV_CC_EnumDevices(
            transport_mask, self.deviceList
        )
        if ret != 0:
            _camera_logger.error(
                "MV_CC_EnumDevices failed (ret=0x%x). Verify camera connection, driver service, and power.",
                ret,
            )
            return False
        if self.deviceList.nDeviceNum == 0:
            _camera_logger.warning(
                "MV_CC_EnumDevices succeeded but returned zero devices. Check cables, hubs, and whether another app is using the camera."
            )
            return False
        _camera_logger.info(
            "EnumDevices discovered %d device(s).", self.deviceList.nDeviceNum
        )
        return True

    def close(self):
        """Release SDK resources for the stages that were reached. Idempotent."""
        released_any = False
        if self._grabbing:
            ret = self.cam.MV_CC_StopGrabbing()
            if ret != 0:
                _camera_logger.warning(
                    "MV_CC_StopGrabbing failed: ret=0x%x", ret & 0xFFFFFFFF
                )
            self._grabbing = False
            released_any = True
        if self._device_opened:
            ret = self.cam.MV_CC_CloseDevice()
            if ret != 0:
                _camera_logger.warning(
                    "MV_CC_CloseDevice failed: ret=0x%x", ret & 0xFFFFFFFF
                )
            self._device_opened = False
            released_any = True
        if self._handle_created:
            ret = self.cam.MV_CC_DestroyHandle()
            if ret != 0:
                _camera_logger.warning(
                    "MV_CC_DestroyHandle failed: ret=0x%x", ret & 0xFFFFFFFF
                )
            self._handle_created = False
            released_any = True
        if released_any:
            _camera_logger.info("相機已安全關閉")

    def create_control_window(self):
        """創建參數調整視窗（根據支援的功能）"""
        cv2.namedWindow("Controls")

        # 基本參數（通常都支援）
        cv2.createTrackbar(
            "Exposure (us)",
            "Controls",
            5000,
            20000,
            lambda x: self.set_exposure_time(x),
        )
        cv2.createTrackbar("Gain", "Controls", 5, 15,
                           lambda x: self.set_gain(float(x)))

        if self.supported_features.get("Contrast", False):
            cv2.createTrackbar(
                "Contrast", "Controls", 100, 200, lambda x: self.set_contrast(
                    x)
            )


if __name__ == "__main__":
    from core.config import DetectionConfig

    config_path = "config.yaml"
    config = DetectionConfig.from_yaml(config_path)
    camera = MVSCamera(config)
    if camera.enum_devices():
        if camera.connect_to_camera():
            try:
                camera.create_control_window()

                print("\n控制說明:")
                print("'s': 保存當前影像")
                print("'a': 切換自動曝光模式")
                print("'q': 退出程式")

                while True:
                    frame = camera.get_frame()
                    if frame is not None:
                        # Display copy only — keep the captured frame clean.
                        display = cv2.resize(frame, (640, 640))
                        cv2.putText(
                            display,
                            f"FPS: {camera.current_fps:.1f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                        if camera.auto_exposure:
                            cv2.putText(
                                display,
                                "Auto Exposure: ON",
                                (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2,
                            )
                        cv2.imshow("Camera Frame", display)

                        key = cv2.waitKey(1) & 0xFF
                        if not camera.process_key(key):
                            break
                    else:
                        print("獲取影像失敗!")

            except KeyboardInterrupt:
                print("\n中斷執行，停止獲取影像")
            finally:
                camera.close()
                cv2.destroyAllWindows()
