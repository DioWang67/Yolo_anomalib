
from yolov5.utils.general import scale_boxes
from MvImport.MvCameraControl_class import *
import numpy as np
import cv2
import time
import os 

class MVSCamera:
    def __init__(self):
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        self.cam = MvCamera()
        self.nPayloadSize = None
        self.supported_features = {}
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0
        self.save_image = False
        self.auto_exposure = False
        self.save_path = "captured_images"  # 新增：指定儲存路徑
        
        # 確保儲存目錄存在
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        
    def _check_feature_support(self, feature_name):
        """檢查相機是否支援特定功能"""
        try:
            stParam = MVCC_INTVALUE()
            ret = self.cam.MV_CC_GetEnumValue(feature_name, stParam)
            return ret == 0
        except:
            return False

    def _initialize_supported_features(self):
        """初始化相機支援的功能列表"""
        features_to_check = {
            'ExposureAuto': '自動曝光',
            'TriggerMode': '觸發模式'
        }
        
        self.supported_features = {}
        for feature, description in features_to_check.items():
            supported = self._check_feature_support(feature)
            self.supported_features[feature] = supported
            print(f"{description}: {'支援' if supported else '不支援'}")


    def connect_to_camera(self, device_index=0):
        """連接相機並初始化參數"""
        if not self._basic_connect(device_index):
            return False
            
        # 初始化支援的功能
        self._initialize_supported_features()
        
        # 設置默認參數
        try:
            # 關閉自動曝光
            if self.supported_features.get('ExposureAuto', False):
                self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)  # 0: 關閉
                print("已關閉自動曝光")

            # 設置曝光時間（例如 10000 微秒）
            self.set_exposure_time(10476)

            # 設置增益（例如 10.0 dB）
            self.set_gain(12.0)
            
        except Exception as e:
            print(f"初始化相機參數時發生錯誤: {str(e)}")
            return False

        print("相機連接成功且參數初始化完成!")
        return True

    def set_exposure_time(self, exposure_time):
        """設定曝光時間（微秒）"""
        if not self.supported_features.get('ExposureTime', False):
            return False
            
        exposure_range = self.get_parameter_range("ExposureTime")
        if exposure_range:
            exposure_time = max(min(exposure_time, exposure_range['max']), exposure_range['min'])
            
        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_time))
        if ret != 0:
            print(f"設定曝光時間失敗! ret[0x{ret:x}]")
            return False
        print(f"已設定曝光時間為 {exposure_time} 微秒")
        return True

    def toggle_auto_exposure(self):
        """切換自動曝光模式"""
        if not self.supported_features.get('ExposureAuto', False):
            print("此相機不支援自動曝光功能")
            return False
            
        self.auto_exposure = not self.auto_exposure
        value = 2 if self.auto_exposure else 0  # 2: Continuous, 0: Off
        ret = self.cam.MV_CC_SetEnumValue("ExposureAuto", value)
        if ret != 0:
            print(f"切換自動曝光模式失敗! ret[0x{ret:x}]")
            return False
        print(f"自動曝光模式: {'開啟' if self.auto_exposure else '關閉'}")
        return True

    def get_frame(self):
            """獲取影像並計算FPS"""
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
                    # 添加FPS和其他資訊到影像上
                    cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # 如果開啟了自動曝光，顯示標誌
                    if self.auto_exposure:
                        cv2.putText(frame, "Auto Exposure: ON", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # 如果設定了保存圖片標誌，保存當前幀
                    if self.save_image:
                        try:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = os.path.join(self.save_path, f"captured_image_{timestamp}.jpg")
                            success = cv2.imwrite(filename, frame)
                            if success:
                                print(f"影像已成功保存: {filename}")
                            else:
                                print(f"保存影像失敗: {filename}")
                        except Exception as e:
                            print(f"保存影像時發生錯誤: {str(e)}")
                        finally:
                            self.save_image = False  # 重置標誌，避免重複保存
                        
                return frame
            except Exception as e:
                print(f"獲取影像時發生錯誤: {str(e)}")
                return None

    def _get_frame_internal(self):
        """內部獲取影像的方法"""
        try:
            stOutFrame = MV_FRAME_OUT()
            ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if ret == 0:
                try:
                    pData = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
                    cdll.msvcrt.memcpy(byref(pData), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)
                    data = np.frombuffer(pData, count=int(stOutFrame.stFrameInfo.nFrameLen), dtype=np.uint8)
                    
                    # 重塑為原始 Bayer 格式影像
                    bayer_img = data.reshape((stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth))
                    
                    # 將 Bayer RG 格式轉換為 RGB
                    rgb_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerRG2RGB)
                    
                    return rgb_img
                except Exception as e:
                    print(f"處理影像時發生錯誤: {str(e)}")
                    return None
                finally:
                    self.cam.MV_CC_FreeImageBuffer(stOutFrame)
            else:
                print(f"獲取影像緩衝區失敗! ret[0x{ret:x}]")
            return None
        except Exception as e:
            print(f"獲取影像幀時發生錯誤: {str(e)}")
            return None

 

    def process_key(self, key):
        """處理鍵盤輸入"""
        if key == ord('s'):  # 按's'保存圖片
            self.save_image = True
            print("準備保存下一幀影像...")
        elif key == ord('a'):  # 按'a'切換自動曝光
            self.toggle_auto_exposure()
        elif key == ord('q'):  # 按'q'退出
            return False
        return True
    
    
    def _basic_connect(self, device_index):
        # 將原始的連接代碼移到這個內部方法
        stDeviceList = cast(self.deviceList.pDeviceInfo[device_index], POINTER(MV_CC_DEVICE_INFO)).contents
        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print("創建相機句柄失敗! ret[0x%x]" % ret)
            return False
        
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print("打開設備失敗! ret[0x%x]" % ret)
            return False
        
        return self._setup_initial_parameters()

    def _setup_initial_parameters(self):
        # 設定像素格式為 Bayer RG 8
        ret = self.cam.MV_CC_SetEnumValue("PixelFormat", 17301513)
        if ret != 0:
            print("設置像素格式失敗! ret[0x%x]" % ret)
            return False

        # 獲取數據包大小
        stParam = MVCC_INTVALUE()
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            print("獲取數據包大小失敗! ret[0x%x]" % ret)
            return False
        self.nPayloadSize = stParam.nCurValue

        # 設定觸發模式為關閉
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print("設置觸發模式失敗! ret[0x%x]" % ret)
            return False

        # 開始取流
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print("開始取流失敗! ret[0x%x]" % ret)
            return False

        print("相機連接成功!")
        return True

    def set_exposure_time(self, exposure_time):
        """設定曝光時間（微秒）"""
        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_time))
        if ret != 0:
            print(f"設定曝光時間失敗! ret[0x{ret:x}]")
            return False
        print(f"已設定曝光時間為 {exposure_time} 微秒")
        return True

    def get_exposure_time(self):
        """獲取當前曝光時間"""
        stParam = MVCC_FLOATVALUE()
        ret = self.cam.MV_CC_GetFloatValue("ExposureTime", stParam)
        if ret != 0:
            print(f"獲取曝光時間失敗! ret[0x{ret:x}]")
            return None
        return stParam.fCurValue

    def set_gain(self, gain):
        """設定增益"""
        ret = self.cam.MV_CC_SetFloatValue("Gain", float(gain))
        if ret != 0:
            print(f"設定增益失敗! ret[0x{ret:x}]")
            return False
        print(f"已設定增益為 {gain}")
        return True

    def get_gain(self):
        """獲取當前增益"""
        stParam = MVCC_FLOATVALUE()
        ret = self.cam.MV_CC_GetFloatValue("Gain", stParam)
        if ret != 0:
            print(f"獲取增益失敗! ret[0x{ret:x}]")
            return None
        return stParam.fCurValue

    def get_parameter_range(self, param_name):
        """獲取參數的有效範圍"""
        stParam = MVCC_FLOATVALUE()
        ret = self.cam.MV_CC_GetFloatValue(param_name, stParam)
        if ret != 0:
            print(f"獲取參數範圍失敗! ret[0x{ret:x}]")
            return None
        return {
            "current": stParam.fCurValue,
            "max": stParam.fMax,
            "min": stParam.fMin
        }


    def enum_devices(self):
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, self.deviceList)
        if ret != 0:
            print("列舉設備失敗! ret[0x%x]" % ret)
            return False
        if self.deviceList.nDeviceNum == 0:
            print("找不到設備!")
            return False
        print(f"找到 {self.deviceList.nDeviceNum} 台設備!")
        return True

    def close(self):
        self.cam.MV_CC_StopGrabbing()
        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()
        print("相機已安全關閉")

    def create_control_window(self):
        """創建參數調整視窗（根據支援的功能）"""
        cv2.namedWindow('Controls')
        
        # 基本參數（通常都支援）
        cv2.createTrackbar('Exposure (us)', 'Controls', 5000, 20000, 
                          lambda x: self.set_exposure_time(x))
        cv2.createTrackbar('Gain', 'Controls', 5, 15, 
                          lambda x: self.set_gain(float(x)))
        
            
        if self.supported_features.get('Contrast', False):
            cv2.createTrackbar('Contrast', 'Controls', 100, 200, 
                             lambda x: self.set_contrast(x))
if __name__ == "__main__":
    camera = MVSCamera()
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
                        frame = cv2.resize(frame, (640, 640))
                        cv2.imshow("Camera Frame", frame)
                        
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