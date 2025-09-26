# -*- coding: utf-8 -*-
from pathlib import Path

text = Path("GUI.py").read_text(encoding="utf-8")
old = '        # 狀態初始化\r\n        self.start_btn.setEnabled(False)\r\n        self.stop_btn.setEnabled(True)\r\n        self.status_widget.set_status("running")\r\n\r\n        # 清除圖像\r\n        self.original_image.clear()\r\n'
new = '        if not self.detection_system:\r\n            QMessageBox.critical(self, "偵測系統", "偵測系統尚未成功初始化，請重新載入設定或重啟應用程式")\r\n            self.init_system()\r\n            if not self.detection_system:\r\n                self.start_btn.setEnabled(True)\r\n                self.stop_btn.setEnabled(False)\r\n                self.status_widget.set_status("error")\r\n                return\r\n\r\n        # 狀態初始化\r\n        self.start_btn.setEnabled(False)\r\n        self.stop_btn.setEnabled(True)\r\n        self.status_widget.set_status("running")\r\n\r\n        # 清除圖像\r\n        self.original_image.clear()\r\n'
if old not in text:
    raise SystemExit("target block not found in start_detection")
Path("GUI.py").write_text(text.replace(old, new), encoding="utf-8")
