# YoloVision

YoloVision 是一個使用 **YOLO** 模型與 **海康威視 MVS 相機** 進行即時目標檢測的範例程式。程式透過 `config.yaml` 進行設定，並可將檢測結果與相關影像存入 `Result/` 目錄下，同時使用 `logs/` 記錄過程。

## 主要功能

- 從 MVS 相機擷取影像並進行前處理
- 使用 Ultralytics YOLO 模型推論
- 根據產品與檢測區域比對預期項目
- 將結果、裁切影像與紀錄存檔

## 安裝

1. 安裝 Python 3.9 以上版本。
2. 安裝必要套件（示例）：
   ```bash
   pip install torch ultralytics opencv-python numpy pandas openpyxl
   ```
3. 將預訓練模型 `best.pt` 放置於專案根目錄。

## 設定檔說明

`config.yaml` 內包含模型路徑、裝置、影像大小、曝光設定與各產品區域應有的元件。例如：

```yaml
weights: "best.pt"
device: "cpu"
imgsz: [640, 640]
expected_items:
  PCBA1:
    A:
      - weld_1
      - weld_2
    B:
      - con1
```

可依需求調整各項參數及 `expected_items` 內容。

## 執行

直接執行 `Inference.py` 即可啟動推論流程：

```bash
python Inference.py
```

程式會依序讀取 `config.yaml`、連接相機、取得畫面並進行檢測。結果影像會輸出至 `Result/` 子目錄，Excel 紀錄檔儲存於同處。

## 目錄結構

- `Inference.py`：主推論程式。
- `MVS_camera_control.py`：MVS 相機操作與控制。
- `core/`：包含偵測、記錄與結果處理等模組。
- `config.yaml`：範例設定檔，可自行修改。
- `Result/`：存放偵測結果與紀錄。
- `logs/`：執行時的 log 檔案。

## 注意事項

- 使用前需安裝並正確設定相機 SDK。
- `best.pt` 及 `yolo11_best.pt` 為模型檔案，容量較大，請依實際需求選擇。
- 若需自訂測試流程，可修改 `Inference.py` 中 `main()` 的示例程式碼。

