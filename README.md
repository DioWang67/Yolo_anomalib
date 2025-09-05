# YoloVision

YoloVision 是示範如何整合 **YOLO** 與 **Anomalib** 進行產品檢測的範例專案。系統會依據 `config.yaml` 與 `models/` 內的設定進行推論，並將結果輸出至 `Result/` 目錄與 `logs/` 日誌檔。

## 主要功能

- 從海康威視 MVS 相機擷取影像，無相機時使用模擬圖像。
- 支援 YOLO 與 Anomalib 推論模式，可依產品與區域切換。
- 儲存原始圖像、前處理圖像、標註結果、裁切區塊與 Excel 紀錄。

## 安裝

1. 安裝 Python 3.9 以上版本。
2. 建議使用虛擬環境（venv 或 conda）。
3. 安裝依賴套件：

   ```bash
   pip install -r requirements.txt
   ```

4. 將 YOLO 權重檔與 Anomalib 模型檔放入 `models/` 對應資料夾。

## 設定

### 全域設定：`config.yaml`

設定相機參數與預設選項，例如曝光、解析度與是否啟用 YOLO 或 Anomalib。

```yaml
exposure_time: "20000"
gain: "15.0"
enable_yolo: false
enable_anomalib: false
enable_color_detection: false
color_model_path: ""
```

若要啟用顏色檢測，請將 `enable_color_detection` 設為 `true`，並在 `color_model_path` 指定顏色模型的路徑。

### 模型設定

模型檔案位於 `models/<產品>/<區域>/<推論類型>/config.yaml`。依需求提供 YOLO 或 Anomalib 設定：

**YOLO 範例：**

```yaml
weights: "yolo_best.pt"
device: "cuda:0"
conf_thres: 0.25
iou_thres: 0.45
imgsz: [640, 640]
expected_items:
  A:
    - weld_1
    - weld_2
```

**Anomalib 範例：**

```yaml
device: "cuda:0"
imgsz: [640, 640]
width: 640
height: 640
anomalib_config: "path/to/anomalib/model/config.yaml"
```

**顏色檢測範例：**

```yaml
color_model_path: "path/to/color_model.pt"
```

## 執行

以 `main.py` 啟動系統：

```bash
python main.py
```

步驟：

1. 輸入機種名稱。
2. 輸入 `區域,推論類型`，例如 `A,yolo` 或 `B,anomalib`。
   若要進行顏色檢測，請輸入 `A,color`，並在 `config.yaml` 中啟用 `enable_color_detection` 並設定 `color_model_path`。
3. 程式會載入對應設定並輸出檢測結果。

## 檔案輸出

所有結果儲存在 `Result/` 目錄，下層依日期與結果狀態（PASS、FAIL）分類，結構如下：

```
Result/<日期>/<狀態>/
 ├─ original/          # 原始影像
 ├─ preprocessed/      # 前處理影像
 ├─ annotated/         # 標註或熱圖
 └─ cropped/           # YOLO 裁切圖像
```

- `results.xlsx`：每次檢測的統計資訊。
- `logs/detection_YYYYMMDD.log`：日誌紀錄。

YOLO 模式會輸出偵測框與裁切影像；Anomalib 模式會輸出異常熱圖與異常分數。

## 注意事項

- 必須先安裝並設定相機 SDK，否則將改用模擬圖像。
- 模型檔案需自行準備並放置於 `models/` 對應位置。
- 若需自訂流程，可修改 `main.py` 內部邏輯。

