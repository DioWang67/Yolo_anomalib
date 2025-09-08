# YOLO11 Inference (YOLO + Anomalib)

整合 YOLO 與 Anomalib 的檢測系統。支援依機種/區域自動切換模型、顏色檢測、結果落盤（影像 + Excel），並提供互動模式與一次性命令列模式。程式已模組化與管線化，便於擴充與維護。

## 特色

- 支援 YOLO 與 Anomalib 推理；依 `models/<product>/<area>/<type>/config.yaml` 自動切換。
- 顏色檢測（LEDQCEnhanced）：可對 YOLO 偵測框或整張圖檢查，結果寫入標註圖與 Excel。
- 結果落盤：原圖/預處理/標註/裁切與 Excel（立即 flush、不中斷即時觀察）。
- 結構化日誌：每次檢測都帶有 `product/area/type/request_id` 便於追蹤。
- 可插拔：模型管理、顏色檢測、結果輸出都已抽象，方便替換或新增。

## 環境需求與安裝

- Python 3.9+
- 建議使用虛擬環境（venv 或 conda）

安裝依賴：

```
pip install -r requirements.txt
```

硬體（可選）：

- 海康威視 MVS 相機驅動（若無相機，系統會自動使用模擬影像）

路徑基準（可選）：

- 透過環境變數 `YOLO11_ROOT` 指定專案根路徑；未設定則自動以程式檔案推導。

## 專案結構

```
core/
  config.py                # DetectionConfig（全域/合併設定）
  detection_system.py      # 系統編排（相機→推理→管線步驟）
  inference_engine.py      # 後端集合（YOLO/Anomalib）
  logger.py                # 日誌工具（簡化使用）
  logging_utils.py         # 結構化日誌 ContextAdapter/Filter
  models.py                # 資料模型（DetectionItem/ColorCheckResult 等）
  pipeline/
    context.py             # DetectionContext（管線上下文）
    steps.py               # ColorCheckStep / SaveResultsStep
  services/
    color_checker.py       # LEDQCEnhanced 服務封裝（顏色檢測）
    model_manager.py       # 模型設定載入 + LRU 快取 + 初始化
    result_sink.py         # 結果落盤（Excel + 影像）封裝
  yolo_inference_model.py  # YOLO 推理（前處理/偵測/位置檢核整合）
  anomalib_inference_model.py          # Anomalib 推理封裝
  anomalib_lightning_inference.py      # Anomalib Lightning 介面（ckpt/熱圖）
camera/
  camera_controller.py     # 相機抽象（MVS）
models/
  <product>/<area>/<type>/config.yaml  # 各模型設定（yolo 或 anomalib）
Result/                      # 輸出結果（自動建立）
logs/                        # 日誌（自動建立）
main.py                      # 薄入口（互動與命令列模式）
```

## 快速開始

互動模式：

```
python main.py
```

一次性命令列模式（可指定影像）：

```
python main.py --product LED --area A --type yolo --image path/to/img.jpg
python main.py --product LED --area A --type anomalib
```

輸出路徑（自動）：

```
Result/<YYYYMMDD>/<product>/<area>/<status>/
  original/<detector>/
  preprocessed/<detector>/
  annotated/<detector>/
  cropped/<detector>/
Result/<YYYYMMDD>/results.xlsx
logs/detection_YYYYMMDD.log
```

## 設定說明

全域設定（專案根 `config.yaml`）：

```
exposure_time: "5000"
gain: "0.0"
MV_CC_GetImageBuffer_nMsec: 10000
timeout: 1
width: 3072
height: 2048
enable_yolo: false
enable_anomalib: false
max_cache_size: 3
```

模型設定（`models/<product>/<area>/<type>/config.yaml`）：

YOLO 範例：

```
weights: "models/LED/A/yolo/LED_best.pt"
device: "cpu"            # 或 "cuda:0"
conf_thres: 0.30
iou_thres: 0.45
imgsz: [640, 640]
output_dir: Result
enable_yolo: true
enable_color_check: true
color_model_path: "models/LED/A/yolo/enhanced_model.json"   # 可相對於專案根目錄

backends: {}

expected_items:
  LED:
    A:
      - White
      - Red
      - Green
      - Blue

position_config:
  PCBA1:
    A:
      enabled: false
      mode: bbox
      tolerance: 2
```

Anomalib 範例（精簡）：

```
device: "cuda:0"
imgsz: [640, 640]
output_dir: Result
enable_anomalib: true
anomalib_config:
  output: "patchcore_outputs/YYYYMMDD"   # 輸出根目錄（自動帶入日期）
  data: { }                              # 畫素轉換等設定（依你的模型）
  models:
    LED:
      A:
        ckpt_path: "path/to/model.ckpt"
        threshold: 0.5

# 可選：管線化設定（步驟與參數）
pipeline: ["color_check", "save_results"]
steps:
  color_check:
    enabled: true
    # 可在此放自訂參數（目前步驟未使用，預留）
  save_results:
    # 亦可在此覆寫輸出相關參數（預留）
```

顏色模型（LEDQCEnhanced）：

- 使用「advanced JSON」格式（含 `colors/config` 等欄位、`avg_color_hist`），路徑可相對於專案根目錄。
- 顏色檢測會將結果（PASS/FAIL、diff 列表）標注在 YOLO 標註影像與 Excel 欄位。

## 架構總覽（擴充點）

- `DetectionSystem`：編排層（相機→推理→管線步驟），與具體實作解耦。
- `ModelManager`：讀取模型設定、初始化引擎、LRU 快取與釋放。
- `ColorCheckerService`：顏色檢測封裝，提供多框檢測與整圖檢測。
- `ExcelImageResultSink`：結果落盤（影像/Excel）。可新增 REST/DB Sink 而不動核心。
- `Pipeline Steps`：ColorCheckStep / SaveResultsStep。可新增 `PositionCheckStep`、`MeasurementStep` 等。
- `Pipeline 可組態`：在各模型 `config.yaml` 以 `pipeline` 決定步驟順序，`steps.*` 可放步驟參數。

位置檢查（可選）

- 將 `PositionValidator` 以步驟形式整合：

```
pipeline: ["position_check", "color_check", "save_results"]
steps:
  position_check:
    force: false   # 預設依 config.position_config 啟用；可設 true 強制執行
  color_check: {}
  save_results: {}
```

- `Logging`：`context_adapter` 讓日誌自帶 `product/area/type/request_id`。

## 常見問題（FAQ）

- 沒有相機可以跑嗎？
  - 可以。系統會用 640x640 的黑圖模擬影像，仍可驗證推理流程與輸出。
- Excel 沒立即更新？
  - 已在每次 `save()` 後呼叫 `flush()`。若仍無法寫入，請檢查 Excel 是否被其他程式鎖定。
- 顏色檢測沒生效？
  - 確認模型設定 `enable_color_check: true` 並提供合法的 `color_model_path`。
- 切換模型記憶體過大？
  - 調整 `max_cache_size`，LRU 會釋放較舊的模型並清理資源。

## 測試（可選）

專案內含部分單元測試（需安裝 pytest）：

```
pip install pytest
pytest -q tests
```

## 授權

內部專案或請依貴司策略設定授權條款。
