# YOLO11 Inference (YOLO + Anomalib)

## 專案簡介
YOLO11 Inference 將 YOLO 目標偵測與 Anomalib 異常偵測整合為單一檢測系統，
在同一個管線中支援多產品、多站點、多種模型格式，並同步產出 Excel 與影像紀錄。
系統以抽象化的服務層與管線步驟為核心，方便快速擴充新模型、客製化後處理或異常判定規則。

## 快速開始
1. 建立虛擬環境並啟用：
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. 安裝套件：
   ```bash
   pip install -r requirements.txt
   ```
3. 取得指令說明或啟動互動模式：
   ```bash
   python main.py --help
   python main.py
   ```

> 選用依賴：若需串接海康威視相機，請安裝 MVS 驅動程式；若需自動推導模型路徑，可以設定環境變數 `YOLO11_ROOT`。

## 依賴與安裝注意
- **PyQt5**：若需使用 GUI，請額外安裝 `pip install PyQt5==5.15.11`。
- **海康威視 MVS SDK**：請依官方說明安裝驅動與 `MvImport` Python 綁定，並確認 SDK 的 Python 路徑已加入 `PYTHONPATH`。
- **測試工具**：開發環境建議安裝 `pytest` 以執行 `tests/` 單元測試。

## 功能亮點
- 同時支援 YOLO 與 Anomalib 推論，並能依產品/站別切換模型。
- 透過 `models/<product>/<area>/<type>/config.yaml` 管理模型設定，採 LRU 快取減少重複載入。
- 內建 LEDQCEnhanced 顏色檢測，可將結果寫入 Excel 與標註圖。
- 可選用的位置檢查步驟，確保瑕疵品能依座標判定 FAIL。
- Excel 與影像輸出自動分門別類，並於日誌附帶 `product/area/type/request_id` 便於追蹤。
- 模型載入、偵測、結果落地皆以服務抽象化，利於擴充 REST、資料庫或自訂步驟。

## 執行模式
- **互動 CLI**：執行 `python main.py`，依提示選擇產品、站別、推論類型。
- **一次性命令**：以參數指定模型與影像，例如：
  ```bash
  python main.py --product LED --area A --type yolo --image path/to/img.jpg
  python main.py --product LED --area A --type anomalib
  ```
- **GUI（選用）**：若需視覺化操作可執行 `python GUI.py`，介面依專案需求客製。
  - GUI 會在影像檔尚未完成寫入時自動重試載入，若長時間仍顯示「尚無影像」，請檢查 `save_*` 設定或磁碟權限。

## 專案架構
```
app/
  cli.py                   # 互動式命令列介面
camera/
  camera_controller.py     # 海康威視相機封裝（若無相機可退回假影像）
core/
  config.py                # 全域 DetectionConfig 讀取與合併
  detection_system.py      # 偵測主控，負責推論與管線協調
  inference_engine.py      # 後端推論引擎（YOLO / Anomalib）
  logger.py                # 統一的偵測 Logger 封裝
  logging_utils.py         # 日誌 context adapter / filter
  models.py                # Domain models（DetectionItem、ColorCheckResult…）
  pipeline/
    context.py             # 管線執行時狀態容器
    steps.py               # 預設步驟：ColorCheck、SaveResults、PositionCheck
  services/
    color_checker.py       # LEDQCEnhanced 顏色檢測服務
    model_manager.py       # 模型設定載入 + LRU 快取
    result_sink.py         # Excel 與影像輸出落地
  position_validator.py    # 位置檢查與評分工具
  result_adapter.py        # 推論結果正規化為統一格式
GUI.py                     # 視覺化操作介面（選用）
logs/                      # 日誌輸出（執行時自動建立）
models/                    # 產品/站別/類型對應的模型設定
Result/                    # 偵測成果輸出（執行時自動建立）
requirements.txt           # Python 套件相依
```

## 偵測流程（Pipeline）
1. `DetectionSystem` 讀取全域設定並初始化相機、模型管理器與結果匯出器。
2. 依 `product/area/type` 讀取模型設定，透過 `ModelManager` 切換 YOLO/Anomalib 引擎。
3. 執行推論，`result_adapter` 將模型輸出轉為標準欄位（detections、missing_items、anomaly_score 等）。
4. 依設定串接管線步驟（位置檢查 → 顏色檢查 → 結果落地）；步驟可在模型 `config.yaml` 中調整、擴充。
5. `ExcelImageResultSink` 產生原圖、標註圖、裁切圖與 `results.xlsx`；日誌同時記錄檢測摘要。

## 設定指引
### 全域 `config.yaml`
| 欄位 | 說明 | 預設 |
| --- | --- | --- |
| `exposure_time` / `gain` | 相機曝光與增益設定 | `5000` / `0.0` |
| `MV_CC_GetImageBuffer_nMsec` | 相機取像 timeout（ms） | `10000` |
| `timeout` | 偵測流程等待秒數 | `1` |
| `width` / `height` | 影像解析度 | `3072` / `2048` |
| `enable_yolo` / `enable_anomalib` | 是否預載各類模型 | `false` |
| `max_cache_size` | 模型 LRU 快取上限 | `3` |
| `output_dir` | 結果輸出根目錄 | `Result` |

> 若部署環境無相機，可將相機相關欄位保留預設值，系統會自動改用輸入影像或假影像。

### 模型設定 `models/<product>/<area>/<type>/config.yaml`
- **YOLO** 主要欄位：
  - `weights`：YOLO 權重路徑（建議使用專案相對路徑）。
  - `device`：`cpu` 或 `cuda:0`。
  - `conf_thres` / `iou_thres`：偵測門檻調整。
  - `imgsz`：推論輸入大小。
  - `enable_color_check`、`color_model_path`：開啟 LEDQCEnhanced 與色彩模型檔案。
  - `expected_items`：定義每站應出現的元件名稱，缺失會標記在結果中。
  - `position_config`：每個工作站的座標容忍度設定，可搭配 `PositionCheckStep`。
- **Anomalib** 主要欄位：
  - `ckpt_path`：Lightning checkpoint 路徑。
  - `threshold`：異常分數門檻，決定 PASS/FAIL。
  - `anomalib_config.output`：Anomalib 模型原生輸出目錄，系統會自動搬移至結果資料夾。
  - 其餘欄位可覆寫輸入大小、預處理及自訂 pipeline。

### 管線覆寫範例
```yaml
pipeline: ["position_check", "color_check", "save_results"]
steps:
  position_check:
    force: false          # 若全域設定未啟用，可於此強制啟用
  color_check:
    max_log_items: 10     # 自訂顏色檢測的日誌輸出數量
  save_results:
    output_dir: "Result"  # 覆寫輸出資料夾
```
若需新增步驟，可在 `core/pipeline/steps.py` 撰寫 Step 類別，並透過 `core/pipeline/registry.register_step()` 註冊後在模型設定加入名稱與參數；預設已內建 `color_check`、`position_check`、`save_results`。
範例：
```python
from core.pipeline.registry import PipelineEnv, register_step
from core.pipeline.steps import Step

class DbSinkStep(Step):
    def run(self, ctx):
        # 將結果寫入資料庫
        pass

register_step("db_sink", lambda env, options: DbSinkStep())
```

## 輸出結果與日誌
```
Result/<YYYYMMDD>/<product>/<area>/<status>/
  original/<detector>/      # 原始影像
  preprocessed/<detector>/  # 預處理影像（若有）
  annotated/<detector>/     # 加上標註或熱度圖的影像
  cropped/<detector>/       # 依偵測框裁切的局部圖
Result/<YYYYMMDD>/results.xlsx  # 每次偵測追加一列紀錄
logs/detection_YYYYMMDD.log      # 日誌附帶 product/area/type/request_id
```
> 備註：影像寫入採用背景佇列，佇列接近滿載或發生錯誤時會記錄警告訊息，關閉系統時亦會輸出佇列統計以便除錯。
Excel 會在每次 `save()` 後呼叫 `flush()`，避免因中途中斷而遺失資料。

## 模型與部署建議
- 將新模型放入 `models/<product>/<area>/<type>/`，並提供對應 `config.yaml`。
- 安裝新模型後，可先執行 `python main.py --product ... --type ... --image ...` 以靜態影像驗證。
- 若需要共用模型資料夾，可利用 `YOLO11_ROOT` 指向外部儲存路徑；程式會以此為基準尋找模型與色彩檔案。
- 異常偵測模型若輸出大量檔案，建議定期清理 `patchcore_outputs` 等暫存目錄。

## FAQ
**Q：執行時顯示模型或檔案不存在？**  
A：確認 `models/<product>/<area>/<type>/config.yaml` 是否存在，路徑是否使用專案根目錄的相對路徑；必要時設定 `YOLO11_ROOT`。

**Q：顏色檢測沒有生效？**  
A：確認模型設定內 `enable_color_check: true`，並提供正確的 `color_model_path`。可透過 `steps.color_check.max_log_items` 增加日誌資訊。

**Q：Excel 沒有更新或被鎖定？**  
A：系統會自動 `flush()`，若仍然無法寫入，請確認檔案沒有被其他應用程式開啟。

**Q：記憶體占用過高？**  
A：調整全域設定 `max_cache_size`，降低同時緩存的模型數量；不常用的產品可在需求時再載入。

## 測試
專案已配置 pytest 測試套件，可於虛擬環境中執行：
```bash
pip install pytest
pytest -q tests
pytest tests/test_pipeline_registry.py
```
若新增管線步驟或服務，建議撰寫對應單元測試確保輸入輸出格式正確。

## 授權
本專案屬於內部檢測系統，僅供授權人員使用，未經許可請勿散佈或公開程式碼與模型。

