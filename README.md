# yolo11_inference

訓練資料回收不需要使用指令：在 GUI 選好產品／站別後點「檔案 → 訓練資料複核與提交」，或雙擊根目錄的 `一鍵蒐集訓練資料.bat`。可選擇預設或自訂時間範圍；畫面先以多圖總覽列出候選失敗案例，勾選後會進入只包含已選圖片的獨立畫面，未選圖片不會混入。選取會立即保存，可關閉後稍後再決定是否逐張分類。逐張分類可另記錄「閾值未達標」及來源（目前為 YOLO／顏色；未來 detector 可使用新的來源代碼），此失敗原因不會取代人工 OK／NG 判定或自行改變送訓路由。系統另提供 PASS 抽樣與「從已保存結果回報漏檢」。外部圖片必須先用目前產線模型檢測一次，避免錯誤產品、站別或類別進入補訓。逐張判定後可開啟「補訓清單」排除誤入資料；誤檢、漏檢及錯類案件會先進入內建 LabelImg，組長只需框選、選類別、按 `Ctrl+S` 並關閉工具。空白的漏檢標註不會被接受；全部驗證完成後才依序執行資料切分、訓練、同 test set 新舊模型品質比較與部署。資料不足或品質不合格時保留舊模型；部署成功後，下一次推理會自動載入新模型。

影像過曝、失焦、遮擋或取像失敗請選「影像過曝／模糊／遮擋」。此類資料會保留稽核紀錄但暫不送訓，也不會被當成需要補標的資料。

推論結果回訓、版本化部署與模型 reload 流程請見
[`../Yolo11_auto_train/docs/SEAMLESS_WORKFLOW.md`](../Yolo11_auto_train/docs/SEAMLESS_WORKFLOW.md)。

工業視覺檢測系統，整合 YOLO 物件偵測與 Anomalib 異常檢測，支援多產品/多站別的品質檢測流程。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

## 主要功能

- 🎯 **YOLO11 物件偵測**: 零件定位、缺件檢查、位置校驗
- 🔍 **Anomalib 異常檢測**: 表面刮傷、髒污、異物檢測
- ⚡ **Fusion 融合檢測**: YOLO 與 Anomalib 聯合推理，支援特徵熱圖與結果雙重疊加 (GUI 限定功能)
- 📷 **工業相機整合**: 支援海康威視 MVS SDK
- 🎨 **LED 顏色檢測**: 統計式顏色驗證
- 🧭 **顏色誤殺閉環**: 顏色專用覆核、校正資料分流及具名批准門檻發布（見 [操作說明](docs/COLOR_REVIEW_CALIBRATION.md)）
- 🖥️ **雙介面支援**: CLI 命令列 + PyQt5 GUI
- 📊 **結果管理**: Excel 報表輸出、影像標註保存
- 🔄 **多產品支援**: 靈活的產品/區域/類型配置體系
- 🚀 **非同步管線 (NEW)**: Producer-Consumer 三階段管線，解耦取像/推論/I/O，適用於高 FPS 產線

## 專案結構

```
yolo11_inference/
├── core/                       # 核心推理引擎
│   ├── yolo_inference_model.py        # YOLO 推理後端
│   ├── anomalib_inference_model.py    # Anomalib 推理後端
│   ├── detection_system.py            # 主編排器 (同步 + 非同步管線)
│   ├── types.py                       # 強型別資料結構 (DetectionResult, DetectionTask)
│   ├── queues.py                      # OverwriteQueue (FIFO + Drop-Oldest)
│   ├── workers.py                     # Pipeline Workers (Acquisition/Inference/Storage)
│   ├── detector.py                    # YOLO 偵測邏輯
│   ├── position_validator.py          # 位置校驗器
│   ├── services/                      # 服務層
│   │   ├── model_manager.py           # 模型管理 (LRU 快取)
│   │   ├── color_checker.py           # 顏色檢查服務
│   │   └── result_sink.py             # 結果持久化
│   └── pipeline/                      # 管道架構
│       ├── registry.py                # 步驟註冊
│       ├── steps.py                   # 處理步驟
│       └── context.py                 # 執行上下文
├── app/                        # 應用層
│   ├── cli.py                         # 命令列介面
│   └── gui/                           # PyQt5 圖形介面
├── camera/                     # 工業相機控制
│   ├── MVS_camera_control.py          # MVS SDK 封裝
│   └── camera_controller.py           # 相機控制器
├── tools/                      # 獨立工具
│   └── color_verifier.py              # LED 顏色檢測工具
├── tests/                      # 測試套件
├── models/                     # 模型權重目錄
│   └── <product>/
│       └── <area>/
│           ├── yolo/
│           │   └── config.yaml
│           └── anomalib/
│               └── config.yaml
├── Result/                     # 輸出結果
├── docs/                       # 文檔
│   ├── DOCUMENTATION_INDEX.md         # 文件入口索引
│   ├── TECH_GUIDE.md                  # 技術深度指南 (~1300 行)
│   ├── WINDOWS_DEPLOYMENT_SOP.md      # Windows 現場部署 SOP
│   └── RELEASE_ROLLBACK_SOP.md        # 發版與回滾 SOP
├── config.yaml                 # 全域配置
├── config.example.yaml         # 配置範本
├── requirements.txt            # 核心依賴
├── requirements-dev.txt        # 開發依賴
├── pyproject.toml              # 專案配置
└── README.md                   # 本文件
```

## 安裝

### 前置需求

- Python 3.10 或更高版本
- CUDA 12.1+ (若使用 GPU)
- 海康威視相機 SDK (若使用實體相機)

### 基本安裝

```bash
# 克隆專案
git clone <repository-url>
cd yolo11_inference

# 建立虛擬環境
python -m venv .venv

# 啟動虛擬環境
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux/macOS:
source .venv/bin/activate

# 安裝核心依賴
pip install -r requirements.txt

# (可選) 安裝 GUI 支援
pip install PyQt5==5.15.11

# (開發模式) 安裝開發工具
pip install -r requirements-dev.txt
pip install -e .
```

### 使用 pyproject.toml 安裝

```bash
# 僅核心功能
pip install -e .

# 包含 GUI
pip install -e .[gui]

# 包含開發工具
pip install -e .[dev,gui]
```

## 快速開始

### 1. 配置設定

複製配置範本並根據您的環境調整：

```bash
cp config.example.yaml config.yaml
# 編輯 config.yaml，設定模型路徑、相機參數等
```

### 2. 準備模型

將訓練好的 YOLO 模型放置到對應目錄：

```
models/
└── LED/                    # 產品名稱
    └── A/                  # 區域名稱
        └── yolo/           # 推理類型
            ├── config.yaml # 模型配置
            └── best.pt     # 模型權重
```

### 3. 執行推理

#### CLI 互動模式

```bash
python main.py
# 根據提示選擇產品、區域和推理類型
```

#### CLI 單次推理

```bash
# 使用相機拍照並推理
python main.py --product LED --area A --type yolo

# 使用指定影像推理
python main.py --product LED --area A --type yolo --image path/to/image.jpg
```

#### GUI 模式

```bash
python GUI.py
```

> **提示:** Fusion 融合推理模式（YOLO + Anomalib 聯合檢測）目前為 GUI 專屬功能。請於圖形介面中選取「Fusion 分析」選項以啟用雙模型疊加檢測。

### 4. 查看結果

結果將保存到 `Result/` 目錄（或 `config.yaml` 中指定的 `output_dir`）：

- 標註影像：`Result/<timestamp>_annotated.jpg`
- Excel 報表：`Result/detection_results.xlsx`

## 測試

```bash
# 執行所有測試
make test

# 快速測試（跳過 GUI）
make test-fast

# 執行特定測試
pytest tests/test_yolo_inference_model.py -v

# 產生覆蓋率報告
pytest --cov=core --cov=app --cov-report=html
```

## 開發

### 程式碼品質檢查

```bash
# Linting (ruff)
ruff check .

# 型別檢查 (mypy)
mypy core app

# 格式化
ruff format .
```

### 建構與發佈

我們提供一鍵打包腳本，將專案封裝為獨立的可執行檔 (EXE)，方便在未安裝 Python 環境的機台上部署：

```bash
# 執行包裹腳本（可用環境變數 YOLO11_PYTHON 覆蓋 Python 路徑）
build_exe.bat
```
封裝完成後，可執行檔會放置在 `dist\yolo11_inference` 目錄下。
您只需將該目錄複製到目標機台，執行裡面的 `yolo11_inference.exe` 即可啟動檢測系統。

打包後的 `yolo11_inference.exe` 入口來自 `GUI.py`，因此部署診斷參數
`--check-hikrobot-runtime`、`--check-camera-grab` 是封裝版 exe / `GUI.py`
支援的參數，不是 `python main.py` 的 CLI 參數。

Hikrobot 相機 DLL（`Runtime/`）已隨包附帶，目標機台**不需要**另行安裝 MVS；
可用以下命令做部署後預檢：

```powershell
.\yolo11_inference.exe --check-hikrobot-runtime
.\yolo11_inference.exe --check-camera-grab
```

請確保：
- 模型路徑與設定檔維持與打包時的相對路徑關係。

完整現場部署流程請看 `docs/WINDOWS_DEPLOYMENT_SOP.md`；
release / rollback 流程請看 `docs/RELEASE_ROLLBACK_SOP.md`。

## 配置說明

### 全域配置 (config.yaml)

主要配置項目：

| 配置項 | 說明 | 範例值 |
|-------|------|--------|
| `weights` | YOLO 模型權重路徑 | `models/LED/A/yolo/best.pt` |
| `enable_yolo` | 啟用 YOLO 推理 (CLI) | `true` |
| `enable_anomalib` | 啟用 Anomalib 推理 (CLI) | `false` |
| `max_cache_size` | 模型快取大小 (LRU) | `3` |
| `output_dir` | 結果輸出目錄 | `./Result` |
| `exposure_time` | 相機曝光時間 (μs) | `51170` |
| `gain` | 相機增益 | `23.0` |

完整配置範例請參考 `config.example.yaml`。

### 模型特定配置

每個產品/區域/類型可有獨立配置：

```yaml
# models/LED/A/yolo/config.yaml
imgsz: 640
conf_thres: 0.25
iou_thres: 0.45
device: "auto"

position_check:
  enabled: true
  tolerance_px: 10
  tolerance_pct: 0.05

expected_items:
  - J1
  - J2
  - LED1
```

### Advanced: Count + Sequence Checks (no retraining)

Configure a per-model pipeline to enforce strict counts and left-to-right order:

```yaml
# models/Cable1/A/yolo/config.yaml
expected_items:
  Cable1:
    A:
      - Red
      - Green
      - Orange
      - Yellow
      - Black
      - Black
      - Black

pipeline:
  - color_check
  - count_check
  - sequence_check
  - save_results

steps:
  count_check:
    strict: true           # true: extra items also FAIL
  sequence_check:
    expected: [Red, Green, Orange, Yellow, Black, Black]
    direction: left_to_right
```

Notes:
- `count_check` validates missing/extra counts using `expected_items`.
- `sequence_check` sorts detections by bbox center X and matches `expected`.

## 位置驗證 (Position Validation)

`position_validator` 用於檢查偵測物件的中心位置是否符合預期範圍。

### 配置範例

```yaml
# models/<product>/<area>/yolo/position_config.yaml
LED:
  A:
    J1:
      cx: 512
      cy: 384
      w: 64
      h: 48
      tolerance_px: 10      # 絕對容差 (像素)
      tolerance_pct: 0.05   # 相對容差 (5%)
```

### 驗證流程

1. YOLO 推理獲得偵測框
2. 計算每個偵測物件的中心座標
3. 與預期位置比對，檢查是否在容差範圍內
4. 輸出驗證報告 (JSON)

詳細說明請參考 `docs/TECH_GUIDE.md`。

## 從 Yolo11_auto_train 部署模型

本系統與 `Yolo11_auto_train` 配套使用。訓練完成後，執行以下步驟部署模型：

```bash
# 在 Yolo11_auto_train 目錄執行
picture-tool-pipeline --config configs/<product>.yaml --tasks deploy
```

或手動複製：

```bash
mkdir -p models/<product>/<area>/yolo

# 訓練產物路徑：runs/detect/<name>/
cp runs/detect/<name>/weights/best.pt           models/<product>/<area>/yolo/best.pt
cp runs/detect/<name>/detection_config.yaml     models/<product>/<area>/yolo/config.yaml
cp runs/detect/<name>/auto_position_config.yaml models/<product>/<area>/yolo/position_config.yaml
```

完整的訓練→部署流程說明請參考 `Yolo11_auto_train/docs/INTEGRATION_GUIDE.md`。

---

## 文檔

建議從 [文件入口索引](docs/DOCUMENTATION_INDEX.md) 開始。常用文件：

| 類別 | 文件 |
|------|------|
| 技術總覽 | [技術深度指南](docs/TECH_GUIDE.md) |
| 模組責任 | [模組架構說明](docs/MODULE_ARCHITECTURE.md) |
| Windows 現場部署 | [Windows Deployment SOP](docs/WINDOWS_DEPLOYMENT_SOP.md) |
| 發版與回滾 | [Release and Rollback SOP](docs/RELEASE_ROLLBACK_SOP.md) |
| PCBA pilot | [PCBA Pilot Runbook](docs/PCBA_PILOT_RUNBOOK.md) |
| 操作員命令 | [PCBA Operator Commands](docs/PCBA_OPERATOR_COMMANDS.md) |
| 上線檢查 | [Production Go-Live Checklist](docs/PRODUCTION_GO_LIVE_CHECKLIST.md) |
| 相機診斷 | [Camera Runtime Diagnostics](docs/CAMERA_RUNTIME_DIAGNOSTICS.md) |
| 模型版本 | [Model Version Management Guide](docs/MODEL_VERSION_GUIDE.md) |
| 安全 | [Security Guide](docs/SECURITY.md) |

目前 PCBA 文件支援 controlled pilot；若要 unattended production，仍需完成
golden board、known NG、dry run review、readiness WARN 接受/修正與 rollback
記錄。

## 常見問題

### Q: 如何添加新產品？

```bash
# 1. 建立目錄結構
mkdir -p models/<new_product>/<area>/yolo

# 2. 放置模型權重
cp your_model.pt models/<new_product>/<area>/yolo/best.pt

# 3. 建立配置檔案
cp config.example.yaml models/<new_product>/<area>/yolo/config.yaml
# 編輯 config.yaml 調整參數

# 4. 執行推理
python main.py --product <new_product> --area <area> --type yolo
```

### Q: 如何優化推理速度？

1. **使用 GPU**: 確保 CUDA 可用
2. **混合精度**: `config.yaml` 中啟用 FP16
3. **非同步管線**: 使用 `start_pipeline()` 解耦取像與推論（見下方說明）
4. **TensorRT**: 匯出模型為 TensorRT 引擎（進階）

詳見 `docs/TECH_GUIDE.md` 第 8 節「效能工程手冊」。

## 非同步管線 (Producer-Consumer Pipeline)

適用於高產能、高 FPS 的產線環境。將同步阻塞流程拆分為三個獨立執行緒：

```
┌──────────────────┐    OverwriteQueue    ┌──────────────────┐    stdlib Queue    ┌──────────────────┐
│ AcquisitionWorker│───────────────────▶│ InferenceWorker  │──────────────────▶│  StorageWorker   │
│   (相機取像)      │   drop-oldest      │   (模型推論)      │   保證不丟      │   (Excel/影像)    │
└──────────────────┘                    └──────────────────┘                    └──────────────────┘
```

### 使用方式

```python
from core.detection_system import DetectionSystem

system = DetectionSystem("config.yaml")

# 啟動非同步管線
system.start_pipeline(
    product="LED",
    area="A",
    inference_type="yolo",
    capture_interval=0.0,  # 0 = 全速取像
)

# 監控管線狀態
stats = system.pipeline_stats()
print(f"已擷取: {stats['frames_captured']}")
print(f"已丟棄: {stats['frames_dropped']}")
print(f"已儲存: {stats['tasks_saved']}")

# 安全停止（毒藥丸逐級傳遞 → 殘留 I/O 全清空）
system.stop_pipeline()
```

### 邊界條件防禦

| 威脅 | 機制 | 說明 |
|------|------|------|
| Queue 堆積 OOM | `OverwriteQueue(maxlen=N)` | 自動丟棄最舊幀 |
| 殭屍幀 | FIFO + Drop-Oldest | 推論端永遠處理最新影像 |
| 關機資料遺失 | Poison Pill 機制 | `StorageWorker` 確保殘留 I/O 全寫入 |
| Worker 崩潰死鎖 | `try-finally` 保證傳遞 | 即使推論崩潰也不卡死管線 |

### 相關配置項

| 配置項 | 說明 | 預設值 |
|--------|------|--------|
| `buffer_limit` | OverwriteQueue 容量 (maxlen) | `10` |
| `timeout` | 推論超時秒數 | `2` |

### Q: 測試失敗怎麼辦？

```bash
# 檢查依賴版本
pip list

# 重新安裝依賴
pip install -r requirements.txt --force-reinstall

# 執行單一測試並查看詳細輸出
pytest tests/test_yolo_inference_model.py -v -s
```

## 安全性

本專案實作了多層安全機制，確保生產環境的穩定性與安全性。

### 路徑安全驗證

**防止目錄遍歷攻擊** (Directory Traversal Protection)：

- 自動驗證所有文件路徑（配置、模型、影像、輸出）
- 阻擋 `../` 等路徑穿越嘗試
- 白名單式訪問控制

```python
# 範例：使用路徑驗證器
from core.security import path_validator, SecurityError

try:
    safe_path = path_validator.validate_path(user_input, must_exist=True)
    # 安全地使用 safe_path
except SecurityError as e:
    logger.error(f"路徑驗證失敗: {e}")
```

### YAML 安全載入

所有 YAML 配置使用 `yaml.safe_load()` 防止任意程式碼執行：

- ✅ `core/config.py` - 全局配置
- ✅ `core/services/model_manager.py` - 模型配置
- ✅ `core/detection_system.py` - 位置配置

### 依賴安全

- 固定版本依賴（342 行 `requirements.txt`）
- 定期安全掃描與更新
- 使用 `pip-compile` 確保可重現構建

### 更多資訊

詳細安全指南請參考：
- **[docs/SECURITY.md](docs/SECURITY.md)** - 完整安全指南
- **[CHANGELOG.md](CHANGELOG.md)** - 安全相關變更記錄
- **測試**: `tests/test_security.py` (12/13 測試通過)



## 授權

Proprietary License - 專有授權，未經許可不得分發或使用。

## 致謝

本專案使用以下開源套件：
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Anomalib](https://github.com/openvinotoolkit/anomalib)
- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://lightning.ai/)

## Local Environment and Fusion Notes

The Windows startup scripts in this repository activate the `yolo_anomalib`
conda environment. Use that environment for YOLO/Anomalib development and
verification on this machine:

```powershell
conda activate yolo_anomalib
python -c "import torch, jsonargparse; print(torch.__version__)"

# Or call the environment Python directly.
D:\miniconda\envs\yolo_anomalib\python.exe -m pytest tests\test_fusion_inference.py
```

If tests fail with `torch` DLL or missing `jsonargparse` errors, confirm that
the active Python is not the conda `base` environment.

Use the GUI Fusion mode or call `DetectionSystem.detect(..., "fusion", ...)`
when a product/area has both YOLO and Anomalib models available:

```text
models/<product>/<area>/yolo/
models/<product>/<area>/anomalib/
```

Fusion runs YOLO and Anomalib concurrently, merges their detections/status, and
draws YOLO boxes over the Anomalib result frame when possible. If the Anomalib
engine is not loaded, fusion falls back to YOLO-only inference. The current
`main.py --type` CLI accepts only `yolo` and `anomalib`; use GUI/API for fusion.

Model-level color checker overrides can be placed in
`models/<product>/<area>/<inference_type>/config.yaml`:

```yaml
color_threshold_overrides:
  red: 0.91
  blue: 0.88

color_rules_overrides:
  red:
    min_area: 3
```

These values override global color-check settings for that model only. If the
file is missing, invalid, or PyYAML is unavailable, the system falls back to the
global configuration.

## Firmware / Edge Runtime Notes

For firmware or constrained edge deployment, keep Python training/validation
separate from runtime artifacts. YOLO can now load PyTorch `.pt`, ONNX files and
OpenVINO export directories through the same inference wrapper; exported
runtimes skip PyTorch-only setup such as `model.to()`, `fuse()` and FP16 module
conversion.

Export YOLO artifacts from the training project before changing production
configs. This GUI/runtime project should only consume and measure those
artifacts:

```powershell
cd D:\Git\robotlearning\yolo11_workspace\Yolo11_auto_train
picture-tool-pipeline --config configs\<product>.yaml --tasks yolo_train,deploy

cd D:\Git\robotlearning\yolo11_workspace\yolo11_inference
python tools\runtime_benchmark.py `
  --backend yolo `
  --model models\Cable1\A\yolo\weights\best.pt `
  --images path\to\images `
  --device cpu `
  --runs 50
```

See `docs/FIRMWARE_RUNTIME_PLAN.md` for the benchmark matrix and acceptance
criteria. Anomalib should remain a training/validation framework until an
exported runtime is proven equivalent to the current Lightning baseline.
