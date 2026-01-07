# yolo11_inference

工業視覺檢測系統，整合 YOLO 物件偵測與 Anomalib 異常檢測，支援多產品/多站別的品質檢測流程。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

## 主要功能

- 🎯 **YOLO11 物件偵測**: 零件定位、缺件檢查、位置校驗
- 🔍 **Anomalib 異常檢測**: 表面刮傷、髒污、異物檢測
- 📷 **工業相機整合**: 支援海康威視 MVS SDK
- 🎨 **LED 顏色檢測**: 統計式顏色驗證
- 🖥️ **雙介面支援**: CLI 命令列 + PyQt5 GUI
- 📊 **結果管理**: Excel 報表輸出、影像標註保存
- 🔄 **多產品支援**: 靈活的產品/區域/類型配置體系

## 專案結構

```
yolo11_inference/
├── core/                       # 核心推理引擎
│   ├── yolo_inference_model.py        # YOLO 推理後端
│   ├── anomalib_inference_model.py    # Anomalib 推理後端
│   ├── detection_system.py            # 主編排器
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
├── tests/                      # 測試套件 (52 個測試)
├── models/                     # 模型權重目錄
│   └── <product>/
│       └── <area>/
│           ├── yolo/
│           │   └── config.yaml
│           └── anomalib/
│               └── config.yaml
├── Result/                     # 輸出結果
├── docs/                       # 文檔
│   └── TECH_GUIDE.md                  # 技術深度指南 (1153 行)
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

### 建構與發布

```bash
# 建構套件
python -m build

# 上傳到 PyPI (若開源)
twine upload dist/*
```

## 配置說明

### 全域配置 (config.yaml)

主要配置項目：

| 配置項 | 說明 | 範例值 |
|-------|------|--------|
| `weights` | YOLO 模型權重路徑 | `models/LED/A/yolo/best.pt` |
| `enable_yolo` | 啟用 YOLO 推理 | `true` |
| `enable_anomalib` | 啟用 Anomalib 推理 | `false` |
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

## 文檔

- 📖 [技術深度指南](docs/TECH_GUIDE.md) - 1153 行從 JR 到 SR 的完整教學
- 📝 配置範本：`config.example.yaml`
- 🧪 測試範例：`tests/` 目錄

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
3. **批次推理**: 對多張影像使用批次處理（進階）
4. **TensorRT**: 匯出模型為 TensorRT 引擎（進階）

詳見 `docs/TECH_GUIDE.md` 第 8 節「效能工程手冊」。

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
