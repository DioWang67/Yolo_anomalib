# picture-tool

影像處理與 YOLO 自動化訓練/驗證工具，提供 CLI 與 PyQt GUI。涵蓋格式轉換、資料增強、分割、訓練/評估、批次推論、顏色檢測、位置驗證等流程，並內建任務預設組合。

## 主要功能
- 影像格式轉換、YOLO/一般影像增強、資料分割、資料品質檢查與增強預覽。
- YOLO11 訓練與評估（GPU/CPU 自動偵測），可匯出檢測設定。
- LED 顏色檢測：以 SAM 建模/多張/批次檢測，產出 JSON/CSV。
- 批次推論、報告生成，以及 PyQt GUI 控制面板。
- 任務管線可由 CLI、GUI 選擇或預設套用。

## 專案結構
```
Yolo11_auto_train/
├─ src/picture_tool/           # 程式主體
│  ├─ anomaly/ augment/ color/ format/ infer/ position/ split/ train/ ...
│  ├─ gui/                     # PyQt GUI
│  ├─ pipeline/                # 管線組裝與工具
│  ├─ resources/               # 內建範例設定
│  └─ main_pipeline.py         # CLI 入口
├─ configs/                    # 可覆蓋的設定 (default_pipeline.yaml, gui_presets.yaml)
├─ models/, data/, reports/, runs/ ...
├─ README.md, pyproject.toml, requirements-dev.txt
```

## 安裝
```bash
python -m pip install .
python -m pip install .[gui]        # 需要 PyQt5 GUI 時
```

開發環境建議：
```bash
python -m venv .venv
. .venv/Scripts/activate  # PowerShell
pip install -r requirements-dev.txt
pip install -e .[dev,gui]
```

## 快速開始
CLI 範例：
```bash
picture-tool-pipeline --config configs/default_pipeline.yaml --tasks full
```

GUI：
```bash
picture-tool-gui --config configs/gui_presets.yaml
```

顏色檢測（LED）：
```bash
picture-tool-color-verify \
  --input-dir data/led_qc/infer \
  --color-stats reports/led_qc/color_stats.json \
  --expected-map reports/led_qc/expected.csv
```

## Position Validation 使用指南
`position_validation` 任務用來檢查偵測中心是否落在預期範圍，會輸出 `position_validation.json`。

1) 設定檔（例如 `configs/default_pipeline.yaml` 或你的 config）填寫：
```yaml
yolo_training:
  position_validation:
    enabled: true
    product: Cable1            # 必填
    area: A                    # 必填
    config_path: ./models/yolo/position_config.yaml  # 或直接填 config: {...}
    sample_dir: ./data/split/val/images              # 選填，預設 dataset_dir/val/images
    weights: null              # 選填，預設 runs/detect/<name>/weights/best.pt
    output_dir: ./reports/position_validation        # 選填
    conf: 0.25                 # 選填
    device: auto               # 選填
    tolerance_override: null   # 選填，百分比
```
2) 確認 `yolo_training.project/name` 指向已訓練的 run 目錄 (預設 `runs/detect/train`) 且有 weights。
3) 執行任務：GUI 勾選「Position Validation」，或 CLI `--tasks position_validation`。
4) 結果輸出：`runs/detect/<name>/position_validation/position_validation.json`（或 `output_dir`），包含每張影像與整體摘要。

若 `enabled: false` 則會略過且不輸出。

## 任務與預設
- 任務 key（需與 `TASK_HANDLERS` 對應）：`format_conversion`, `anomaly_detection`, `yolo_augmentation`, `image_augmentation`, `dataset_splitter`, `yolo_train`, `yolo_evaluation`, `generate_report`, `dataset_lint`, `aug_preview`, `color_inspection`, `color_verification`, `batch_inference`, `position_validation`。
- GUI/CLI 可用 `configs/default_pipeline.yaml` 的 `pipeline.tasks` 或 `configs/gui_presets.yaml` 的 presets 來一次套用任務。

## 測試與開發
```bash
ruff check src tests
pytest --cov=picture_tool
python -m build
```

## 授權
預設為專案內標示的 Proprietary License；如需開源請同步更新 pyproject.toml 與 LICENSE。
