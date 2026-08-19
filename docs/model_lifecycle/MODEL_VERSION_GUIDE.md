# Model Version Management Guide

## Overview
本專案採用 **Git LFS + 語義化版本命名** 策略管理模型權重，確保版本可追溯、可回滾。

---

## 版本命名規範

### 檔案命名格式
```
{model_name}_v{major}.{minor}.{patch}_{YYYYMMDD}.pt

範例:
- LED_best_v1.0.0_20260201.pt
- Cable_detector_v1.1.0_20260208.pt
- PCBA_anomaly_v2.0.0_20260210.pt
```

### 版本號規則 (Semantic Versioning)
- **Major (X.0.0)**: 架構變更、不向後相容
  - 範例: YOLOv11 → YOLOv12
- **Minor (0.X.0)**: 新增類別、改善準確率 (向後相容)
  - 範例: 新增 LED_Green 類別
- **Patch (0.0.X)**: Bug 修復、微調參數
  - 範例: 調整 conf_threshold

---

## 目錄結構

### 舊結構 (已棄用)
```
models/
└── LED/A/yolo/
    ├── best.pt          ❌ 無版本資訊
    └── config.yaml
```

### 新結構 (推薦)
```
models/
└── LED/A/yolo/
    ├── weights/
    │   ├── LED_best_v1.0.0_20260201.pt
    │   ├── LED_best_v1.1.0_20260208.pt
    │   └── LED_best_v1.2.0_20260210.pt  ← 最新版本
    └── config.yaml  # 指定當前版本
```

---

## config.yaml 配置

### 指定模型版本
```yaml
# models/LED/A/yolo/config.yaml
model_version: "1.2.0"  # 僅需版本號
# 或完整路徑覆蓋
weights_path: "weights/LED_best_v1.2.0_20260210.pt"

# (可選) 最低相容版本
min_supported_version: "1.0.0"
```

### 全域配置 (向後相容)
```yaml
# config.yaml (根目錄)
weights: "models/LED/A/yolo/weights/LED_best_v1.2.0_20260210.pt"
# 或使用版本指定
model_version: "1.2.0"
```

---

## Git LFS 配置

### 初始化 (首次設置)
```bash
# 1. 安裝 Git LFS
git lfs install

# 2. 追蹤模型檔案
git lfs track "models/**/*.pt"
git lfs track "models/**/*.pth"

# 3. 提交 .gitattributes
git add .gitattributes
git commit -m "chore: configure Git LFS for model weights"
```

### 驗證 LFS 狀態
```bash
# 查看 LFS 追蹤的檔案
git lfs ls-files

# 查看 LFS 儲存資訊
git lfs env
```

---

## 遷移指南

### 步驟 1: 備份現有模型
```bash
# 備份到 models_backup/
cp -r models/ models_backup/
```

### 步驟 2: 重新組織結構
```bash
# 為每個產品/區域建立 weights/ 目錄
mkdir -p models/LED/A/yolo/weights
mkdir -p models/Cable/B/yolo/weights

# 移動並重命名模型
mv models/LED/A/yolo/best.pt \
   models/LED/A/yolo/weights/LED_best_v1.0.0_$(date +%Y%m%d).pt
```

### 步驟 3: 更新 config.yaml
```yaml
# models/LED/A/yolo/config.yaml
model_version: "1.0.0"  # 初始版本
```

### 步驟 4: 提交變更
```bash
git add models/
git commit -m "refactor: migrate to versioned model management"
git push
```

---

## 使用範例

### 發布新版本
```bash
# 1. 訓練新模型
python train.py --product LED --area A

# 2. 命名並放置
mv runs/train/weights/best.pt \
   models/LED/A/yolo/weights/LED_best_v1.1.0_20260210.pt

# 3. 更新 config.yaml
# model_version: "1.1.0"

# 4. 提交
git add models/LED/A/yolo/
git commit -m "feat(LED): release v1.1.0 - improved accuracy to 95%"
git tag models/LED/A/yolo/v1.1.0  # (可選) 打標籤
git push --follow-tags
```

### 回滾到舊版本
```bash
# 方法 1: 修改 config.yaml
model_version: "1.0.0"

# 方法 2: Git checkout 特定 commit
git checkout \u003ccommit-hash\u003e -- models/LED/A/yolo/weights/LED_best_v1.0.0_*.pt
```

---

## ModelManager 支援

### 自動版本解析
```python
# core/services/model_manager.py 已支援
from core.version_utils import parse_model_version, check_compatibility

version = parse_model_version("LED_best_v1.2.0_20260210.pt")
# 返回: (1, 2, 0)

is_compatible = check_compatibility(
    current_version=(1, 2, 0),
    min_version=(1, 0, 0)
)
# 返回: True
```

### 版本驗證流程
```python
# 1. 載入模型時自動檢查
if model_version < min_supported_version:
    raise ModelVersionError(
        f"Model v{model_version} is below minimum v{min_supported_version}"
    )

# 2. 記錄到日誌
logger.info(f"Loaded model: {product}/{area} v{model_version}")
```

---

## 最佳實踐

### ✅ 推薦
1. **每次訓練都建立新版本** - 避免覆蓋舊模型
2. **保留至少 3 個歷史版本** - 用於 A/B 測試
3. **使用 Git Tags 標記里程碑** - `git tag models/LED/v1.0.0`
4. **在 Commit Message 記錄變更** - 包含準確率、訓練資料集

### ❌ 避免
1. ❌ 直接覆蓋 `best.pt` - 會失去歷史記錄
2. ❌ 手動編輯版本號 - 使用自動化腳本
3. ❌ 混用版本與非版本檔案 - 完全遷移到新結構

---

## 故障排除

### Q: Git LFS 上傳失敗 (檔案過大)
```bash
# 調整 LFS 快取大小
git config lfs.storage 10GB
```

### Q: 如何清理舊版本模型?
```bash
# 僅保留最新 3 個版本
cd models/LED/A/yolo/weights/
ls -t | tail -n +4 | xargs rm
```

### Q: ModelManager 找不到模型?
檢查 `config.yaml` 中的 `model_version` 與實際檔案名稱是否一致:
```bash
ls models/LED/A/yolo/weights/
# 應顯示: LED_best_v1.2.0_*.pt
```

---

## 參考資源
- [Git LFS 官方文檔](https://git-lfs.github.com/)
- [Semantic Versioning 2.0.0](https://semver.org/)
- [本專案 CHANGELOG.md](../../CHANGELOG.md)
