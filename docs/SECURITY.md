# 安全指南 (Security Guide)

本文檔說明 `yolo11_inference` 專案的安全機制、最佳實踐和安全配置。

---

## 📋 目錄

1. [路徑安全驗證](#路徑安全驗證)
2. [YAML 安全載入](#yaml-安全載入)
3. [依賴管理](#依賴管理)
4. [生產環境部署](#生產環境部署)
5. [安全測試](#安全測試)
6. [常見安全問題](#常見安全問題)

---

## 🔒 路徑安全驗證

### 概述

從 v0.1.0 開始，專案實作了完整的路徑安全驗證機制，防止**目錄遍歷攻擊** (Directory Traversal Attack)。

**What is Directory Traversal?**  
攻擊者通過構造包含 `../` 的路徑來訪問系統中未授權的文件。例如：
```
../../../etc/passwd          # Linux
..\..\..\Windows\System32    # Windows
```

### 核心模組

**檔案**: `core/security.py`

**主要類別**:
- `SecurityError`: 路徑安全相關異常
- `PathValidator`: 路徑驗證器類別

### 防護範圍

系統自動驗證以下路徑：

| 路徑類型 | 來源 | 防護方式 |
|---------|------|---------|
| 配置文件 | `config.yaml` | `config.py` 自動驗證 |
| 影像輸入 | `--image` 參數 | `main.py` 自動驗證 |
| 模型權重 | 模型路徑配置 | 全局驗證器 |
| 輸出目錄 | `output_dir` 配置 | 全局驗證器 |

### 使用範例

#### 基本用法

```python
from core.security import path_validator, SecurityError

try:
    # 驗證路徑是否在允許的根目錄內
    safe_path = path_validator.validate_path(
        user_input_path,
        must_exist=True  # 可選：要求路徑必須存在
    )
    # 使用 safe_path 進行檔案操作
    with open(safe_path, 'r') as f:
        data = f.read()
except SecurityError as e:
    logger.error(f"路徑安全驗證失敗: {e}")
except FileNotFoundError:
    logger.error("檔案不存在")
```

#### 自定義驗證器

```python
from core.security import PathValidator

# 為特定功能創建專用驗證器
custom_validator = PathValidator(
    allowed_roots=[
        "/path/to/project",
        "/path/to/models",
        "/path/to/data",
        "/mnt/shared/results"  # 網路共享目錄
    ]
)

# 使用自定義驗證器
safe_path = custom_validator.validate_path(
    user_input,
    must_exist=False  # 允許不存在的路徑（用於創建新文件）
)
```

### 防護機制詳解

#### 1. 目錄遍歷防護

**檢查項目**:
- 路徑正規化（解析 `.` 和 `..`）
- 檢查正規化後的路徑是否在允許的根目錄內

**範例**:
```python
# ❌ 拒絕這些路徑
path_validator.validate_path("../../etc/passwd")
path_validator.validate_path("config/../../../secrets.txt")

# ✅ 允許這些路徑（假設在專案目錄內）
path_validator.validate_path("models/LED/A/best.pt")
path_validator.validate_path("./Result/output.jpg")
```

#### 2. 符號連結檢查

**行為**:
- 解析符號連結到真實路徑
- 檢查真實路徑是否在允許的根目錄內

**範例**:
```python
# 假設 /tmp/link -> /etc/passwd
path_validator.validate_path("/tmp/link")  # ❌ 拒絕，真實路徑在 /etc
```

#### 3. 白名單機制

**預設允許的根目錄**:
```python
# 在 core/security.py 中定義
PROJECT_ROOT = Path(__file__).parent.parent  # 專案根目錄

path_validator = PathValidator(
    allowed_roots=[
        PROJECT_ROOT,                    # 專案根目錄
        PROJECT_ROOT / "models",        # 模型目錄
        PROJECT_ROOT / "Result",        # 結果輸出目錄
    ]
)
```

**自定義允許的目錄**:
```python
# 在生產環境中可能需要額外的目錄
from core.security import PathValidator

production_validator = PathValidator(
    allowed_roots=[
        "/opt/yolo11_prod",
        "/var/lib/yolo11/models",
        "/mnt/nfs/shared", 
        "/data/camera_images"
    ]
)
```

### 整合到現有代碼

#### 在配置載入中使用

```python
# 已整合在 core/config.py 中
def load_config(path: str) -> dict:
    from core.security import path_validator, SecurityError
    
    try:
        safe_path = path_validator.validate_path(path, must_exist=True)
    except SecurityError as exc:
        raise ConfigLoadError(f"安全錯誤: {exc}") from exc
    
    with safe_path.open('r') as f:
        return yaml.safe_load(f)
```

#### 在文件操作中使用

```python
def save_result(output_path: str, data: dict):
    from core.security import path_validator, SecurityError
    
    try:
        safe_path = path_validator.validate_path(
            output_path,
            must_exist=False  # 輸出文件可能不存在
        )
    except SecurityError as e:
        raise SecurityError(f"不允許寫入路徑 {output_path}: {e}")
    
    # 確保父目錄存在
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    
    with safe_path.open('w') as f:
        json.dump(data, f)
```

---

## 🛡️ YAML 安全載入

### 為什麼需要 `safe_load`?

使用 `yaml.load()` 而不是 `yaml.safe_load()` 可能導致**任意程式碼執行**漏洞。

**危險範例**:
```yaml
# malicious.yaml
!!python/object/apply:os.system
args: ['rm -rf /']
```

```python
# ❌ 危險！會執行系統命令
with open('malicious.yaml') as f:
    data = yaml.load(f)  # 會執行 rm -rf /
```

### 已驗證的安全載入

專案中所有 YAML 載入已使用 `yaml.safe_load()`:

| 文件 | 行號 | 用途 |
|------|------|------|
| `core/config.py` | 206 | 載入全局配置 |
| `core/services/model_manager.py` | 179 | 載入模型配置 |
| `core/detection_system.py` | 165 | 載入位置配置 |

### 最佳實踐

```python
import yaml

# ✅ 正確：使用 safe_load
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# ❌ 錯誤：永遠不要使用 load
with open('config.yaml') as f:
    config = yaml.load(f)  # 危險！

# ✅ 正確：如果需要 Loader，使用 SafeLoader
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
```

---

## 📦 依賴管理

### requirements.txt 更新流程

**使用 pip-compile 管理依賴**:

```bash
# 1. 修改 pyproject.toml 中的依賴
vim pyproject.toml

# 2. 重新編譯 requirements.txt
pip-compile pyproject.toml -o requirements.txt

# 3. 在開發環境中測試
pip install -r requirements.txt

# 4. 提交變更
git add requirements.txt pyproject.toml
git commit -m "deps: update dependencies"
```

### 安全漏洞檢查

**定期檢查已知漏洞**:

```bash
# 使用 pip-audit (推薦)
pip install pip-audit
pip-audit -r requirements.txt

# 或使用 safety
pip install safety
safety check -r requirements.txt
```

### 依賴固定版本

`requirements.txt` 中所有依賴都固定了版本號：

```text
torch==2.4.1                    # ✅ 固定版本
ultralytics==8.3.156            # ✅ 固定版本
numpy>=1.26.0                   # ❌ 避免使用範圍
```

**原因**:
- 確保可重現的構建
- 避免意外的破壞性更新
- 便於安全審計

---

## 🚀 生產環境部署

### 安全檢查清單

#### 文件權限

```bash
# 配置文件只允許擁有者讀寫
chmod 600 config.yaml

# 模型權重只允許讀取
chmod 444 models/**/*.pt

# 可執行文件
chmod 755 main.py GUI.py

# 結果目錄允許写入
chmod 755 Result/
```

#### 環境變數

```bash
# 不要在配置文件中硬編碼敏感信息
# 使用環境變數

# .env 文件 (不要提交到 Git)
PROJECT_ROOT=/opt/yolo11_prod
MODEL_PATH=/var/lib/models
API_KEY=your_secret_key_here
DATABASE_URL=postgresql://user:pass@localhost/db
```

```python
# 使用 python-dotenv 載入
from dotenv import load_dotenv
import os

load_dotenv()
model_path = os.getenv('MODEL_PATH', './models')
```

#### 網路安全

如果部署 API 服務：

```python
# 使用 HTTPS
# 限制來源 IP
# 實作速率限制
# 添加認證機制

from flask import Flask, request, abort

app = Flask(__name__)

ALLOWED_IPS = ['192.168.1.100', '10.0.0.50']

@app.before_request
def limit_remote_addr():
    if request.remote_addr not in ALLOWED_IPS:
        abort(403)  # Forbidden
```

### Docker 部署建議

```dockerfile
# Dockerfile
FROM python:3.10-slim

# 不要以 root 運行
RUN useradd -m -u 1000 yolo11
USER yolo11

# 只複製必要的文件
COPY --chown=yolo11:yolo11 requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=yolo11:yolo11 . /app
WORKDIR /app

# 使用非特權端口
EXPOSE 8080

CMD ["python", "main.py"]
```

---

## ✅ 安全測試

### 運行安全測試

```bash
# 運行所有安全相關測試
pytest tests/test_security.py -v

# 預期結果
# 12 passed, 1 skipped (符號連結測試在 Windows 上跳過)
```

### 測試覆蓋範圍

| 測試案例 | 描述 |
|---------|------|
| `test_allows_valid_path_within_allowed_root` | 允許合法路徑 |
| `test_allows_subdirectory_path` | 允許子目錄 |
| `test_blocks_directory_traversal_with_dotdot` | 阻擋 `..` 遍歷 |
| `test_blocks_absolute_path_outside_allowed_roots` | 阻擋外部絕對路徑 |
| `test_must_exist_flag_enforces_existence` | 文件存在性檢查 |
| `test_multiple_allowed_roots` | 多個允許根目錄 |
| `test_handles_relative_paths` | 相對路徑處理 |
| `test_blocks_symlink_escape` | 符號連結逃逸阻擋 |
| `test_error_message_includes_allowed_roots` | 錯誤訊息品質 |
| `test_global_validator_*` | 全局驗證器測試 (4個) |

### 手動滲透測試

**測試目錄遍歷**:
```bash
# 嘗試訪問系統文件
python main.py --image "../../etc/passwd"
# 預期：SecurityError

python main.py --image "..\\..\\Windows\\System32\\config\\SAM"
# 預期：SecurityError
```

**測試符號連結逃逸**:
```bash
# Linux/macOS
ln -s /etc/passwd malicious_link
python main.py --image "./malicious_link"
# 預期：SecurityError

# Windows (需管理員權限)
mklink malicious_link C:\Windows\System32\config\SAM
python main.py --image ".\malicious_link"
# 預期：SecurityError
```

---

## ⚠️ 常見安全問題

### Q1: 如何允許存取網路共享目錄?

**解決方案**:

```python
# 方法 1: 修改 core/security.py 的全局驗證器
from pathlib import Path

path_validator = PathValidator(
    allowed_roots=[
        PROJECT_ROOT,
        Path("/mnt/nfs/shared"),  # NFS mount
        Path("//server/share"),   # Windows network share
    ]
)

# 方法 2: 為特定功能創建專用驗證器
network_validator = PathValidator(
    allowed_roots=[Path("//192.168.1.100/models")]
)
```

### Q2: 部署在容器中時路徑驗證失敗

**原因**: 容器內的路徑與主機不同

**解決方案**:

```yaml
# docker-compose.yml
services:
  yolo11:
    volumes:
      - ./models:/app/models       # 掛載到容器內的 /app/models
      - ./Result:/app/Result
    environment:
      - PROJECT_ROOT=/app          # 明確指定專案根目錄
```

```python
# 更新 core/security.py
import os

PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT', Path(__file__).parent.parent))
```

### Q3: 如何在保持安全性的同時簡化開發?

**開發環境配置**:

```python
# core/security.py - 添加開發模式
import os

if os.getenv('DEV_MODE') == '1':
    # 開發模式：更寬鬆的路徑限制
    path_validator = PathValidator(
        allowed_roots=[
            Path.home(),  # 允許整個用戶目錄
            Path('/'),    # ⚠️ 僅用於開發！
        ]
    )
else:
    # 生產模式：嚴格限制
    path_validator = PathValidator(
        allowed_roots=[PROJECT_ROOT]
    )
```

```bash
# 開發時啟用
export DEV_MODE=1
python main.py

# 生產環境不設置 DEV_MODE
python main.py
```

### Q4: 路徑驗證影響效能嗎?

**效能分析**:

```python
import time
from core.security import path_validator

paths = [f"models/product_{i}/model.pt" for i in range(1000)]

start = time.time()
for p in paths:
    path_validator.validate_path(p, must_exist=False)
end = time.time()

print(f"驗證 1000 個路徑: {(end-start)*1000:.2f}ms")
# 預期: < 10ms (negligible overhead)
```

**結論**: 路徑驗證的效能開銷極小（微秒級），不會影響系統吞吐量。

---

## 📚 延伸閱讀

- [OWASP Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal)
- [CWE-22: Improper Limitation of a Pathname](https://cwe.mitre.org/data/definitions/22.html)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security_warnings.html)
- [YAML Safe Loading](https://security.openstack.org/guidelines/dg_avoid-dangerous-input-parsing-libraries.html)

---

## 📝 安全問題回報

如果發現安全漏洞，請**不要**公開 Issue，而是聯繫:

- **Email**: a0983743448@gmail.com
- **主題**: [SECURITY] yolo11_inference vulnerability report

---

**最後更新**: 2026-01-07  
**版本**: v0.1.0  
**維護者**: DioWang
