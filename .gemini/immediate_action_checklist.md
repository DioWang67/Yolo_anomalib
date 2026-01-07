# 立即行動清單 - YOLO11 推理系統

**優先級**: P0 - 緊急修復（本週內完成）

---

## 🔴 Critical Issues（立即處理）

### 1. 修復 requirements.txt ⚡ [2 小時]

**問題**: 
```txt
# requirements.txt 當前內容
-e .
```
這會導致外部環境無法正確安裝依賴。

**解決方案**:
```bash
# 安裝 pip-tools
pip install pip-tools

# 生成完整的 requirements.txt
pip-compile pyproject.toml -o requirements.txt

# 驗證
python -m venv test_env
test_env\Scripts\activate
pip install -r requirements.txt
pytest -v
```

**檔案位置**: `requirements.txt`

---

### 2. 移除 CI continue-on-error ⚡ [4 小時]

**問題**: 
`.github/workflows/ci.yml` 中多處使用 `continue-on-error: true`，會隱藏真實問題。

**需要修改的地方**:

```yaml
# 第 32 行
- name: Lint with ruff
  run: ruff check .
  continue-on-error: true  # ❌ 移除這行

# 第 37 行  
- name: Type check with mypy
  run: mypy core app --ignore-missing-imports...
  continue-on-error: true  # ❌ 移除這行

# 第 73 行
- name: Run fast tests (skip GUI)
  run: pytest -v -m "not gui"
  continue-on-error: ${{ runner.os == 'Linux' }}  # ❌ 移除這行

# 第 79 行
- name: Run all tests with coverage
  run: pytest -p pytest_cov...
  continue-on-error: true  # ❌ 移除這行
```

**修復步驟**:
1. 本地執行所有 CI 檢查，找出失敗的測試
2. 修復失敗的測試或使用 `@pytest.mark.xfail` 標記已知問題
3. 移除所有 `continue-on-error`
4. 確認 CI 全綠

---

### 3. YAML 安全載入 🔐 [2 小時]

**問題**: 可能使用了不安全的 `yaml.load()`

**檢查所有使用**:
```bash
# 搜尋所有 yaml.load 的使用
grep -rn "yaml.load" --include="*.py" core/ app/

# 應該搜尋到的檔案（需逐一檢查）:
# - core/config.py
# - core/services/model_manager.py
# - 其他
```

**修改範例**:
```python
# ❌ 不安全
import yaml
with open(config_path) as f:
    config = yaml.load(f)

# ✅ 安全
import yaml
with open(config_path) as f:
    config = yaml.safe_load(f)
```

**需要檢查的檔案**:
- [ ] `core/config.py`
- [ ] `core/services/model_manager.py`
- [ ] `app/cli.py`
- [ ] 其他使用 yaml 的地方

---

### 4. 路徑安全驗證 🔐 [6 小時]

**新增檔案**: `core/security.py`

```python
"""Security utilities for safe file operations."""
from pathlib import Path
from typing import Union

class SecurityError(Exception):
    """Raised when a security check fails."""
    pass

class PathValidator:
    """Validate file paths to prevent directory traversal attacks."""
    
    def __init__(self, allowed_roots: list[Path]):
        self.allowed_roots = [Path(root).resolve() for root in allowed_roots]
    
    def validate_path(self, path: Union[str, Path], *, must_exist: bool = False) -> Path:
        """Validate that a path is safe to access.
        
        Args:
            path: Path to validate
            must_exist: If True, verify path exists
            
        Returns:
            Resolved absolute path
            
        Raises:
            SecurityError: If path is outside allowed roots
            FileNotFoundError: If must_exist=True and path doesn't exist
        """
        resolved = Path(path).resolve()
        
        # Check if path is within allowed roots
        if not any(self._is_relative_to(resolved, root) for root in self.allowed_roots):
            raise SecurityError(
                f"Access denied: {path} is outside allowed directories"
            )
        
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        
        return resolved
    
    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        """Check if path is relative to parent."""
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

# 全局實例
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
path_validator = PathValidator(allowed_roots=[
    PROJECT_ROOT,
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "Result",
])
```

**修改現有代碼** - `core/config.py`:
```python
# 在檔案頂部加入
from core.security import path_validator

class DetectionConfig:
    def load_config(self, config_path: str | Path):
        # ✅ 加入路徑驗證
        safe_path = path_validator.validate_path(config_path, must_exist=True)
        
        with open(safe_path) as f:
            raw = yaml.safe_load(f)  # 也改用 safe_load
            ...
```

**需要加入驗證的地方**:
- [ ] `core/config.py` - `load_config()`
- [ ] `core/services/model_manager.py` - 模型路徑載入
- [ ] `core/detection_system.py` - 影像輸入路徑
- [ ] `main.py` - `--image` 參數處理

---

### 5. 加入測試 🧪 [4 小時]

為以上修復加入測試：

**新增檔案**: `tests/test_security.py`

```python
"""Security feature tests."""
import pytest
from pathlib import Path
from core.security import PathValidator, SecurityError

def test_path_validator_allows_valid_path(tmp_path):
    """Test that valid paths are allowed."""
    validator = PathValidator(allowed_roots=[tmp_path])
    
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")
    
    # Should not raise
    result = validator.validate_path(test_file, must_exist=True)
    assert result == test_file.resolve()

def test_path_validator_blocks_directory_traversal(tmp_path):
    """Test that directory traversal is blocked."""
    validator = PathValidator(allowed_roots=[tmp_path / "safe"])
    
    # Try to access parent directory
    malicious_path = tmp_path / "safe" / ".." / "secret.txt"
    
    with pytest.raises(SecurityError):
        validator.validate_path(malicious_path)

def test_path_validator_blocks_absolute_outside_path(tmp_path):
    """Test that absolute paths outside allowed roots are blocked."""
    validator = PathValidator(allowed_roots=[tmp_path / "allowed"])
    
    outside_path = tmp_path / "forbidden" / "file.txt"
    
    with pytest.raises(SecurityError):
        validator.validate_path(outside_path)

def test_path_validator_must_exist(tmp_path):
    """Test that must_exist flag works."""
    validator = PathValidator(allowed_roots=[tmp_path])
    
    non_existent = tmp_path / "does_not_exist.txt"
    
    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        validator.validate_path(non_existent, must_exist=True)
    
    # Should not raise without must_exist
    result = validator.validate_path(non_existent, must_exist=False)
    assert result == non_existent.resolve()
```

執行測試：
```bash
pytest tests/test_security.py -v
```

---

## 📋 檢查清單

完成以下所有項目後，專案的關鍵風險將大幅降低：

### Phase 1: 依賴管理
- [ ] 安裝 `pip-tools`
- [ ] 執行 `pip-compile` 生成完整 requirements.txt
- [ ] 在新虛擬環境測試安裝
- [ ] 更新 CI 使用新的 requirements.txt
- [ ] 更新 README 安裝指引

### Phase 2: CI 穩定性
- [ ] 本地執行 `ruff check .` 並修復所有錯誤
- [ ] 本地執行 `mypy core app` 並修復類型錯誤
- [ ] 本地執行 `pytest -v` 確保所有測試通過
- [ ] 移除 `.github/workflows/ci.yml` 中所有 `continue-on-error`
- [ ] Push 並確認 CI 全綠

### Phase 3: 安全性
- [ ] 搜尋所有 `yaml.load` 並替換為 `yaml.safe_load`
- [ ] 建立 `core/security.py` 檔案
- [ ] 在 `core/config.py` 加入路徑驗證
- [ ] 在 `core/services/model_manager.py` 加入路徑驗證
- [ ] 在 `main.py` 加入影像路徑驗證
- [ ] 建立 `tests/test_security.py` 並執行測試

### Phase 4: 驗證
- [ ] 執行完整測試套件: `pytest -v`
- [ ] 執行 linting: `ruff check .`
- [ ] 執行類型檢查: `mypy core app`
- [ ] 檢查 CI 狀態: 所有檢查應為綠色 ✅
- [ ] 建立 Git commit 並 push

---

## 🎯 成功標準

完成後，應該達到：

1. ✅ **依賴可復現**: 在新環境執行 `pip install -r requirements.txt` 成功
2. ✅ **CI 穩定性**: GitHub Actions 顯示全綠，無警告
3. ✅ **安全基礎**: 路徑注入攻擊被阻止，YAML 反序列化安全
4. ✅ **測試通過**: 所有測試通過，包括新增的安全測試

---

## 📞 需要幫助？

如果遇到以下情況：

1. **CI 中有測試持續失敗**
   - 使用 `@pytest.mark.xfail(reason="Known issue #123")` 標記
   - 建立 GitHub Issue 追蹤

2. **不確定如何修復某個 Lint 錯誤**
   - 執行 `ruff check . --explain E501` 查看詳細說明
   - 或暫時加入 `# noqa: E501` 註解（但要有 TODO）

3. **路徑驗證破壞了現有功能**
   - 檢查 `path_validator.allowed_roots` 是否包含所需目錄
   - 臨時擴大允許範圍，後續再收緊

---

## ⏱️ 預估時間

| 任務 | 預估時間 | 實際時間 |
|------|----------|----------|
| 修復 requirements.txt | 2h | |
| 移除 CI continue-on-error | 4h | |
| YAML 安全載入 | 2h | |
| 路徑安全驗證 | 6h | |
| 加入測試 | 4h | |
| **總計** | **18h** | |

建議分 2-3 天完成，避免一次性修改過多導致難以除錯。

---

## 📝 Commit Message 建議

```bash
# Day 1
git commit -m "fix(deps): use pip-compile for reproducible requirements"

# Day 2
git commit -m "fix(ci): remove continue-on-error to prevent hiding failures"
git commit -m "fix(ci): resolve all linting and type checking errors"

# Day 3
git commit -m "security(yaml): use safe_load to prevent code injection"
git commit -m "security(path): add PathValidator to prevent directory traversal"
git commit -m "test(security): add comprehensive security tests"
```

---

**優先級**: 🔴 P0 - 立即處理  
**預估完成**: 2026-01-13（本週內）  
**負責人**: 開發團隊

---

**下一步**: 完成此清單後，參考 `improvement_roadmap.md` 進行 Phase 2（可觀測性增強）
