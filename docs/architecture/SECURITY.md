# 安全設計與操作界線

本文件記錄 `yolo11_inference` 目前已實作、可由程式與測試驗證的安全機制。
它不是作業系統權限、網路隔離或公司資安政策的替代品。

最後核對日期：2026-08-14；目前系統正式版本：1.1.0。
v1.1.0 未變更第 1 節的 `core/security.py` 路徑邊界，也未變更第 2 節的 YAML 載入
機制。新增的是第 1.1 節：驗收證據檔案在既有路徑邊界之上另有一層更嚴格的判定。

## 1. 路徑邊界

核心實作位於 `core/security.py`：

| API | 用途 | 失敗行為 |
| --- | --- | --- |
| `PathValidator.validate_path()` | 正規化路徑並限制在允許根目錄內 | 越界時拋出 `SecurityError` |
| `safe_segment()` | 驗證 product、area、status 等單一路徑片段 | 空值、`.`、`..`、分隔符、磁碟前綴或非法字元均拒絕 |
| `ensure_subpath()` | 確認輸入／輸出仍位於指定 root 下 | 越界時拋出 `SecurityError` |
| `resolve_result_output_dir()` | 將 Result alias 或相對輸出解析到專用 Result root | 拒絕 traversal、drive-relative 與 root 外絕對路徑 |

`Path.resolve()` 會先解析 `.`、`..` 與符號連結，再檢查是否位於允許 root，因而
可阻擋一般目錄遍歷與 symlink escape。

全域 validator 的允許範圍由工作區設定解析，包含：

- inference 專案根目錄；
- station data root、models、results 與 logs；
- inference 專案內的 `Runtime` 與 `MvImport`。

目前工作區的實際值定義在根目錄 `workspace.yaml`。不要在文件或程式中假設所有
可變資料都仍位於 source repository 內。

### 使用原則

```python
from core.security import ensure_subpath, safe_segment

product = safe_segment(raw_product, field_name="product")
target = ensure_subpath(result_root / product / "evidence.json", result_root)
```

- 外部輸入形成資料夾名稱前先呼叫 `safe_segment()`。
- 寫檔前以該功能的最小 root 呼叫 `ensure_subpath()`。
- 讀取既有檔案時使用 `must_exist=True`。
- 不要把整個使用者家目錄、磁碟根目錄或網路分享根目錄加入白名單。
- 專案沒有 `DEV_MODE` 放寬路徑限制；不得自行加入全磁碟 bypass。

### 1.1 驗收證據檔案的額外判定

第 1 節的邊界回答「這個路徑是否在允許範圍內」。驗收證據還要回答一個不同的問題：
**這個檔案是否就是當初記錄雜湊值的那一個**。`Path.resolve()` 會跟隨符號連結，
因此一個指向 root 內部的連結可以通過 `ensure_subpath()`，但它不是原始檔案——
指向的目標可以在事後被換掉，而記錄下來的雜湊值不會改變。

因此下列 API 對符號連結採**拒絕**而非跟隨：

| API | 位置 | 判定 |
| --- | --- | --- |
| `verified_acceptance_image_path()` | `core/services/model_acceptance.py` | 逐一走過相對路徑的每個組成部分，任一部分是符號連結即拒絕；再檢查仍位於 dataset root 內；預設另比對 `image_sha256` |
| `artifact_ref()` | `core/services/acceptance_artifacts.py` | 在 `resolve()` **之前**檢查最後一個組成部分是否為符號連結，並拒絕空檔案 |
| `cross_process_file_lock()` | `tools/cross_process_lock.py` | 在 `resolve()` 之前檢查鎖檔與其父目錄是否為符號連結 |

符號連結檢查必須寫在 `resolve()` **之前**。`resolve()` 之後的 `is_symlink()` 恆為
`False`，那種寫法看起來像防護但永遠不會生效。

`verify_acceptance_artifact_bundle()` 在每次驗收推論的開始與結束各執行一次，
比對已釘住的路徑、SHA-256 與檔案大小，並確認 model config 仍解析到同一個權重
檔案。推論期間被替換的檔案會使該次結果被拒絕，而不是產生一份指向已變更檔案的
報告。

### 1.2 跨行程互斥

驗收 manifest（`ground_truth.csv`）的每一次寫入都在 `cross_process_file_lock()`
保護下進行：同一行程內以 `threading.RLock` 序列化，跨行程以位元組範圍鎖
（Windows `msvcrt.locking`、POSIX `fcntl.flock`）序列化。批次提交另外採用
checksum compare-and-swap——若 manifest 在推論期間被改動，該批次會被拒絕而不是
覆蓋他人的結果。

鎖檔位於 station data 的 `locks/` 目錄，不含任何內容，且被排除於備份 ZIP 之外。

## 2. YAML 與設定輸入

現行設定載入使用 `yaml.safe_load()`，寫回使用 `yaml.safe_dump()`。主要呼叫點包含
`core/config.py`、`core/workspace.py`、模型／release services 與 production tools。

安全載入只阻止 YAML 物件反序列化；欄位型別與業務範圍仍須由 config schema、
service validation 或 readiness check 驗證。新增 YAML 入口時必須同時補負面測試，
不可使用沒有 `SafeLoader` 的 `yaml.load()`。

## 3. 憑證與公司同步

公司同步 token 只透過設定指定的環境變數名稱讀取；預設為
`YOLO11_INSPECTION_SYNC_TOKEN`。設定檔只保存環境變數名稱，不保存 token 值。

- 不得把 API token、密碼或私鑰提交到 Git、Excel 修訂紀錄、log 或截圖。
- 啟用同步前，以 `tools/production_preflight.py` 驗證 endpoint 與 token 環境變數。
- token 缺失時同步應失敗並保留本地 outbox，不得降級成未驗證請求。
- 憑證輪替與主機權限由部署環境／公司 IT 管理，本專案不提供 secret vault。

## 4. 部署控制

- 使用核准的 Windows release bundle，不直接在產線機台執行任意開發分支。
- 模型、config、runtime manifest 應成組發布並保留可驗證 rollback。
- station data、Result、database 與 logs 應由作業系統 ACL 限制到服務帳號及授權人員。
- 公司 API 應使用 HTTPS、伺服器憑證驗證與公司網路存取控制。
- 依賴更新需在受控環境測試並執行弱點掃描；本文件不宣稱所有套件永遠無漏洞。

相關程序見 [Windows 部署 SOP](../operations/WINDOWS_DEPLOYMENT_SOP.md) 與
[發布／回滾 SOP](../operations/RELEASE_ROLLBACK_SOP.md)。

## 5. 驗證

路徑安全回歸測試：

```powershell
D:\miniconda\envs\yolo_anomalib\python.exe -m pytest tests\test_security.py -q
```

測試涵蓋合法子路徑、多 root、`..` traversal、root 外絕對路徑、symlink escape、
Result alias、drive-relative 路徑與單一路徑片段。Windows 無建立 symlink 權限時，
該案例可被 pytest 明確標示為 skipped；不得把 skip 寫成已驗證通過。

第 1.1／1.2 節的證據邊界回歸測試：

```powershell
D:\miniconda\envs\yolo_anomalib\python.exe -m pytest tests\test_acceptance_artifacts.py tests\test_acceptance_runs.py tests\test_model_acceptance.py -q
```

另有一項測試環境保護：`tests/conftest.py` 會在 `--basetemp` 落在含
`workspace.yaml` 的目錄之內時**拒絕啟動測試**。workspace 探索是往上層尋找
`workspace.yaml`，因此 workspace 內的 basetemp 會讓每個測試的 `tmp_path` 解析到
正式 station data。此事曾發生：一個測試用的殘缺顏色模型被寫入正式 color profile
store，並在驗收工具中成為可選取的顏色方案。另有一道 per-test 寫入偵測作為後備。

文件與單元測試不能證明主機 ACL、公司 API 或現場網路已完成滲透測試。正式上線
仍須執行 [上線檢查表](../operations/PRODUCTION_GO_LIVE_CHECKLIST.md)。

## 6. 已知限制與變更規則

- `PathValidator` 只保護有實際呼叫它的檔案操作，不是全程序 sandbox。
- inference 專案根目錄是全域允許 root；高風險功能仍應使用更小的專用 root。
- 路徑驗證不取代 NTFS ACL、防毒、端點管控、TLS 或網段隔離。
- 新增外部路徑、網路分享或新憑證來源時，必須先完成威脅評估、負面測試與部署
  文件更新，不得只修改白名單。

發現疑似漏洞時，不要在公開紀錄貼出憑證或可利用細節；依公司內部通報管道交由
專案 Owner 與 IT／資安人員處理。
