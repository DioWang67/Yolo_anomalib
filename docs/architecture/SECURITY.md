# 安全設計與操作界線

本文件記錄 `yolo11_inference` 目前已實作、可由程式與測試驗證的安全機制。
它不是作業系統權限、網路隔離或公司資安政策的替代品。

最後核對日期：2026-08-05；系統正式版本：1.0.0。

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
