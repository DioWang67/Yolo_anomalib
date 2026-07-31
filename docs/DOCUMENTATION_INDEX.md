# 文件總索引

這是本專案的文件入口。所有 Markdown 使用 UTF-8；Windows PowerShell 若顯示
亂碼，請使用：

```powershell
Get-Content -Encoding utf8 <file>
```

## 依角色開始

| 角色 | 第一份文件 | 接著閱讀 |
| --- | --- | --- |
| 產線操作者 | [AI 檢測系統操作手冊](OPERATOR_MANUAL.md) | [誤判與漏檢 SOP](MISJUDGE_TRIAGE_SOP.md) |
| 班組長／資料複核人員 | [操作手冊：異常複核](OPERATOR_MANUAL.md#9-異常複核與資料回收) | [補訓閉環](../../Yolo11_auto_train/docs/SEAMLESS_WORKFLOW.md) |
| 設備／製程工程 | [工程維運手冊](ENGINEERING_MANUAL.md) | [相機診斷](CAMERA_RUNTIME_DIAGNOSTICS.md) |
| AI／軟體工程 | [工程維運手冊](ENGINEERING_MANUAL.md) | [模組架構](MODULE_ARCHITECTURE.md) |
| IT／公司 API | [公司同步](COMPANY_SERVER_SYNC.md) | [工程手冊：公司同步](ENGINEERING_MANUAL.md#13-公司伺服器同步) |
| 發版人員 | [發布與回滾 SOP](RELEASE_ROLLBACK_SOP.md) | [正式上線檢查表](PRODUCTION_GO_LIVE_CHECKLIST.md) |

## 依工作尋找

| 需求 | 主文件 | 補充文件／工具 |
| --- | --- | --- |
| 日常檢測、PASS／NG／ERROR | [操作者手冊](OPERATOR_MANUAL.md) | — |
| 查詢紀錄、良率、匯出 Excel | [操作者手冊：檢測紀錄](OPERATOR_MANUAL.md#7-檢測紀錄) | [資料庫](INSPECTION_DATABASE.md) |
| 複核誤殺、漏檢、錯框 | [操作者手冊：異常複核](OPERATOR_MANUAL.md#9-異常複核與資料回收) | [誤判 SOP](MISJUDGE_TRIAGE_SOP.md) |
| 1.0.6 跨類別重複框處理與 Pilot | [重複框改善企畫書](CROSS_CLASS_DUPLICATE_DETECTION_PROPOSAL.md) | 已實作；現場 Gate 待完成 |
| 補訓、續訓、安全停止 | [工程手冊：補訓](ENGINEERING_MANUAL.md#7-補訓閉環) | [補訓閉環](../../Yolo11_auto_train/docs/SEAMLESS_WORKFLOW.md) |
| 位置補訓與首次啟用 | [工程手冊：位置檢測](ENGINEERING_MANUAL.md#8-位置檢測補訓) | [位置補訓部署](../../Yolo11_auto_train/docs/POSITION_RETRAINING_DEPLOYMENT.md) |
| 顏色誤殺與門檻校正 | [顏色覆核與校正](COLOR_REVIEW_CALIBRATION.md) | [誤判 SOP](MISJUDGE_TRIAGE_SOP.md) |
| 模型／顏色組合驗收與上線 | [模型組合驗收與發布](MODEL_COMBINATION_ACCEPTANCE.md) | [發布與回滾 SOP](RELEASE_ROLLBACK_SOP.md) |
| Windows 機台部署 | [Windows 部署 SOP](WINDOWS_DEPLOYMENT_SOP.md) | [工程手冊：首次部署](ENGINEERING_MANUAL.md#4-首次部署) |
| 相機 Runtime／取像問題 | [相機診斷](CAMERA_RUNTIME_DIAGNOSTICS.md) | `tools/diagnostics/diagnose_camera.bat` |
| 模型版本切換或回滾 | [模型版本指南](MODEL_VERSION_GUIDE.md) | [發布與回滾 SOP](RELEASE_ROLLBACK_SOP.md) |
| SQLite、備份、保存、還原 | [資料庫文件](INSPECTION_DATABASE.md) | `tools/maintain_inspection_data.py` |
| 公司伺服器同步 | [公司同步](COMPANY_SERVER_SYNC.md) | `tools/inspection_sync_admin.py` |
| 上線前完整預檢 | [正式上線檢查表](PRODUCTION_GO_LIVE_CHECKLIST.md) | `tools/production_preflight.py` |
| PCBA 受控試產 | [PCBA 試產手冊](PCBA_PILOT_RUNBOOK.md) | [試產驗收表](PCBA_PILOT_ACCEPTANCE_TEMPLATE.md) |
| 程式模組與執行緒責任 | [模組架構](MODULE_ARCHITECTURE.md) | `core/detection_system.py` |
| 安全要求 | [安全文件](SECURITY.md) | `tests/test_security.py` |

## 文件責任

### 角色手冊

- `OPERATOR_MANUAL.md`：操作者與班組長的唯一日常操作入口。
- `ENGINEERING_MANUAL.md`：工程、部署、補訓、資料與同步的交接入口。
- `README.md`：專案簡介、開發安裝與快速命令，不取代角色 SOP。

### 專題文件

- `INSPECTION_DATABASE.md`：SQLite schema、備份、保存與還原。
- `COMPANY_SERVER_SYNC.md`：站點 outbox 與公司 HTTP 契約。
- `WINDOWS_DEPLOYMENT_SOP.md`：Windows release bundle 部署。
- `RELEASE_ROLLBACK_SOP.md`：runtime/model/config 成對發布與回滾。
- `CAMERA_RUNTIME_DIAGNOSTICS.md`：Hikrobot Runtime 與現場取像診斷。
- `MISJUDGE_TRIAGE_SOP.md`：誤殺、漏檢、原因碼與變更控制。
- `CALIBRATION_CHANGE_LOG.md`：閾值、模型與保存政策的追加式變更紀錄。
- `COLOR_REVIEW_CALIBRATION.md`：顏色資料路由與具名批准門檻。
- `MODEL_COMBINATION_ACCEPTANCE.md`：獨立驗收集、組合矩陣、五色基準、
  指標限制及完整檢測組合的啟用與回滾。
- `CROSS_CLASS_DUPLICATE_DETECTION_PROPOSAL.md`：Cable1/A 1.0.6
  跨類別重複框的根因、保守消除規則、重播證據、設定與現場 Pilot Gate。
- `MODEL_VERSION_GUIDE.md`：模型命名、版本與回復原則。
- `MODULE_ARCHITECTURE.md`：模組、執行緒與狀態安全。
- `SECURITY.md`：路徑、YAML、依賴與憑證安全。

### PCBA 與驗收

- `PCBA_INSPECTION_PLAN.md`：檢測範圍與能力限制。
- `PCBA_PILOT_RUNBOOK.md`：受控試產流程。
- `PCBA_OPERATOR_COMMANDS.md`：`pcba.bat` 工程／試產輔助命令。
- `PCBA_PILOT_ACCEPTANCE_TEMPLATE.md`：Golden、known NG、dry run 驗收表。
- `PRODUCTION_GO_LIVE_CHECKLIST.md`：上線 Gate 與目前警告。

### 開發與治理

- `TECH_GUIDE.md`：深度技術教材。
- `FIRMWARE_RUNTIME_PLAN.md`：受限裝置 runtime artifact 與 benchmark。
- `CHANGELOG.md`：release 變更。
- `PROJECT_MEMORY.md`：決策、假設與風險。
- `PROJECT_TODO.md`：尚未完成事項；不得把 TODO 當成現行功能。

### 歷史封存

- `archive/project/PROGRESS_LOG.md`：舊開發進度紀錄，不作為現行 SOP。
- `archive/validation/FPS_OVERLAY_AB_VALIDATION.md`：FPS 疊字移除的歷史 A/B
  驗證結論與證據限制。

## 上線聲明

文件完整不等於產品／工位已通過生產驗收。無人值守生產前仍必須具備：

1. readiness 沒有未處理的 `FAIL`；
2. 所有 `WARN`已修正或有具名書面接受；
3. Golden OK 重複性證據；
4. 已知 NG 驗證證據；
5. 位置、顏色與公司同步依實際啟用範圍完成 Gate；
6. 已填寫試產驗收紀錄；
7. 可用且演練過的完整回滾 release。

