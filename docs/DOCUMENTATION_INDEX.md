# 文件總索引

這是 `yolo11_inference` 的唯一文件入口。文件依用途分類；`docs` 根目錄只保留
本索引與正式版修訂紀錄 Excel。Markdown 一律使用 UTF-8。

## 依角色開始

| 角色 | 第一份文件 | 後續文件 |
| --- | --- | --- |
| 產線操作者／班組長 | [AI 檢測系統操作手冊](manuals/OPERATOR_MANUAL.md) | [誤判與漏檢 SOP](operations/MISJUDGE_TRIAGE_SOP.md) |
| 設備／製程工程 | [工程維運手冊](manuals/ENGINEERING_MANUAL.md) | [相機診斷](operations/CAMERA_RUNTIME_DIAGNOSTICS.md) |
| AI／軟體工程 | [工程維運手冊](manuals/ENGINEERING_MANUAL.md) | [模組架構](architecture/MODULE_ARCHITECTURE.md) |
| IT／公司 API | [公司同步](data/COMPANY_SERVER_SYNC.md) | [安全設計](architecture/SECURITY.md) |
| 發版人員 | [發布與回滾 SOP](operations/RELEASE_ROLLBACK_SOP.md) | [正式上線檢查表](operations/PRODUCTION_GO_LIVE_CHECKLIST.md) |

## 依工作尋找

| 需求 | 主文件 | 補充文件 |
| --- | --- | --- |
| 日常檢測、紀錄、Excel、異常複核 | [操作者手冊](manuals/OPERATOR_MANUAL.md) | [資料庫](data/INSPECTION_DATABASE.md) |
| 工程設定、補訓、部署與恢復 | [工程維運手冊](manuals/ENGINEERING_MANUAL.md) | [訓練閉環](../../Yolo11_auto_train/docs/SEAMLESS_WORKFLOW.md) |
| 顏色誤殺與門檻校正 | [顏色覆核與校正](model_lifecycle/COLOR_REVIEW_CALIBRATION.md) | [誤判 SOP](operations/MISJUDGE_TRIAGE_SOP.md) |
| 模型／顏色組合驗收 | [模型組合驗收與發布](model_lifecycle/MODEL_COMBINATION_ACCEPTANCE.md) | [發布與回滾](operations/RELEASE_ROLLBACK_SOP.md) |
| 模型命名、版本切換與回復 | [模型版本指南](model_lifecycle/MODEL_VERSION_GUIDE.md) | [發布與回滾](operations/RELEASE_ROLLBACK_SOP.md) |
| Windows 現場部署 | [Windows 部署 SOP](operations/WINDOWS_DEPLOYMENT_SOP.md) | [上線檢查表](operations/PRODUCTION_GO_LIVE_CHECKLIST.md) |
| 相機 Runtime／取像問題 | [相機診斷](operations/CAMERA_RUNTIME_DIAGNOSTICS.md) | `tools/diagnostics/diagnose_camera.bat` |
| SQLite 備份、保存與還原 | [資料庫文件](data/INSPECTION_DATABASE.md) | `tools/maintain_inspection_data.py` |
| 公司伺服器同步 | [公司同步](data/COMPANY_SERVER_SYNC.md) | `tools/inspection_sync_admin.py` |
| PCBA 受控試產 | [PCBA 試產指南](pilot/PCBA_PILOT_GUIDE.md) | [試產驗收紀錄](pilot/PCBA_PILOT_ACCEPTANCE_TEMPLATE.md) |
| 跨類別重複框 Pilot | [Cable1/A 1.0.6 改善企畫](pilot/CROSS_CLASS_DUPLICATE_DETECTION_PROPOSAL.md) | 現場 Gate 尚須完成 |
| 程式模組與執行緒責任 | [模組架構](architecture/MODULE_ARCHITECTURE.md) | [技術指南](architecture/TECH_GUIDE.md) |
| 路徑、YAML 與憑證安全 | [安全設計](architecture/SECURITY.md) | `tests/test_security.py` |
| 受限裝置 runtime 評估 | [Firmware Runtime Plan](architecture/FIRMWARE_RUNTIME_PLAN.md) | 需以 benchmark gate 決定 |
| 門檻／模型／保存政策歷程 | [Calibration Change Log](records/CALIBRATION_CHANGE_LOG.md) | 只追加，不覆寫歷史 |

## 目錄責任

- `manuals/`：操作者與工程角色主手冊。
- `operations/`：部署、回滾、上線、診斷與誤判處理 SOP。
- `data/`：檢測資料庫及公司同步契約。
- `model_lifecycle/`：模型、顏色校正與組合驗收。
- `architecture/`：架構、安全與技術參考；`TECH_GUIDE.md` 是教材，不是操作 SOP。
- `pilot/`：仍需現場 Gate 的受控試產文件。
- `records/`：追加式工程紀錄。
- `archive/`：歷史資料，只供追溯，不作為現行操作依據。

版本演進、工程問題、驗證證據與已知限制記錄於
[`yolo_inference_RevisionNote_v1.0.0_rev2.xlsx`](yolo_inference_RevisionNote_v1.0.0_rev2.xlsx)。
系統畫面顯示版本的唯一來源是 `core/_version.py`，目前正式版為 `1.0.0`。

## 上線界線

文件完整不等於工位已通過生產驗收。無人值守上線前仍須滿足：readiness 無未處理
`FAIL`、所有 `WARN` 有具名接受、Golden OK 重複性通過、已知 NG 驗證完成、
實際啟用的模型／位置／顏色／同步 Gate 完成，以及 rollback 已演練。
