# AI 檢測系統工程維運手冊

文件對象：設備、製程、AI、軟體與 IT 工程人員
適用範圍：`yolo11_inference`、與其連接的 `Yolo11_auto_train`
文件擁有者：AI／設備工程
最後核對：2026-07-29

日常檢測與班組長複核操作請先看
[操作者手冊](OPERATOR_MANUAL.md)。本手冊是工程交接主入口；專題細節仍以
文末連結的單一主題文件為準。

## 1. 系統邊界

系統採本機優先設計：

```text
相機／保存影像
    -> yolo11_inference
    -> 判定、證據影像、SQLite
    -> 人工複核
    -> 檔案式 handoff
    -> Yolo11_auto_train
    -> 資料 Gate、YOLO Gate、Position Gate
    -> 版本化權重／設定
    -> yolo11_inference 安全重載

SQLite outbox
    -> 公司 HTTPS API（選配、背景、可離線）
```

關鍵原則：

- 檢測先寫入本機，網路中斷不阻擋產線。
- 權重、設定、位置報告與部署 manifest 是同一版本單位。
- 新模型未通過 Gate 時保留目前產線版本。
- GUI 背景工作透過 Qt signal/slot 回到 UI thread。
- 資料庫短交易內只做狀態更新；HTTP 與大型 I/O 不在交易鎖內執行。

架構細節見 [模組架構](MODULE_ARCHITECTURE.md)。

## 2. 目錄與資料責任

| 路徑 | 用途 | 備份／版本策略 |
| --- | --- | --- |
| `config.yaml` | 站點全域設定 | 隨 release 保存 |
| `config.local.yaml` | 本機覆寫；不進版控 | 站點加密備份 |
| `models/<產品>/<工位>/<類型>/` | 權重、模型設定、位置與顏色設定 | 版本化、保留 checksum |
| `Result/` | 檢測影像、JSON、報表與 SQLite | 依保存政策備份 |
| `Result/inspection_records.sqlite3` | 可查詢檢測、複核與同步索引 | 線上 SQLite 備份 |
| `Result/database_backups/` | 已驗證的資料庫備份 | 另存公司備份位置 |
| `../Yolo11_auto_train/data/` | 補訓資料與 handoff | 不可任意清理 |
| `../Yolo11_auto_train/runs/` | 訓練、驗證與報告 | 依模型 release 保存 |
| `dist/yolo11_inference/` | 完整 Windows 部署包 | 整個資料夾為 release |

影像檔保留在檔案系統，SQLite 只保存可查詢 metadata 與路徑，不把大圖 BLOB
塞進資料庫。

## 3. 設定載入與變更控制

設定優先順序由低至高：

1. 根目錄 `config.yaml`；
2. 同層 `config.local.yaml` 的站點覆寫；
3. `models/<product>/<area>/<type>/config.yaml` 的模型覆寫。

模型設定只套用到該次切換產生的私有設定副本，不直接修改共享全域設定。
新引擎初始化成功後才原子切換；初始化失敗時舊引擎仍可用。

### 3.1 應放在哪個設定檔

| 設定 | 建議位置 |
| --- | --- |
| 公司同步、備份週期、Result 根目錄 | 全域 `config.yaml` |
| 機台名稱、工單、相機序號、本機輸出路徑 | `config.local.yaml` |
| 權重、conf、IoU、imgsz、expected_items | 模型 `config.yaml` |
| 位置 expected boxes／容差 | 模型設定或同模型的 `position_config.yaml` |
| 顏色規則／門檻 | 模型 `config.yaml`／`color_stats.json` |
| Token、密碼、私鑰 | Windows 環境或公司 Secret 管理；禁止寫 YAML |

修改流程：

1. 停止檢測。
2. 備份目前設定與模型版本。
3. 只修改核准範圍。
4. 執行 YAML、路徑與 readiness 驗證。
5. 用 Golden OK 與已知 NG 測試。
6. 記錄修改人、原因、差異與回滾版本。
7. 由 GUI `檢視 > 重新載入模型`或重啟程式啟用。

## 4. 首次部署

### 4.1 接收或建立 release

開發機執行：

```powershell
$env:CI = "1"
$env:YOLO11_PYTHON = "D:\miniconda\envs\yolo_anomalib\python.exe"
.\build_exe.bat
```

必須部署整個：

```text
dist\yolo11_inference\
```

不可只部署 `yolo11_inference.exe`。完整程序見
[Windows 部署 SOP](WINDOWS_DEPLOYMENT_SOP.md)。

### 4.2 封裝與相機預檢

在部署資料夾執行：

```powershell
.\yolo11_inference.exe --check-hikrobot-runtime
.\yolo11_inference.exe --check-camera-grab
```

第一個命令驗證隨附 Runtime；第二個命令需要相機未被其他程式占用。
失敗時依 [相機診斷](CAMERA_RUNTIME_DIAGNOSTICS.md)處理。

### 4.3 資料與恢復預檢

先產生至少一筆已保存檢測，再執行：

```powershell
python -m tools.production_preflight `
  --result-root Result `
  --config config.yaml `
  --backup-restore-drill
```

若該站已要求公司同步，使用 `--strict`，所有檢查必須 PASS：

```powershell
python -m tools.production_preflight `
  --result-root Result `
  --config config.yaml `
  --backup-restore-drill `
  --strict
```

若公司同步尚不在本次 rollout 範圍，`company_sync_configuration` 會是 WARN。
可進行有書面接受的監督式試產，但不可宣稱公司端資料鏈已完成。

## 5. 工程設定頁

從主畫面按`工程設定 >`並輸入 PIN：

- 首次使用預設 PIN 時，系統要求立即更換。
- 連續 5 次錯誤會鎖定約 30 秒。
- PIN 狀態跨視窗／程序序列化，儲存失敗時採 fail-closed。
- 按`返回檢測並鎖定`或`鎖定工程模式`後，下次必須重新驗證。

工程頁包含：

### 5.1 模型補訓

- `開啟補訓資料與送出`：複核、補標、設定參數與建立工作。
- `補訓進度`：查看、續訓、安全停止或清除已終止的畫面紀錄。

### 5.2 檢測評估版本

此工作區統一管理`元件版本`、`候選組合`、`組合驗證`與`上線紀錄`：

- 元件可包含 YOLO、Anomalib、完整顏色基準與顏色方案；
- 候選組合可自由選擇 Pipeline Template 允許的元件，不直接影響產線；
- 組合驗證使用獨立且已人工確認的驗收照片；
- 上線與回滾以完整組合為原子單位，不個別切換 YOLO 或顏色檔；
- 有驗證警告的候選只能限定試用或具名接受風險，`BLOCKED`不得啟用。

五色基準重建、YOLO × 顏色矩陣、指標與啟用規則見
[模型組合驗收與發布](MODEL_COMBINATION_ACCEPTANCE.md)。

### 5.3 相機

- 使用／停用相機；
- 重新連接或中斷；
- 選擇保存影像進行單張工程測試。

相機、產品、模型或設定變更前先停止檢測。

主視窗底部會常駐顯示相機生命週期狀態：

- 黃色代表初始化或重新連線進行中；
- 藍色`相機已連線`只代表 SDK／裝置連線成功，尚未證明取像正常；
- 綠色`相機就緒`必須在 GUI 收到第一張有效影像後才會顯示；
- 紅色代表初始化失敗或執行中失聯，可在停止檢測後直接按狀態旁的
  `重新連線`；
- 灰色代表手動中斷或圖片模式。

相機離線時只停用相機必要的`自動模式`，不全域鎖住單張圖片流程。
若 GUI 顯示已連線但長時間無法進入就緒，先確認沒有其他程式占用相機，
再依[相機診斷](CAMERA_RUNTIME_DIAGNOSTICS.md)執行硬體擷取預檢。

### 5.4 除錯／設定

- 編輯目前模型設定；
- 模型版本／還原；
- 查看目前輸出路徑；
- 顯示／隱藏檢測框、原始分頁與處理後分頁。

模型設定中的兩個 IoU 用途不同：

- `YOLO NMS IoU（同類框）`是模型原本的分類別 NMS 門檻；
- `跨類別重疊 IoU`只用於顏色複核後的保守重複框處理。

`跨類別重複框`可選`僅觀察`或`消除`。自動消除必須同時通過高 IoU、
中心距離、面積相似度、不同 YOLO 原類別、相同 verified class 與兩筆顏色
PASS。位置檢測啟用時，首版會 fail-closed 顯示
`blocked_position_enabled`，不會消除框。不要以降低門檻或關閉 strict count
取代此規則。

歷史資料唯讀重播：

```powershell
python tools/audit_cross_class_duplicates.py Result\20260729\Cable1\A
```

需要程式讀取時加`--json`。停用或回滾時，在模型設定取消啟用；Pipeline 會
同步移除`cross_class_duplicate_filter`，已保存的 raw detections 不受影響。
Cable1/A 1.0.6 的現行設定與版本設定快照都包含相同 policy，避免切換版本後
遺失本次修正。

### 5.5 自動觸發校正

1. 選好產品與工位。
2. 連接相機並啟動自動模式預覽。
3. 治具清空後按`取樣空框`。
4. 放入標準產品後按`取樣產品`。
5. 確認畫面目標仍是同一產品／工位。
6. 檢查系統計算的中間閾值。
7. 按`套用閾值`。
8. 重啟循環並以空框、正常品與移除動作驗證。

更換產品／工位後先前取樣會失效。校正只處理產品有無的觸發閾值，不等於
模型精度或位置 Golden Set 校正。

## 6. 新產品或新工位

1. 建立：

```text
models/<product>/<area>/<type>/
  config.yaml
  weights/
    <versioned-weight>.pt 或 .onnx
```

2. 模型設定至少核對：

- `weights`；
- `imgsz`、`conf_thres`、`iou_thres`；
- `expected_items`；
- `pipeline`與各 `steps`；
- 位置、顏色與 fail-closed 行為；
- `defect_coverage` 的已涵蓋／未涵蓋缺陷。

3. 確保產品、工位、類別名稱與順序在推論、訓練與部署 manifest 中完全一致。
4. 執行產品 readiness gate。
5. 以 Golden OK、已知 NG 及整班 dry run 驗收。
6. 保存接受報告與回滾版本。

不得只放入 `best.pt` 就視為完整模型部署。

## 7. 補訓閉環

### 7.1 正式入口

目前正式入口為：

```text
主 GUI
  -> 選擇產品／工位
  -> 停止檢測
  -> 工程設定
  -> 模型補訓
  -> 開啟補訓資料與送出
```

舊文件中的`檔案 > 訓練資料複核與提交`已不是目前 GUI 入口。
根目錄的一鍵批次檔仍可作為獨立備援入口，但日常應從主 GUI 保留正確產品脈絡。

### 7.2 資料路由

| 人工結果 | 路由 |
| --- | --- |
| 框正確的確認 NG、確認空圖 | YOLO 直接訓練 |
| 漏檢、誤檢、錯框、錯類 | LabelImg 補標後進 YOLO |
| 顏色確認 NG／顏色誤殺 | 顏色校正資料 |
| 純位置誤殺 | 位置 Golden OK |
| 只有 `POSITION_SHIFT` 的確認 NG | 位置 Golden NG |
| 模糊、過曝、遮擋、無法判定 | 保留稽核，不送訓 |

同一影像以 SHA-256 穩定識別，重複提交不重複累積；人工改判會撤銷過時的
raw/label。詳細語意見
[`Yolo11_auto_train/docs/SEAMLESS_WORKFLOW.md`](../../Yolo11_auto_train/docs/SEAMLESS_WORKFLOW.md)。

### 7.3 補訓參數

建議值：

| 項目 | 建議值 | 邊界 |
| --- | ---: | --- |
| Epochs | 20 | 以 GUI 驗證範圍為準 |
| 每張原圖增強 | 20 | 0 表示只用原圖 |
| Batch | 8 | GPU 記憶體不足時才下調 |
| 圖片尺寸 | 640 | 改動需重新做精度／效能驗收 |

Epochs、增強數、Batch、imgsz 可記住；位置訓練與啟用選項刻意不記住，
每筆工作都要重新決策。

### 7.4 安全 Gate

固定流程：

```text
人工覆核
-> 標註驗證
-> YOLO 標註感知增強
-> dataset lint/readiness
-> family-aware train/val/test split
-> 從配對且 checksum 正確的現行 PT 續訓
-> YOLO challenger/incumbent 同 test set 比較
-> 選配 Position calibration/validation/gate
-> ONNX/PT 成對驗證
-> 版本化部署
```

最低資料安全下限包括每類有效 instance、train/val/test 圖片與 split 隔離。
產品設定只能提高，不可降低硬性安全下限。

## 8. 位置檢測補訓

位置與 YOLO 共用部署交易，但分別通過 YOLO Gate 與 Position Gate。

### 8.1 GUI 選項

`啟用位置檢測補訓`：

- 關閉：`position_training_mode=yolo_only`；
- 開啟：`position_training_mode=calibrate_validate`。

`位置驗證通過後啟用現場位置檢測`：

- 關閉：`position_activation=preserve`；
- 開啟：`position_activation=enable_after_gate`。

若現場位置檢測已啟用，YOLO-only 會在訓練前被阻擋，避免權重改變後沿用不相容
的位置基準。

### 8.2 校正與 Golden Set 必須分離

| 資料 | 用途 |
| --- | --- |
| calibration images + human YOLO labels | 計算中心、尺寸與容差 |
| Golden OK | 正常品誤殺率 |
| Golden NG | 純位置異常召回率 |

預設要求至少 10 張可用 Golden OK。Golden manifest 只接受：

- `position_false_reject`；
- 只有 `POSITION_SHIFT` 的 `confirmed_ng`。

缺件、錯類、錯框、顏色或混合原因不納入 Position Gate。
校正集與 Golden Set 以 SHA-256 證明不重疊。

### 8.3 `no eligible samples`處理

看到：

```text
Position golden manifest has no eligible samples in the holdout image directory
```

依序檢查：

1. 本次工作是否誤勾位置補訓。
2. manifest 是否有 `position_false_reject`。
3. 確認 NG 是否只有 `POSITION_SHIFT`。
4. 合格影像是否實際存在於 test holdout。
5. calibration 與 holdout 是否因 SHA-256 重疊被排除。
6. 正常樣本是否達 Gate 下限。

資料不足時取消位置補訓或先收集證據，不得把整個 test split 假設為位置 OK。

### 8.4 首次啟用

只有完成以下項目才可用 `enable_after_gate`：

- 真實治具量測；
- 相機、鏡頭、焦距、解析度與方向鎖定；
- Golden OK 重複性；
- 真實位置 NG 或經批准的受控偏移樣本；
- 誤殺率與召回率門檻；
- 權重、位置設定、報告與 manifest 的成對回滾。

完整設定與輸出見
[`POSITION_RETRAINING_DEPLOYMENT.md`](../../Yolo11_auto_train/docs/POSITION_RETRAINING_DEPLOYMENT.md)。

## 9. 補訓工作狀態與恢復

| 狀態 | 工程判讀 |
| --- | --- |
| 等待處理 | 已建立工作，尚未進入處理 |
| 累積改善案例 | 資料安全保存但不足以訓練 |
| 等待補標 | LabelImg 工作尚未完成 |
| 準備資料 | 正在 lint、增強或切分 |
| 模型訓練 | 訓練程序持有工作 |
| 品質驗證 | 正在評估 challenger 與 incumbent |
| 部署模型 | 正在發布版本化產物 |
| 部署完成 | 全部 Gate 通過且已發布 |
| 失敗 | 檢查目前說明與 error；舊模型仍保留 |
| 已停止 | 可安全續訓 |
| 心跳中斷 | 先確認原程序，不得立即開第二份 |
| 紀錄損壞 | 保存工作目錄後進行工程調查 |

續訓前確認原 PID 不再活動。`繼續這筆補訓`只在 queued、
waiting_annotation、failed 或 cancelled 等可處理狀態啟用。
安全停止以控制要求交給訓練程序，不用 Task Manager 強殺。

常見失敗：

| 訊息 | 原因與處置 |
| --- | --- |
| `already has an active training job` | 已有同產品／工位工作；回到進度頁續用原工作 |
| `Training was not started` | 檢查 handoff、標註與啟動程序 |
| `Identical images have conflicting labels` | 同 SHA-256 圖片標註衝突；人工統一真值 |
| `Position golden manifest has no eligible samples` | 依第 8.3 節補位置專用證據 |
| 品質比較未通過 | 保留 incumbent；分析 precision/recall/mAP 與案例 |
| 缺少 PT／部署紀錄 | ONNX 沒有配對 training PT；禁止冒用基礎模型續訓 |

## 10. 檢測組合版本、切換與回滾

檢測評估版本頁只在停止檢測後操作。正式切換以完整組合為單位：

1. 選產品、工位與 Pipeline Template。
2. 核對每個元件版本、來源、checksum、驗收報告與啟用限制。
3. 依驗證結果選擇完整上線、限定試用或具名風險接受。
4. 清除模型 cache 並重新載入。
5. 以 Golden OK 與已知 NG 驗證。

回滾必須退回前一完整發布組合。位置檢測組合至少同時包含：

- runtime weight；
- training PT；
- 模型 `config.yaml`；
- position config；
- deployment manifest；
- Position Gate／validation 報告。

不可只把檔名改回 `best.pt`。完整程序見
[模型組合驗收與發布](MODEL_COMBINATION_ACCEPTANCE.md)、
[模型版本指南](MODEL_VERSION_GUIDE.md)及
[發布與回滾 SOP](RELEASE_ROLLBACK_SOP.md)。

## 11. 檢測資料庫與 Excel

資料庫：

```text
Result/inspection_records.sqlite3
```

目前 schema v3 主要資料表：

- `inspections`；
- `ai_predictions`；
- `inspection_artifacts`；
- `review_events`；
- `maintenance_events`；
- `inspection_sync_outbox`。

啟動時會做 integrity check、版本化 migration，並在 migration 前備份。
檢測與複核更新和 outbox revision 在同一交易提交。

歷史結果重建索引：

```powershell
python -m tools.rebuild_inspection_database --result-root Result
```

GUI Excel 匯出讀取 SQLite 一致性快照，分成摘要、明細、原因三張工作表。
它不取代 SQLite、原圖或正式備份。

## 12. 備份、保存與還原

### 12.1 手動驗證備份

```powershell
python -m tools.maintain_inspection_data `
  --result-root Result `
  --backup-only
```

### 12.2 保存政策預覽

預設不自動刪圖。先執行 dry-run：

```powershell
python -m tools.maintain_inspection_data --result-root Result
```

審核 JSON 候選後才可：

```powershell
python -m tools.maintain_inspection_data --result-root Result --apply
```

Apply 前會建立資料庫備份。工具只處理 Result 下已知影像 artifact，不刪除
SQLite metadata、review events、JSON、Excel 或 Result 外檔案。

### 12.3 還原

先驗證，不變更目前資料庫：

```powershell
python -m tools.restore_inspection_database `
  --result-root Result `
  --backup Result\database_backups\<backup>.sqlite3.bak
```

確認所有使用同一 Result 的 GUI 都已關閉，再執行：

```powershell
python -m tools.restore_inspection_database `
  --result-root Result `
  --backup Result\database_backups\<backup>.sqlite3.bak `
  --confirm-application-closed
```

還原前會再備份目前資料庫。若無法證明所有程式已關閉，不得執行 restore。

## 13. 公司伺服器同步

### 13.1 站點設定

```yaml
inspection_sync_enabled: true
inspection_sync_endpoint: "https://inspection-api.company/api/v1/inspections"
inspection_sync_api_token_env: "YOLO11_INSPECTION_SYNC_TOKEN"
inspection_sync_timeout_seconds: 10
inspection_sync_interval_seconds: 30
inspection_sync_batch_size: 20
inspection_sync_max_attempts: 12
inspection_sync_allow_insecure_http: false
```

Token 由 IT 放在 Windows 環境，不寫進 YAML：

```powershell
setx YOLO11_INSPECTION_SYNC_TOKEN "<IT issued token>"
```

重新登入 Windows 或重啟承載 GUI 的程序後才會讀到新的環境變數。

### 13.2 API 必備契約

- HTTPS（HTTP 只允許 localhost 測試或明確工程例外）；
- Bearer Token；
- `Idempotency-Key=<inspection_id>`；
- `X-Inspection-Revision=<revision>`；
- 同 inspection/revision 重送必須回同一成功結果；
- 只有較新 revision 可更新公司資料；
- 2xx 必須代表公司端交易已提交。

目前只同步 metadata、predictions、複核狀態與站點檔案路徑，不傳影像 bytes。
影像上傳需另設 checksummed object-storage 契約。

### 13.3 管理與故障處理

```powershell
python -m tools.inspection_sync_admin --result-root Result
```

先修復網路、憑證、Token 或 API 後，才重排 dead-letter：

```powershell
python -m tools.inspection_sync_admin --result-root Result --retry-dead
```

不要為了清除紅色狀態直接刪除 outbox。完整伺服器契約與試點步驟見
[公司同步文件](COMPANY_SERVER_SYNC.md)。

## 14. 日常監控

每日：

- Result 可寫；
- 磁碟空間；
- 最近備份時間；
- ERROR、連續 NG 與異常原因分布；
- 同步 pending/dead；
- 補訓是否存在 unresponsive 或 active 重複工作。

每週：

- Golden OK 重複性；
- 已知 NG 召回；
- 誤殺／漏檢趨勢；
- 模型版本與現場通知一致；
- 備份可讀性抽查。

每次 release：

```powershell
python -m tools.production_preflight `
  --result-root Result `
  --config config.yaml `
  --backup-restore-drill
```

並保存 JSON 版本：

```powershell
python -m tools.production_preflight `
  --result-root Result `
  --config config.yaml `
  --backup-restore-drill `
  --json > production_preflight.json
```

## 15. 事故處理

### 15.1 產線優先原則

1. 隔離最後已知正常時間之後的產品。
2. 停止自動模式並保留畫面、log、inspection ID 與版本。
3. 不刪除或覆蓋失敗 release。
4. 先判斷是實物、光學、相機、模型、設定、磁碟、資料庫或網路。
5. 需要回滾時以完整 release 單位回復。
6. 驗證 Golden OK 與已知 NG 後再恢復。

### 15.2 不同故障域

| 故障域 | 證據 |
| --- | --- |
| 相機／網路 | camera diagnostics、SDK error、grab preflight |
| 模型／設定 | model version、config checksum、load error |
| 品質 | inspection ID、原圖、標註圖、原因、人工真值 |
| 儲存／DB | free disk、integrity check、最近備份 |
| 補訓 | job ID、status.json、current task、heartbeat、error |
| 公司同步 | pending/dead、HTTP 狀態、伺服器 request ID |

禁止在支援 log 中記錄 API Token 或其他憑證。

## 16. 發布驗收

程式可啟動不等於可無人值守生產。每個產品／工位至少完成：

```text
[ ] runtime 與相機預檢 PASS
[ ] production preflight 無未接受的 FAIL/WARN
[ ] 模型與設定 checksum 已保存
[ ] Golden OK 重複性完成
[ ] 已知 NG 原因符合預期
[ ] 位置／顏色 Gate 依實際啟用功能完成
[ ] 一班 dry run 與人工複核完成
[ ] Excel／SQLite／備份可讀
[ ] 公司同步（若在範圍內）完成離線重連與去重測試
[ ] 回滾 release 已保存並演練
[ ] 操作者、工程、IT 責任人與聯絡方式已填寫
```

## 17. 文件維護規則

以下變更必須同一個 release 更新文件：

- GUI 入口、按鈕或角色權限；
- config key、預設值或 schema；
- 補訓／位置 Gate；
- SQLite schema、保存或還原；
- 公司 API schema、認證或重試規則；
- release、部署或回滾方式。

文件核對方式：

1. 按操作者手冊完成一次圖片單次檢測。
2. 按操作者手冊查詢並匯出一次 Excel。
3. 建立測試複核工作但不部署。
4. 按工程手冊執行 preflight 與備份驗證。
5. 對照 `config.example.yaml`與 GUI 實際文字。

## 18. 專題文件

- [文件總索引](DOCUMENTATION_INDEX.md)
- [Windows 部署 SOP](WINDOWS_DEPLOYMENT_SOP.md)
- [發布與回滾 SOP](RELEASE_ROLLBACK_SOP.md)
- [正式上線檢查表](PRODUCTION_GO_LIVE_CHECKLIST.md)
- [資料庫、備份與保存](INSPECTION_DATABASE.md)
- [公司伺服器同步](COMPANY_SERVER_SYNC.md)
- [相機診斷](CAMERA_RUNTIME_DIAGNOSTICS.md)
- [誤判與漏檢處理](MISJUDGE_TRIAGE_SOP.md)
- [顏色覆核與校正](COLOR_REVIEW_CALIBRATION.md)
- [模型組合驗收與發布](MODEL_COMBINATION_ACCEPTANCE.md)
- [模型版本指南](MODEL_VERSION_GUIDE.md)
- [補訓閉環](../../Yolo11_auto_train/docs/SEAMLESS_WORKFLOW.md)
- [位置補訓與部署](../../Yolo11_auto_train/docs/POSITION_RETRAINING_DEPLOYMENT.md)
