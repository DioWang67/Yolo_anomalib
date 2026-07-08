# 誤判/漏判分流 SOP（Misjudge Triage SOP）

本文件定義產線發生誤判（過殺，false FAIL）與漏判（漏檢，false PASS）時的
處置流程、資料保留規則與參數變更管制。目標是：任何一筆判定都能離線重現、
任何一次參數調整都有回歸證據。

適用對象：站別工程師（第一線處置）、演算法/系統工程師（週期分流與參數變更）。

---

## 1. 名詞定義

| 名詞 | 定義 |
| --- | --- |
| 過殺（OVERKILL） | 良品被判 FAIL。直接損失：重測工時、誤報 |
| 漏判（UNDERKILL） | 不良品被判 PASS。直接損失：流出，成本最高 |
| 真實 FAIL（TRUE_FAIL） | 判 FAIL 且確為不良品，屬正常攔截 |
| 覆判 | 人工對機器判定結果的再確認 |

---

## 2. 資料基礎：每筆檢測的可回溯記錄

每次檢測會在 `Result/<日期>/.../metadata/<detector>/` 落一份
`*_config_snapshot.json`（schema_version 2），內容包含：

- `status` / `fail_reasons`：最終判定與**機器可讀失敗原因碼**（見下表）
- `detections`：各偵測框（含 `verified_class`、位置檢查欄位）
- `missing_items`、`color_result`、`sequence_check`、`anomaly_score`
- `model_info`：權重路徑、`model_version`、conf/iou 閾值
- `artifacts`：原圖 / 前處理圖 / 標註圖 / 熱圖 / 裁剪圖路徑
- `config_hash` 與完整 `config`：當下生效的所有設定

`fail_reasons` 原因碼（定義於 `core/services/decision_engine.py`）：

| 代碼 | 意義 | 常見根因方向 |
| --- | --- | --- |
| `MISSING` | 應有零件未檢出 | 漏檢（conf 過高）、遮擋、光源 |
| `WRONG_COMPONENT` | 槽位偵測到錯誤類別 | 上錯料、模型混淆 |
| `POSITION_SHIFT` | 位置超出容差 | 治具鬆動、板彎、容差過嚴 |
| `BOARD_ALIGNMENT` | 整板對位失敗 | 治具/相機位移 |
| `UNEXPECTED_COMPONENT` | 出現不該有的類別 | 多料、誤檢 |
| `COLOR_MISMATCH` | 顏色檢查不符 | 光源漂移、白平衡、色域參數 |
| `SEQUENCE_MISMATCH` | 線序/順序錯誤 | 實際錯線或顏色誤驗證 |
| `ANOMALY_DETECTED` | anomalib 異常分數超標 | 實際缺陷或閾值過嚴 |
| `INFERENCE_ERROR` | 推論本身失敗 | 模型/相機/系統問題，走設備異常流程 |

---

## 3. 現場當下處置（站別工程師）

1. 機台判 FAIL、人工覆判認為是良品時：**不得**當場修改任何閾值或設定。
2. 在覆判記錄（紙本或站別表單）記下：時間、產品/站別、覆判結論。
3. 確認該筆的標註圖與 metadata 已存在於 `Result/` 對應日期目錄
   （系統自動保存，不需手動操作）。
4. 同一班次同一原因碼過殺 **≥ 3 次**：通報演算法工程師，並優先檢查光源
   （亮度、色溫）與治具，再考慮參數。
5. `INFERENCE_ERROR`：不屬誤判，直接走設備異常/當機處理流程
   （見 `docs/CAMERA_RUNTIME_DIAGNOSTICS.md`）。

漏判（客訴/下游站退回）發生時：依據流水時間回查 `Result/` 當日 PASS 記錄，
取出該筆 `*_config_snapshot.json` 與原圖，進入第 4 節分流。

---

## 4. 週期分流（演算法工程師，建議每週）

### 4.1 收集待覆判案例

```powershell
# FAIL 案例（預設）；要含 PASS（查漏判）加 --include-pass
python tools/collect_review_cases.py --result-root Result `
    --output-csv review/review_manifest.csv
```

產出的 `review_manifest.csv` 每列一筆案例，含 `decision_reasons`、
`model_version`、標註圖與裁剪圖路徑。

### 4.2 人工標記

在 `review_manifest.csv` 的 `review_label` 欄填入（`review_note` 填佐證）：

- `TRUE_FAIL`：正確攔截
- `OVERKILL`：過殺（良品被判 FAIL）
- `UNDERKILL`：漏判（僅出現在 `--include-pass` 的回查）
- `UNCLEAR`：影像證據不足，需補拍或現場確認

### 4.3 匯出成資料集 / 回歸集

```powershell
python tools/export_review_dataset.py `
    --manifest-csv review/review_manifest.csv `
    --output-dir datasets/review_batch_<日期> `
    --source-kind both --include-label OVERKILL --include-label UNDERKILL
```

匯出的 OVERKILL / UNDERKILL 影像有兩個用途：

1. 併入**回歸集**（見第 5 節），之後任何參數變更必跑；
2. 累積達重訓門檻時作為增量訓練樣本。

---

## 5. 參數變更管制

任何閾值/設定變更（conf、iou、顏色參數、位置容差、anomalib 閾值）：

1. 變更前記下當前 `config_hash`（任一近期 snapshot 內有）。
2. 變更只能改 config/模型檔，**禁止**改程式碼中的常數。顏色判定參數
   （黃色色相窗、黑色門檻、橘紅平手邊界等）已可在
   `models/<product>/<area>/<type>/config.yaml` 的 `color_decision_tuning`
   區塊設定（鍵名見 `config.example.yaml`），存檔即生效（mtime 熱重載），
   不需重新打包。
3. 變更後必跑回歸集：歷史 OVERKILL/UNDERKILL 案例 + 金板（golden sample），
   確認「舊過殺不復發、舊攔截不放行」。
4. 記錄於 `docs/PROGRESS_LOG.md`：日期、變更項、新舊值、回歸結果、批准人。
5. 部署到機台走 `docs/RELEASE_ROLLBACK_SOP.md`，不直接在機台上手改。

重訓觸發條件（滿足其一）：

- 同一原因碼的 OVERKILL 連續兩週 > 全部檢測數的 2%；
- 出現任何一筆確認的 UNDERKILL 且無法以閾值解釋；
- 產品外觀/物料變更（換供應商、換色、換板）。

---

## 6. FAIL 影像保留政策

| 資料 | 保留規則 |
| --- | --- |
| FAIL：原圖 + 標註圖 + 裁剪圖 + metadata JSON | 全數保留 ≥ 90 天 |
| PASS：metadata JSON | 全數保留 ≥ 90 天（體積小，供漏判回查） |
| PASS：影像 | 至少保留 30 天供漏判回查；磁碟緊張時最先清理 |
| results.xlsx | 隨月份歸檔，永久保留 |

磁碟清理順序（先清 1，最後清 4）：

1. 逾 30 天的 PASS 影像
2. 逾 90 天的 FAIL 預處理圖（保留原圖與標註圖）
3. 逾 180 天的 FAIL 全部影像（metadata JSON 仍保留）
4. metadata JSON 與 Excel：非經主管批准不清理

注意：`config.yaml` 的 `save_fail_only: true` 會停存 PASS 影像。啟用前
必須確認該站別已無漏判回查需求，並在 PROGRESS_LOG 記錄批准人。

---

## 7. 責任分工速查

| 情境 | 動作 | 負責人 |
| --- | --- | --- |
| 單次過殺 | 覆判記錄，不改參數 | 站別工程師 |
| 同班次同原因 ≥ 3 次過殺 | 通報 + 光源/治具檢查 | 站別 → 演算法工程師 |
| 漏判（客訴/退回） | 24 小時內回查 metadata 與原圖 | 演算法工程師 |
| 週期分流 | collect → 標記 → export → 回歸集 | 演算法工程師（每週） |
| 參數變更 | 回歸集全過 + 記錄 + 走發版流程 | 演算法工程師 + 批准人 |
