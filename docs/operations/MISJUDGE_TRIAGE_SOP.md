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
- `raw_detections`：重複框處理前的原始框；沒有抑制時與 detections 相同
- `duplicate_filter`：候選、保留／排除 index、IoU、policy 與阻擋原因
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
   （亮度、色溫）與治具，再考慮參數。若疑似環境光/亮度漂移，用選單
   **燈光控制 → 光源校正** 對照「目前亮度」與記錄的目標值，必要時按
   **自動校正**（見第 8 節）。
5. `INFERENCE_ERROR`：不屬誤判，直接走設備異常/當機處理流程
   （見 `docs/operations/CAMERA_RUNTIME_DIAGNOSTICS.md`）。

若結果圖有紫色`DUP`：

1. 先比對`raw_detections`與有效`detections`，不要把紫色框直接標成真實多件；
2. 確認 kept／suppressed 框是否落在同一實體，並核對 IoU、verified class；
3. 若兩個實體被錯誤合併，立即把該產品／工位切回`report_only`並列為
   blocking UNDERKILL 事故；
4. 若同一實體確實產生跨類別雙框，標記為模型重複框案例，送入模型改善資料。

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
2. 變更只能改 config/模型檔，**禁止**改程式碼中的常數。顏色判定參數可在
   `models/<product>/<area>/<type>/config.yaml` 的 `color_decision_tuning`
   區塊設定，**有效鍵名一律以 `config.example.yaml` 為準**：未知的鍵會被
   靜默忽略，不會報錯，所以照著舊筆記填一個已移除的鍵，看起來像改了、
   其實沒有。

   已移除的鍵：黑色門檻（`black_s_threshold` / `black_v_threshold` /
   `black_min_coverage`）不再存在。黑色改由基準學到的 S/V 與 LAB 範圍決定，
   分數是符合範圍的最大連通區塊佔整個偵測框的比例（`stats-robust-v6`；不再
   除以 `coverage_mean`，這個欄位已停用）——沒有可調的手寫門檻，黑色判定要
   移動只能重建基準。`min_blob_pixels`（連通區塊的最小可信像素數，每色共用
   一個值）是 v6 新增的鍵，同樣受下面「不是熱重載」規則約束。

   > **`color_decision_tuning` 不是熱重載即生效的無痛調整。**
   > 已部署的顏色基準會記錄它是在哪一組完整 tuning 下驗證的，執行期會逐鍵
   > 比對。任何一個值改掉，`color_baseline_algorithm_enforcement: strict`
   > 的站台會**拒絕載入基準並停線**，`warn` 的站台則會繼續用不相符的參數
   > 評分。調 tuning 等同於一次重新校正：必須連帶重建基準、重跑驗收矩陣，
   > 並依第 4 點留下紀錄。
3. 變更後必跑回歸集：歷史 OVERKILL/UNDERKILL 案例 + 金板（golden sample），
   確認「舊過殺不復發、舊攔截不放行」。
4. 記錄於 `docs/records/CALIBRATION_CHANGE_LOG.md`：日期、變更項、新舊值、雜湊／版本、
   回歸證據、結果、執行人與批准人。
5. 部署到機台走 `docs/operations/RELEASE_ROLLBACK_SOP.md`，不直接在機台上手改。

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
必須確認該站別已無漏判回查需求，並在
`docs/records/CALIBRATION_CHANGE_LOG.md` 記錄批准人。

---

## 7. 責任分工速查

| 情境 | 動作 | 負責人 |
| --- | --- | --- |
| 單次過殺 | 覆判記錄，不改參數 | 站別工程師 |
| 同班次同原因 ≥ 3 次過殺 | 通報 + 光源/治具檢查 | 站別 → 演算法工程師 |
| 漏判（客訴/退回） | 24 小時內回查 metadata 與原圖 | 演算法工程師 |
| 週期分流 | collect → 標記 → export → 回歸集 | 演算法工程師（每週） |
| 參數變更 | 回歸集全過 + 記錄 + 走發版流程 | 演算法工程師 + 批准人 |
| 亮度/環境光漂移 | 光源校正（記錄目標 / 自動校正） | 站別 → 演算法工程師 |

---

## 8. 光源校正（亮度閉環）

顏色檢查對照的是固定 HSV/LAB 範圍，環境光或 LED 老化造成的**亮度漂移**會
讓判定失準。選單 **燈光控制 → 光源校正** 提供亮度閉環，把當下影像平均亮度
(luma) 拉回記錄的目標值。

**建立基準（每個機種一次，換光源/搬遷後重做）**
1. 選好產品/站別/模型類型，確認相機已連線、檢測未在跑。
2. 開 **光源校正**，畫面顯示「目前亮度」。
3. 在良品/金板、正常光源下按 **記錄目前值**：系統把當下曝光、增益、LED
   亮度（%）與目前亮度寫進該機種 `config.yaml`（`calibration.target_luma`
   等，含 `.bak` 備份）。

**日常校正 / 開線檢查**
1. 放金板、開 **光源校正**，看「目前亮度」是否落在目標 ± 容差內。
2. 偏離就按 **自動校正**：系統以 LED 粗調、曝光微調兩段式逼近目標，收斂後
   把新曝光/增益/LED 寫回機種設定並熱更新；未收斂則不寫入並回報原因。
3. 亮度進容差後，金板不要拿走，接著做第 9 節的**顏色開線檢查**。亮度只是
   顏色的代理指標，量到的顏色才是產線實際比對的東西。

**注意事項**
- 亮度校正只處理明暗，不校色溫。若懷疑是色偏（`COLOR_MISMATCH` 為主）而非
  單純明暗，仍須從光源硬體（遮光、光源老化、白平衡）著手 —— 但先跑第 9 節
  確認到底是不是色偏，不要憑印象判斷。
- 曝光/增益會在**切換模型時自動套用**該機種記錄值（無記錄的機種維持全域值）。
- 校正屬設定變更：大幅調整後仍建議跑一次誤判回歸集（第 5 節）再放行量產。

---

## 9. 顏色開線檢查（餘裕比對）

亮度閉環之後仍有一個沒人量的東西：**顏色實際讀出來對不對，以及離門檻還有多遠**。
以前色偏的第一個證據是一片誤判的板子。這一節把它提前到開線。

**它不會改任何東西。** 用開線時一片板子的讀數去動顏色基準，等於產生一份沒有
證據集、沒有 holdout、沒有具名批准的統計，正是基準契約要擋的東西。沒通過時
只有兩條路：回去做第 8 節的亮度校正／查治具與光源，或升級為顏色基準重建 +
重跑驗收 + 簽核。

操作員走選單 **燈光控制 → 顏色開線檢查**；工程師用
`tools/color_preflight.py`。兩者共用同一個評估服務，判定一致，只有趨勢檢視
（`--trend`）目前只在命令列。

**建立參考餘裕（每個機種一次，換光源/治具/基準後重做）**
1. 亮度已在容差內、金板在治具上，按現有的手動檢測跑**一次**。
2. 開 **顏色開線檢查**，確認畫面上的「拍攝時間」就是剛才那一次
   （會顯示「幾分鐘前」），再按 **記錄為參考餘裕**，並輸入姓名。

   命令列等效：
   ```powershell
   python tools/color_preflight.py --product Cable1 --area A `
       --record-reference --operator "<你的姓名>"
   ```
3. 每色的餘裕（分數減門檻）會寫進該機種 `config.yaml` 的
   `color_preflight`（含 `.bak` 備份），並記下當下基準檔的 sha256。
   **判讀有誤的板子會被拒絕記錄** —— 否則那個瑕疵就成了之後每一班的目標。

**每班開線**
1. 同樣放金板跑一次手動檢測，然後開 **顏色開線檢查**（或執行
   `python tools/color_preflight.py --product Cable1 --area A`）。
   > 對話框讀的是**已存檔的最新一筆檢測**，所以務必核對「拍攝時間」。
   > 忘記跑手動檢測的話，你看到的是上一次那片板子。
2. 看判定：

   | 判定 | 意思 | 處置 |
   | --- | --- | --- |
   | `OK` | 每色都讀對，且餘裕維持在參考的保留率之上 | 放行 |
   | `WARN` | 讀對了，但有事情要知道：某色餘裕大幅衰退（`MARGIN_LOW`）、還沒有參考、參考已過期，或已部署基準早於現行契約 | 可以開線。畫面上的處置說明會指出是哪一種 |
   | `NG` | 有色別讀錯、沒量到，或已低於門檻 | 產線現在就會誤判。先查治具與光源，再考慮重建 |

   `NG` 只保留給「這片板子讀錯了」—— 當班能處理的事。**已部署基準早於現行契約
   算 `WARN`**：那是站台的既有狀態，在基準遷移完成前每一班都成立，天天報 `NG`
   只會訓練出「NG 沒意義」的習慣。真正該為此停線的機制是
   `color_baseline_algorithm_enforcement: strict`，它會直接拒絕載入基準。

   > **參考餘裕會綁定它所量測的那份基準檔（sha256）。** 所以基準一旦重建，
   > 參考自動標記為過期（`REFERENCE_STALE`）並要求重錄 —— 舊數字不會偷偷繼續
   > 被拿來比較。反過來，這也表示**基準還沒遷移到現行契約的站台今天就能用**
   > 這個檢查來偵測漂移，不必等遷移完成。

3. 每次執行都會寫進開線紀錄。看趨勢：
   ```powershell
   python tools/color_preflight.py --product Cable1 --area A --trend 14
   ```

**為什麼用保留率而不是絕對容差**

各色餘裕差一個數量級：紅色可能在門檻之上 0.5，黑色曾只有 0.01。單一絕對容差
會讓黑色掉到零都不報，同時對紅色的正常波動報警。所以判準是
`餘裕 / 參考餘裕 < minimum_margin_retention`（預設 0.6）。參考餘裕本身太薄
（≤ 0.02）時不做比值 —— 那種比值靠雜訊擺動，會回報 `NO_REFERENCE` 而不是
拿一個沒有意義的數字下判斷。

**這個檢查偵測不到什麼**

金板每色只有一到兩個樣本，所以它偵測的是**系統性漂移**（光源老化、治具位移、
部署了錯的基準），不是良率變化。良率仍要靠第 5 節的誤判回歸集。
