# Cable1/A 1.0.6 跨類別重複框問題與改善企畫書

文件狀態：**已實作／曾於 2026-08-05 至 08-19 靜默失效並已修復／Cable1/A 維持 `report_only`／現場 Gate 尚待完成**
適用範圍：`Cable1 / A / YOLO 1.0.6`
建立日期：2026-07-29
最後更新：2026-08-19（見 §17 回歸事件紀錄）
變更類型：推論後處理、結果可視化、品質 Gate
目前執行設定：`Cable1/A 1.0.6` 維持 `report_only`；未修改 YOLO NMS
`iou_thres=0.45`，也未修改線序。位置檢測啟用時會 fail-closed 停止自動消除。

§17 的修復讓本機制**第一次能真正產生 report-only 候選資料**，但不改變
§10／§15 的切換條件：`suppress` 仍須完成 500 次、完整班次與具名批准。
`tests/test_pilot_config_safety.py` 會強制 config 維持 `report_only`。

> **⚠️ 讀本文件前必看**：本文件在 2026-08-05 至 08-19 期間所描述的機制**實際上無法
> 觸發**。§6.2 條件 6 與 §6.3 規則 6 所寫的「兩框顏色檢查都必須通過」，在顏色檢查
> 加入「類別須與量測一致」的閘門後變成**邏輯上不可滿足**，使整個過濾器成為死碼。
> §6.3 規則 2「以 YOLO confidence 決定保留框」也已證實會誤殺良品。兩者均已於
> 2026-08-19 修正，相關條文已在下文就地更新。**完整事件經過、證據與修正見 §17。**

## 1. 審核摘要

### 1.1 問題

YOLO 1.0.6 會在同一個實體端子上輸出兩個座標幾乎相同、但原始類別不同的
偵測框。Ultralytics 預設使用分類別 NMS；不同類別的框不互相抑制。顏色校驗
又可能把兩個框修正為相同顏色，造成：

- 嚴格數量檢查判定多一件；
- 排列檢查收到七個項目而判定長度不符；
- 後畫的框完全蓋住先畫的框，操作者看不出多在哪；
- 保存的決策只記錄「多一個 Orange」，沒有指出對應框號與重疊關係。

### 1.2 建議

不直接全域啟用低門檻的 class-agnostic NMS。建議在`顏色校驗`之後、
`數量／排列檢查`之前加入可關閉、可觀察、依產品／工位啟用的
`cross_class_duplicate_filter`：

1. 只把高度重疊、中心與尺寸皆近似、且顏色校驗後類別相同的框列為候選；
2. 支援`report_only`觀察，不改判定；
3. Cable1/A 經 2026-07-29 歷史紀錄離線重播後，維持`report_only`收集受控 Pilot 證據；
4. 保留原始框、有效框與被抑制框，不能破壞追溯證據；
5. 結果圖與詳細結果必須指出重複框編號、位置及 IoU。

### 1.3 本次不處理

- 不關閉嚴格數量檢查；
- 不以提高 confidence 門檻掩蓋問題；
- 不直接修改全域 NMS 行為；
- 不在未確認產品規格前修改線序；
- 不把此修正宣稱為模型補訓完成。

## 2. 問題證據

### 2.1 代表案例

保存紀錄：

```text
Result/20260729/Cable1/A/DETECTION_FAIL/metadata/yolo/
yolo_Cable1_A_163152_462272_ece1db75c2f9_config_snapshot.json
```

模型與設定：

```text
model_version = 1.0.6
weights = Cable1_A_v1.0.6_20260727.onnx
conf_thres = 0.40
iou_thres = 0.45
```

同一橘色端子的兩個原始框：

| 編號 | YOLO 原始類別 | YOLO confidence | 顏色校驗後 | bbox |
| --- | --- | ---: | --- | --- |
| `#5` | Orange | 0.6670 | Orange | `[114, 353, 141, 395]` |
| `#6` | Red | 0.5054 | Orange | `[114, 353, 141, 396]` |

兩框 IoU 為 `0.977`。`#6`後畫且座標幾乎相同，因此覆蓋`#5`。

本次保存決策：

```text
UNEXPECTED_COMPONENT: Orange
SEQUENCE_MISMATCH: length_mismatch
observed: Red, Orange, Orange, Green, Yellow, Black, Black
```

### 2.2 發生頻率

實作後讀取全日 PASS／NG 紀錄的重播範圍：

```text
Result/20260729/Cable1/A/**/*_config_snapshot.json
```

共掃描 252 筆紀錄；用正式候選條件（IoU ≥ 0.90、中心距離、面積相似度、
相同 verified class、兩框顏色均通過）找到 15 筆候選，全部位於
DETECTION_FAIL，PASS 紀錄為 0 筆候選。其中 14 筆為 1.0.6、1 筆為 1.0.5，
IoU 範圍 `0.913～1.00`。多數為`Orange / Red`最終都校驗成 Orange；另有
`Black / Yellow`最終都校驗成 Black 的案例。因此問題不是單張影像或單一
顏色的偶發事件。

此統計只代表目前已保存的失敗紀錄，不可直接當作整班誤判率；完整誤判率
必須以全部 PASS、NG、ERROR 紀錄為分母重新計算。

### 2.3 現行流程根因

現行主要流程：

```text
YOLO 推論
  → 分類別 NMS
  → detections
  → color_check（可能修正 verified_class）
  → count_check
  → sequence_check
  → save_results
```

根因鏈：

1. 同一端子同時輸出 Orange 與 Red 框；
2. 因原始類別不同，分類別 NMS 保留兩者；
3. color check 將兩者的 `verified_class`都判成 Orange；
4. count check 以 `verified_class`計數，得到兩個 Orange；
5. sequence check 也收到兩個 Orange；
6. annotation 依序畫框，後一個框遮住前一個框。

## 3. 影響分析

### 3.1 品質影響

- 正常品可能被誤判為`UNEXPECTED_COMPONENT`。
- 排列長度連帶失敗，形成第二個衍生原因。
- 若直接關閉嚴格數量檢查，真正多件可能被錯放，禁止採用。
- 誤判影像若未正確複核，可能污染後續補訓資料。

### 3.2 操作影響

- 操作者看得到七筆明細，但結果圖只看得到六個標籤。
- 現行訊息沒有指出哪兩個框重疊。
- 操作者無法分辨模型重複框與實物真正多件。

### 3.3 資料與追溯影響

- decision 保存`Orange`，但缺少候選框編號、IoU 與保留理由。
- 同一實體會保存兩份高度重複 crop。
- Excel、SQLite 與補訓複核只看到衍生結果，缺少後處理證據。

## 4. 需求與安全邊界

### 4.1 功能需求

1. 辨識高度可信的跨類別重複框候選。
2. 允許依產品／工位獨立開關。
3. 支援`off`、`report_only`、`suppress`三種模式。
4. 原始 detections 必須不可變保存。
5. 首版 count、sequence、保存與 GUI 使用同一份 effective detections。
6. 結果圖和詳細結果能定位重複框。
7. 每次抑制都必須有可稽核原因。
8. 首版只允許用於 position check 關閉的產品／工位；position 啟用時
   fail-closed，不得自動抑制。
9. filter 不得直接設定 PASS／NG；它只產生 effective detections，最終狀態
   仍由既有 count、sequence、color 與其他 Gate 合併決定。

### 4.2 邊界案例

- 兩個不同實體真的互相遮擋；
- 相鄰端子框有部分重疊但中心不同；
- 原始類別不同，顏色校驗後仍不同；
- 同類別重複框已被一般 NMS 處理；
- 三個以上框形成同一重疊群組；
- 顏色校驗停用或沒有有效結果；
- position expected boxes 尚未啟用或基準不可靠；
- 真正多一個元件恰好放在既有元件上方；
- ONNX、PT、OpenVINO 的輸出有微小座標差異。

任何條件不足或資料矛盾的候選都不得自動抑制。

## 5. 方案比較

| 方案 | 優點 | 主要風險 | 結論 |
| --- | --- | --- | --- |
| A. 全域 `agnostic_nms=True` | 實作最少 | 沿用 IoU 0.45 時，可能誤刪真正相鄰的不同類別元件；缺少產品範圍與稽核資料 | 不採用 |
| B. 提高 confidence | 可能移除本案例的低分 Red | 其他案例的錯誤框可能分數更高，也可能漏掉真正低分元件 | 不採用 |
| C. 關閉 strict count | 可消除多件 NG | 真正多件會被放行，屬安全倒退 | 禁止 |
| D. 只改善顯示 | 操作者看得懂 | 仍持續誤判 | 僅作必要配套 |
| E. 補訓後再處理 | 可降低模型類別混淆 | 週期較長，無法保證完全不再重複 | 長期措施 |
| F. 顏色後保守條件式去重 | 可利用 verified class，能分產品啟用並保留證據 | 需建立 Gate，避免誤刪真正重疊實體 | 建議方案 |

## 6. 建議架構

### 6.1 流程位置

```text
YOLO raw detections
  → 現行 position validation（Cable1/A 目前停用）
  → color_check
  → cross_class_duplicate_filter
      ├─ raw_detections（不可變證據）
      ├─ effective_detections（供後續判定）
      └─ suppressed_duplicates（稽核資料）
  → count_check
  → sequence_check
  → annotation
  → persistence
```

此步驟不得同時負責數量判定、排列判定或 GUI 畫圖，以維持單一職責。

現行 position validation 位於模型推論內，早於 pipeline color check。
`Cable1/A`目前 position check 關閉，因此首版不存在重跑 position 的問題。
若未來要在 position 啟用的工位使用，必須另案把 raw／effective detections
契約前移，或以 effective detections 重新執行 position validation；不得直接
沿用 raw position 結果。

### 6.2 候選條件

建議初始門檻，仍須由 shadow data 核准：

```yaml
cross_class_duplicate_filter:
  enabled: false
  mode: report_only       # report_only | suppress；停用請設 enabled: false
  iou_threshold: 0.90
  center_distance_ratio_max: 0.10
  area_similarity_min: 0.80
  require_same_verified_class: true
  require_color_check_pass: true
  require_different_raw_class: true
  require_position_disabled: true
```

一對框必須同時符合：

1. bbox IoU `>= 0.90`；
2. 中心距離除以較小框對角線 `<= 0.10`；
3. 較小面積／較大面積 `>= 0.80`；
4. YOLO 原始類別不同；
5. 顏色校驗後 `verified_class`相同且非空；
6. 兩框的顏色**量測**都必須可信（`measurement_is_ok`，即量測本身通過自己的
   門檻）；
7. 兩框皆有有效 bbox、confidence 與類別；
8. 沒有證據顯示兩者屬於不同實體槽位。

> **條件 6 的修訂（2026-08-19）**：原文為「兩框的顏色檢查都必須通過」，實作讀取
> `is_ok`。但 `is_ok` 同時要求「量測可信」**且**「量測與 YOLO 類別一致」，而條件 4
> 要求兩框類別不同 — 兩框蓋同一個實體只會量到一個顏色，最多只有一個能與自己的
> 類別一致。因此條件 4 與原條件 6 **互斥**，此過濾器對顏色類別永不可能產生候選。
> 現改為只要求「量測可信」：本閘門真正要擋的是**用猜出來的顏色去刪框**；「兩框是否
> 同一物」由條件 5 負責，而條件 5 在修正後才真正比較的是量測到的顏色。

目前門檻已寫入 Cable1/A 1.0.6 現行 config；production checkout 固定為
`report_only`。歷史版本快照若仍記錄 `suppress`，只視為 Pilot 候選，未完成
500 次、完整班次與具名批准前不得啟用；其他產品／工位預設不啟用。
條件不完整時保留全部框，交由 strict count 判 NG，採 fail-closed。

### 6.3 群組與保留規則

1. 對每一對框計算條件，只有直接符合的 pair 才可候選；
2. **優先保留「YOLO 類別與量測到的顏色一致」的框**；兩框一致性相同時才依 YOLO
   confidence 由高到低進行 deterministic greedy 保留；
3. confidence 相同時以 bbox、類別、原始 index 作穩定 tie-break；
4. 不使用 transitive chain 間接刪框，保留框與被消除框必須直接通過全部條件；
5. 其餘框只從 effective detections 排除，raw detections 不刪除；
6. verified class 不一致、顏色**量測不可信**或資料無效的 pair 不抑制；
7. 抑制後，屬於被消除框的顏色 NG 必須從整體顏色判定中撤回（該框已不屬於板子），
   但其 item 仍留在 `color_result.items` 作為工程證據。

> **規則 2 的修訂（2026-08-19）**：原文為純 confidence 排序。實測發現這會**保留
> 標籤被像素推翻的框、丟掉標籤正確的框**（Cable1/A 16:03：`#5 Red` conf 0.808 vs
> `#6 Orange` conf 0.664，實際量測為 Orange）。存活框帶著自己的顏色 NG，於是板子
> 失敗在模型的錯而非自身狀況；而同樣的重複框若正確標籤那個 confidence 較高就會通過
> — **同一物理狀況由 confidence 擲硬幣決定判定，良品遇到即誤殺**。
> 依據：detector confidence 是自報值，顏色量測是證據，且跨類別重複框中恰好只有一個
> 能與量測一致。1486 筆快照重放顯示 66 個配對完全不變（0 增 0 減），僅 1 筆
> confidence 反向 — 即上述事故。

不建立混合的「YOLO 分數 + 顏色分數」權重，除非後續有獨立校準證據。規則 2 並非
權重混合：它是二階排序（先一致性、後 confidence），不對兩種分數做算術結合。

### 6.4 輸出契約

每筆抑制紀錄至少包含：

```json
{
  "reason": "CROSS_CLASS_DUPLICATE",
  "kept_index": 5,
  "suppressed_index": 6,
  "kept_raw_class": "Orange",
  "suppressed_raw_class": "Red",
  "verified_class": "Orange",
  "kept_confidence": 0.666983,
  "suppressed_confidence": 0.505362,
  "iou": 0.977,
  "center_distance_ratio": 0.012,
  "area_similarity": 0.977,
  "policy_version": 1
}
```

result JSON 會一起保存`raw_detections`、有效`detections`與
`duplicate_filter`。SQLite 的`predictions_json`維持有效 detections，並以
`snapshot_path`連回完整 raw／suppression 證據，因此未新增 schema migration。

## 7. GUI 與操作者呈現

### 7.1 Report-only

結果仍依現行規則判定，但詳細結果增加：

```text
疑似重複框：#5 Orange 與 #6 Red
顏色校驗後：Orange
重疊：97.7%
目前為觀察模式，未修改判定
```

### 7.2 Suppress

結果圖：

- 保留框使用正常實線；
- 被抑制框以紫色虛線或偏移標籤顯示；
- 標籤顯示`#6 DUP→#5 (97.7%)`；
- 避免兩個標籤使用相同座標互相覆蓋。

詳細結果：

```text
已排除重複候選：#6 Red → Orange
保留：#5 Orange（IoU 97.7%）
有效檢測數：6；原始檢測數：7
```

若本次仍為 NG，必須繼續顯示真正的 NG 原因，不能只顯示去重資訊。

## 8. 線序設定的獨立問題

目前 config 預期：

```text
Red → Green → Orange → Yellow → Black → Black
```

本案例去除重複框後觀察到：

```text
Red → Orange → Green → Yellow → Black → Black
```

這是產品規格／設定問題，不屬於去重演算法。處理規則：

1. 由製程或產品工程提供正式線序；
2. 以 Golden 標準品及工位左右方向確認；
3. 單獨建立 config 變更與核准紀錄；
4. 不得為了讓目前照片 PASS 而直接改 expected sequence；
5. 去重功能驗證時同時報告 count 結果與 sequence 結果，兩者不可混算。

## 9. 測試計畫

### 9.1 純邏輯單元測試

- 同 verified class、IoU 0.977：形成候選並保留高 confidence。
- IoU 低於門檻：不抑制。
- 中心距離過大：不抑制。
- 面積比超界：不抑制。
- verified class 不同：不抑制。
- verified class 缺失：不抑制。
- 三框重疊群組：只保留一框且結果穩定。
- detection 順序改變：保留結果不因非決定性排序改變。
- 無效 bbox／NaN／負面積：回報診斷但不抑制。
- `off`、`report_only`、`suppress`契約分離。

### 9.2 Pipeline 整合測試

- color check 後才執行候選分析。
- count／sequence 使用 effective detections。
- raw detections 未被修改。
- decision、fail reasons、annotations、JSON、SQLite 使用一致 revision。
- position 啟用時 suppress 必須 fail-closed，不能沿用不一致的 raw position。
- report-only 不改變既有判定。
- suppress 模式能修正本案例的衍生多件與長度不符，但不掩蓋其他 NG。
- filter 本身不能直接把狀態改成 PASS。

### 9.3 回歸資料集

至少包含：

| 資料 | 最低要求 | 目的 |
| --- | ---: | --- |
| 本次已知重複框案例 | 全部 15 筆 | 確認候選召回與可追溯 |
| Cable1/A 現有 holdout | 全部 199 張 | 比較修正前後判定 |
| Golden OK | 至少 100 次離線重播 | 確認不新增 false reject |
| 現場正常品重複取像 | 至少 100 cycle | 驗證光線與微振動 |
| 真正缺件 | 至少 10 張 | 不可誤放 |
| 真正錯色／錯線 | 每類至少 10 張 | 不可被去重掩蓋 |
| 真正多件／遮擋件 | 至少 10 張 | 驗證最高風險案例 |

資料不足時只能停留在 report-only。

### 9.4 效能

每張圖通常小於十個框，pairwise 比較為 O(n²)，不在鎖內執行 I/O。

驗收目標：

- `n <= 20`時新增處理延遲 p95 小於 `5 ms`；
- 不增加模型推論次數；
- 不複製原始影像；
- 不新增跨執行緒共享可變狀態。

## 10. 驗收標準

全部成立才可從 report-only 切換到 suppress：

- [x] 15 筆歷史候選均產生正確直接 pair 與 confidence 保留結果。
- [ ] 經正式線序核准的 Golden OK 沒有新增 NG。
- [ ] Known NG 沒有因 suppress 變成 false PASS。
- [ ] 真正多件／遮擋件測試全部維持 NG。
- [ ] 真正缺件、錯色、錯線不因去重而消失。
- [ ] 本案例去重後 count check 為六件；若線序尚未核准，sequence 結果仍
      獨立列示，不得宣稱整張 PASS。
- [x] raw、effective、suppressed 三份資料可以互相追溯。
- [x] GUI 與結果圖能指出本案例的`#5/#6`與`97.7%`重疊。
- [ ] report-only 與 suppress 的結果差異有 A/B 報告。
- [ ] 延遲符合 p95 目標。
- [x] 模型版本、config hash、filter policy version 已保存。
- [ ] 製程／AI／軟體三方具名批准。
- [ ] 回滾已演練。

任何一項未完成都不得宣稱已解決或直接無人值守上線。

## 11. 上線計畫

### Phase 0：文件核准

- 核准問題定義、方案與測試數量；
- 確認正式線序的負責人；
- 不修改生產 config。

### Phase 1：實作與離線測試

- 實作純函式 analyzer；
- 接入 pipeline registry；
- 補齊保存契約與 GUI；
- 完成單元及整合測試。

### Phase 2：Shadow／report-only

- 僅在`Cable1/A`啟用；
- 不改判定，只記錄「若啟用會排除哪些框」；
- 至少一個完整班次或 500 次檢測，取較晚完成者；
- 每一筆候選由工程抽查或依風險分層抽查。

### Phase 3：受控 suppress Pilot

- 具名批准後切換；
- 僅限有人監督的試產；
- 比較 false reject、false pass、候選率與推論延遲；
- 異常時立即切回 report-only 或 off。

### Phase 4：正式啟用

- 模型與 config 成對建立版本；
- 更新 CHANGELOG、變更紀錄與操作／工程手冊；
- 保存 A/B 報告與批准人。

## 12. 回滾

回滾不需要資料庫 migration：

1. 停止檢測；
2. 將 filter mode 改為`report_only`或`off`；
3. 還原上一份具 hash 的 config；
4. 清除 runtime model/config cache；
5. 以一張 Golden OK、一張真正多件、一張錯線重新驗證；
6. 保存回滾原因與執行人。

已保存的 raw detections 不得因回滾刪除。回滾只改變後續判定行為。

## 13. 可觀測性

建議結構化欄位：

```text
duplicate_filter_mode
duplicate_candidate_count
suppressed_duplicate_count
raw_detection_count
effective_detection_count
max_candidate_iou
filter_policy_version
```

建議監控：

- 候選率突然升高：模型、光線或相機位置可能改變；
- 某一 raw class pair 集中發生：優先加入補訓；
- suppress 後 false PASS：立即停用並進入事故處理；
- report-only 候選長期為零：評估是否仍需保留該設定。

## 14. 後續補訓

後處理修正不取代補訓。應將下列案例獨立標記：

- `Orange / Red`同一端子的跨類別雙框；
- `Black / Yellow`最終校驗成 Black 的雙框；
- 真正相鄰、遮擋但屬不同實體的負面案例；
- 低 confidence 正確元件，避免提高 confidence 造成漏檢。

補訓驗收除了 mAP，還要新增：

- 每張圖重複實體框數；
- 跨類別高 IoU pair rate；
- strict count false reject rate；
- known extra false pass rate。

## 15. 已採用決策與待完成 Gate

請逐項勾選或修改：

- [x] 已實作`report_only`與`suppress`，Cable1/A 目前維持 report-only；完成 500 次與完整班次 Gate 後才可切換。§17 的修復只讓 report-only 開始產生候選資料，未改變本條件。
- [x] 首版僅在`Cable1/A 1.0.6`現行設定啟用。
- [x] 初始 IoU 候選門檻為`0.90`。
- [x] 中心距離與面積相似度必須同時通過。
- [x] ~~只有 verified class 相同且兩者顏色通過才可候選~~ → **修訂**：verified class 相同且兩者顏色**量測可信**才可候選（2026-08-19，原條件邏輯不可滿足，見 §6.2）。
- [x] ~~以 YOLO confidence 決定保留框~~ → **修訂**：優先保留與量測一致的框，confidence 僅作同級 tie-break（2026-08-19，見 §6.3）。
- [ ] 現場至少一班或 500 次監督資料，取較晚者。
- [ ] 同意真正多件／遮擋件為 blocking Gate。
- [ ] 指定正式線序核准人：________________。
- [ ] 指定製程批准人：________________。
- [ ] 指定 AI 批准人：________________。
- [ ] 指定軟體批准人：________________。

審核意見：

```text


```

## 16. 審核簽核

| 角色 | 姓名 | 結論 | 日期 | 備註 |
| --- | --- | --- | --- | --- |
| 製程／產品工程 |  | 待審 |  |  |
| AI／模型工程 |  | 待審 |  |  |
| 軟體工程 |  | 待審 |  |  |
| 產線代表 |  | 待審 |  |  |

結論選項：`核准`、`修改後再審`、`不核准`。

## 17. 2026-08-19 回歸事件與修正紀錄

### 17.1 事件摘要

本文件所描述的機制在 **2026-08-05 至 2026-08-19 期間靜默失效**，無任何錯誤訊息、
無測試失敗、config 顯示啟用。現場症狀與 §1.1 所列一字不差 — 也就是這份企畫書要
解決的問題原封不動回來了。

| 日期 | commit | 事件 |
| --- | --- | --- |
| 2026-07-30 | `0852398` | 過濾器上線，`mode: suppress`，**運作正常** |
| 2026-07-30 ~ 08-04 | — | 22 筆真實抑制紀錄，含 1 筆讓良品順利 PASS |
| **2026-08-05** | `0e9e6ad` | 「feat: harden color baselines and camera lifecycle」加入 `_is_expected_color_match`，**同時打斷兩處機制** |
| 2026-08-05 ~ 08-18 | — | **稽核零命中**，斷點精準落在 08-05 |
| 2026-08-19 | `8582a8b` | 改為 `report_only`，並新增 `tests/test_pilot_config_safety.py` 鎖住，明訂未完成 500 次、完整班次與具名批准前不得切換 |
| 2026-08-19 | `931e0cf` `7668525` `4262c3c` | 修復三處缺陷；**`mode` 維持 `report_only`**（切換屬 §15 治理決定，不隨修復變動）|

### 17.2 根因：一個 fail-closed 強化造成的雙重死亡

`0e9e6ad` 讓 `ColorCheckItemResult.is_ok` 同時要求「量測通過自己的門檻」**且**
「量測與 detector 類別一致」。這個 flag 被三處消費，其中兩處因此失效：

1. **`verified_class` 的修正路徑變成死碼。**
   `verified_class = best_color if is_ok else class`。但 `is_ok=True` 已隱含
   `best_color == class`，所以這行只會把類別指派給自己；每次 detector 真的標錯時，
   都退回**剛被量測推翻的那個標籤**。[finalize.py](../../core/pipeline/finalize.py)
   的模組註解所依賴的「顏色校驗會修正類別」從此不成立。
   連帶後果：錯誤標籤經 `verified_class` 流入
   [export_review_dataset.py](../../tools/export_review_dataset.py) 的複判資料集，
   **重訓時會強化 detector 自己的錯誤**。

2. **`require_color_check_pass` 變成不可滿足。** 見 §6.2 條件 6 的修訂說明。

兩者疊加：兩個重複框的 `verified_class` 因 (1) 永不相同 → 條件 5 失敗；即使相同也
會因 (2) 條件 6 失敗。**雙重死亡。**

### 17.3 為什麼測試沒抓到

[test_cross_class_duplicate_filter.py](../../tests/test_cross_class_duplicate_filter.py)
的 fixture 給 `class="Red"` 的框配上 `best_color="Orange"` 且 `is_ok=True` — 這是
真實 `ColorCheckerService` **不可能產生**的狀態（`_is_expected_color_match("Red",
"Orange")` 恆為 False）。測試對著一個虛構狀態通過，因此綠燈是假的。

**教訓**：fixture 描述的狀態必須是被測服務真的會輸出的狀態。此類 fixture 現已改為
生產環境實際會出現的組合，並加入斷言固定「該框確實 `is_ok=False`」。

### 17.4 離線重放證據

`tools/audit_cross_class_duplicates.py`（唯讀）對 `Result/` 全量重放：

| 項目 | 數值 |
| --- | --- |
| 掃描快照 | 1486 |
| 實際分析 | 942（其餘無偵測或不符產品篩選） |
| 命中配對 | **66** |
| IoU 範圍 | **0.911 ~ 1.000**（政策門檻 0.90，無勉強過關者） |
| 主要配對 | Red→Orange 34（55%）、Orange→Green 11，其餘 Yellow/Black 長尾 |
| confidence 反向者 | 1（即 §6.3 規則 2 修訂所依據的事故） |

重點：**這不只是紅橘混淆**，而是 detector 的通用重複框問題，補訓範圍不應只涵蓋
紅橘（見 §14）。

### 17.5 修正內容

`is_ok` 原本混在一起的兩個判定已拆開，並確立三個互不相同的消費者：

| 消費者 | 應讀取 | 語意 |
| --- | --- | --- |
| §6.2 條件 6（配對閘門） | `measurement_is_ok` | 量測本身可信 |
| `verified_class`（標籤修正） | `measurement_is_ok` | 可信量測可覆寫 detector |
| §6.3 規則 2（保留決策） | `is_ok` | 標籤與量測一致 |

另修正兩項：

- **抑制後未撤回顏色判定**（§6.3 規則 7）：被消除框的顏色 NG 仍釘住整體 FAIL，
  會誤殺「良品 + 一個重複框」。
- **現場判定卡未過濾被抑制框**：操作員被告知一個已不在板子上的框的顏色錯誤。詳細
  面板與疊圖本來就有過濾，此卡是唯一漏網者（§7 所述三介面一致性要求）。

### 17.6 驗證

以 2026-08-19 15:18 與 16:03 兩筆**真實快照**重放完整 pipeline：

```text
修正前：raw=7  LR=[Orange, Orange, Green, Red, Yellow, Black, Black]
        count FAIL (over=['Orange'])  sequence FAIL (length_mismatch)
        Color: FAIL / NG idx 顯示已被消除的框

修正後：抑制 1 框（保留與量測一致者，即使其 confidence 較低）
        LR=[Orange, Green, Red, Yellow, Black, Black]
        count PASS  unexpected=[]  color_result.is_ok=True
        sequence FAIL (order_mismatch)  ← 板子真正的不良
        FINAL STATUS = DETECTION_FAIL
```

判定維持 FAIL（該板線序確實錯誤），但**只站在真實原因上**。

上述「修正後」是**在 `suppress` 下離線重放**得到的結果，用以證明修正有效；
production 仍為 `report_only`，因此線上不會消除框，`over=['Orange']` 與
`length_mismatch` 仍會出現。這是 §15 治理條件未完成的預期後果，不是缺陷 —
消除框需要具名批准，不是再改一次程式。此重放正是 §10 所要求的
「report-only 與 suppress 的結果差異 A/B 報告」可用的第一批素材。

新增測試 20 項、取代 2 項，重點涵蓋：疊圖與判定序列必須同源、可信量測可覆寫
detector 標籤、低信心量測仍須退回、顏色不符不得使判定變寬鬆、detector 不一致本身
不得取消配對資格、保留與量測一致的框（含 confidence 反向案例）、抑制後撤回／不撤回
顏色判定、舊 payload 相容、現場判定卡過濾（含 report_only 與全畫面檢查的反例）。

被取代的 2 項是原本固化錯誤行為的測試：
`test_rejected_color_mismatch_does_not_replace_detector_class`（固化 §17.2 缺陷 1）
與 `test_both_color_checks_must_pass`（固化 §6.2 原條件 6）。

### 17.7 尚未處理

- **detector 補訓**（§14）：66 筆重複框的路徑、框號與 IoU 已在稽核報告中，可直接
  作為複判／重訓輸入。這是唯一的根因修復，其餘皆為後處理補償。
- §15 現場 Gate（500 次／完整班次／具名批准）仍未完成。
