# 顏色檢測工作紀錄 — 2026-08-27

接手用。涵蓋 `yolo11_inference`、`Yolo11_auto_train` 與 workspace 三個 repo。
**全部已提交、尚未 push。**

---

## 一、目前狀態

| repo | 分支 | 測試 | 本日提交 |
|---|---|---|---|
| yolo11_inference | `fix/restore-color-verification-and-duplicate-suppression` | 1971 passed, 6 skipped | 9 |
| Yolo11_auto_train | `feature/operator-retraining-workflow-20260715` | 971 passed, 5 skipped | 2 |
| workspace | `agent/fix-onnx-runtime-bootstrap` | 31 passed, coverage 85% | 5 |

ruff 全過，mypy 無新增錯誤，跨實作 conformance fixture 為最新。

`yolo11_inference` 工作區仍有**與本次無關的既有變更**（duplicate-filter A/B：
`position_validator.py`、`annotations.py`、`handler.py`、`docs/pilot/...`、
`models/Cable1/A/yolo/config.yaml`、3 個未追蹤檔）。本次提交刻意避開，未動一行。

---

## 二、已修的缺陷

### 執行期（`core/stats_color_checker.py` 等）

| 問題 | 影響 |
|---|---|
| 無證據區域被塞 `black: 0.7` 假分數 | 洗白/未點亮的區域**通過**顏色檢查，還會覆寫偵測器 class 汙染重訓資料 |
| 色相在 0/179 接縫用線性平均 | 跨接縫的暗紅被判成 **Green** |
| 退化 / 出框的 bbox | `cv2.error` 逸出 → 整張 frame ERROR；非同步管線會觸發 stop event **停線** |
| fusion 的 `processed_image` 指向疊圖 | 顏色量到的是熱力圖偽色 + 標註框邊 |
| `global` 顏色 revision 對 `color_qc` 無效 | 已簽核的 revision 靜默失效，log 卻顯示已套用 |
| 統計缺失反而加分 | 缺 `hsv_mean` 的顏色贏過統計完整的顏色 |
| red/orange/green 閘門寫死在模組 | 一個產品的校正值綁死全部產品 |

### 基準授權路徑（`core/services/color_baseline_recalibration.py`）

- 有效像素不足時**丟棄遮罩改用全部像素** → 基準由純背景建成，而記錄的 coverage 寫著 0.00
- 彩色取樣只用 `S >= 20`，**不看顏色** → 鄰線被吃進基準（見下方數據）
- 改為**環形直方圖定位主色**、只保留 ±15°；跨接縫的紅色會被當成一個叢集完整保留
- `ALGORITHM_VERSION` → `stats-robust-v3`（改動前後基準不可比較）
- 新增兩個互補的人工審核觸發：`hue_spread`（分佈寬度）與 `chroma_retention`（飽和度保留率），
  以及 `dominant_fraction`（主色佔比）。回溯套用：8 份問題候選全數觸發、已核可的乾淨基準零誤報

### 訓練 repo（`Yolo11_auto_train/src/picture_tool/color/`）

同源實作，且**是產出 `color_stats.json` 的上游作者**與訓練管線的顏色關卡。

- Orange/Red 決勝把信心 **×1.3** → 黑線被報成橘色（生產端已有回歸測試禁止此事）
- `BlackStrategy` 把 hue 相似度寫死 1.0 仍計 0.2 權重 → Black 每次白拿 0.2
- 色相線性平均**三處**（策略評分、逐張 `hsv_mean`、跨張聚合）
- 有效像素 < 50 時捏造 `Black: 0.7`
- 空區域 `ZeroDivisionError`

### 診斷工具（`tools/color_verifier.py`）

原本自帶一套**不同政策**的計分，與生產端只有 8/15 一致（純紅回報 `Unknown`、
紅色帶橘邊回報 `Orange`）。已改為委派 `StatsColorChecker`，包絡檢查保留成
獨立訊號 `envelope_ratios` / `envelope_match`。平行判定鏈已刪除（-214 行）。

### 防漂移

兩個 repo 共用逐位元相同的 `tests/fixtures/color_conformance.json`（顏色模型內嵌，
兩邊測試皆自給自足），各自把實作釘在案例上；workspace CI 比對兩份副本相同。
產生器：`scripts/generate_color_conformance.py`（`--check` 供 CI）。
已記錄的合理差異：訓練關卡用 `max(每色門檻, 預設)`，比產線嚴格。

---

## 三、我造成並修好的回歸（重要教訓）

驗證準確率時發現 **100% 誤殺**（173 片好板全被擋）。原因是我的兩個「整理」動作：

1. 黑色捷徑信心值從 `coverage` 改成「觸發規則的邊際值」
2. `_is_black_image` 的中心裁切統一成 `min(h,w)`

黑色門檻在真實裁切上**只有約 1% 餘裕**，兩者各自把分數推過線：真實黑色區域
**0.500 → 0.016**（門檻 0.45）。

**1971 個測試全程綠燈** —— 單元測試全用均勻色塊，那種情況 coverage 規則會觸發、
兩個數字剛好相同；真實裁切不均勻，只有 mean/median 觸發才會現形。

已還原數值與裁切，改用 `debug.black_rules` 記錄觸發規則（原本的批評成立，
但正確修法是加註記而非動判定所依賴的數字），並補上會抓到此回歸的測試
（細長、有反光的非均勻黑色裁切）。

> **教訓**：門檻餘裕小的地方，「一致性整理」等於偷偷重新校正。統一那三個中心裁切
> 仍值得做，但必須和門檻在同一個變更裡一起調。

---

## 四、實測結果（Cable1/A，250 個已確認驗收樣本）

### 重建（215 張已確認 OK 照片、1291 個裁切、status = READY）

| color | 已部署 (n=3~6) | 舊演算法候選 | 新重建 (n=169) |
|---|---:|---:|---:|
| Red 平均色相 | 3.9 | **20.7**（琥珀色） | **4.7** |
| Red 色相寬度 | 1.6 | 64.0 | **6.0** |
| Orange 寬度 | 2.3 | 61.0 | 12.0 |
| Yellow 寬度 | 9.6 | 47.0 | 11.0 |
| Green 寬度 | 7.5 | 34.0 | 22.0 |

`dominant_fraction` 0.50~0.70 —— **每個裁切有 30~50% 的有色內容是鄰線**。

四個彩色 `REBUILT`、holdout 準確率 1.00；**Black 被安全檢查退回**（色相偏移 37.9）。

### 端到端（帶產線 black 門檻 0.4）

| | 誤殺 | 漏放 |
|---|---:|---:|
| 已部署基準 | 16/173 = **9.2%** | 0/77 = **0%** |
| 新重建基準 | 15/173 = **8.7%** | 0/77 = **0%** |

**統計修得徹底，端到端只差 1 片板。** 瓶頸不在彩色，在 Black —— 而 Black 在兩個
基準裡完全一樣（被安全檢查保留）。

> 重建報告裡的「holdout 準確率 1.00」**不是準確率**。證據以 `expected_verdict == OK`
> 篩選，77 個 NG 樣本從未使用，該數字對「壞品抓不抓得到」零資訊。

### NG 樣本組成（77 個）

`SEQUENCE_MISMATCH` 68、`MISSING` 6、`POSITION_SHIFT` 2、`OTHER` 1。
主要缺陷是線序而非顏色 —— 但**序列檢查必須先正確辨識每條線的顏色**，顏色準確度是上游依賴。

---

## 五、待辦（按槓桿排序）

1. **框的準確度**（最大槓桿）。`dominant_fraction` 0.50~0.70 是硬數據：檢出框有
   30~50% 裝的是隔壁那條線。取樣端已治標，治本要用分割遮罩或每 class 向內縮的子 ROI。
2. **Black 重新設計**。它不是學出來的（寫死規則 `s<50 & v<80` + 手設門檻 0.45，
   餘裕 1%），統計基準在自己的 holdout 上只有 **3.6% 準確率**，等於裝飾品。
   誤殺主要來自這裡。現在是「兩頭不到岸」。
3. **安全檢查加絕對下限**。「新的沒比舊的好就保留舊的」在舊的本來就 3.6% 時，
   會把壞基準永久凍結。
4. **NG 樣本至少用於評估**。`collect_confirmed_ok_evidence` 只收 OK。
5. **red/orange/green 手調閘門**仍主導判定，重新訓練基準的效果有上限。
6. 統一三個中心裁切 —— 連同門檻一起重新校正（見第三節）。
7. `26a6e2c7` 那份五色統計相同的壞候選仍在 `station_data`，建議標記或移除避免誤核可。
8. 把重建走一次 GUI，產生**有簽核人與原因**的候選（本次刻意寫到 scratch，
   未寫入 `station_data`，避免留下無歸屬的稽核紀錄）。

---

## 六、環境陷阱

- **ONNX Runtime**：`import core.anomalib_inference_model`（含經由 `core.detection_system`）
  之後 onnxruntime 就載不進 DLL。單獨 import `anomalib` / `lightning` / `torch` / `cv2`
  都正常。**先 `import onnxruntime` 再 import core 即可避開**。
  （workspace 分支 `agent/fix-onnx-runtime-bootstrap` 正在處理這題，此為精確重現條件。）
- **C: 磁碟已滿**（205G/205G）。pytest `tmp_path` 在 C: 上，會讓
  `tests/test_result_handler.py` 有 13 個測試因 handler 的 1 GiB 磁碟保留檢查而失敗。
  用 `--basetemp` 指到 D: 上**且在 workspace 之外**（repo 有防護會拒絕 workspace 內的 basetemp）。
- `Yolo11_auto_train` 的 `.gitignore` 有 `data/`，會吃掉 `tests/data/` —— fixture 因此放 `tests/fixtures/`。

## 七、產出位置

`.tmp/color_rebuild_Cable1_A/`（已 gitignore）：
- `color_stats.json` — 重建的基準
- `report.json` — 完整重建報告
- `rebuild_cable1.py` — headless 重建腳本
- `eval_baselines.py` — 端到端評分（兩個基準各跑一次全部 250 個樣本的混淆矩陣）
- `audit_baselines.py` — 掃描既有 `color_stats.json`，標出色相跨度可疑的顏色

三個腳本都需要「先 import onnxruntime」那道處理，見第六節。

---

## 八、續作結果（ROI 與 Black v4）

### 已完成

- 新增共用且不可變的 `ColorRoiPolicy`；Cable1/A 採每側水平內縮 20%、垂直不縮、最小邊長 8 px。線上推論、重建 GUI、headless 重建與 model config 使用同一份 policy。
- Black 不再使用手寫 S/V shortcut；推論 repo 與訓練 repo 都改為 learned S/V + LAB 範圍的聯合匹配，並用基準的 `coverage_mean` 正規化。缺少或無效的 `coverage_mean` 會 fail closed。
- 重建器升級為 `stats-robust-v4`，Black hue drift 明確為 `null`；新增逐色 holdout 絕對準確率下限 0.90，避免「新舊一樣差」仍通過相對退化檢查。
- 部署會保留經簽核的 `color_roi_policy`，不讓訓練輸出默默覆蓋站點取樣幾何。

### v4 scratch 候選

- 證據：215 張 confirmed-OK、1291 crops。
- 狀態：`READY`；五色全部 `REBUILT`，無 safety preserve、無 review-required color、無 absolute-accuracy failure。
- Holdout：Black 84/84；Green、Orange、Red、Yellow 各 42/42。
- ROI purity（x inset 0.20）：Red 0.816、Green 0.701、Orange 0.735、Yellow 0.758 dominant fraction。
- 產出：`.tmp/color_rebuild_Cable1_A/color_stats_v4.json`、`report_v4.json`、`eval_v4.json`、`roi_black_analysis.json`。

### 250 張 confirmed acceptance 端到端

| 基準 | OK→OK | OK→NG | NG→OK | NG→NG | 正確率 |
|---|---:|---:|---:|---:|---:|
| deployed（新程式碼） | 169 | 4 | 0 | 77 | 98.4% |
| rebuilt v4（新程式碼） | 170 | 3 | 0 | 77 | 98.8% |

v4 修好 `ACC-24DAF6D4E8EF`，沒有新增誤殺或漏放。剩餘三個 OK false reject 中，有樣本同時帶有 count / sequence / missing / unexpected-component 原因，不應用放寬顏色門檻掩蓋。

### 最終驗證

- `yolo11_inference`：1982 passed、6 skipped。
- `Yolo11_auto_train`：973 passed、5 skipped（使用含 FastAPI/Uvicorn 的 base conda；`yolo_anomalib` 缺這兩個 optional dependencies）。
- workspace：31 passed。

### 唯一尚需人工完成

候選仍未寫入 `station_data`。請由有權限的簽核人從 GUI 執行重建／審查，填寫真實身分與原因；GUI 會套用相同 ROI policy 與 v4 安全門檻。不得用 headless 腳本建立無歸屬的正式候選。

## 九、關聯程式盤點（`stats-robust-v2` → `v4` 的下游影響）

起因是驗收畫面出現「無法使用：演算法 stats-robust-v2（目前為 stats-robust-v4）」。
驗收並沒有被鎖死 —— Cable1/A 當時仍有 6 個可選變體 —— 但沿著這條線盤點，發現
版本閘門只擋了三扇門裡的一扇。

### 9.1 最嚴重的一項：線上配對的 Black 寬鬆了約 2 倍

| 來源 | Black `coverage_mean` | count | provenance |
|---|---:|---:|---|
| 已部署 `models/Cable1/A/yolo` | 0.370 | 6 | 無 `recalibration` 區塊 |
| v2 候選 `e5097c9a` | 0.370（原封繼承） | 6 | stats-robust-v2 |
| v4 scratch 候選 | 0.744 | 336 | stats-robust-v4 |

v4 的 Black 是 `score = raw_ratio / coverage_mean`，其中 `raw_ratio` 在**內縮 20%
之後**的 ROI 上量測，而 0.370 是**內縮之前**的寬框上量的（框內約半數是背景，因此
恰好差約一倍）。通過條件為 `raw_ratio >= threshold × coverage_mean`：

- 現行配對：0.45 × 0.370 = 0.167 → 框內只要 16.7% 像素落在 Black 包絡就通過
- v4 應有：0.45 × 0.744 = 0.335

方向是**漏放**，而 250 張驗收測不出來：77 張 NG 有 68 張是 `SEQUENCE_MISMATCH`，
幾乎沒有考驗顏色漏放。`coverage_mean` 只被 Black 規則使用，其他四色不受影響。

因此「GUI 簽核」不是最後的形式手續，而是 v4 程式上線的**前置條件**。

### 9.2 三扇門，原本只鎖了一扇

- 候選：有檢查（`discover_color_variants`）。
- 顏色方案 profile：**沒有檢查**。`75a77b2a` 是從被排除的 v2 候選 `83c8b909` 打包
  的，`a8af98dc` 來自 v2 `860aed5f`；兩者的 `color_model.json` 裡就寫著
  `recalibration.algorithm = "stats-robust-v2"`，provenance 從未遺失，是閘門沒去讀。
  兩者五色 `coverage_mean` 齊全，所以不會 fail-closed，而是安靜給出看起來正常的
  錯誤結論。
- 發布：**沒有檢查**。builder 只認 schema 1/2 並核對 sha256。磁碟上 14 份驗收報告
  全在 v2 時代寫成；最新那份（2026-08-14）比較的三欄正是上述三個 v2 產物。也就是
  說在修好之前，仍可拿該報告把 v2 基準發布並啟用。

`docs/model_lifecycle/MODEL_COMBINATION_ACCEPTANCE.md` 早已寫著「`INCOMPATIBLE`：
舊演算法…不再供選擇」。文件先寫了程式沒做到的事。

### 9.3 已修

1. 新增 `core/color_baseline_contract.py`：版本常數與「讀取 artifact 自己記錄的
   演算法」只有一份。它刻意位於重建器之下 —— 重建器匯入檢查器，所以版本不能住在
   重建器裡，否則執行期得匯入整個重建服務才能知道自己要求什麼。
2. 三處閘門一律改為呼叫同一個判定函式，不再各自比較自己匯入的常數。第一版我只換了
   理由字串、判定仍讀舊常數，測試立刻抓到兩者不一致 —— 正是要消滅的漂移。
3. profile 依 artifact 自己的記錄檢查；未記錄者與舊版分開報告，因為補救方式不同。
4. 發布拒絕不相容基準，判定順序為「報告記錄優先、artifact 備援」。
5. 驗收報告的每個顏色變體開始記錄 `algorithm`，舊報告沒有此欄位，讀為「無法確認」。
6. 新增站點設定 `color_baseline_algorithm_enforcement`（`warn` 預設 / `strict`）。
   無法辨識的值解讀為 `strict`。
7. 重建視窗顯示 `review_reasons`（先前只寫進 report.json），量測值放在 tooltip。

### 9.4 確認不需跟進

- 已部署基準五色 `coverage_mean` 齊全，Black v4 不會 fail-closed。
- `unmeasurable_roi` 下游唯一消費者是 `core/pipeline/steps.py`，以「狀態不等於
  `evaluated` 就不重算」處理，fail-closed 正確。
- `color_roi_policy` 消費者齊全，訓練端 deploy 也已列入 `STATION_LOCAL_FIELDS`。
- `tools/color_calibration_packages.py` 的 `picture-tool-threshold-v1` 是門檻修訂
  的軸，與基準演算法無關。

### 9.5 尚未處理（補訓路徑）

訓練端 `picture_tool/color/color_inspection.py` 產出的 `quality/color/stats.json`
**有** `coverage_mean`，但它是 SAM 遮罩面積比（`mask_nonzero / mask.size`），與站點
重建的「內縮 ROI 中通過取樣遮罩的比例」不是同一個量，且訓練端沒有 `ColorRoiPolicy`
概念。`bundle.py` 會把它打包成 `color_stats.json`，而 `DEPLOYMENT_OWNED_FIELDS`
擁有 `color_model_path`，因此補訓部署可以換掉線上顏色基準。

該檔案沒有 provenance，所以在 `strict` 之下會被拒絕載入 —— 這是正確的 fail-closed，
但也意味著**補訓之後站點必須重跑一次顏色基準重建**。刻意不在訓練端補寫
`recalibration.algorithm`：那會是謊稱它與站點重建同一套幾何。是否要讓訓練端 deploy
在替換 `color_model_path` 時直接拒絕或警告，留待決定。

## 十、GUI 重建「未更新，沿用舊基準」的追查

操作員從 GUI 重建，五色全部回報 `PRESERVED_SAFETY_REJECTED`（候選
`9f3bf5ee9eb530715801592c`）。

### 10.1 真正的原因不是安全門檻

該次重建的 `evidence_sources.color_roi_policy` 是 `inset_x_ratio: 0.0`。ROI policy
讀自 `self.model.config_snapshot_path`（模型版本 config 快照），而
`models/Cable1/A/yolo/versions/*.config.yaml` 七份快照**全部沒有** `color_roi_policy`
這個鍵 —— 它是我當天才加到線上 config 的站點本地欄位。所以解析成預設值「不內縮」，
重建改用整個偵測框取樣。

與我先前的 scratch 重建對照，兩者輸入**只差這一項**（同 215 樣本＝173 驗收 OK＋42
顏色覆核 OK、同一個 base、同樣的計數）：

| inset_x_ratio | 結果 |
|---|---|
| 0.2（線上 config） | 五色全部 REBUILT、READY |
| 0.0（版本快照） | 五色全部 PRESERVED_SAFETY_REJECTED |

連鎖過程：Red 新統計 holdout 41/42 = 0.976，舊的 42/42 = 1.0，退步 0.024 超過
`maximum_accuracy_regression = 0.02`（holdout 只有 42 片，1 片就是 0.0238，等於零
容忍）→ Red 被否決回退 → 一個「已保留」的顏色退步時，
`CROSS_COLOR_HOLDOUT_REGRESSION` 分支會把當輪所有還在用新統計的顏色一次全部否決。

**刻意沒有放寬那個門檻。** 門檻做對了事 —— 錯的是輸入幾何。在幾何修正後，五色
holdout 全 100%，沒有任何退步可觸發連鎖。為了讓錯誤的輸入通過而放寬安全限制，正是
第三章那個回歸的翻版。

### 10.2 附帶暴露的洗白漏洞

該候選的 `summary` 與已部署基準**逐位元相同**（`coverage_mean` Black 0.370、
count 6，五色皆同），卻標記 `algorithm: stats-robust-v4`、`status: READY`。它會通過
第九章新加的三道閘門，然後 Black 繼續 2 倍寬鬆，而且看起來完全合規 —— 比原本的問題
更難發現，因為閘門會替它背書。

### 10.3 已修

1. **重建的 ROI policy 改讀線上站點設定**；站點設定不存在則拒絕重建，不套預設值。
   報告新增 `color_roi_policy_source` 記錄幾何來源檔案。
2. **候選記錄 `preserved_colors` 與 `base_algorithm`**；契約在有沿用顏色時，要求舊
   基準本身也是現行演算法。含舊欄位 `preserved_by_safety` 的備援，否則檢查碰不到
   正是暴露這個漏洞的那個檔案。
3. 候選 `9f3bf5ee` 現在會被正確拒絕，不需人工刪除。

### 10.4 對目標的意義

修正後的幾何正是紅橘分辨與黑色檢測要的東西：dominant fraction Red 0.816、
Orange 0.735（先前 0.50–0.70，代表框內有 30–50% 是鄰線），Black 的
`coverage_mean` 0.744 才是 v4 公式該用的分母。

### 10.5 下一步

請重新從 GUI 執行一次重建。輸入條件與 scratch 那次已完全一致，預期五色 REBUILT、
狀態 READY；若仍出現「沿用」，把報告貼出來再查，不要調門檻。
