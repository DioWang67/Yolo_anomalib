# 顏色檢測工作紀錄 — 2026-09-02

接手用。涵蓋 `yolo11_inference`、`Yolo11_auto_train` 與 workspace 三個 repo。
**全部已修改、尚未 commit，也未 push。**

上一份是 [`COLOR_DETECTION_WORK_LOG_20260827.md`](../archive/COLOR_DETECTION_WORK_LOG_20260827.md)，
屬 `stats-robust-v4` 時代，只當歷史紀錄讀。

---

## 一、目前狀態

| repo | 分支 | 測試 |
|---|---|---|
| yolo11_inference | `fix/restore-color-verification-and-duplicate-suppression` | 2129 passed, 6 skipped |
| Yolo11_auto_train | `feature/operator-retraining-workflow-20260715` | 980 passed, 5 skipped；ruff + mypy 通過 |
| workspace | `agent/fix-onnx-runtime-bootstrap` | 31 passed, coverage 84.68%；`--check` 通過 |

契約版本以 `core/color_baseline_contract.py` 的 `BASELINE_ALGORITHM_VERSION` 為準
（目前 `stats-robust-v5`）。**v4 的 artifact 一律不相容。**

---

## 二、這次做完的事

### v5 連動修復（九項）

v5 只改了推論端的取樣幾何，相依程式沒跟上。九個缺口都已修並有測試：

1. **版本切換／回滾會洗掉顏色欄位** —
   `core/services/model_version_registry.py` 的 `STATION_LOCAL_FIELDS` 沒有任何
   color 欄位，而啟用會覆寫線上 `config.yaml`。線上有
   `color_roi_policy.inset_x_ratio: 0.2`，7 個版本快照裡 0 個有。新增
   `STATION_OWNED_COLOR_FIELDS`，對這三個欄位「線上優先，含缺席」。**最高風險**。
2. **`color_decision_tuning` 兩邊部署路徑都沒保留** — 已加入訓練端
   `deploy.py` 的 `STATION_LOCAL_FIELDS`。
3. **重建器能產出沒有幾何戳記的 artifact** — 執行期一定會拒收，改為寫入時就拒絕。
4. **conformance fixture 看不見幾何差異** — `render()` 只產生正方形，而正方形時
   `int(min(h,w)*r) == int(h*r) == int(w*r)`。新增 `height`/`width`/`orientation`
   與 5 個案例（兩種帶向的細長線材、黃色少數、紅多綠少）；
   `generate_color_conformance.py --check` 進 workspace CI；每個 divergence 必須有
   自己的理由，否則腳本失敗。
5. **訓練端 gate 三類漂移**（實測會改判定）— `min(h,w)` 裁切改逐軸、Yellow
   shortcut 移除提早返回、`green.py` 的 green-dominance override 移除。
6. **一致性測試可被測試順序關掉** — `tests/test_color_integration.py` 在 import 時
   把 `color_verifier` 換成 MagicMock 且未還原（`sys.modules` 與父套件屬性都要還原，
   因為 `picture_tool.color` 是延遲載入）。
7. **新增顏色是「整片板子一直 FAIL」的操作** — 啟動時 config 驗證會指名該色別；
   重建改為覆蓋工位 `expected_items` 實際列出的色別（原本寫死五色）。
8. **驗收與發布只問演算法標籤** — 兩道門都改為傳入線上站點的
   `color_roi_policy` 與完整 tuning，由 `core/services/station_color_settings.py`
   統一解析；候選那道門另外改讀 artifact 自己的 provenance。
9. **`_supported_candidates` 的空調色盤語意**、**`tools/color_verifier.py` 殘留的
   v4 shortcut 與 0.12 邊距**。

### 顏色開線檢查（新功能）

`燈光控制 → 顏色開線檢查`（`app/gui/color_preflight_dialog.py`）與
`tools/color_preflight.py`，共用 `core/services/color_preflight_runner.py`。
流程與判準見 [`operations/MISJUDGE_TRIAGE_SOP.md`](../operations/MISJUDGE_TRIAGE_SOP.md) 第 9 節。

兩個不可打破的規則：

- **它永遠不寫顏色基準。** 開線一片板子的讀數沒有證據集、沒有 holdout、沒有具名
  批准，正是基準契約要擋的東西。`--record-reference` 只寫參考餘裕。
- **參考餘裕綁定基準檔的 sha256**，基準重建後自動標記過期並要求重錄。

判準是**餘裕保留率**而不是絕對容差：各色餘裕差一個數量級（Cable1/A 目前
Black +0.541、Yellow +0.075），單一絕對容差會讓 Yellow 掉到零都不報。

---

## 三、量測污染分析（重要，尚未處理）

開線檢查的視覺面板揭露的：**線材是斜的，而量測框是 bbox 正中央的軸對齊矩形**，
所以每個量測區塊都含板子背景。Cable1/A 實測（2026-09-02，
`Result/20260902/.../162631` 那批）：

| 色別 | 通過飽和度閘門 | 在 envelope 內（執行期分母） |
|---|---|---|
| Red | 91% | 72% |
| Orange | 81% | 73% |
| Yellow | 83% | 71% |
| Green | 67% | 67% |
| Black | —（不適用） | 57%（整框） |

基準自己也記錄了同一件事：`coverage_mean` 0.169–0.408。

**污染落在哪裡（已逐一查證程式碼）**：

- **彩色**（`stats_color_checker.py` 的 `sat_mask` / `valid_hsv`）分母是通過飽和度
  門檻（20）的像素，所以**約 30% 的分母不是線材**。envelope 是學在遮罩過的像素上
  （`_sample_color_pixels` 用主導色相遮罩），所以背景大多落在 envelope **外面** ——
  它稀釋分母，不造成誤判。代價是**靈敏度**：真正的偏色要更大才推得動分數；
  反過來，治具位移改變背景比例會讓分數在顏色沒變的情況下移動。
- **黑色**（`_black_baseline_match`）把整框的匹配率除以 `coverage_mean`，而那是
  「校準當時框裡有多少是黑的」= **治具幾何的性質，不是顏色的性質**。線材位移、
  bbox 大小改變、ROI policy 改變 → 除數就錯，所有黑色分數一起偏移。這是
  2026-08-27 紀錄的「Black 約 2 倍過寬」的機制。**這是脆弱的那個。**
- **訓練端乾淨**：`compute_hsv_lab_stats` 只在 SAM mask 內計算（`hsv[mask_bool]`）。
  所以兩邊污染程度**不同** —— 這也是部署端拒絕把訓練 `quality/color/stats.json`
  當 runtime baseline 的根本原因：同名的 `coverage` 是兩種不同的量。
- **Detector 訓練**：框住斜線材必然含背景，對偵測是正常的，不是問題。

**處置**：要改善只有讓量測區塊跟隨線材（調 `color_roi_policy`，或改用 mask）。
但 `color_roi_policy` 綁在基準契約裡，**一改就讓已部署基準失效** —— 那是一次
重新校正，不是微調。**建議併進 v5 重建一起做，不要做兩次。**

---

## 四、待辦（按槓桿排序）

1. **每個工位跑一次 v5 重建 + 驗收 + 發布 + 設 `strict`**
   （見 [`model_lifecycle/MODEL_COMBINATION_ACCEPTANCE.md`](../model_lifecycle/MODEL_COMBINATION_ACCEPTANCE.md) 第 5.3 節）。
   在第 4 步之前該工位並未受保護。注意排程相依：portable detector bundle 會強制
   帶入 `strict`，所以**站台必須先完成 v5 重建才能吃新 bundle**。
2. **決定要不要在同一次重建裡調整 `color_roi_policy`**（第三節）。若要，先在真實
   資料上量門檻餘裕，不要憑感覺調。
3. **取樣遮罩不對稱**（刻意未做）：校準端遮罩到主導色相並降採樣後存 coverage，
   執行期在未縮放裁切上量 `raw_ratio` 再除以它 —— 分子分母來自不同量測，而契約
   自己把 sampling mask 列為版本遞增觸發條件。這需要再一次版本遞增 + 在真實資料上
   量測餘裕，是獨立提案。
4. **趨勢檢視只在 CLI**（`tools/color_preflight.py --trend`），GUI 尚未有。
5. **三個 repo 全部未 commit**，需整理成有邏輯邊界的 commit。

---

## 五、環境陷阱

- **ONNX Runtime**：import 任何會拉進 `core.anomalib_inference_model` 的東西
  （含 `core.detection_system`）之後，`onnxruntime` 就載不了擴充 DLL。
  必要時先 `import onnxruntime`。
- **C: 已滿**：pytest 的 `tmp_path` 在 C:，會讓 `test_result_handler.py` 因 1 GiB
  磁碟保留而失敗。跑測試請加 `--basetemp` 指向 D: 且**在 workspace 之外**。
- **`Yolo11_auto_train/.gitignore` 有 `data/`**，會吞掉 `tests/data/`。共用 fixture
  放 `tests/fixtures/`。
- **離屏 Qt 沒有字型**：`QT_QPA_PLATFORM=offscreen` 下 `QFontDatabase().families()`
  是 0，截圖不會有文字。要看 GUI 就用預設平台後端、不 `show()`、直接 `grab()`。

---

## 六、這次犯過並修好的錯（教訓）

- **用「所有量測像素的平均」判斷色域位置**，得出「Black 的 V 跑出 envelope」的
  假警報。那個平均被 43% 的背景像素拉走；主導群集其實貼著基準
  （Black 中位數 [75,16,32] vs 基準 [75,21,24]）。檢查器判的是**主導色**，
  所以對整個區塊取的統計不是被比較的那個統計。**密度圖**才讓這件事看得出來。
- **`命中` 一開始用整框當分母**，但執行期對彩色用的是飽和度子集 —— 描述了產線
  沒做的計算。已改為用執行期自己的分母（差別：45–66% → 67–73%）。
- **把開線檢查的可用性綁在 v5 遷移上**（基準 provenance 失敗就禁止記錄參考），
  結果每個站台在遷移完成前都不能用來偵測漂移 —— 那正是漂移最看不見的時期。
  改為綁定基準 sha256。
- **裁切圖用索引配對到顏色量測結果**，但裁切只在 bbox 有效時才存，跳號會讓後面
  全部錯位、把別條線的圖配到這條線的數字上。改為用檔名裡的偵測索引配對，並驗證
  類別。
