---

# yolo11\_inference 深入教學手冊（JR→SR 版）

> 目標：把新進工程師（JR）帶到能獨立設計、優化、佈署、除錯的資深工程師（SR）水準，
> 以本專案為載體，系統性講清楚「技術、演算法、架構設計、效能與品保」。

---

## 0. 如何使用本手冊

* **學習路線**（建議 2–4 週）

  1. 快速開始 → 2) 觀念建立 → 3) YOLO 推論深潛 → 4) 異常偵測深潛 → 5) 架構與設計模式 → 6) 測試/效能 → 7) 佈署 → 8) 實作練習與檢核。
* **角色導向**：每章末尾附 *JR→SR 檢核清單* 與 *實務演練*。
* **程式碼位置對照**：若你把專案放在 `yolo11_inference/`，文中檔案路徑以此為相對根目錄。

---

## 1. 專案概觀與核心目標

* **任務型態**：

  * 物件偵測（Object Detection）：零件定位、缺件檢查、定位校驗。
  * 異常偵測（Anomaly Detection）：表面刮傷、髒污、異物、異常形變。
* **使用技術**：

  * YOLOv8（Ultralytics）作為偵測後端。
  * anomalib（Lightning 生態）作為異常偵測後端：PatchCore / PaDiM / STFPM / DRAEM 等。
  * OpenCV / NumPy 作為前後處理。
  * YAML 設定 + Registry/Pipeline 模組化設計。
* **高階設計目標**：可切換多產品/站別，具快取與暖機、批次處理、可觀測性（logging/metrics）、可測試性與可佈署性。

**JR→SR 檢核**

* ✅ 能清楚說明為何同時需要 *偵測* 與 *異常* 兩條線。
* ✅ 能用不依賴 GUI 的 CLI 腳本重現一條完整流程。

---

## 2. 觀念建立：演算法與系統的「正確切點」

### 2.1 影像資料與顏色空間

* BGR↔RGB：OpenCV 預設 BGR；PyTorch/YOLO 常用 RGB。
* HSV、Lab：

  * HSV：色調(H) 對燈光敏感低，適合顏色判斷；
  * Lab：近人眼感知，適合均勻性/亮度校正。
* 幾個常見前處理：

  * **Letterbox**：維持長寬比縮放並補邊（YOLO 常用）。
  * **標準化**：`img/255.0`、均值方差正規化（依模型而定）。

### 2.2 偵測基礎

* **Anchor-free 概念（YOLOv8）**：直接回歸中心/寬高，減少 anchor 設定與計算。
* **NMS（Non-Maximum Suppression）**：Greedy NMS 或 DIoU-NMS，移除高度重疊的框。
* **IoU / GIoU / DIoU / CIoU**：框重疊度指標；推論時多用 IoU 做 NMS，訓練時用改良 IoU 當 loss。

### 2.3 異常偵測家族一覽

* **Embedding-based**：

  * *PatchCore*：用 ImageNet 預訓練特徵取樣建典，近鄰距離當 anomaly score。
  * *PaDiM*：每像素位置建高斯分布（均值/共變），以馬氏距離出分數。
* **知識蒸餾（STFPM）**：以教師模型特徵蒸餾到學生模型，偏差即為異常。
* **重建式（AE/DRAEM）**：以生成/重建重現正常樣，重建誤差高的區域即異常。

**JR→SR 檢核**

* ✅ 能說出何時選 PatchCore vs PaDiM（資料量、類別變化、記憶體/延遲限制）。
* ✅ 知道 Letterbox 對實際像素座標的影響與還原方法。

---

## 3. YOLOv8 推論深潛（從 API 到原理）

### 3.1 推論資料流

1. 讀圖（BGR）→ 2) 轉 RGB/resize/letterbox → 3) 張量化（NCHW、FP16/FP32）→ 4) 前傳 →
2. 後處理（解碼 + NMS）→ 6) 視覺化/驗證/儲存。

### 3.2 關鍵技巧

* **混合精度（AMP/FP16）**：GPU 上常見提速；注意運算溢位與 NMS 精度。
* **Warmup**：以 dummy tensor 前傳一次，觸發 CUDA kernel 最佳化，降低首張延遲。
* **CUDNN Benchmark**：針對固定大小輸入可提速。
* **Batch 推論**：資料夾批次、DataLoader 拼批；但要注意後處理的 per-image 邏輯。

### 3.3 後處理細節

* **分數與 IoU 閾值**（`conf_thres` / `iou_thres`）：

  * 分數太高 → 漏檢；太低 → 誤檢。
  * IoU 太高 → 保守合併；太低 → 殘留重疊框。
* **尺度還原**：從 letterbox 坐標還原到原圖。
* **多標的校驗（PositionValidator）**：

  * 以期望的 `(cx, cy, w, h)` 與容差（px 或百分比）比對；
  * 可加入角度/順序/相對布局規則（例如「J1 左於 J2，上於 J3」）。

### 3.4 效能優化清單（GPU）

* 啟用 `model.fuse()`、`half()`；
* 預先 `to(device)` 並常駐模型；
* 控制 cache（LRU）避免 VRAM 爆掉；
* 影像尺寸選擇 640/704/736 等與模型 stride 對齊；
* 若需求穩定，在生產環境禁用 `cudnn.benchmark`（輸入尺寸變動時）。

### 3.5 進階：匯出與加速

* **ONNX 匯出** → **TensorRT**（FP16/INT8）；
* **動態 batch** 與 **固定輸入大小** 的取捨；
* 推論前/後處理盡量下沉到 C++/CUDA 或 TensorRT plugin（高階需求）。

**JR→SR 檢核**

* ✅ 能量化說明 `conf_thres` / `iou_thres` 對 Precision/Recall 的影響。
* ✅ 能把 YOLO 模型匯出成 ONNX 並在 TensorRT 上跑通、對齊數值差異。

**實務演練**

* 寫一段批次推論（資料夾 → CSV/Excel），量測 **延遲/吞吐/VRAM**，並比較 FP32 vs FP16。

---

## 4. 異常偵測深潛（anomalib）

### 4.1 典型流程（以 PatchCore 為例）

1. 教師特徵抽取（正常品）→ 2) 記憶庫建立（取樣/壓縮）→ 3) 推論時取最近鄰距離作為 anomaly 分數 → 4) 出熱圖（像素分數）→ 5) 閾值化成遮罩。

### 4.2 分數與閾值的學問

* **全圖分數**：影像是否異常（image-level）。
* **像素分數**：異常區域位置（pixel-level）。
* 閾值選擇：

  * ROC 曲線最大化 Youden’s J；
  * 固定 FPR 下最大 TPR；
  * 根據產線成本（誤殺 vs 放過）做 *成本敏感* 閾值。

### 4.3 後處理與視覺化

* 正規化 `(x - min)/(max - min)`；
* `cv2.applyColorMap` 疊色；
* 形態學（開/閉/膨脹）去除噪點；
* 小區域過濾（`area < thr`）。

### 4.4 模型選型建議

* **資料少、類別單一、速度優先**：PatchCore。
* **位置對齊良好、同構件大量**：PaDiM。
* **需要語義分離**：STFPM。
* **異常樣式複雜**：重建式（DRAEM/AE），但調參較難。

**JR→SR 檢核**

* ✅ 能用一份正常集做出可用模型，並以 ROC-AUC、PRO、F1 報告。
* ✅ 能從「熱圖很花」追根究柢：特徵層選擇、正規化、補光/鏡頭/對焦。

**實務演練**

* 在同一資料上比較 PatchCore 與 PaDiM，記錄速度/記憶體/精度/熱圖穩定性。

---

## 5. 架構與設計模式（為什麼這樣寫）

### 5.1 Strategy（策略）

* **場景**：多後端（YOLO / anomalib / mock）同一界面。
* **作法**：`InferenceEngine` 根據 `config` 的 `class_path` 動態載入（`import_module` + `getattr`）。
* **好處**：替換後端不動上層流程；單元測試可注入假實作。

### 5.2 Factory + Registry

* **場景**：Pipeline steps 可插拔（`SaveResultsStep` / `ColorCheckStep` / `APIStep`）。
* **作法**：以 `register_step(name, factory)` 登錄；`pipeline: ["save_results", "api_step"]`。
* **好處**：新增功能零侵入、易 A/B 測試。

### 5.3 LRU Cache + Warmup

* **場景**：多產品/站別模型輪轉。
* **作法**：`OrderedDict` 保最近使用；超限彈出最舊，並記錄 `warmup_registry` 防重複暖機。

### 5.4 Producer–Consumer

* **場景**：影像/Excel IO 成本高。
* **作法**：Queue + 背景執行緒寫檔，避免主流程阻塞。

### 5.5 設定與驗證

* **作法**：YAML → dataclass / Pydantic；啟動時做 schema 檢查（路徑存在、型態、範圍）。

**JR→SR 檢核**

* ✅ 能畫出模組互動圖（Engine↔Detector↔Validator↔ResultSink↔Pipeline）。
* ✅ 能替換任一環節而不引入循環依賴與大型 side-effect。

**實務演練**

* 新增 `WebhookStep`：將結果 POST 到 REST 端點（含重試與退避）。

---

## 6. 可觀測性與回溯（Logging/Metrics/Tracing）

* **Structured Logging**：log line 必含 `product, area, inference_type, image_id, latency_ms, status`。
* **Error Taxonomy**：`ResultImageWriteError / ResultPersistenceError / ModelLoadError / ConfigValidationError`。
* **Metrics**：

  * 速度：`preprocess_ms / infer_ms / post_ms / total_ms`；
  * 資源：`vram_mb / rss_mb / cache_hit_rate`；
  * 品質：`precision / recall / f1 / auroc / pro`（離線評估）。
* **Tracing**（選配）：以 `trace_id` 串聯每一步（CLI/GUI/後端）。

**JR→SR 檢核**

* ✅ 能由一段 log 快速定位瓶頸與失敗點。
* ✅ 針對常見故障路徑提供「SOP + 設備/光源檢查表」。

---

## 7. 測試策略與品質保證

### 7.1 單元測試（Unit）

* **案例**：

  * `PositionValidator` 容差計算；
  * `ImageWriteQueue` 正確 flush；
  * `LRU Cache` 驅逐策略。
* **技巧**：對 IO/模型以介面注入 fake；測試不依賴真實 GPU/權重。

### 7.2 整合/端到端（IT/E2E）

* **案例**：資料夾批次 → 產出 Excel/影像 → 校驗檔案存在與行數。
* **資料治具**：小影像包 + 假權重（或官方最小模型）。

### 7.3 效能與回歸

* 固定隨機種子（torch/np/random）；
* 建立「基準報告」：版本、GPU、平均延遲、峰值 VRAM、mAP/ROC-AUC；
* 任何 PR 需附對比表。

**JR→SR 檢核**

* ✅ 會寫 table-driven / property-based 測試（如座標還原雙向一致）。

**實務演練**

* 為 YOLO 後處理寫 property test：encode→decode 框不變性（誤差 < 1px）。

---

## 8. 效能工程手冊

* **輸入大小**：與模型 stride 對齊（32 的倍數）；
* **序列→併發**：IO 走併發（thread/async），GPU 前傳盡量批次；
* **資料排程**：先排序大圖/小圖降低 reallocation；
* **記憶體**：預先分配、緩衝池；
* **半精度/INT8**：FP16 對視覺常足夠，INT8 需校準資料集；
* **Amdahl 定律**：優化熱點，度量後再改。

*效能檢核清單*

* [ ] 首張延遲 < 300ms（GPU）
* [ ] 穩態 throughput ≥ 15 FPS（640）
* [ ] VRAM 峰值 < 60% 裝置上限

---

## 9. 佈署與介面（CLI/GUI/Docker）

### 9.1 CLI

* 支援：單張/資料夾、輸出到 Excel/CSV、保存標註影像、調參（conf/iou/size/device）。
* 參數命名慣例：`--product --area --type --input --output --conf --iou --imgsz --device`。

### 9.2 GUI（PyQt）

* 元件：產品/站別選單、檔案對話框、影像顯示、熱圖切換、log 面板、狀態列（FPS/延遲）。
* 執行緒：推論執行緒 + UI 執行緒分離，避免 GUI 卡死。

### 9.3 Docker

* 基礎映像 `python:3.10`；
* 建議把模型與結果掛載 volume；
* GPU 版需 `--gpus all` 與對應 PyTorch CUDA。

---

## 10. 資料與評估（從開發走向生產）

* **資料版本化**：DVC / git-lfs（選擇其一）；
* **標註規範**：類別表、命名、邊界框容忍度、忽略區域標記；
* **抽樣驗收**：每周抽樣 50 張離線回歸；
* **漂移監控**：輸入統計（亮度/對比/色偏）、錯誤拓撲（漏檢/誤檢分佈）。

---

## 11. 安全與健壯性

* 輸入檔案驗證（副檔名/內容、路徑穿越防護）；
* 逾時與取消（長任務可中止）；
* 資源清理（thread join、queue drain、Excel 關閉）；
* 錯誤等級分明（可恢復 vs 致命）。

---

## 12. 實作練習（完整課綱）

> 每題目標：可在 1–3 小時內完成，附驗收標準。

1. **批次推論管線**：資料夾→推論→`results.xlsx`；**驗收**：CSV/Excel 行數=影像數，含每張 `latency_ms`。
2. **PositionValidator 擴充**：加入角度與相對關係檢查；**驗收**：對 10 張畸變影像仍準確。
3. **anomalib 閾值選擇器**：輸出 ROC、F1\@thr、PR 曲線（離線）；**驗收**：自動產生最佳 thr.json。
4. **WebhookStep**：POST 結果到測試伺服器（含 3 次退避重試）；**驗收**：斷網/異常能正確回報與重試。
5. **TensorRT 加速**：ONNX→TRT→數值對齊（mAP 差 < 0.5%）；**驗收**：延遲下降 ≥ 30%。
6. **回歸測試**：為關鍵模組撰寫 pytest；**驗收**：CI 過、涵蓋率 > 80%。

---

## 13. JR→SR 成長地圖

* **設計**：能做需求澄清 → 畫模組互動圖 → 選模型 → 定 KPI → 交付實作與說明文件。
* **工程**：懂抽象界面、解耦、測試邊界、觀測性、故障預案。
* **數據**：會設計實驗、讀 ROC/PR、成本敏感調參。
* **溝通**：能對非技術同仁說清楚「為何這麼做」。

*面試/內訓問題庫（自我檢核）*

1. 為什麼 anchor-free 能簡化偵測？代價是什麼？
2. NMS 與 Soft-NMS/DIoU-NMS 差在哪？何時需要？
3. PatchCore 與 PaDiM 內存占用與速度誰優？為何？
4. 你會如何定位「熱圖過度擴散」的根因？
5. 如何設計可滾動擴充的新站別？哪些檔案/設定要動？

---

## 14. 常見故障 → 排障樹（Troubleshooting Tree）

* **模型載入失敗** → 路徑/權限/版本相容 → `torch.cuda.is_available()` → 以 CPU 退化試跑。
* **首張超慢** → 確認 warmup 執行 → 檢查磁碟/網路延遲 → 關閉除錯影像保存。
* **熱圖不穩** → 攝影/光源/曝光 → 正規化與形態學 → 模型特徵層與閾值。
* **Excel 損壞或寫入不全** → flush/關閉順序 → 背景執行緒 join → 檔名唯一性（time+uuid）。

---

## 15. 附錄

### 15.1 術語速查（Glossary）

* **IoU / DIoU / CIoU**：框重疊度指標及其改良版本。
* **NMS / Soft-NMS**：非極大抑制；Soft 透過衰減分數而非硬刪除。
* **ROC-AUC / PRO**：二元分類與像素級評估指標。
* **Letterbox**：等比例縮放與補邊。

### 15.2 參考程式片段（可直接移植）

* **Warmup**（偽碼）

```python
if device != "cpu" and (product, area) not in warmup:
    x = torch.randn(1, 3, imgsz, imgsz, device=device)
    with torch.inference_mode():
        _ = model(x)
    warmup.add((product, area))
```

* **DIoU-NMS（概念流程）**

```text
對每個類別：
  1) 按分數排序
  2) 選最高者為 keep
  3) 其餘與 keep 計算 DIoU，若 > thr → 抑制
  4) 重覆直到列表空
```

* **ROC 找閾值（Youden’s J）**

```python
thr = np.linspace(0,1,1001)
J = tpr(thr) - fpr(thr)
best_thr = thr[J.argmax()]
```

---

## 16. 收尾

本手冊聚焦「把演算法原理落地成可維運的工程系統」。若你能完成所有 *實務演練* 並通過 *檢核*，你已具備 SR 級的分析、設計與交付能力。

> 下一步：把你的實作（腳本、報告、基準數據）整理成一份 10–15 分鐘的分享簡報，
> 給團隊做一次技術導讀與 Q\&A。

---

---

## 17. 相機/光源 SOP＋校正全攻略（可移植版）

> 目的：把「影像來源」標準化，降低資料漂移與誤檢/漏檢；可跨專案複用。

### 17.1 何時需要校正

* **需求特徵**：判讀幾何尺寸、相對位置、顏色/亮度一致性、長期穩定性。
* **介入點**：新站建置、鏡頭/光源調整、產線搬遷或週期性維護（建議每季/每 5000h）。

### 17.2 幾何校正（內外參）

* **內參（intrinsics）**：焦距 `(fx, fy)`、主點 `(cx, cy)`、畸變 `(k1, k2, p1, p2, k3)`；魚眼另有 `k1..k4`。
* **方法**：Zhang’s Calibration（棋盤/Charuco）。
* **流程**：

  1. 以固定曝光/焦距拍攝 ≥ 15 張棋盤（覆蓋不同姿態）。
  2. 角點擷取 → `calibrateCamera()` / 魚眼用 `fisheye::calibrate()`。
  3. 保存 `camera_intrinsics.json`（含解析度、矩陣、畸變、重投影誤差）。
  4. 推論前做 `undistort()`；必要時保留 `valid ROI` 避免邊緣黑邊。
* **外參（extrinsics）**：世界座標到相機座標（`R|t`）。

  * 以已知尺寸治具 + `solvePnP()` 建立像素→毫米（pix2mm）換算；
  * 保存 `extrinsics.json` 與 `pix2mm` 比例（X/Y 可能不同）。
* **ROI 與座標還原**：若前處理使用 **letterbox**，需記錄 `scale, pad`，以便把偵測框還原到原圖。

### 17.3 顏色/亮度校正（Color & Illumination）

* **白平衡**：

  * 灰卡（18% gray）拍攝 → 計算 RGB 比例 → 更新相機 WB 或在前處理做 `gain` 修正。
* **ColorChecker**：

  1. 在產線光下拍攝 X-Rite 24 色卡（RAW/無自動增益）；
  2. 擷取每格均值，解算 3×3 顏色校正矩陣（Least Squares）對齊到 sRGB；
  3. 保存 `color_correction.json`（矩陣 + gamma）。
* **光照/陰影校正（Flat-Field/Shading）**：

  * 以均勻白板拍攝 `I_flat` 與暗場 `I_dark`，保存 `gain = median(I_flat - I_dark) / (I_flat - I_dark)`；
  * 推論前：`I_corr = (I_raw - I_dark) * gain`。
* **亮度與均勻性指標**：

  * 平均亮度（V 或 L），**均勻性** `U = (min/avg)` 或 `1 - (max-min)/avg`；
  * 建議門檻：`U ≥ 0.85`，鄰區差異 ΔV ≤ 10（8-bit）。

### 17.4 光學/曝光/鏡頭參考

* **景深（DOF）**：收縮光圈（大 f-number）提升 DOF，但需提高光照或曝光時間。
* **快門**：偏動態建議 **全局快門**；滾動快門需降低震動/速度。
* **MTF/對焦 SOP**：在高對比斜邊上量 `MTF50`；以峰值作為對焦完成判準。

### 17.5 照明幾何與偏振

* **背光**：輪廓量測、缺件判斷。
* **明場（bright-field）**：表面色差、印刷。
* **暗場（dark-field）**：刮傷、凹凸紋（以斜角入射強化缺陷反差）。
* **同軸光/環形/穹頂光**：減反光/熱點；
* **偏振**：交叉偏振壓抑鏡面反射、平行偏振保留反射。

### 17.6 驗收指標與表單（模板）

* **幾何**：重投影誤差 ≤ 0.3 px；pix2mm 偏差 ≤ 0.5%；
* **亮度**：平均 V 於 \[200±15]（8-bit），均勻性 `U ≥ 0.85`；
* **顏色**：色卡 ΔE00 ≤ 3（關鍵色 ≤ 2）。
* **輸出**：`/calibration/<date>/` 目錄保存 `intrinsics.json / extrinsics.json / color_correction.json / flatfield.npz` 與 `report.md`。

### 17.7 與專案結合

* **設定檔** `calibration:` 區塊（範例）：

```yaml
calibration:
  enable: true
  undistort: true
  intrinsics: path/to/intrinsics.json
  extrinsics: path/to/extrinsics.json
  color_correction: path/to/color_correction.json
  flatfield: path/to/flatfield.npz
```

* 前處理順序：`flatfield → WB/CCM → resize/letterbox → to_tensor`。

### 17.8 常見問題（FAQ）

* **校正後邊角發黑？** 調整 `valid ROI` 或擴大棋盤覆蓋角度重做。
* **顏色不穩？** 固定曝光/WB，禁用自動增益；檢查光源老化與 AC 閃爍。

**JR→SR 檢核**

* ✅ 能在新站 2 小時內產出完整校正包與 `report.md`。
* ✅ 能說明「為何此光型」與「若更換光型需調的參數」。

**實作練習**

1. 建立你的 `calibration/` 目錄並填入三份 JSON + 一份 NPZ；
2. 以 10 張樣本驗證校正前後：幾何誤差下降、均勻性提升、ΔE 降低。

---

## 18. TensorRT / INT8 量化與校準實務

> 目的：在不犧牲精度的前提下，將延遲與資源壓到最佳化，形成可複用的加速流水線。

### 18.1 選型決策

* **FP32**：開發/對齊基準。
* **FP16**：首選加速，數值穩定度高。
* **INT8**：延遲/功耗最佳，但需校準；對低對比/小物體較敏感。

### 18.2 匯出與轉換

* **ONNX 匯出（Ultralytics）**：

```python
from ultralytics import YOLO
m = YOLO('best.pt')
m.export(format='onnx', opset=12, dynamic=False, imgsz=640)
```

* **TensorRT 轉換（trtexec 範例）**：

```bash
trtexec --onnx=best.onnx --saveEngine=best_fp16.engine --fp16 --workspace=4096
# INT8（使用校準集與快取）
trtexec --onnx=best.onnx --saveEngine=best_int8.engine --int8 \
        --workspace=4096 --shapes=<input_name>:1x3x640x640 \
        --calib=path/to/cali.cache
# 註：不同 TensorRT 版本對 --calib 的支援略有差異；若不支援，需用 API 實作校準器並生成快取。請以 `trtexec --help` 確認實際旗標。
```

> 注意：YOLO 解碼/NMS 可能以 plugin 內嵌；若外部 NMS，需在推論後自行處理。

### 18.3 校準資料

* **數量**：300–1000 張覆蓋實際分佈（亮暗、角度、批次差）。
* **一致性**：**前處理必須一致**（letterbox、normalize、色彩校正）。
* **輸出**：產生 `cali.cache`，納入版本控管。

### 18.4 推論與對齊

* **TensorRT Runtime** 或 **ONNX Runtime（TensorRT EP）** 都可：

```python
import onnxruntime as ort
so = ort.SessionOptions()
# ONNX Runtime + TensorRT EP：載入 .onnx，TRT 引擎會在第一次執行時自動建置與快取
sess = ort.InferenceSession('best.onnx',
                            providers=['TensorrtExecutionProvider','CUDAExecutionProvider'])

# 若要直接載入 .engine，請使用 TensorRT Python Runtime：
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
with open('best_int8.engine', 'rb') as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
```

* **數值回歸**：以同一批驗證集比較 `mAP@0.5`、`mAP@0.5:0.95`、延遲、VRAM；記錄差異。

### 18.5 精度守門

* **門檻**：`ΔmAP ≤ 0.5%`（相對）；關鍵類別 `ΔmAP ≤ 1%`。
* **失真排查**：

  * 檢查前處理一致；
  * 調整校準集涵蓋；
  * 改用 per-channel 量化（若可）；
  * 部分層鎖定為 FP16（mixed precision）。

### 18.6 效能量測 SOP

* **首張/穩態延遲**、**throughput（img/s）**、**VRAM 峰值**、**溫度/功耗**；
* 工具：內建 profiler 或外部 `nsys`；輸出 `perf_report.md`。

### 18.7 常見坑

* **動態形狀**：指定 `--shapes`；輸入長寬需與 stride 對齊（32 的倍數）。
* **NMS 差異**：TRT 與 PyTorch NMS 實作不同導致框有微差；以 IoU\@0.95 對齊再比 mAP。
* **精度掉太多**：收斂到 FP16；或重製更代表性的校準集。

### 18.8 與專案結合（設定）

```yaml
runtime:
  backend: tensorrt  # 或 onnxruntime-trt
  engine: path/to/best_int8.engine
  input:
    size: [640, 640]
    letterbox: true
    normalize: [0.0, 1.0]
  nms:
    type: standard   # Ultralytics 預設 IoU-based NMS；如需 DIoU/Soft-NMS 需自訂實作或使用對應 plugin
    conf: 0.25
    iou: 0.45
```

### 18.9 發佈物與驗收

* 交付：`best_fp16.engine / best_int8.engine / cali.cache / perf_report.md / accuracy_report.md`。
* 驗收：PR 需附 **精度對齊表** 與 **效能對齊表**（FP32/FP16/INT8）。

**JR→SR 檢核**

* ✅ 能在開發機完成 FP16/INT8 兩種引擎並生成性能/精度報告。
* ✅ 能說明量化導致的誤差來源與對策。

**實作練習**

1. 用 500 張校準集產生 INT8 引擎，填寫 `accuracy_report.md`；
2. 撰寫一個 `engine_benchmark.py`，輸出平均延遲/VRAM 與 95% 尾延遲（p95）。

---

## 19. 參考程式：可直接落地的範例（可複用模板）

> 說明：以下程式片段皆可獨立存為檔案運行（除非特別標註與本專案耦合）。
> 盡量使用「無版本鎖定」的 API；若你的套件版本不同，請對照官方文件微調。

### 19.1 CLI：資料夾批次推論（YOLOv8/11，輸出 CSV＋標註圖）

```python
#!/usr/bin/env python3
# batch_infer.py
import argparse, time, csv, os
from pathlib import Path
import cv2
from ultralytics import YOLO

def run(model_path, input_dir, output_dir, conf=0.25, iou=0.45, imgsz=640, device=None):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)
    if device:  # e.g., '0' or 'cpu'
        model.to(device)
    # header
    csv_path = out / 'predictions.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image','cls','conf','x1','y1','x2','y2','latency_ms'])
        for p in sorted(Path(input_dir).glob('*.jpg')) + sorted(Path(input_dir).glob('*.png')):
            img = cv2.imread(str(p)); t0 = time.time()
            res = model.predict(img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
            latency_ms = (time.time()-t0)*1000
            # 取第一張結果
            r = res[0]
            # 畫框
            im_anno = r.plot()  # 已標註 BGR numpy
            cv2.imwrite(str(out / p.name), im_anno)
            # 寫 CSV
            for b in r.boxes:
                xyxy = b.xyxy[0].tolist(); cls_id = int(b.cls[0]); confv = float(b.conf[0])
                w.writerow([p.name, r.names[cls_id], f"{confv:.3f}", *[f"{v:.1f}" for v in xyxy], f"{latency_ms:.1f}"])
    print(f"Saved: {csv_path}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default=None)
    run(**vars(ap.parse_args()))
```

### 19.2 位置校驗器（PositionValidator）：容差與相對關係

```python
# position_validator.py
from dataclasses import dataclass
from typing import Dict, Tuple, List

@dataclass
class TargetSpec:
    cx: float; cy: float; w: float; h: float
    tol_px: float = 5.0  # 絕對容差（像素）
    tol_pct: float = 0.05  # 相對容差（佔期望寬高）

class PositionValidator:
    def __init__(self, specs: Dict[str, TargetSpec]):
        self.specs = specs

    @staticmethod
    def _ok(v, ref, tol_px, tol_pct):
        tol = max(tol_px, tol_pct * ref)
        return abs(v - ref) <= tol

    def validate(self, preds: Dict[str, Tuple[float,float,float,float]]):
        # preds: {name: (cx,cy,w,h)}
        errors: List[str] = []
        for name, spec in self.specs.items():
            if name not in preds:
                errors.append(f"missing:{name}"); continue
            cx,cy,w,h = preds[name]
            if not self._ok(cx, spec.cx, spec.tol_px, spec.tol_pct):
                errors.append(f"{name}.cx")
            if not self._ok(cy, spec.cy, spec.tol_px, spec.tol_pct):
                errors.append(f"{name}.cy")
            if not self._ok(w,  spec.w,  spec.tol_px, spec.tol_pct):
                errors.append(f"{name}.w")
            if not self._ok(h,  spec.h,  spec.tol_px, spec.tol_pct):
                errors.append(f"{name}.h")
        # 相對關係（示例）：A 必須左於 B，上於 C
        if 'A' in preds and 'B' in preds:
            if preds['A'][0] >= preds['B'][0]:
                errors.append('A_left_of_B')
        if 'A' in preds and 'C' in preds:
            if preds['A'][1] >= preds['C'][1]:
                errors.append('A_above_C')
        return len(errors) == 0, errors
```

### 19.3 WebhookStep（退避重試＋JSON 結果上報）

```python
# webhook_step.py
import json, time, urllib.request
from urllib.error import URLError, HTTPError

def post_with_retry(url: str, payload: dict, retries=3, backoff=0.5, timeout=5.0):
    data = json.dumps(payload).encode('utf-8')
    for i in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read().decode('utf-8')
        except (URLError, HTTPError) as e:
            if i == retries-1: raise
            time.sleep(backoff * (2 ** i))
```

### 19.4 幾何校正：棋盤標定與 JSON 輸出

```python
# camera_calib.py
import json, glob, cv2
import numpy as np

def calibrate(img_dir, square_size=10.0, pattern=(9,6), out_json='intrinsics.json'):
    objp = np.zeros((pattern[0]*pattern[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1,2)
    objp *= square_size
    objpoints, imgpoints, shape = [], [], None
    for p in glob.glob(f"{img_dir}/*.jpg"):
        img = cv2.imread(p); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, pattern)
        if ok:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001))
            objpoints.append(objp); imgpoints.append(corners2)
    ret, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
    with open(out_json,'w') as f:
        json.dump({'K':K.tolist(),'dist':dist.tolist(),'reproj_err':float(ret),'shape':shape}, f, indent=2)
    print('saved', out_json)
```

### 19.5 平場（Flat-Field）與暗場校正

```python
# flat_field.py
import numpy as np, cv2

def build_flatfield(white_path, dark_path, out_npz='flatfield.npz'):
    white = cv2.imread(white_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    dark  = cv2.imread(dark_path,  cv2.IMREAD_GRAYSCALE).astype(np.float32)
    eps = 1e-6
    gain = np.median(white - dark) / np.maximum(white - dark, eps)
    np.savez_compressed(out_npz, gain=gain, dark=dark)

def apply_flatfield_gray(img, ff_npz):
    d = np.load(ff_npz); gain, dark = d['gain'], d['dark']
    imgf = img.astype(np.float32)
    corr = (imgf - dark) * gain
    return np.clip(corr, 0, 255).astype(np.uint8)

def apply_flatfield_color(img_bgr, ff_npz):
    # 以灰階增益近似校正三通道（簡化），若有色偏建議計算「每通道」增益
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    corr_gray = apply_flatfield_gray(gray, ff_npz)
    ratio = (corr_gray.astype(np.float32) + 1e-6) / (gray.astype(np.float32) + 1e-6)
    b,g,r = cv2.split(img_bgr.astype(np.float32))
    b = np.clip(b*ratio,0,255); g = np.clip(g*ratio,0,255); r = np.clip(r*ratio,0,255)
    return cv2.merge([b,g,r]).astype(np.uint8)
```

> 註：最佳做法是為 **B/G/R 各自**建立 `gain_b/g/r`，這裡提供灰階近似作為簡化版本。

### 19.6 簡易色彩校正（3×3 CCM 最小平方法）

````python
# color_ccm.py
import numpy as np
# src: n×3（相機量測 RGB），tgt: n×3（目標 sRGB 線性空間）
# *注意*：若使用 sRGB，請先從 gamma 空間解碼至線性，再回寫時再編碼

def srgb_to_linear(x):
    x = x.astype(np.float32)
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x+a)/(1+a))**2.4)

def linear_to_srgb(x):
    a = 0.055
    return np.where(x <= 0.0031308, 12.92*x, (1+a)*np.power(x,1/2.4)-a)

def solve_ccm(src_srgb: np.ndarray, tgt_srgb: np.ndarray):
    src = srgb_to_linear(src_srgb)
    tgt = srgb_to_linear(tgt_srgb)
    A = np.hstack([src, np.ones((src.shape[0],1))])  # 加偏置
    X, _, _, _ = np.linalg.lstsq(A, tgt, rcond=None) # 4×3，前三列 3×3，最後 1×3 偏置
    M = X[:3,:]; b = X[3,:]
    return M, b

def apply_ccm(img_bgr, M, b):
    img = img_bgr[...,::-1].astype(np.float32)/255.0  # BGR→RGB
    h,w,_ = img.shape
    x = img.reshape(-1,3)
    y_lin = x @ M + b
    y = linear_to_srgb(np.clip(y_lin,0,1)).reshape(h,w,3)
    return (y[...,::-1]*255).astype(np.uint8)  # 回到 BGR
```python
# color_ccm.py
import numpy as np
# src: n×3（相機量測 RGB），tgt: n×3（目標 sRGB）

def solve_ccm(src: np.ndarray, tgt: np.ndarray):
    A = np.hstack([src, np.ones((src.shape[0],1))])  # 加偏置
    X, _, _, _ = np.linalg.lstsq(A, tgt, rcond=None) # 4×3，前三列是 3×3，最後一列是偏置
    M = X[:3,:]; b = X[3,:]
    return M, b

def apply_ccm(img, M, b):
    h,w,_ = img.shape
    x = img.reshape(-1,3).astype(np.float32)/255.0
    y = x @ M + b
    y = np.clip(y, 0, 1)
    return (y.reshape(h,w,3)*255).astype(np.uint8)
````

### 19.7 ROC 找最佳閾值（Youden’s J）

```python
# best_threshold.py
import numpy as np

def best_thr_by_youden(scores: np.ndarray, labels: np.ndarray):
    # scores: 越大越異常；labels: 1 異常, 0 正常
    thr = np.linspace(scores.min(), scores.max(), 1001)
    tpr = [( (scores[labels==1] >= t).mean() if (labels==1).any() else 0) for t in thr]
    fpr = [( (scores[labels==0] >= t).mean() if (labels==0).any() else 0) for t in thr]
    J = np.array(tpr) - np.array(fpr)
    i = int(J.argmax())
    return float(thr[i]), float(tpr[i]), float(fpr[i])
```

### 19.8 TensorRT/ONNX Runtime 基準量測（延遲/吞吐/VRAM）

```python
# engine_benchmark.py
import time, numpy as np
import onnxruntime as ort

def ort_trt_session(onnx_path):
    return ort.InferenceSession(onnx_path, providers=['TensorrtExecutionProvider','CUDAExecutionProvider'])

def bench(sess, input_name=None, runs=100, hw=(1,3,640,640)):
    if input_name is None:
        input_name = sess.get_inputs()[0].name
    x = (np.random.rand(*hw).astype(np.float32))
    # warmup
    for _ in range(5):
        _ = sess.run(None, {input_name: x})
    ts = []
    for _ in range(runs):
        t0=time.time(); _ = sess.run(None, {input_name:x}); ts.append((time.time()-t0)*1000)
    ts = np.array(ts)
    return {'mean_ms':float(ts.mean()), 'p95_ms':float(np.percentile(ts,95))}

if __name__=='__main__':
    sess = ort_trt_session('best.onnx')
    print(bench(sess))
```

### 19.9 pytest：座標還原可逆性（Property-based 思維）

```python
# test_letterbox_coords.py
import numpy as np

def to_letterbox(xyxy, in_wh, out_wh):
    # xyxy in 原圖座標
    (w,h),(W,H)=in_wh,out_wh
    r = min(W/w, H/h)
    nw, nh = int(w*r), int(h*r)
    padw, padh = (W-nw)//2, (H-nh)//2
    x1,y1,x2,y2 = xyxy
    return [x1*r+padw, y1*r+padh, x2*r+padw, y2*r+padh], (r, padw, padh)

def from_letterbox(xyxy_l, meta):
    r, padw, padh = meta
    x1,y1,x2,y2 = xyxy_l
    return [(x1-padw)/r, (y1-padh)/r, (x2-padw)/r, (y2-padh)/r]

def test_roundtrip():
    in_wh=(1920,1080); out_wh=(640,640)
    for _ in range(100):
        x1,y1 = np.random.randint(0,1000,2)
        x2,y2 = x1+np.random.randint(10,500), y1+np.random.randint(10,300)
        a, meta = to_letterbox([x1,y1,x2,y2], in_wh, out_wh)
        b = from_letterbox(a, meta)
        assert max(abs(b[0]-x1),abs(b[1]-y1),abs(b[2]-x2),abs(b[3]-y2)) < 1e-3
```

### 19.10 YAML：最小可運行設定（偵測＋校正＋結果輸出）

```yaml
runtime:
  backend: ultralytics
  weights: runs/detect/train/weights/best.pt
  device: 0            # 或 'cpu'
  imgsz: 640
  conf: 0.25
  iou: 0.45

calibration:
  enable: true
  undistort: true
  intrinsics: calibration/intrinsics.json
  color_correction: calibration/color_correction.json
  flatfield: calibration/flatfield.npz

pipeline:
  - load_image
  - apply_calibration
  - detect
  - position_validate
  - save_results

position_validate:
  specs:
    A: {cx: 320, cy: 240, w: 80, h: 60, tol_px: 5, tol_pct: 0.05}
    B: {cx: 480, cy: 240, w: 80, h: 60, tol_px: 5, tol_pct: 0.05}

results:
  save_dir: results/
  save_csv: true
  save_images: true
```

### 19.11 Logging：結構化欄位（內建 logging）

```python
# logging_setup.py
import logging, sys

def setup_logger():
    h = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    logger = logging.getLogger('inference')
    logger.setLevel(logging.INFO); logger.addHandler(h)
    return logger

# 使用
# log.info('infer', extra={'product':p,'area':a,'latency_ms':12.3,'status':'OK'})
```

> 小提醒：若你要把這些程式直接併入本專案，可將 `19.1/19.2/19.3` 放入 `core/cli/`, `core/validators/`, `core/services/`，並補上相對應的單元測試（見 19.9 範例）。

---

## 20. 資料集與標註（可遷移標準）

### 20.1 目錄規範（YOLO 標準）

```
datasets/
  projectX/
    images/{train,val,test}/xxx.jpg
    labels/{train,val,test}/xxx.txt  # 每列: class cx cy w h (歸一化)
```

* 檔名對齊：`images/xxx.jpg` ↔ `labels/xxx.txt`。
* 座標範圍：`[0,1]`，空白檔表示 **無物件**。

### 20.2 標註檢查腳本（基本衛生）

```python
# validate_yolo_dataset.py
from pathlib import Path

def validate(root='datasets/projectX'):
    img_dir = Path(root)/'images/train'
    lab_dir = Path(root)/'labels/train'
    issues=[]
    for p in img_dir.glob('*.jpg'):
        q = lab_dir/(p.stem+'.txt')
        if not q.exists():
            issues.append(f'missing label: {q}')
        else:
            for i,l in enumerate(q.read_text().splitlines()):
                parts = l.strip().split()
                if len(parts)!=5: issues.append(f'bad line {q}:{i+1}')
                else:
                    try:
                        cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                        for v in (cx,cy,w,h):
                            if not (0.0<=v<=1.0): issues.append(f'out of range {q}:{i+1}')
                    except Exception:
                        issues.append(f'parse error {q}:{i+1}')
    print('
'.join(issues) if issues else 'OK')

if __name__=='__main__':
    validate()
```

### 20.3 COCO ↔ YOLO 簡易轉換（示意）

* 推薦工具：`roboflow`/`fiftyone`/`ultralytics convert`；若自寫，請處理：

  * COCO bbox（x,y,w,h，絕對座標）→ YOLO（cx,cy,w,h，歸一化）。
  * 分類 id 映射（class map）。

### 20.4 擴增（Augmentation）原則

* 幾何（flip/rotate/scale/perspective）與光度（brightness/contrast/hue/sat/gamma/gauss-noise）。
* 「可解釋」優先：避免破壞關鍵幾何（量測任務減少旋轉）。
* 驗收：以固定驗證集觀察 mAP/召回變化；避免 over-augment。

---

## 21. 可重現性與 CI/CD

### 21.1 決定論設定（PyTorch）

```python
# determinism.py
import os, random, numpy as np, torch
random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass
```

### 21.2 Dockerfile（最小可用，CUDA 版）

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir ultralytics onnxruntime-gpu==1.18.* opencv-python-headless==4.*
WORKDIR /app
COPY . /app
CMD ["python3","batch_infer.py","--help"]
```

### 21.3 GitHub Actions（pytest + flake8）

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -r requirements.txt || true
      - run: pip install pytest flake8
      - run: pytest -q
      - run: flake8 . --max-line-length=120
```

### 21.4 Jenkins（Pipeline 範例，簡版）

```groovy
// Jenkinsfile（簡版）
pipeline {
  agent any
  stages {
    stage('Test') {
      steps {
        sh 'pip install -r requirements.txt || true'
        sh 'pip install pytest'
        sh 'pytest -q'
      }
    }
  }
}
```

---

## 22. 結果格式與 Schema（跨專案一致）

### 22.1 CSV 欄位（偵測）

* `image, cls, conf, x1, y1, x2, y2, latency_ms`（與 19.1 一致）。

### 22.2 JSON Schema（偵測＋校驗）

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["image","detections","latency_ms","status"],
  "properties": {
    "image": {"type":"string"},
    "latency_ms": {"type":"number"},
    "status": {"type":"string", "enum":["OK","FAIL","ERROR"]},
    "detections": {"type":"array","items":{
      "type":"object",
      "required":["cls","conf","bbox"],
      "properties":{
        "cls":{"type":"string"},
        "conf":{"type":"number"},
        "bbox":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4}
      }
    }},
    "position_errors": {"type":"array","items":{"type":"string"}}
  }
}
```

---

## 23. 專案移植 Checklist（30 分鐘自檢）

* [ ] 影像來源與校正包（17 章）已建立，驗收指標達標
* [ ] 模型與輸入尺寸/letterbox 對齊
* [ ] `batch_infer.py` 跑通，生成 CSV/標註圖
* [ ] PositionValidator 規格化，誤差容忍定義
* [ ] ROC/F1 閾值選定並記錄版本
* [ ] Dockerfile build/跑通；CI 測試綠燈
* [ ] 結果 JSON/CSV/Excel 欄位符合 22 章 Schema
* [ ] 效能報告（首張/穩態/VRAM）與版本紀錄

---

## 24. 已知相容性與易踩坑（速查）

* TensorRT 旗標：不同版本 `--calib/--shapes` 差異，請以 `trtexec --help` 為準。
* ONNX 輸入名稱：以 `sess.get_inputs()[0].name` 讀取，不要寫死 'images'。
* Logging 需考慮 JSON 序列化（numpy 數值先轉 float）。
* OpenCV 與 numpy 版本關係：某些舊版 OpenCV 對新 numpy 會有 ABI 警告。
