# PCBA 受控試產指南

本文件整合檢測範圍、工程設定、常用命令與試產流程。適用於目前以 YOLO
判定元件缺件、錯件、多件與明顯位移的受控試產；它不是完整 AOI 能力聲明。

## 1. 能力邊界

目前決策層可輸出下列可追溯原因碼：

- `MISSING`：預期元件未出現。
- `WRONG_COMPONENT`：預期位置出現其他類別。
- `UNEXPECTED_COMPONENT`：出現設定外元件。
- `POSITION_SHIFT`：元件位置超出允收範圍。

焊點、極性字樣、刮傷、污染與 OCR 類檢查仍需 ROI 第二階段模型、傳統影像
處理、OCR 或模板比對；不得只靠目前 YOLO 結果宣稱已驗證。

目前位置校正採用由可信配對框估計的共用平移量，適合固定相機、固定治具且
旋轉／透視誤差很小的情境。若板件會旋轉、左右偏移不同，或共用平移後仍有
局部位置誤差，應改用 fiducial affine／homography 並重新驗證。

## 2. 試產前設定

產品設定至少要包含：

```yaml
current_product: PCBA1
current_area: A

weights: models/PCBA1/A/yolo/weights/best.onnx
conf_thres: 0.4
iou_thres: 0.45
output_dir: Result
fail_on_unexpected: true
save_original: true
save_annotated: true
save_crops: true

expected_items:
  PCBA1:
    A:
      - R101
      - C205

position_config:
  PCBA1:
    A:
      enabled: true
      mode: center
      tolerance: 12
      tolerance_unit: pixel
      alignment:
        enabled: true
        min_source_count: 2
      missing_slot_check:
        enabled: true
      expected_boxes:
        R101: {x1: 100, y1: 120, x2: 140, y2: 150}
        C205: {x1: 220, y1: 260, x2: 250, y2: 290}
```

`configs/products/pcba_example.yaml` 只供結構參考。正式設定的 expected boxes、
門檻、曝光、焦距與光源必須來自實際工位量測，不可直接複製範例或其他治具。

2026-05-18 的 PCBA1 A／B 設定檢查沒有 blocking `FAIL`，但只代表可進入受控
試產候選，不代表可無人值守上線。當時仍有：

- A 的 IoU 位置容許值過寬，且 `missing_slot_check` 關閉。
- B 的位置容許值為 `10.27%`，且未設定 `missing_slot_check`。

開始每一輪試產前都要重跑 readiness；上述舊結果不能取代當次證據。

## 3. 常用命令

從 `yolo11_inference` 根目錄執行 `pcba.bat`。批次檔優先使用
`D:\miniconda\envs\yolo_anomalib\python.exe`，找不到時才使用目前的 `python`。

```powershell
# A／B 區 readiness
.\pcba.bat readiness A
.\pcba.bat readiness B

# 收集 NG；加入 PASS 供 Golden board 複核
.\pcba.bat collect --result-root ..\Result
.\pcba.bat collect --result-root ..\Result --include-pass

# 將證據限制在單一產品／區域，以及實站 pilot 的精確時間窗。
.\pcba.bat collect --result-root ..\Result --product PCBA1 --area A `
  --start-time <ISO-8601> --end-time <ISO-8601> --include-pass --strict-evidence `
  --output-csv ..\release_artifacts\yolo11_inference\review_manifest_PCBA1_A.csv `
  --output-json ..\release_artifacts\yolo11_inference\review_manifest_PCBA1_A.json

# 依人工標註後的 manifest 建立摘要
.\pcba.bat summary A `
  --review-manifest-csv ..\release_artifacts\yolo11_inference\review_manifest_PCBA1_A.csv
# B 區需先以相同方式收集 B 的 scoped manifest，再把該 CSV 明確傳給 summary B。

# 一次執行 readiness、精確時窗收集與 pre-pilot 摘要
.\pcba.bat pilot A --product PCBA1 --result-root ..\Result --include-pass `
  --start-time <ISO-8601> --end-time <ISO-8601>
.\pcba.bat pilot B --product PCBA1 --result-root ..\Result --include-pass `
  --start-time <ISO-8601> --end-time <ISO-8601>
```

上例的 `..\Result` 對應目前 `workspace.yaml` 的 `inference_results: Result`。
若部署環境使用其他工作區設定，改傳該設定解析後的實際結果目錄。
任何 product/area/time filter 都會使用一對 deterministic scope-specific CSV/JSON
預設檔名；若明確指定其中一個 CSV，JSON 會自動使用同 stem。輸出禁止放在
`Result` 樹內，避免覆寫檢測 snapshot、SQLite 或備份。

輸出位置：

- standalone readiness：`readiness_report_A.json`／`readiness_report_B.json`
- 未篩選 collect：在 station review root 產生
  `review_manifest.csv`／`review_manifest.json`
- 有 product／area／time filter 的 collect 或 one-step pilot：在 `workspace.yaml`
  指定的 station review root（目前 `..\station_data\yolo11_inference`）產生
  `review_manifest_<PRODUCT>_<AREA>_<scope-hash>.csv`／`.json`
- one-step pilot 在目前目錄產生
  `readiness_report_<PRODUCT>_<AREA>_<scope-hash>.json` 與
  `pilot_acceptance_summary_<PRODUCT>_<AREA>_<scope-hash>.json`／`.md`

`pilot_acceptance_summary` **只代表 pilot 前篩檢**。其 `merge_eligible` 永遠為
`false`，不能代表現場驗收完成或人工核准。Exit code `0` 只表示可開始受監督
pilot；readiness 的 `WARN` 在具名工程接受紀錄完成前仍維持 HOLD。

## 4. 試產順序

1. 從 `Yolo11_auto_train` 部署核准的模型與設定到
   `models/<product>/<area>/yolo/`。
2. 填入本工位量測的 `expected_items` 與 `expected_boxes`。
3. 執行 readiness；所有 `FAIL` 必須修正，所有 `WARN` 必須有具名工程說明。
4. 以核准的 Golden OK 板做重複取像，確認結果與治具穩定性。
5. 以每個宣稱支援的已知 NG 類型驗證原因碼與證據保存。
6. 在不阻擋產線判定的模式執行 dry run，人工複核全部 FAIL 與抽樣 PASS。
7. 在該輪 scoped manifest CSV 填寫 `review_label` 與 `review_note`，並將同一路徑
   明確傳給 `summary --review-manifest-csv`。
8. 需要補訓時，以 `tools/export_review_dataset.py` 匯出後人工標註；空標籤檔
   不可直接當成真值。
9. 產生 acceptance summary，並填寫
   [PCBA 試產驗收紀錄](PCBA_PILOT_ACCEPTANCE_TEMPLATE.md)。
10. 保存核准版本與可用的模型／設定回滾組合。

## 5. 證據要求

每個 FAIL 至少保留：最終狀態、`decision.reasons`、缺件／多件清單、slot mismatch、
alignment metadata、標註圖、可用的 crop、config snapshot、模型路徑與版本、
confidence／IoU／影像尺寸門檻、product 與 area。

人工複核標籤建議使用 `confirmed_ng`、`false_positive`、`false_negative`、
`uncertain`，並保留複核人員與時間。測試資料不得同時當作獨立驗收資料。

## 6. Go／No-Go

可進入 supervised pilot 的最低條件：

- readiness 無 `FAIL`，每個 `WARN` 有書面接受。
- Golden board 重複性通過。
- 每個已宣稱 NG 類型都以實體或已人工確認影像驗證。
- 每個 FAIL 都能回查影像、設定與模型。
- dry run 的誤殺／漏檢率符合該工位事先定義的門檻。
- rollback 組合可用且已記錄。

下列任一情況均不得進入 unattended production：expected boxes 尚未量測、相機／
光源仍在調整、板件變化超出平移校正能力、review manifest 無法生成、沒有誤殺／
漏檢複核流程，或 rollback 未演練。

## 7. 已知限制

- readiness 是設定完整性檢查，不證明模型準確率或現場吞吐量。
- `pcba.bat` 不會替代實體 OK／NG 板驗證。
- `pilot_acceptance_summary` 只整理輸入證據；輸入缺漏時不能視為正式批准。
- 輸出目錄取決於工作區設定；執行前應確認命令指向本工位的實際 Result root。
