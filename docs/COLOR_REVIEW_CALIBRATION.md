# 顏色失敗覆核與門檻校正

本流程處理「框與類別都正確，但顏色分數沒有通過門檻」的案例。顏色回饋與 YOLO 框選資料完全分流，避免把門檻問題誤送進物件偵測模型。

## 作業員看到的按鈕

當結果包含 `COLOR_MISMATCH` 時，畫面只顯示下列相關選項：

| 按鈕 | 實物真值 | 後續路由 |
|---|---|---|
| 顏色確實 NG（顏色判定正確） | 實物顏色不合格 | 顏色校正資料 |
| 顏色其實 OK（門檻過嚴） | 實物顏色合格，系統過殺 | 顏色校正資料 |
| 顏色覆核＋框需修正 | 同時記錄顏色真值及框位置／數量或類別錯誤 | 顏色校正＋YOLO 補標 |
| 圖片無法判定（不採用） | 證據不足 | 排除，不送入任何訓練 |

畫面會顯示每個失敗項目的預期顏色、預測顏色、`diff` 與執行時門檻。`diff <= threshold` 代表門檻本身已通過；若整體仍失敗，畫面會標示為其他顏色規則未過，按鈕也改顯示「顏色規則過嚴」。這類資料會保存供規則分析，但不會被門檻最佳化器使用。舊快照若沒有項目分數，匯出會 fail closed，不會猜測門檻或污染校正資料。

「閾值未達標」是獨立的失敗原因分類，不等同於顏色真值。紀錄會分開保存 `failure_category=threshold_not_met` 與 `failure_source`；來源可為 `yolo`、`color` 或未來 detector 的穩定代碼。實際 OK／NG 與送訓路由仍由覆核答案決定，不能只靠此分類自動推論。

## 資料分類

按下「送出顏色校正資料」後，資料寫入：

```text
Yolo11_auto_train/data/<product>/<area>/color_review/
  images/        # 以影像 SHA-256 產生穩定 sample ID
  feedback.csv   # item-level 真值、diff、舊門檻、checker 類型與稽核欄位
```

顏色單一路由不會建立 `raw/images` 或 `raw/labels`，也不會啟動 YOLO 補訓。相同影像再次覆核時，以最新決定覆蓋同一個 sample/item，不會重複累積。

## 工程師校正與批准

先在 `Yolo11_auto_train` 產生 shadow report；此步不修改產線設定：

```powershell
picture-tool-color-calibrate recommend data --output color-threshold-report.json
```

只有 `failure_kind=threshold` 的回饋會參與門檻建議；顏色規則失敗會在報告中另行計數。預設每一個產品／站別／模型／顏色群組至少需要 30 筆，其中實際 OK、NG 各至少 5 筆。最佳候選以誤放 NG 成本 10、誤殺 OK 成本 1 評分，而且 NG 誤放率不得高於 0。資料不足、無安全改善或類別分布重疊時，只產生原因，不產生可部署建議。

工程師完成報告與現場樣本抽驗後，才可具名批准：

```powershell
picture-tool-color-calibrate apply color-threshold-report.json `
  --models-root ..\yolo11_inference\models `
  --approver "OP-王小明"
```

套用前會驗證報告 checksum、目標路徑及目前門檻是否仍與報告一致。通過後先備份 `config.yaml`，再以原子替換發布；設定漂移、鎖定、無批准人或任何寫入錯誤都會中止。稽核紀錄保存在模型目錄的 `color_threshold_history.json`，備份保存在 `color_threshold_backups/`。

`stats` checker 的執行結果以 diff 表示，但設定檔使用相似度分數；部署工具會自動執行 `config_value = 1 - diff_threshold`。其他 checker 直接使用 diff 門檻，作業員與工程師不需要自行換算。
