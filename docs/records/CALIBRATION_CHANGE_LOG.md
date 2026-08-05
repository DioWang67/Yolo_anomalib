# Calibration Change Log

本文件是檢測閾值與站別設定的追加式稽核紀錄，不是一般開發進度表。
所有 `conf`、`iou`、顏色參數、位置容差、Anomalib 閾值及影像保存政策變更，
都必須先完成回歸驗證再記錄於此。

## 填寫規則

- 一次部署或回滾一列，不覆寫既有紀錄。
- `變更前雜湊` 與 `變更後雜湊` 優先填 `config_hash`；模型變更另填模型版本。
- 回歸證據必須指向可追溯的報表、Golden/known-NG 測試紀錄或任務編號。
- 批准人不得與執行人皆留空；不得記錄密碼、PIN、Token 或其他機密。
- 緊急回滾也必須補列一筆，並在「結果」說明回滾原因。

## 變更紀錄

| 日期時間 | 產品／工位 | 變更項 | 舊值 | 新值 | 變更前雜湊／版本 | 變更後雜湊／版本 | 回歸證據 | 結果 | 執行人 | 批准人 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YYYY-MM-DD HH:MM | Product / Area | 參數或模型 | before | after | hash/version | hash/version | report/path/job ID | PASS/ROLLBACK | 姓名 | 姓名 |
| 2026-07-29 13:43 | Cable1 / A | `color_threshold_overrides.black`（畫面 diff 上限） | 內建 `0.45`（`thr=0.55`） | `0.42`（`thr=0.58`） | `2e5c3e7efba4e5f8c4c667e0bdc083e2b341e695911206bab66a8536f04850b7` | `7391a5dc7b0cd68a0f1512e24f723d8957c83fd1b409a88412a5e54f5c30f13b` | runtime loader 驗證；顏色 override／stats checker 測試 14 passed | 自動驗證 PASS；正常 Black 與錯色品現場回歸待完成 | Codex（依使用者指示） | 待現場具名批准 |
