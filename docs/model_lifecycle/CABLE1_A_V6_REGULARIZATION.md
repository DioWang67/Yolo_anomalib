# Cable1/A stats-robust-v6 補正作業清單

> 一次性作業說明，不是常規流程。常規流程見
> `MODEL_COMBINATION_ACCEPTANCE.md` 與 `COLOR_REVIEW_CALIBRATION.md`。
> 狀態核對日期：2026-09-09。動手前請先重跑「第 0 步」確認現況沒變。

## 這份清單要解決什麼

v6 顏色基準**已經在線上**，但它是以「風險接受」模式、帶著空白理由推上去的，
而且沒有任何機制在守護演算法一致性。要補的是**正當性與防呆**，不是部署。

## 第 0 步：確認現況（唯讀，可隨時重跑）

在 `yolo11_inference/` 下執行：

```powershell
python -c "import json;d=json.load(open('models/Cable1/A/yolo/color_stats.json'));print('deployed has recalibration:', 'recalibration' in d)"
```

預期 `False`。另外確認三件事：

| 項目 | 位置 | 目前值 |
|---|---|---|
| 現行 release | `station_data/yolo11_inference/.inspection_releases/active/61420e40ba6b0fcd7c075617.json` | `inspection-v1.0.10`，`mode: RISK_ACCEPTED`，理由欄 `"1"` |
| release 內容 | `.../releases/61420e40.../e99215e3.../release.json` | `status: DRAFT`，`validation.run_id: UNVALIDATED`，`sample_count: 0` |
| 綁定的顏色設定 | `.color_profiles/73f89061597f82056b963cfb/color_model.json` | `stats-robust-v6`（正確，Black `coverage_mean` 0.5153） |
| Black 門檻修訂 | `.color_revisions/active/edd13955f263e012f61cd973.json` | `color-v1.0.3`（0.20），已 active，有完整理由 |
| 演算法強制 | `models/Cable1/A/yolo/config.yaml` | **沒有 `color_baseline_algorithm_enforcement` 這個 key** → 實際生效 `warn` |

**線上跑的組合本身是對的**（v6 程式 + v6 基準 + v6 校出來的 Black 門檻），
就是那個量到 98.0% 的組合。問題在下面兩點。

## 先認清一個硬限制：正式上線目前走不到

`inspection_release_store.py:107-110` 規定：只要組合含顏色元件且
`color_escape_known` 為 false，就會產生警告；而 `allowed_modes`
（`:89-93`）只要有任何警告就**移除「正式上線」選項**，只留「限定試用」和
「風險接受」。

現有驗收資料 `station_data/yolo11_inference/acceptance/Cable1/A/ground_truth.csv`
共 250 筆、全部人工確認（OK 173 / NG 77），但

```
COLOR_MISMATCH 標記數 = 0
```

**沒有任何一筆被標成顏色不良**。所以就算驗收矩陣跑得完美，`color_escape_known`
仍是 false，正式上線依然選不到。這不是跑一次矩陣能解決的，是資料標註缺口。

`MODEL_COMBINATION_ACCEPTANCE.md:403` 也明講：缺顏色 NG 真值時
**「不得依上述數字直接宣告正式完整上線」** —— 我們手上的「漏檢 0」其實是
「未知」，不是「零」。

---

## 路線 A：今天就能做完（目標＝限定試用 + 防呆）

把目前這個 `RISK_ACCEPTED` + 空白理由的狀態，換成有驗收證據、有真實理由的
`LIMITED_TRIAL`，並把演算法防呆打開。

### A1. 跑正式驗收矩陣

GUI：工程設定 → PIN → 版本與上線 → **「驗收選取版本」**（或
**「開啟完整驗收工具」**）。
資料沿用既有那 250 筆，不需重新標註（畫面上的「驗收資料」摘要會這樣提示）。

產出會寫進不可變的驗收報告，並把 release 的 `validation` 從 `UNVALIDATED`
換成真實 run。

> 這一步取代 `.tmp/color_rebuild_Cable1_A_v6/` 裡那些臨時腳本跑出來的數字。
> 那些數字是對的，但依 §5.1 不是可採信的驗收證據，因為不是驗收閘門產出的。

### A2. 重新發布並啟用

GUI：**「建立候選組合」**（若需要新的 release）→ **「發布並啟用選取組合」**。

- **必須先停止檢測**，否則會被擋（`inspection_version_workspace.py:1177`）。
- 上線模式選 **「限定試用」**。
- 具名操作者 + 理由欄請寫真正的理由，例如：
  `stats-robust-v6 基準補正上線；驗收 250 片 OK173/NG77，顏色 NG 真值尚缺，
  故以限定試用模式運行並持續觀察。`
  （目前欄位裡是 `"1"`／`"2"`，那是佔位字元。）

### A3. 打開演算法防呆

在 `models/Cable1/A/yolo/config.yaml` 加入：

```yaml
color_baseline_algorithm_enforcement: strict
```

**這一步是這份清單裡最重要的。** 目前那個 3 月的
`models/Cable1/A/yolo/color_stats.json` 連 `recalibration` 區塊都沒有，等於
不宣告演算法。只要 release 被回退或解析失敗，runtime 就會靜靜掉回去用它
（`detection_system.py:701-706` 只在 release 存在時才 override），
**而 Black 的 0.20 門檻仍然 active** —— 一個為 v6 整框計分校出來的門檻，
套在舊幾何統計上。在 `warn` 之下沒有任何東西會攔下這個組合。

設成 `strict` 後，同樣情境會變成 `RuntimeError`「拒絕載入顏色基準」
（`color_checker.py:276-281`），寧可停線也不誤判。

改完請重啟檢測程式並確認能正常載入（此時 release 有效，載的是 v6 profile，
應該要過）。

### A4. 重錄 preflight 基準邊界

```powershell
python tools/color_preflight.py --record-reference --operator <你的識別碼>
```

會更新站別 config 的 `color_preflight:` 區塊。v6 改變了 Black 的計分尺度，
舊的參考邊界不能沿用。

> 註：這個工具只讀不寫基準（檔頭自述 "It never writes a baseline"），
> 只寫 preflight 參考值，安全。

---

## 路線 B：要走到正式上線（需要先補資料）

唯一的路是讓驗收集合裡有**真實的顏色不良板**：

1. 找出實際發生過顏色不良的板子（或刻意保留的不良樣本）。
2. 在複核流程中把它們標成 `expected_verdict = NG` 且
   `expected_reasons` 含 `COLOR_MISMATCH`。
3. 併入 `acceptance/Cable1/A/ground_truth.csv` 後重跑 A1 的驗收矩陣。
4. `color_escape_known` 變 true 且無其他警告後，「正式上線」選項才會出現。

在那之前，維持限定試用是誠實的狀態，不是妥協。

---

## 回退方式

GUI 同一頁的 **「回退至前一正式組合」**（同樣要先停檢測、要具名理由）。
前一個 release 是 `c7b57cc0-15b4-454a-8e8d-104f8f1ffd17`。

**但注意**：回退只換 release 指標，**不會**同時停用 Black 的 0.20 修訂。
若要完整回到 v6 之前的狀態，顏色修訂必須另外處理。這也是 A3 先做完比較安全的原因。

---

## 不要做的事

- 不要手動覆寫 `models/Cable1/A/yolo/color_stats.json`。
  正式切換以完整組合為單位（`MODEL_COMBINATION_ACCEPTANCE.md:15-16`），
  重建產出的是不可變候選，本來就不該直接改模型設定（`:226`）。
- 不要拿 `.tmp/color_rebuild_Cable1_A_v6/` 裡的檔案當發布來源。
  那份候選內容雖然與正式候選 `3d3383833b80ff926100e97b` 位元組相同，
  但它沒有 manifest 與候選 ID，矩陣選不到它。要用就用正式候選。
- 重建、驗收、發布、啟用**都沒有 CLI**，只能在 GUI 做，且都需要具名操作者。

---

## 資料處理提醒

驗收產物、station config 與 revision 記錄含有操作者識別碼與產品／站別資訊。
往外部系統貼上前，請先確認場合是否適當。
