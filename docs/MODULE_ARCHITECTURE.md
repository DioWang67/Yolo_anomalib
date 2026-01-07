# 軟體模組架構說明書 (Module Architecture)

本文件說明 `yolo11_inference` 專案的軟體架構設計，重點介紹各模組的職責以及它們之間的互動關係。

## 1. 系統高階架構 (High-Level Architecture)

系統採用 **分層架構 (Layered Architecture)**，主要分為兩層：
1.  **Core Layer (`core/`)**: 負責核心商業邏輯、AI 推論、硬體控制與結果處理。此層不依賴任何 GUI 函式庫。
2.  **App/GUI Layer (`app/gui/`)**: 負責使用者介面、非同步任務排程與狀態管理。

```mermaid
graph TD
    User[使用者] --> GUI[GUI Layer (Qt)]
    GUI --> Controller[DetectionController]
    Controller --> Core[Core Layer (DetectionSystem)]
    
    subgraph "App / GUI Layer"
        GUI
        Controller
        Workers[Async Workers]
    end
    
    subgraph "Core Layer"
        Core
        YOLO[YOLO Detector]
        Anom[Anomalib Detector]
        Cam[Camera Control]
        Res[Result Handler]
    end

    Controller --> Workers
    Workers -.->|Signal| GUI
    Core --> YOLO
    Core --> Anom
    Core --> Cam
    Core --> Res
```

---

## 2. Core Layer (核心層)

位於 `core/` 目錄，是系統的引擎。

### 2.1 `DetectionSystem` (Facade)
*   **檔案**: `core/detection_system.py`
*   **用途**: 這是核心層的單一入口點 (Facade Pattern)。
*   **職責**:
    *   載入設定檔 (`config.yaml`)。
    *   初始化硬體 (相機) 與 AI 模型。
    *   提供統一的 `detect()` 方法供上層呼叫。
    *   管理資源釋放 (`shutdown`)。

### 2.2 偵測策略 (`detectors/`)
*   **用途**: 定義具體的檢測演算法。
*   **`InferenceEngine`**: 根據設定動態載入不同的偵測器 (YOLO 或 Anomalib)。
*   **`YOLODetector`**: 封裝 Ultralytics YOLOv11 推論邏輯。
*   **`AnomalibDetector`**: 封裝 Anomalib (PatchCore/PaDiM) 推論邏輯。

### 2.3 服務元件 (`services/`)
*   **`CameraService`**: 相機控制抽象層，支援 OpenCV 模擬或 MVS 工業相機。
*   **`ResultHandler`**: 負責將推論結果 (JSON/Excel) 與影像寫入磁碟。
*   **`PositionValidator`**: 負責多物件的幾何位置校驗邏輯。

### 2.4 安全組件 (`core/security.py`)
*   **`PathValidator`**: 負責防止路徑遍歷攻擊，驗證所有檔案 I/O 操作的路徑安全性。
*   **Check Points**: 整合於 `Config` 載入、`DetectionSystem` 初始化及影像讀取流程中。

---

## 3. GUI Layer (介面層)

位於 `app/gui/` 目錄，採用 **MVC (Model-View-Controller)** 的變體設計。

### 3.1 `DetectionSystemGUI` (View/Main Window)
*   **檔案**: `app/gui/main_window.py`
*   **用途**: 主視窗，負責組裝各個 UI 面板並協調它們。
*   **職責**:
    *   **Layout Management**: 使用 `QSplitter` 將介面分為控制、影像、資訊三區。
    *   **Event Handling**: 接收按鈕點擊、選單操作。
    *   **Signal Routing**: 將底層信號 (如偵測完成) 轉發給對應的顯示元件。

### 3.2 UI 面板 (Panels)
為了降低耦合度，UI 被拆分為三個獨立模組 (位於 `app/gui/panels/`)：

1.  **`ControlPanel`**: 
    *   **用途**: 左側控制區。
    *   **內容**: 產品/區域下拉選單、開始/停止按鈕、相機連接控制。
    *   **輸出**: 發出 `start_requested`, `stop_requested` 等信號。
    
2.  **`ImagePanel`**:
    *   **用途**: 中間影像顯示區。
    *   **內容**: Tab 頁籤 (原始影像/處理後/結果圖)，內含 `ImageViewer`。
    
3.  **`InfoPanel`**:
    *   **用途**: 右側資訊區。
    *   **內容**: `BigStatusLabel` (PASS/FAIL 大燈號)、系統狀態顯示整合、`ResultDisplayWidget` (詳細數據)、日誌視窗。

### 3.3 `DetectionController` (Controller)
*   **檔案**: `app/gui/controller.py`
*   **用途**: 連接 GUI 與 Core 的橋樑。
*   **職責**:
    *   持有 `DetectionSystem` 實例 (單例模式)。
    *   管理 `ModelCatalog` (掃描模型目錄結構)。
    *   **Worker Factory**: 負責建立非同步的工作執行緒 (Worker)。

### 3.4 非同步工作者 (Workers)
*   **檔案**: `app/gui/workers.py`
*   **用途**: 將耗時操作移出 UI 執行緒 (避免介面凍結)。
*   **類別**:
    *   `ModelLoaderWorker`: 背景掃描模型目錄。
    *   `CameraInitWorker`: 背景連接相機與初始化系統。
    *   `DetectionWorker`: 執行單次或連續偵測任務。

---

## 4. 模組互動流程圖 (Signal/Slot Flow)

### 4.1 系統初始化 (System Init)
1.  **GUI** 啟動，呼叫 `init_system()`。
2.  **Controller** 建立 `CameraInitWorker`。
3.  **Worker** 在背景執行 `DetectionSystem` 初始化 (載入權重、連接相機)。
4.  **Worker** 發出 `finished` 信號。
5.  **GUI** 接收信號，更新狀態燈號為 "READY"。

### 4.2 執行偵測 (Detection Loop)
1.  **User** 點擊 `ControlPanel` 的「開始」按鈕。
2.  **GUI** 鎖定按鈕，並請求 **Controller** 建立 `DetectionWorker`。
3.  **Worker** 啟動線程，進入迴圈：
    *   從相機擷取影像。
    *   呼叫 `DetectionSystem.detect()`。
    *   取得結果 `dict` 並轉換為 `DetectionResult` 物件。
    *   發出 `result_ready(DetectionResult)` 信號。
4.  **GUI** 的 `on_detection_complete` 槽函數被觸發：
    *   呼叫 `ImagePanel` 顯示結果圖。
    *   呼叫 `InfoPanel` 更新 PASS/FAIL 燈號與詳細數據。
5.  若為單次模式，Worker 結束；若為連續模式，Worker 繼續下一張。

---

## 5. 設計決策 (Design Decisions)

1.  **非同步優先 (Async-First)**: 所有涉及 I/O (磁碟讀取、硬體通訊、AI 推論) 的操作一律封裝在 `QThread` Worker 中，確保 GUI 永遠流暢。
2.  **信號驅動 (Signal-Driven)**: Panel 之間互不知曉，全透過 `DetectionSystemGUI` 轉發信號，降低模組間的耦合 (Decoupling)。
3.  **依賴注入 (Dependency Injection)**: `DetectionController` 被注入到 GUI 中，使 GUI 不需要知道 `Core` 如何實作，方便未來替換或測試 mock。
