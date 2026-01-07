# 改進路線圖 - YOLO11 推理系統

**版本**: v1.0  
**創建日期**: 2026-01-06  
**目標**: 從生產可用邁向企業級系統

---

## 🎯 整體目標

將專案從當前 **7.5/10** 提升至 **9.0/10**，重點改進：
- 可靠性（Reliability）
- 可維護性（Maintainability）
- 可觀測性（Observability）
- 安全性（Security）

---

## 📅 Phase 1: 緊急修復（第 1 週）

### 目標：修復關鍵風險，確保 CI 穩定

#### Task 1.1: 修復依賴管理 ⚡ P0
**問題**: `requirements.txt` 只有 `-e .`，外部環境無法安裝

**解決方案**:
```bash
# 1. 安裝 pip-tools
pip install pip-tools

# 2. 生成完整依賴
pip-compile pyproject.toml -o requirements.txt

# 3. 生成開發依賴
pip-compile pyproject.toml --extra dev --extra gui -o requirements-dev-full.txt
```

**驗證**:
```bash
# 在新環境測試
python -m venv test_env
source test_env/bin/activate  # Windows: test_env\Scripts\activate
pip install -r requirements.txt
pytest -v
```

**工時**: 2 小時  
**負責人**: DevOps/開發團隊

---

#### Task 1.2: 移除 CI continue-on-error ⚡ P0
**問題**: CI 中多處使用 `continue-on-error: true`，隱藏真實問題

**解決方案**:

```yaml
# .github/workflows/ci.yml - 修改前
- name: Lint with ruff
  run: ruff check .
  continue-on-error: true  # ❌ 移除

# 修改後
- name: Lint with ruff
  run: ruff check .
  # 如果有已知問題，使用 ignore 或 baseline

- name: Run tests (Ubuntu)
  run: pytest -v -m "not gui"
  # ❌ 移除 continue-on-error
```

**修復步驟**:
1. 執行本地測試找出失敗原因
2. 修復失敗的測試或標記為 `xfail`
3. 移除所有 `continue-on-error`

**工時**: 4 小時  
**負責人**: QA/開發團隊

---

#### Task 1.3: 加入路徑安全驗證 🔐 P0

**新增文件**: `core/security.py`

```python
"""Security utilities for safe file operations."""
from pathlib import Path
from typing import Union

class SecurityError(Exception):
    """Raised when a security check fails."""
    pass

class PathValidator:
    """Validate file paths to prevent directory traversal attacks."""
    
    def __init__(self, allowed_roots: list[Path]):
        self.allowed_roots = [Path(root).resolve() for root in allowed_roots]
    
    def validate_path(self, path: Union[str, Path], *, must_exist: bool = False) -> Path:
        """Validate that a path is safe to access.
        
        Args:
            path: Path to validate
            must_exist: If True, verify path exists
            
        Returns:
            Resolved absolute path
            
        Raises:
            SecurityError: If path is outside allowed roots
            FileNotFoundError: If must_exist=True and path doesn't exist
        """
        resolved = Path(path).resolve()
        
        # Check if path is within allowed roots
        if not any(self._is_relative_to(resolved, root) for root in self.allowed_roots):
            raise SecurityError(
                f"Access denied: {path} is outside allowed directories"
            )
        
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        
        return resolved
    
    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        """Check if path is relative to parent (Python 3.9+ has this built-in)."""
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

# 全局實例
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
path_validator = PathValidator(allowed_roots=[
    PROJECT_ROOT,
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "Result",
])
```

**修改 `core/config.py`**:
```python
from core.security import path_validator

class DetectionConfig:
    def load_config(self, config_path: str | Path):
        # 驗證路徑安全
        safe_path = path_validator.validate_path(config_path, must_exist=True)
        with open(safe_path) as f:
            ...
```

**工時**: 6 小時（包含測試）  
**負責人**: 安全/開發團隊

---

#### Task 1.4: YAML 安全載入 🔐 P0

**搜尋並替換所有 `yaml.load()`**:
```bash
# 找出所有使用 yaml.load 的地方
grep -r "yaml.load" --include="*.py" .
```

**替換為**:
```python
# ❌ 危險
config = yaml.load(f)

# ✅ 安全
config = yaml.safe_load(f)
```

**工時**: 2 小時  
**負責人**: 安全/開發團隊

---

### Phase 1 總結
- **總工時**: 14 小時（2 工作日）
- **交付成果**: CI 穩定通過、依賴可復現、基礎安全加固
- **驗收標準**: 
  - ✅ CI 全綠
  - ✅ `pip install -r requirements.txt` 在新環境可用
  - ✅ 路徑注入測試通過

---

## 📅 Phase 2: 可觀測性增強（第 2-4 週）

### 目標：增加性能監控、日誌增強、告警系統

#### Task 2.1: 整合 Prometheus 指標 📊

**安裝依賴**:
```bash
pip install prometheus-client
```

**新增文件**: `core/monitoring.py`

```python
"""Prometheus metrics for monitoring."""
from prometheus_client import Counter, Histogram, Gauge, Info
import time

# 推理延遲直方圖
INFERENCE_LATENCY = Histogram(
    'inference_latency_seconds',
    'Inference latency in seconds',
    ['product', 'area', 'inference_type', 'status'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# 推理計數器
INFERENCE_COUNT = Counter(
    'inference_total',
    'Total number of inferences',
    ['product', 'area', 'inference_type', 'status']
)

# 模型載入計數
MODEL_LOAD_COUNT = Counter(
    'model_load_total',
    'Total number of model loads',
    ['product', 'area', 'inference_type']
)

# 當前載入的模型數
LOADED_MODELS = Gauge(
    'loaded_models_count',
    'Number of currently loaded models'
)

# 檢測結果統計
DETECTION_RESULTS = Counter(
    'detection_results_total',
    'Detection results by status',
    ['product', 'area', 'result_status']  # PASS/FAIL
)

# GPU 記憶體使用（如果可用）
GPU_MEMORY_USAGE = Gauge(
    'gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    ['device']
)

class InferenceTimer:
    """Context manager for timing inference operations."""
    
    def __init__(self, product: str, area: str, inference_type: str):
        self.product = product
        self.area = area
        self.inference_type = inference_type
        self.status = "unknown"
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        # 根據是否有異常設定狀態
        if exc_type is None:
            self.status = "success"
        else:
            self.status = "error"
        
        # 記錄指標
        INFERENCE_LATENCY.labels(
            product=self.product,
            area=self.area,
            inference_type=self.inference_type,
            status=self.status
        ).observe(duration)
        
        INFERENCE_COUNT.labels(
            product=self.product,
            area=self.area,
            inference_type=self.inference_type,
            status=self.status
        ).inc()
```

**修改 `core/yolo_inference_model.py`**:
```python
from core.monitoring import InferenceTimer, INFERENCE_COUNT

class YOLOInferenceModel:
    def infer(self, image: np.ndarray, product: str, area: str, ...):
        with InferenceTimer(product, area, "yolo"):
            # 現有推理邏輯
            ...
```

**提供 Prometheus Endpoint** (如果有 API):
```python
# app/api.py (未來)
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

**工時**: 16 小時  
**負責人**: DevOps/開發團隊

---

#### Task 2.2: 結構化日誌 📝

**目標**: 從文本日誌轉為 JSON 格式，便於解析和搜尋

**安裝依賴**:
```bash
pip install python-json-logger
```

**修改 `core/logging_config.py`**:
```python
from pythonjsonlogger import jsonlogger

def configure_logging(log_level: str = "INFO"):
    """配置結構化日誌."""
    
    # JSON 格式化器
    json_handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s',
        timestamp=True
    )
    json_handler.setFormatter(formatter)
    
    # 根日誌器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(json_handler)
```

**使用範例**:
```python
logger.info(
    "Inference completed",
    extra={
        "product": product,
        "area": area,
        "inference_type": inference_type,
        "inference_time_ms": 123.4,
        "result_status": "PASS",
        "detection_count": 5
    }
)

# 輸出 JSON:
# {
#   "asctime": "2026-01-06T10:30:00.123Z",
#   "name": "core.detection_system",
#   "levelname": "INFO",
#   "message": "Inference completed",
#   "product": "LED",
#   "area": "A",
#   "inference_type": "yolo",
#   "inference_time_ms": 123.4,
#   "result_status": "PASS",
#   "detection_count": 5
# }
```

**工時**: 8 小時  
**負責人**: 開發團隊

---

#### Task 2.3: 建立效能基準測試 ⚡

**新增文件**: `tests/test_benchmarks.py`

```python
"""Performance benchmark tests."""
import pytest
import numpy as np
from core.yolo_inference_model import YOLOInferenceModel

@pytest.fixture
def sample_image():
    """Generate a sample test image."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

@pytest.fixture
def yolo_model():
    """Initialize YOLO model for benchmarking."""
    # 使用測試配置
    ...

def test_inference_latency(benchmark, yolo_model, sample_image):
    """Benchmark YOLO inference latency."""
    result = benchmark(yolo_model.infer, sample_image, "test", "test")
    
    # SLA: 推理應在 200ms 內完成（CPU）
    stats = benchmark.stats.stats
    assert stats.mean < 0.2, f"Mean latency {stats.mean:.3f}s exceeds 200ms"

def test_model_load_time(benchmark):
    """Benchmark model loading time."""
    def load_model():
        model = YOLOInferenceModel(config)
        model.initialize("test", "test")
        return model
    
    result = benchmark(load_model)
    
    # SLA: 模型載入應在 5s 內完成
    stats = benchmark.stats.stats
    assert stats.mean < 5.0, f"Model load time {stats.mean:.3f}s exceeds 5s"

@pytest.mark.slow
def test_throughput(yolo_model, sample_image):
    """Test inference throughput."""
    import time
    
    start = time.time()
    count = 100
    
    for _ in range(count):
        yolo_model.infer(sample_image, "test", "test")
    
    duration = time.time() - start
    fps = count / duration
    
    # SLA: 應達到 10 FPS
    assert fps >= 10, f"Throughput {fps:.1f} FPS below target 10 FPS"
```

**執行基準測試**:
```bash
# 執行並生成報告
pytest tests/test_benchmarks.py --benchmark-only --benchmark-json=benchmark_results.json

# 與歷史比較（檢測回歸）
pytest tests/test_benchmarks.py --benchmark-compare=0001 --benchmark-compare-fail=mean:10%
```

**工時**: 12 小時  
**負責人**: QA/開發團隊

---

### Phase 2 總結
- **總工時**: 36 小時（4.5 工作日）
- **交付成果**: 完整的監控體系、結構化日誌、效能基準
- **驗收標準**:
  - ✅ Prometheus 指標可導出
  - ✅ JSON 日誌可被 ELK 解析
  - ✅ 效能基準測試通過並建立 baseline

---

## 📅 Phase 3: 配置系統重構（第 5-8 週）

### 目標：使用 Pydantic V2 重構配置，提升類型安全

#### Task 3.1: 遷移到 Pydantic V2

**安裝 Pydantic V2**:
```bash
pip install "pydantic>=2.0"
```

**新增文件**: `core/config_v2.py`

```python
"""Refactored configuration using Pydantic V2."""
from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

class CameraConfig(BaseModel):
    """Camera configuration."""
    exposure_time: float = Field(gt=0, description="Exposure time in microseconds")
    gain: float = Field(ge=0, le=24, description="Camera gain")
    timeout_ms: int = Field(default=10000, gt=0)
    width: int = Field(default=3072, gt=0)
    height: int = Field(default=2048, gt=0)

class ModelConfig(BaseModel):
    """Model-specific configuration."""
    imgsz: int | list[int] = Field(default=640)
    conf_thres: float = Field(default=0.25, ge=0, le=1)
    iou_thres: float = Field(default=0.45, ge=0, le=1)
    device: str = Field(default="auto")
    
    @field_validator('imgsz')
    @classmethod
    def validate_imgsz(cls, v):
        """Ensure imgsz is valid."""
        if isinstance(v, list):
            if not all(x > 0 and x % 32 == 0 for x in v):
                raise ValueError("imgsz must be multiples of 32")
        elif v <= 0 or v % 32 != 0:
            raise ValueError("imgsz must be multiple of 32")
        return v

class PositionCheckConfig(BaseModel):
    """Position validation configuration."""
    enabled: bool = Field(default=False)
    tolerance: float = Field(default=5.0, ge=0)
    tolerance_unit: Literal["pixel", "percent"] = "percent"
    mode: Literal["bbox", "region", "bbox_region"] = "bbox"
    expected_boxes: dict[str, dict[str, float]] = Field(default_factory=dict)

class DetectionConfigV2(BaseSettings):
    """Main detection configuration with validation."""
    
    # Model paths
    weights: Path = Field(description="Path to YOLO weights")
    
    # Feature flags
    enable_yolo: bool = True
    enable_anomalib: bool = False
    
    # Cache settings
    max_cache_size: int = Field(default=3, ge=1, le=10)
    
    # Output
    output_dir: Path = Field(default=Path("./Result"))
    
    # Camera
    camera: CameraConfig = Field(default_factory=CameraConfig)
    
    # Model defaults
    model: ModelConfig = Field(default_factory=ModelConfig)
    
    # Position check
    position_check: PositionCheckConfig = Field(default_factory=PositionCheckConfig)
    
    @model_validator(mode='after')
    def validate_paths(self):
        """Validate and resolve paths."""
        # 解析相對路徑
        if not self.weights.is_absolute():
            self.weights = (Path.cwd() / self.weights).resolve()
        if not self.output_dir.is_absolute():
            self.output_dir = (Path.cwd() / self.output_dir).resolve()
        
        return self
    
    @field_validator('weights')
    @classmethod
    def weights_must_exist(cls, v: Path) -> Path:
        """Validate weights file exists."""
        if not v.exists():
            raise ValueError(f"Weights file not found: {v}")
        return v
    
    class Config:
        env_prefix = "YOLO_"
        env_file = ".env"
        env_file_encoding = "utf-8"

# 載入配置
def load_config(config_path: str | Path) -> DetectionConfigV2:
    """Load and validate configuration."""
    import yaml
    from core.security import path_validator
    
    safe_path = path_validator.validate_path(config_path, must_exist=True)
    
    with open(safe_path) as f:
        data = yaml.safe_load(f)
    
    # Pydantic V2 自動驗證
    return DetectionConfigV2(**data)
```

**工時**: 24 小時  
**負責人**: 開發團隊

---

#### Task 3.2: 配置合併追蹤

**新增**: `core/config_merger.py`

```python
"""Configuration merging with audit trail."""
from typing import Any
import logging

logger = logging.getLogger(__name__)

class ConfigMerger:
    """Merge configurations with tracking."""
    
    def __init__(self):
        self.merge_history: list[dict] = []
    
    def merge(
        self, 
        base: dict[str, Any], 
        override: dict[str, Any], 
        source: str
    ) -> dict[str, Any]:
        """Merge override into base, track changes."""
        result = base.copy()
        changes = []
        
        for key, value in override.items():
            if key in result and result[key] != value:
                changes.append({
                    "key": key,
                    "old": result[key],
                    "new": value,
                    "source": source
                })
            result[key] = value
        
        if changes:
            self.merge_history.append({
                "source": source,
                "changes": changes
            })
            logger.info(
                f"Applied {len(changes)} config overrides from {source}",
                extra={"changes": changes}
            )
        
        return result
    
    def get_audit_trail(self) -> list[dict]:
        """Get configuration merge history."""
        return self.merge_history
```

**工時**: 8 小時

---

### Phase 3 總結
- **總工時**: 32 小時（4 工作日）
- **交付成果**: 類型安全的配置系統、配置合併追蹤
- **驗收標準**:
  - ✅ 所有配置欄位有型別驗證
  - ✅ 配置錯誤在載入時即被捕獲
  - ✅ 配置合併過程可追蹤

---

## 📅 Phase 4: 測試增強（第 9-10 週）

### 目標：提升測試覆蓋率到 85%+，加入突變測試

#### Task 4.1: 提升覆蓋率

**目前狀態**: 未知（需先測量）

**測量基線**:
```bash
pytest --cov=core --cov=app --cov-report=html --cov-report=term-missing
```

**目標**: 85% 覆蓋率

**策略**:
1. 識別未覆蓋的關鍵路徑
2. 為錯誤處理添加測試
3. 為邊界條件添加測試

**範例**:
```python
# tests/test_position_validator_edge_cases.py
def test_position_validator_with_negative_coordinates():
    """Test validator handles negative coordinates."""
    validator = PositionValidator(...)
    detections = [{"cx": -10, "cy": 20, "class": "test"}]
    
    result = validator.validate(detections)
    assert result[0]["position_status"] == "INVALID"

def test_position_validator_with_out_of_bounds():
    """Test validator handles out-of-bounds coordinates."""
    ...
```

**工時**: 16 小時  
**負責人**: QA/開發團隊

---

#### Task 4.2: 突變測試（Mutation Testing）

**安裝**:
```bash
pip install mutmut
```

**執行突變測試**:
```bash
# 對 core/position_validator.py 進行突變測試
mutmut run --paths-to-mutate=core/position_validator.py

# 查看結果
mutmut results

# 查看倖存的突變（測試應該捕獲但沒有）
mutmut show
```

**目標**: 突變分數 > 80%

**工時**: 8 小時  
**負責人**: QA團隊

---

#### Task 4.3: 契約測試（Contract Testing）

**針對模型輸出格式**:

```python
# tests/test_contracts.py
from pydantic import BaseModel, Field

class YOLODetection(BaseModel):
    """Contract for YOLO detection output."""
    class_name: str = Field(alias="class")
    confidence: float = Field(ge=0, le=1)
    bbox: list[float] = Field(min_length=4, max_length=4)
    cx: float
    cy: float

class YOLOInferenceResult(BaseModel):
    """Contract for YOLO inference result."""
    status: str
    detections: list[YOLODetection]
    inference_time_ms: float = Field(gt=0)
    missing_items: list[str]

def test_yolo_inference_output_contract():
    """Ensure YOLO output matches contract."""
    result = run_inference(test_image)
    
    # Pydantic 自動驗證契約
    validated = YOLOInferenceResult(**result)
    
    assert validated.status in {"PASS", "FAIL", "ERROR"}
```

**工時**: 12 小時

---

### Phase 4 總結
- **總工時**: 36 小時（4.5 工作日）
- **交付成果**: 85%+ 覆蓋率、突變測試、契約測試
- **驗收標準**:
  - ✅ 測試覆蓋率 >= 85%
  - ✅ 突變分數 > 80%
  - ✅ 所有關鍵契約有測試保護

---

## 📅 Phase 5: API 開發（第 11-16 週）

### 目標：提供 REST API，支援非同步推理

#### Task 5.1: FastAPI 基礎架構

**安裝依賴**:
```bash
pip install fastapi uvicorn[standard] python-multipart
```

**新增文件**: `api/main.py`

```python
"""FastAPI application for YOLO inference."""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import cv2
from core.detection_system import DetectionSystem

app = FastAPI(
    title="YOLO11 Inference API",
    description="Industrial vision inspection system",
    version="1.0.0"
)

# 初始化檢測系統（啟動時）
detection_system = DetectionSystem()

class InferenceRequest(BaseModel):
    """Inference request model."""
    product: str
    area: str
    inference_type: str = "yolo"

class InferenceResponse(BaseModel):
    """Inference response model."""
    status: str
    detections: list[dict]
    missing_items: list[str]
    inference_time_ms: float

@app.post("/api/v1/infer", response_model=InferenceResponse)
async def infer(
    image: UploadFile = File(...),
    product: str = Form(...),
    area: str = Form(...),
    inference_type: str = Form("yolo")
):
    """Run inference on uploaded image."""
    # 讀取影像
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(400, "Invalid image format")
    
    # 執行推理
    result = detection_system.detect(product, area, inference_type, frame=frame)
    
    return InferenceResponse(**result)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown."""
    detection_system.shutdown()
```

**啟動服務**:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**工時**: 24 小時  
**負責人**: 後端開發團隊

---

#### Task 5.2: 非同步推理隊列

**問題**: FastAPI 是 async，但 YOLO 推理是 CPU 密集型同步操作

**解決方案**: 使用 ThreadPoolExecutor

```python
from fastapi import BackgroundTasks
from concurrent.futures import ThreadPoolExecutor
import asyncio

# 全局執行器
executor = ThreadPoolExecutor(max_workers=4)

async def run_inference_async(
    frame: np.ndarray,
    product: str,
    area: str,
    inference_type: str
) -> dict:
    """Run inference in thread pool."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        detection_system.detect,
        product,
        area,
        inference_type,
        frame
    )
    return result

@app.post("/api/v1/infer/async")
async def infer_async(image: UploadFile, ...):
    """Async inference endpoint."""
    frame = await load_image(image)
    result = await run_inference_async(frame, product, area, inference_type)
    return result
```

**工時**: 16 小時

---

#### Task 5.3: OpenAPI 文檔和客戶端

**自動生成 OpenAPI spec**:
```bash
# FastAPI 自動提供
curl http://localhost:8000/openapi.json > openapi.json
```

**生成 Python 客戶端**:
```bash
pip install openapi-python-client
openapi-python-client generate --url http://localhost:8000/openapi.json
```

**生成 TypeScript 客戶端**:
```bash
npm install -g @openapitools/openapi-generator-cli
openapi-generator-cli generate -i openapi.json -g typescript-axios -o ./client-ts
```

**工時**: 8 小時

---

### Phase 5 總結
- **總工時**: 48 小時（6 工作日）
- **交付成果**: 完整的 REST API、非同步推理、客戶端
- **驗收標準**:
  - ✅ API 可處理並發請求
  - ✅ OpenAPI 文檔完整
  - ✅ 客戶端可正常調用

---

## 📅 Phase 6: 容器化與部署（第 17-18 週）

### Task 6.1: Docker 化

**新增檔案**: `Dockerfile`

```dockerfile
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用代碼
COPY . .

# 安裝應用
RUN pip install --no-cache-dir -e .

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  yolo-inference:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
      - ./Result:/app/Result
    environment:
      - YOLO_LOG_LEVEL=INFO
      - YOLO_MAX_CACHE_SIZE=3
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

**工時**: 16 小時  
**負責人**: DevOps團隊

---

## 📊 總體時間規劃

| Phase | 工作週 | 工時 | 關鍵交付 |
|-------|--------|------|----------|
| Phase 1: 緊急修復 | 第 1 週 | 14h | CI 穩定、依賴修復、安全加固 |
| Phase 2: 可觀測性 | 第 2-4 週 | 36h | 監控、日誌、基準測試 |
| Phase 3: 配置重構 | 第 5-8 週 | 32h | Pydantic V2、配置追蹤 |
| Phase 4: 測試增強 | 第 9-10 週 | 36h | 85% 覆蓋率、突變測試 |
| Phase 5: API 開發 | 第 11-16 週 | 48h | REST API、非同步推理 |
| Phase 6: 容器化 | 第 17-18 週 | 16h | Docker、K8s ready |

**總計**: ~182 小時（約 23 工作日，即 4.5 個月以每週 10 小時計算）

---

## ✅ 驗收標準總覽

### 技術指標
- ✅ CI/CD: 所有檢查全綠，無 `continue-on-error`
- ✅ 測試覆蓋率: >= 85%
- ✅ 突變分數: > 80%
- ✅ API 響應時間: P95 < 200ms（CPU）/ P95 < 50ms（GPU）
- ✅ 容器啟動時間: < 30 秒

### 安全指標
- ✅ 無高危漏洞（`safety check` 全過）
- ✅ 所有路徑驗證通過
- ✅ 所有 YAML 使用 `safe_load`

### 可靠性指標
- ✅ 錯誤率 < 0.1%（生產環境）
- ✅ 推理 SLA 達成率 > 99%
- ✅ 模型載入失敗自動重試成功

---

## 🎓 團隊培訓需求

### 建議培訓主題
1. **Pydantic V2 深度應用** (4h)
2. **FastAPI 非同步最佳實踐** (8h)
3. **Prometheus + Grafana 監控** (4h)
4. **Docker 與 Kubernetes 基礎** (8h)
5. **契約測試與 API 設計** (4h)

---

## 📈 成功指標

完成此路線圖後，專案應達到：

| 維度 | 目前 | 目標 |
|------|------|------|
| 整體評分 | 7.5/10 | 9.0/10 |
| 測試覆蓋率 | ~60% | 85%+ |
| CI 穩定性 | 有 warnings | 全綠 |
| 部署複雜度 | 手動 | 容器化自動 |
| 可觀測性 | 基礎日誌 | 完整監控 |
| API 可用性 | 無 | REST + async |

---

**文檔版本**: 1.0  
**最後更新**: 2026-01-06  
**下次審查**: 2026-04-06
