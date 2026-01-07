# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Path Security Validation Module** (`core/security.py`)
  - `PathValidator` class for preventing directory traversal attacks
  - Global `path_validator` instance for project-wide use
  - Support for multiple allowed root directories
  - Symlink resolution and validation
  - Comprehensive test suite (12/13 tests passing, 1 skipped on Windows)

- **Security Tests** (`tests/test_security.py`)
  - Directory traversal attack prevention tests
  - Path validation boundary tests
  - Global validator integration tests
  - Multiple allowed roots scenario tests

### Changed
- **`core/config.py`**: Integrated path security validation
  - Added path validation in `DetectionConfig.from_yaml()`
  - Prevents loading configs from外部 untrusted paths
  - Graceful fallback if security module unavailable

- **`main.py`**: Enhanced CLI security
  - Added security validation for `--image` parameter
  - Prevents image loading from untrusted paths
  - Improved error messages for path validation failures

- **`requirements.txt`**: Updated to complete dependency list
  - Expanded from simple `-e .` to 342 lines of pinned dependencies
  - Generated using `pip-compile` from `pyproject.toml`
  - Ensures reproducible installations across environments

### Security
- **防止目錄遍歷攻擊** (Directory Traversal Protection)
  - Blocks paths containing `..` sequences
  - Validates paths are within allowed root directories
  - Protects configuration files, model weights, and input images

- **YAML 安全載入** (Safe YAML Loading)
  - All YAML loading uses `yaml.safe_load()`
  - Verified in: `core/config.py`, `core/services/model_manager.py`, `core/detection_system.py`
  - Prevents arbitrary code execution via malicious YAML files

- **白名單式路徑控制** (Whitelist-based Path Access)
  - Only allows access to predefined project directories
  - Default allowed roots: project root, models directory, Result directory
  - Configurable for different deployment scenarios

## [0.1.0] - 2026-01-06

### 初始版本功能 (Initial Release Features)

#### Core Functionality
- **YOLO11 物件偵測** (Object Detection)
  - Integration with Ultralytics YOLO11
  - Support for custom trained models
  - Confidence and IoU threshold configuration
  - Model caching with LRU eviction (3 models default)
  - GPU warmup for reduced first-frame latency

- **Anomalib 異常檢測** (Anomaly Detection)
  - PatchCore, PaDiM, STFPM, DRAEM support
  - Pixel-level and image-level anomaly detection
  - Heatmap generation and visualization
  - Configurable anomaly thresholds

- **位置驗證** (Position Validation)
  - Expected position checking for detected objects
  - Absolute (pixel) and relative (percentage) tolerance
  - Support for multiple products and areas
  - Auto-generation of position configs from training data

- **顏色檢測** (Color Detection)
  - Statistical color checking for LED components
  - HSV-based color classification
  - Support for multiple color targets per product
  - Color sequence validation for cable products

#### Interfaces
- **命令列介面** (CLI)
  - Interactive mode for product/area selection
'  - Single-shot inference mode with arguments
  - Batch processing support
  - Configurable output formats (Excel, JSON, images)

- **圖形介面** (PyQt5 GUI)
  - Live camera preview
  - Model hot-swapping
  - Real-time inference visualization
  - Results logging and export
  - Multi-threading for responsive UI

#### Industrial Camera Support
- **海康威視 MVS SDK 整合**
  - Automatic device enumeration
  - Exposure and gain control
  - Image acquisition with timeout
  - ROI (Region of Interest) support

#### Pipeline Architecture
- **Modular Step System**
  - Registry-based step loading
  - Configurable pipeline per model
  - Built-in steps: color_check, count_check, sequence_check, position_validation
  - Easy custom step development

#### Quality & Development
- **測試套件** (Test Suite)
  - 52 comprehensive tests
  - Unit, integration, and E2E test coverage
  - Mock camera support for CI environments
  - Performance and robustness tests

- **程式碼品質工具** (Code Quality Tools)
  - Ruff for linting
  - MyPy for type checking  
  - Pytest with coverage reporting
  - Pre-commit hooks configuration

#### Documentation
- **README.md**: 371-line comprehensive project documentation
- **docs/TECH_GUIDE.md**: 1153-line deep-dive technical guide (JR→SR level)
- **docs/MODULE_ARCHITECTURE.md**: Architecture diagrams and design patterns
- **config.example.yaml**: Full configuration template with comments

#### Configuration & Flexibility
- **多產品/多站別支援** (Multi-product/Multi-area Support)
  - Hierarchical model organization: `models/{product}/{area}/{type}/`
  - Per-model configuration files
  - Dynamic model loading based on product/area selection

- **靈活的配置系統** (Flexible Configuration)
  - YAML-based global and model-specific configs
  - Environment variable support via `python-dotenv`
  - Pydantic schema validation
  - Hot-reload capabilities

#### Results Management
- **Excel 報表輸出** (Excel Reports)
  - Detailed detection results with timestamps
  - Pass/Fail status per item
  - Color check results
  - Position validation results
  - Anomaly scores and heatmaps

- **影像標註與保存** (Image Annotation)
  - Bounding box visualization
  - Confidence score labels
  - Color-coded pass/fail indicators
  - Original and processed image pairs

---

## Migration Guide

### From Pre-0.1.0 Development Versions

If you're upgrading from an early development version:

1. **Update Dependencies**:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **Review Configuration**:
   - Compare your `config.yaml` with `config.example.yaml`
   - Add any new required fields (especially security-related)

3. **Test Path Validation**:
   - Ensure your model paths, output directories are within project root
   - Or configure custom allowed roots if needed

4. **Run Tests**:
   ```bash
   pytest -v
   ```

---

## Roadmap

### Planned for v0.2.0
- [ ] Remove CI `continue-on-error` flags (after fixing all linting/type errors)
- [ ] TensorRT INT8 quantization support
- [ ] Docker deployment guide and Dockerfile
- [ ] REST API service mode
- [ ] Calibration wizard for new cameras
- [ ] Performance profiling dashboard

### Under Consideration
- [ ] Support for additional anomaly models (FastFlow, Reverse Distillation)
- [ ] Multi-camera orchestration
- [ ] Cloud model repository integration
- [ ] Automated retraining pipeline
- [ ] Web-based configuration interface

---

## Contributors

- **DioWang** - Initial development and architecture
- **AI Assistant** - Documentation, testing, and security enhancements

---

## License

Proprietary License - Unauthorized distribution or use is prohibited.

---

## Acknowledgments

This project uses the following open-source packages:
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Anomalib](https://github.com/openvinotoolkit/anomalib)
- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://lightning.ai/)
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/)
