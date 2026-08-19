# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Acceptance color discovery now reports the in-scope artifacts it withholds
  instead of dropping them silently. `discover_color_variants()` returns a
  `ColorVariantDiscovery` carrying both the selectable variants and a
  `ColorVariantExclusion` for each in-scope baseline built by a superseded
  algorithm, and the matrix dialog lists those as unselectable rows stating
  why. Out-of-scope artifacts are deliberately not reported, and a revoked
  revision stays unlisted because the operator already knows it was revoked.
- The matrix dialog states whether the selected scope has any stored color
  model at all, so a station that never had a color baseline is no longer
  indistinguishable from a broken tool.
- The acceptance window's main page now has a 顏色模型 selector, so a single
  inference run can be pinned to one stored color model instead of always using
  whatever the station has active. The default entry keeps the previous
  behavior, unusable entries are listed but cannot be selected, and the chosen
  model travels with its version-matched model config because the override
  cannot be staged without one. Nothing here activates or edits a color
  version.

### Fixed
- Release activation, rollback history, and retention cleanup now use
  recoverable commit points so database, audit, and filesystem failures cannot
  leave an active blocked release or silently orphan inspection evidence.
- Pilot, preflight, and duplicate-audit evidence now fails closed on wrong
  scope, incomplete inputs, stale configuration, unsafe output paths, and
  runtime policy mismatches.
- Pytest now isolates workspace discovery from live station data and rejects
  environment changes that redirect tests into production paths.
- The acceptance window kept one full-resolution annotated preview per inferred
  sample and never released any of them, so running a batch of a few hundred
  station images accumulated several GB of QPixmap and the process died with no
  traceback. Previews are now stored at display size and capped, evicting the
  least recently viewed. A record whose preview was released says so instead of
  reporting itself as never inferred.
- An acceptance run using a selected color model recorded the deployed color
  model's hash instead of the one it actually used, because the run supplied no
  model identity and the service fell back to the identity on disk. The
  override itself was applied all along; only the recorded evidence disagreed,
  which made a working override look ignored.
- The matrix result table and the version workspace's validation table printed
  a whole-verdict count (誤殺／漏檢) next to a color-only rate (顏色誤殺率／
  顏色逃逸率), so no denominator a reader could guess turned the one into the
  other and identical, reproducible runs read as non-deterministic. Every
  metric now renders as `張數（比率）` in a single cell, the headers name which
  family they belong to and carry the exact denominator as a tooltip, and a
  count whose denominator is empty reports UNKNOWN with the reason instead of
  a zero that reads as a clean sheet.
- The matrix table showed `0` in 相較首組變動 both for the reference row itself
  and for a row that matched the reference. The reference row now says 基準組.
- Committing an acceptance run discarded every result outside that run, so
  推論目前圖片 and 推論未完成圖片 silently wiped the machine verdicts of every
  other sample while reporting a successful atomic commit. The two partial
  buttons could therefore never converge on a complete manifest: each one
  re-created the pending set it had just cleared, leaving 全部重新推論 as the
  only route to a snapshot. A committed batch now clears only results whose
  `artifact_bundle_sha256` differs from the incoming one, because the property a
  formal snapshot needs is that every result came from one identical artifact
  combination — not from one invocation. A formal snapshot accordingly accepts
  several completed runs of one bundle, and still rejects a mixed bundle, an
  untracked result, or a run that never reached COMPLETED. Starting a partial
  run that would discard another combination's results now asks first, so the
  operator decides instead of discovering it from a table that emptied itself.
- `calculate_acceptance_metrics()` raised on a confirmed sample carrying no
  OK/NG verdict, and the acceptance window called it unguarded on every render.
  One hand-edited or restored-from-old-backup row therefore threw out of a Qt
  slot each time the scope was opened, taking the summary bar and snapshot
  comparison with it. Such rows are now counted as `malformed`, excluded from
  `confirmed` and from every rate so no denominator is inflated, and shown in
  the summary bar as 真值異常. Rejection stays at the two decision points that
  can act on it: the acceptance gate and a formal snapshot.
- The fusion→yolo color-scope rule existed as five separate copies across the
  acceptance window, its inference worker, the gate, the matrix, and the matrix
  dialog, and they had drifted: the window's copy did not lowercase the type it
  returned, so a scope lookup could search a differently named scope than the
  same lookup made from the matrix dialog and report a station as having no
  stored color model. All five now call one `color_scope_model_type()`. The
  model-directory lookups in `build_model_variant()` and `load_model_identity()`
  deliberately keep their own copy: they resolve filesystem paths rather than a
  color scope, and lowercasing a directory name is not theirs to do.
- Three symlink guards could never fire, because they tested a path that had
  already been through `resolve()`, which follows the link. They read as
  protection while protecting nothing. The lock helper now checks before
  resolving, where a link is still visible, and the redundant post-resolve test
  in `artifact_ref()` is gone.
- The acceptance gate and the release builder compared bundle values that had
  been stripped and lowercased when the bundle was built against raw caller and
  report values, so a target differing only in whitespace or letter case was
  reported as a mismatched bundle. Both sides are now normalized identically.
- `export_backup_zip()` archived the manifest lock directory. Each file there
  carries one zero-information byte, and on Windows a lock held by another
  process makes it unreadable, which aborted an otherwise valid backup.
  `locks/` is now skipped alongside `backups/`.
- A matrix combination in which every sample failed was publishable. Only the
  combination-level `error` field was checked, but the inference service turns
  each per-image failure into an ERROR outcome and carries on, so the realistic
  failure shape is `error: ""` alongside `errors: 250` and four zeroed confusion
  counts. A release could therefore bind itself to evidence in which nothing was
  ever decided. Publishing now also refuses a combination carrying any per-sample
  error or no decided samples at all.
- Acceptance color discovery offered stored color models without checking they
  can be loaded, so an unusable one was selectable and only failed after every
  image of its combination had been inferred — two of them wasted 500 inferences
  in one run and produced two all-ERROR combinations. A model the stats checker
  cannot load is now reported as a `ColorVariantExclusion` naming the missing
  statistic. Validation runs the real loader rather than re-listing the keys it
  needs, because a separate schema check is free to drift from the loader and
  then a model passes validation and still fails at inference. Candidate
  *status* is deliberately not an exclusion criterion: an INCOMPLETE
  recalibration can still hold usable statistics for the colors it did finish.
- The test suite could publish into live station data. Workspace discovery walks
  *upwards* for `workspace.yaml`, so a pytest `--basetemp` inside the workspace
  resolves a test's own `tmp_path` to the real station data, and code that
  rediscovers paths from a caller-supplied root — as the release builder does
  from `models_root.parent` — writes production evidence. A stub color model
  reached the real profile store this way and was offered as an acceptance
  variant. Two guards now cover this. The suite refuses to start when
  `--basetemp` resolves inside a workspace, which is the whole mechanism and is
  an easy mistake here because this workspace keeps its scratch directories in
  `.tmp/`, inside the workspace. As a backstop for any other route, an autouse
  fixture fails the individual test that creates an entry in a live
  station-data directory, instead of leaving it to be traced weeks later from a
  manifest's recorded source path.
- Every inference outcome cleared and refilled the whole acceptance sample list,
  so the cost of watching a run grew with the square of its length: a few
  hundred station images discarded tens of thousands of rows to show the same
  list back. An outcome can only change its own row and never the order,
  because the visible order follows the manifest, so one row is now repainted in
  place. The full rebuild is still used for the one case a repaint cannot
  express — a new verdict that moves the record in or out of the active filter,
  which shifts every row after it.

- Independent model acceptance workspace with reusable human truth, immutable
  snapshots, backup export, FP/FN metrics and YOLO × color combination tests.
- Versioned inspection-component catalog and atomic inspection releases for
  YOLO, Anomalib, fusion and full Stats Color profiles, including guarded
  activation and complete-combination rollback.
- Five-color Stats Color baseline rebuilding from confirmed OK evidence with
  train/holdout separation, HSV/Lab drift review and immutable candidates.
- Conservative cross-class duplicate-box handling after color verification,
  with report-only/suppress modes, position-check fail-closed protection,
  raw/effective result traceability, GUI evidence overlays and a read-only
  historical replay audit tool.
- Role-based production documentation:
  - `docs/manuals/OPERATOR_MANUAL.md` for daily inspection, history, Excel, review,
    retraining and escalation;
  - `docs/manuals/ENGINEERING_MANUAL.md` for configuration, position gates, deployment,
    SQLite recovery, server synchronization and release acceptance.
- PIN-protected full-width engineering settings and in-window retraining
  workspace.
- Inspection-history page with database-backed filters, evidence preview,
  synchronization status and cancellable Excel export.
- Versioned SQLite inspection database, verified backup/restore tooling,
  retention dry-run and production preflight.
- Local-first company-server synchronization outbox with idempotency,
  revisions, retry leases and dead-letter administration.
- Explicit per-job position-retraining and post-gate activation controls.

- Documentation for the local `yolo_anomalib` conda environment, fusion
  inference usage, and model-level color checker overrides.

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
- Consolidated engineering version operations into component, candidate,
  validation and deployment views; UI metrics now use 誤殺／漏檢 terminology.
- Acceptance color crops now use the processed-image coordinate space emitted
  by inference. Earlier `stats-robust-v1` candidates are retained for audit but
  marked incompatible; new candidates use `stats-robust-v2`.
- Release timestamps are stored as ISO 8601 UTC and rendered in the station's
  local timezone instead of displaying raw UTC wall time.
- Renamed the existing model IoU control to `YOLO NMS IoU (same-class)` and
  added separately scoped duplicate IoU/geometry controls under model settings.
- Removed unused legacy GUI panel builders and the superseded ad-hoc performance
  benchmark; retained the current GUI, packaging and camera compatibility
  entrypoints.
- Replaced the historical project progress file as a calibration record with
  `docs/records/CALIBRATION_CHANGE_LOG.md`, and moved historical evidence under
  `docs/archive/`.
- Retraining is opened from `工程設定 > 模型補訓`; legacy documentation that
  pointed to the File menu has been corrected.
- Position validation runs only when explicitly selected for the current
  retraining job. Position choices are intentionally not persisted.
- Result reporting uses `inspection_records.sqlite3` as the searchable source
  and GUI-filtered Excel snapshots instead of one fixed workbook.

- Fusion inference and color override handling are documented as model-level
  pipeline behavior, including the fallback behavior when Anomalib or PyYAML is
  unavailable.

- **`core/config.py`**: Integrated path security validation
  - Added path validation in `DetectionConfig.from_yaml()`
  - Prevents loading configs from untrusted external paths
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

## [1.1.0] - 2026-08-13

### Added
- Explicit `status` field on `ColorCheckResult` / the serialized `color_check`
  payload (`evaluated`, `no_detections`), and on `count_check` /
  `position_check` results (`expected_items_lookup_failed`,
  `position_config_lookup_failed`), so downstream consumers can tell a check
  that actually ran from one that could not, instead of reading an unrelated
  FAIL as a pass.
- Shared color-failure classification (`classify_color_check_failure` in
  `core/services/results/customer_message.py`) distinguishing a color
  **mismatch** (wrong color detected) from **low confidence** (right color,
  score below its own threshold), with matching wording reused across the
  customer-facing message, the GUI detail panel and the ASCII image overlay,
  plus new `color_mismatch` / `color_low_confidence` translation strings
  (EN/ZH).
- `ColorQCEnhanced.apply_runtime_configuration()` /
  `reset_runtime_configuration()` and the equivalent `StatsColorChecker`
  methods, which replace (rather than merge) the active thresholds/rules in a
  single validated, all-or-nothing call.
- Expanded pipeline and color-checker test coverage for the new fail-closed
  paths (`tests/test_pipeline_steps.py`,
  `tests/test_color_checker_service.py`).

### Changed
- Color check on a frame with zero detections now fails closed
  (`status=no_detections`) instead of estimating a verdict from the full-frame
  background.
- `ColorCheckerService` reapplies the complete runtime configuration (default
  threshold, overrides, rules) on every invocation, including calls that
  supply none, so a checker instance cached across products can no longer
  carry a previous product's tuning into the next inspection.
- Color-check failure text throughout the GUI and customer message now states
  whether the color was mismatched or just under-confident, instead of
  naming only the detected class.

### Fixed
- Count check and position check no longer silently pass when their
  configuration lookup fails (unreadable expected-items or position-enable
  config); both now fail closed and record why, and `finalize_status` carries
  that verdict through instead of re-deriving a false PASS from the absence of
  a signal.
- Color check candidate lookup failures now respect `color_fail_closed`
  instead of silently falling back to an unrestricted color vocabulary.
- `apply_threshold_overrides`, `apply_color_rules_overrides` and
  `set_default_threshold` now raise on invalid values instead of silently
  discarding them.

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
- **docs/architecture/TECH_GUIDE.md**: 1153-line deep-dive technical guide (JR→SR level)
- **docs/architecture/MODULE_ARCHITECTURE.md**: Architecture diagrams and design patterns
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

### Unscheduled
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
