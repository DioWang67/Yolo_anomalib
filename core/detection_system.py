"""Detection system orchestrator.

Responsibilities:
- Load and merge per-product/area/type model configs via ModelManager
- Initialize camera and inference engine backends
- Build a per-run pipeline (position/color/save) and execute
- Normalize outputs and persist results via sinks
- (Async mode) Manage Producer-Consumer pipeline workers

Public entrypoints:
  - DetectionSystem.detect(...)          — synchronous, single-shot
  - DetectionSystem.start_pipeline(...)  — async continuous mode
  - DetectionSystem.stop_pipeline()      — graceful shutdown
"""

import copy
import logging
import os
import shutil
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from camera.camera_controller import CameraController
from core.async_pipeline import AsyncPipelineManager
from core.config import DetectionConfig
from core.fusion_inference import FusionInferenceRunner
from core.inference_engine import InferenceEngine
from core.inference_tokens import InferenceTypeToken
from core.logging_config import DetectionLogger
from core.logging_utils import context_adapter
from core.path_utils import project_root, resolve_path
from core.pipeline.context import DetectionContext
from core.pipeline.finalize import finalize_status
from core.pipeline.registry import PipelineEnv, build_pipeline, default_pipeline
from core.pipeline.steps import SaveResultsStep
from core.result_adapter import normalize_result
from core.runtime_preflight import validate_runtime_for_model
from core.security import ensure_subpath, resolve_result_output_dir
from core.services.color_checker import ColorCheckerService
from core.services.color_override_loader import ColorOverrideLoader
from core.services.inspection_release_models import InspectionRelease
from core.services.inspection_release_store import (
    InspectionReleaseResolver,
    InspectionReleaseStore,
)
from core.services.model_manager import ModelManager
from core.services.result_sink import ExcelImageResultSink
from core.station_data import load_station_data_paths
from core.types import DetectionItem, DetectionResult

PROJECT_ROOT = project_root()


class DetectionSystem:
    _CAMERA_SETTINGS_RETRY_ATTEMPTS = 3
    _CAMERA_SETTINGS_RETRY_DELAY_SECONDS = 0.05
    _CAMERA_SETTINGS_SETTLE_DELAY_SECONDS = 0.10
    _CAMERA_SETTINGS_SETTLE_FRAMES = 2

    def __init__(
        self,
        config_path: str = "config.yaml",
        *,
        initialize_camera: bool = True,
        models_root: str | Path | None = None,
        color_revisions_root: str | Path | None = None,
        color_revision_overrides: dict[str, str] | None = None,
        include_active_color_revisions: bool = True,
        model_config_overrides: Mapping[
            tuple[str, str, str], str | Path
        ]
        | None = None,
        inspection_releases_root: str | Path | None = None,
        include_active_inspection_release: bool = True,
    ):
        """Initializes the DetectionSystem with project settings.

        Args:
            config_path: Relative or absolute path to the global config.yaml.
            initialize_camera: Whether to initialize production camera hardware.
                Offline tools pass ``False`` and provide image frames explicitly;
                production callers keep the default behavior.
            models_root: Optional explicit model bundle root. Candidate
                acceptance uses a job-scoped bundle instead of the deployed
                ``models`` directory.
            color_revisions_root: Optional production color-revision store.
                Candidate validation keeps this separate from its model bundle.
            color_revision_overrides: Optional acceptance-only mapping from
                scope hash to an immutable revision ID or display version.
            include_active_color_revisions: Whether unselected active color
                pointers participate. Matrix tests set this to ``False``.
            model_config_overrides: Optional acceptance-only exact model
                config snapshots. This does not change deployed config files.
            inspection_releases_root: Optional immutable release-store root.
            include_active_inspection_release: Disable when an offline tool
                supplies its own exact model and color combination.
        """

        root_dir = PROJECT_ROOT
        self.data_paths = load_station_data_paths(root_dir)
        self.logger = DetectionLogger(log_dir=str(self.data_paths.logs))
        if config_path:
            resolved_config = Path(config_path).resolve()
        else:
            # resolve_path checks sys.executable parent, sys._MEIPASS, then fallback
            resolved = resolve_path("config.yaml")
            resolved_config = resolved if resolved and resolved.exists() else root_dir / "config.yaml"
        self.config_path = resolved_config
        # _base_config holds the pristine global config; switching models
        # never mutates it. self.config is the active (model-merged) view and
        # is replaced wholesale on each switch — an atomic reference swap, so
        # concurrent readers always see a fully-built config object.
        self._base_config = self.load_config(resolved_config)
        self.config = copy.deepcopy(self._base_config)

        self.camera: CameraController | None = None
        # Last (exposure, gain) pushed to the camera, so per-model settings are
        # only re-applied on change (never on the per-frame hot path).
        self._applied_camera_settings: tuple[str, str] | None = None
        self._camera_settings_need_settle = False
        self.result_sink: ExcelImageResultSink | None = None
        self._sink_base_dir: Path | None = None
        self._refresh_result_sink()

        self.inference_engine: InferenceEngine | None = None
        # Native inference backends are not assumed to be re-entrant. This lock
        # also prevents a stopped-but-not-yet-returned call from overlapping a
        # new single-shot or auto inspection.
        self._inference_lock = threading.RLock()
        self.current_inference_type: str | None = None
        self.models_root = (
            Path(models_root).expanduser().resolve()
            if models_root is not None
            else self.data_paths.models
        )
        manager_kwargs: dict[str, Any] = {
            "models_root": self.models_root,
            "results_root": self.data_paths.results,
        }
        if model_config_overrides is not None:
            manager_kwargs["model_config_overrides"] = model_config_overrides
        self.model_manager = ModelManager(
            self.logger,
            max_cache_size=self.config.max_cache_size,
            **manager_kwargs,
        )
        self.color_service = ColorCheckerService()
        self.color_override_loader = ColorOverrideLoader(
            self.models_root,
            revisions_root=(
                Path(color_revisions_root).expanduser().resolve()
                if color_revisions_root is not None
                else self.data_paths.color_revisions
            ),
            revision_overrides=color_revision_overrides,
            include_active_revisions=include_active_color_revisions,
        )
        self._inspection_release_resolver: InspectionReleaseResolver | None = None
        self._active_inspection_release: InspectionRelease | None = None
        if include_active_inspection_release:
            release_root = (
                Path(inspection_releases_root).expanduser().resolve()
                if inspection_releases_root is not None
                else self.data_paths.inspection_releases
            )
            self._inspection_release_resolver = InspectionReleaseResolver(
                InspectionReleaseStore(release_root)
            )

        # --- Producer-Consumer pipeline (delegated to AsyncPipelineManager) ---
        self._pipeline = AsyncPipelineManager()

        if initialize_camera:
            self.initialize_camera()

    def _resolve_output_dir(self) -> Path:
        output_dir = resolve_result_output_dir(
            self.config.output_dir,
            result_root=self.data_paths.results,
        )
        self.config.output_dir = str(output_dir)
        return output_dir

    def _refresh_result_sink(self) -> None:
        output_dir = self._resolve_output_dir()
        if getattr(self, "_sink_base_dir", None) == output_dir:
            if self.result_sink:
                update_config = getattr(self.result_sink, "update_config", None)
                if callable(update_config):
                    update_config(self.config)
            return
        if self.result_sink:
            try:
                self.result_sink.close()
            except Exception:
                pass
        self.result_sink = ExcelImageResultSink(
            self.config, base_dir=str(output_dir), logger=self.logger
        )
        self._sink_base_dir = output_dir

    def load_config(self, config_path: str | Path) -> DetectionConfig:
        """Load global config from YAML into DetectionConfig dataclass."""
        return DetectionConfig.from_yaml(str(Path(config_path)))

    def reload_model_settings(
        self,
        product: str | None = None,
        area: str | None = None,
        inference_type: str | None = None,
    ) -> None:
        """Clear cached model engines so future detection uses updated configs.

        Args:
            product: Optional product filter for cache invalidation.
            area: Optional area filter for cache invalidation.
            inference_type: Optional backend filter for cache invalidation.

        Raises:
            RuntimeError: If the async pipeline is currently running.
        """
        if self.pipeline_running:
            raise RuntimeError("Cannot reload model settings while detection is running")
        if self.inference_engine:
            try:
                self.inference_engine.shutdown()
            except Exception:
                pass
            self.inference_engine = None
        self.current_inference_type = None
        self._active_inspection_release = None
        self.model_manager.clear_cache(product, area, inference_type)
        self._base_config = self.load_config(self.config_path)
        self.config = copy.deepcopy(self._base_config)
        self._refresh_result_sink()

    def shutdown(self) -> None:
        """Release resources: pipeline workers, models, camera, sinks."""
        # Stop async pipeline first (if running)
        self.stop_pipeline()
        if self.pipeline_running:
            self.logger.logger.error(
                "Pipeline workers are still active; deferring native resource "
                "shutdown to avoid tearing down an in-flight backend"
            )
            return
        if not self._inference_lock.acquire(blocking=False):
            self.logger.logger.error(
                "An inspection is still active; deferring native resource shutdown"
            )
            return
        try:
            if self.inference_engine:
                self.inference_engine.shutdown()
                self.inference_engine = None
            if self.camera:
                self.camera.shutdown()
                self.camera = None
            if self.result_sink:
                try:
                    self.result_sink.close()
                except Exception:
                    pass
                self.result_sink = None
            self._sink_base_dir = None
        finally:
            self._inference_lock.release()

    # ------------------------------------------------------------------
    # Producer-Consumer Pipeline API
    # ------------------------------------------------------------------

    def start_pipeline(
        self,
        product: str,
        area: str,
        inference_type: str = "yolo",
        *,
        capture_interval: float = 0.0,
        mode: str = "continuous",
        on_task_captured=None,
        on_task_inferred=None,
        on_task_processed=None,
        on_camera_lost=None,
        cancel_cb=None,
    ) -> None:
        """Start a pipeline only when no other inspection owns the backend."""
        if not self._inference_lock.acquire(blocking=False):
            raise RuntimeError("Cannot start pipeline: inference backend is busy")
        try:
            self._start_pipeline_locked(
                product,
                area,
                inference_type,
                capture_interval=capture_interval,
                mode=mode,
                on_task_captured=on_task_captured,
                on_task_inferred=on_task_inferred,
                on_task_processed=on_task_processed,
                on_camera_lost=on_camera_lost,
                cancel_cb=cancel_cb,
            )
        finally:
            self._inference_lock.release()

    def _start_pipeline_locked(
        self,
        product: str,
        area: str,
        inference_type: str = "yolo",
        *,
        capture_interval: float = 0.0,
        mode: str = "continuous",
        on_task_captured=None,
        on_task_inferred=None,
        on_task_processed=None,
        on_camera_lost=None,
        cancel_cb=None,
    ) -> None:
        """Start the async Producer-Consumer detection pipeline.

        Delegates to :class:`AsyncPipelineManager` which manages the
        three-stage worker lifecycle (Acquisition → Inference → Storage).

        Args:
            product: Name of product.
            area: Name of area.
            inference_type: 'yolo', 'anomalib', or 'fusion'.
            capture_interval: Seconds between captures.
            mode: 'single' for one-shot camera detection, or 'continuous'
                for manual-stop monitoring.
            on_task_captured: Optional callback for each captured task.
            on_task_inferred: Optional callback when a verdict is available.
            on_task_processed: Optional callback for each stored task.
            on_camera_lost: Optional callback when camera disconnects.

        Raises:
            RuntimeError: If the pipeline is already running or camera
                is unavailable.
        """
        if self._is_canceled(cancel_cb):
            return

        if not self.camera:
            raise RuntimeError(
                "Cannot start pipeline: no camera available. "
                "Call initialize_camera() first."
            )

        _logger = logging.getLogger(__name__)

        try:
            # Pre-load model configs and validate runtime before acquisition starts.
            self.load_model_configs(product, area, inference_type)
            self._validate_runtime_for_current_model(product, area, inference_type, _logger)
            self._ensure_camera_settings_ready_for_capture()
            if self._is_canceled(cancel_cb):
                return
            self._prepare_resources(
                product, area, inference_type, _logger, load_model_config=False
            )
            if self._is_canceled(cancel_cb):
                return
        except Exception:
            _logger.exception("Pipeline start aborted.")
            raise

        self._pipeline.start(
            camera=self.camera,
            detection_system=self,
            product=product,
            area=area,
            inference_type=inference_type,
            buffer_limit=getattr(self.config, "buffer_limit", 10),
            storage_queue_limit=getattr(self.config, "storage_queue_maxsize", 8),
            capture_interval=capture_interval,
            mode=mode,
            on_task_captured=on_task_captured,
            on_task_inferred=on_task_inferred,
            on_task_processed=on_task_processed,
            on_camera_lost=on_camera_lost,
            camera_lost_threshold=getattr(self.config, "camera_lost_threshold", 5),
            camera_reconnect_attempts=getattr(
                self.config, "camera_reconnect_attempts", 0
            ),
            camera_reconnect_backoff=getattr(
                self.config, "camera_reconnect_backoff", 2.0
            ),
        )

    def stop_pipeline(self, timeout: float = 10.0) -> None:
        """Gracefully shut down the async pipeline.

        Delegates to :class:`AsyncPipelineManager` which handles
        poison-pill propagation and worker join.

        Args:
            timeout: Max seconds to wait for each worker to finish.
        """
        self._pipeline.stop(timeout=timeout)

    @property
    def pipeline_running(self) -> bool:
        """Whether the async pipeline is currently active."""
        return self._pipeline.running

    def pipeline_stats(self) -> dict[str, Any]:
        """Return a snapshot of pipeline counters for monitoring."""
        return self._pipeline.stats()

    def capture_image(self) -> np.ndarray | None:
        """Capture a single frame from the active camera."""
        if self.camera:
            return self.camera.capture_frame()
        return None

    def initialize_camera(self) -> None:
        """Initialize camera if available; log and fall back to dummy on failure."""
        self.logger.logger.info("Initializing camera...")
        try:
            self.camera = CameraController(self.config)
            self.camera.initialize()
            self._applied_camera_settings = None
            self._camera_settings_need_settle = False
            self.logger.logger.info("Camera is ready")
        except Exception as e:
            self.logger.logger.error(f"Camera init failed: {str(e)}")
            self.logger.logger.warning(
                "Camera disabled; detection will fail unless an explicit frame is provided"
            )
            self.camera = None

    def disconnect_camera(self) -> None:
        """Release the active camera instance and mark it unavailable."""
        self.logger.logger.info("Disconnecting camera...")
        if self.camera:
            try:
                self.camera.shutdown()
            except Exception as e:
                self.logger.logger.warning(f"Camera shutdown raised: {e}")
        self.camera = None

    def reconnect_camera(self) -> bool:
        """Attempt to reinitialize the camera after a manual disconnect."""
        self.disconnect_camera()
        self.logger.logger.info("Reconnecting camera...")
        self.initialize_camera()
        connected = self.is_camera_connected()
        if connected:
            self.logger.logger.info("Camera reconnected successfully")
        else:
            self.logger.logger.error("Camera reconnect failed")
        return connected

    def is_camera_connected(self) -> bool:
        """Return True if a camera controller is initialized and ready."""
        return bool(
            self.camera
            and getattr(self.camera, "is_initialized", False)
            and getattr(self.camera, "is_healthy", True)
        )

    def load_model_configs(self, product: str, area: str, inference_type: str) -> None:
        """Resolve one release snapshot and switch all model components."""
        release = (
            self._inspection_release_resolver.resolve(product, area, inference_type)
            if self._inspection_release_resolver is not None
            else None
        )
        self._active_inspection_release = release
        config_overrides = release.model_config_overrides() if release else {}
        if release is not None:
            self.logger.logger.info(
                "Resolved inspection release: product=%s, area=%s, type=%s, "
                "release=%s, version=%s",
                product,
                area,
                inference_type,
                release.release_id,
                release.display_version,
            )

        if inference_type.lower() == "fusion":
            # The release snapshot is resolved once above. Both backends are
            # loaded from that same immutable combination.
            _, ano_merged = self._switch_model_component(
                product, area, "anomalib", config_overrides.get("anomalib")
            )
            _, merged = self._switch_model_component(
                product, area, "yolo", config_overrides.get("yolo")
            )
            if ano_merged.anomalib_config is not None:
                merged.anomalib_config = ano_merged.anomalib_config
            self.config = merged
            self.inference_engine = None
            self.current_inference_type = "fusion"
        else:
            engine, merged = self._switch_model_component(
                product,
                area,
                inference_type,
                config_overrides.get(inference_type.lower()),
            )
            self.config = merged
            self.inference_engine = engine
            self.current_inference_type = inference_type.lower()
        self._refresh_result_sink()
        self._apply_camera_settings_from_config()

    def _switch_model_component(
        self,
        product: str,
        area: str,
        inference_type: str,
        config_override: Path | None,
    ):
        """Preserve the legacy manager contract when no release is active."""
        if config_override is None:
            return self.model_manager.switch(
                self._base_config, product, area, inference_type
            )
        return self.model_manager.switch(
            self._base_config,
            product,
            area,
            inference_type,
            config_path_override=config_override,
        )

    def prepare_auto_inspection(
        self, product: str, area: str, inference_type: str
    ) -> None:
        """Prepare the selected model and camera before auto-preview starts.

        Auto inspection passes a frame captured by the preview worker directly
        to :meth:`detect`. Therefore the model-specific exposure/gain must be
        applied, settled, and its stale SDK frames cleared *before* that worker
        reads its first frame.

        Raises:
            RuntimeError: If the model runtime or camera settings cannot be
                prepared safely.
        """
        run_logger = context_adapter(
            self.logger.logger, product, area, inference_type
        )
        self.load_model_configs(product, area, inference_type)
        self._validate_runtime_for_current_model(
            product, area, inference_type, run_logger
        )
        self._prepare_resources(
            product,
            area,
            inference_type,
            run_logger,
            load_model_config=False,
        )
        self._ensure_camera_settings_ready_for_capture()

    def _apply_camera_settings_from_config(self) -> bool:
        """Push the active config's exposure/gain to the camera when changed.

        Models calibrated via the calibration dialog carry their own
        ``exposure_time``/``gain``; models without those keys keep the
        currently-set (global) values. Change-detection guarantees this never
        touches hardware unless a value actually differs, so it is safe on the
        model-switch path that runs before every inspection.
        """
        camera = self.camera
        if camera is None or not getattr(camera, "is_initialized", False):
            return False
        exposure = getattr(self.config, "exposure_time", None)
        gain = getattr(self.config, "gain", None)
        desired = (str(exposure), str(gain))
        if desired == self._applied_camera_settings:
            return True

        for attempt in range(1, self._CAMERA_SETTINGS_RETRY_ATTEMPTS + 1):
            try:
                applied = True
                if exposure is not None:
                    applied = bool(camera.set_exposure(float(exposure))) and applied
                if gain is not None:
                    applied = bool(camera.set_gain(float(gain))) and applied
            except (ValueError, TypeError) as exc:
                self.logger.logger.warning("套用相機曝光/增益失敗: %s", exc)
                return False

            if applied:
                self._applied_camera_settings = desired
                self._camera_settings_need_settle = True
                return True

            self.logger.logger.warning(
                "相機拒絕曝光/增益設定（第 %d/%d 次）: exposure=%s gain=%s",
                attempt,
                self._CAMERA_SETTINGS_RETRY_ATTEMPTS,
                exposure,
                gain,
            )
            if attempt < self._CAMERA_SETTINGS_RETRY_ATTEMPTS:
                time.sleep(self._CAMERA_SETTINGS_RETRY_DELAY_SECONDS)

        return False

    def _ensure_camera_settings_ready_for_capture(self) -> None:
        """Apply active camera settings and best-effort discard stale frames.

        Raises:
            RuntimeError: If the camera is unavailable or rejects the settings.
        """
        camera = self.camera
        if camera is None or not getattr(camera, "is_initialized", False):
            raise RuntimeError("Camera is unavailable before inspection startup")
        if not self._apply_camera_settings_from_config():
            raise RuntimeError("Unable to apply model camera exposure/gain settings")
        if not self._camera_settings_need_settle:
            return

        # Allow the hardware to finish its current exposure before flushing its
        # queue. A transient empty buffer is normal on some cameras, so this
        # transition must not turn an otherwise retryable acquisition failure
        # into a failed inspection.
        time.sleep(self._CAMERA_SETTINGS_SETTLE_DELAY_SECONDS)
        clear_buffer = getattr(camera, "clear_image_buffer", None)
        if callable(clear_buffer) and clear_buffer():
            self._camera_settings_need_settle = False
            return

        discarded_frames = 0
        for _ in range(self._CAMERA_SETTINGS_SETTLE_FRAMES):
            if camera.capture_frame() is not None:
                discarded_frames += 1
        if discarded_frames < self._CAMERA_SETTINGS_SETTLE_FRAMES:
            self.logger.logger.warning(
                "Camera setting transition discarded only %d/%d stale frames; "
                "continuing with normal acquisition retries",
                discarded_frames,
                self._CAMERA_SETTINGS_SETTLE_FRAMES,
            )
        self._camera_settings_need_settle = False

    def _validate_runtime_for_current_model(
        self, product: str, area: str, inference_type: str, run_logger
    ) -> None:
        """Validate backend runtime dependencies before acquisition starts."""
        if inference_type.lower() not in {"yolo", "fusion"}:
            return
        try:
            validate_runtime_for_model(self.config.weights)
        except Exception as exc:
            run_logger.error(
                "ONNX Runtime preflight failed: model_path=%s error=%s",
                self.config.weights,
                exc,
            )
            raise

    def _is_canceled(self, cancel_cb) -> bool:
        """Helper to safely check the cancellation callback."""
        try:
            return bool(cancel_cb and cancel_cb())
        except Exception:
            return False

    def _build_result(
        self, product: str, area: str, inference_type: str, status: str,
        *, error: str | None = None,
    ) -> DetectionResult:
        """Builds a standardized DetectionResult for empty, error, or canceled cases."""
        return DetectionResult(
            status=status,  # type: ignore[arg-type]
            product=product,
            area=area,
            inference_type=inference_type,
            error=error,
        )

    def _prepare_resources(
        self,
        product: str,
        area: str,
        inference_type: str,
        run_logger,
        *,
        load_model_config: bool = True,
    ):
        """Load model configs and initialize the color checker."""
        if load_model_config:
            self.load_model_configs(product, area, inference_type)
            self._validate_runtime_for_current_model(
                product, area, inference_type, run_logger
            )
        color_model_path = self.config.color_model_path
        if self._active_inspection_release is not None:
            color_model_path = (
                self._active_inspection_release.color_model_override()
                or color_model_path
            )
        if self.config.enable_color_check and color_model_path:
            try:
                if self._active_inspection_release is None:
                    overrides, rules_over, decision_tuning = (
                        self.color_override_loader.load(
                            self.config,
                            product,
                            area,
                            inference_type,
                            self.logger.logger,
                        )
                    )
                else:
                    overrides, rules_over, decision_tuning = (
                        self.color_override_loader.load(
                            self.config,
                            product,
                            area,
                            inference_type,
                            self.logger.logger,
                            revision_overrides=(
                                self._active_inspection_release.color_revision_overrides()
                            ),
                            include_active_revisions=False,
                        )
                    )
                checker_type = (
                    getattr(self.config, "color_checker_type", "color_qc") or "color_qc"
                )
                default_threshold = getattr(self.config, "color_score_threshold", None)
                self.color_service.ensure_loaded(
                    color_model_path,
                    overrides=overrides,
                    rules_overrides=rules_over,
                    checker_type=checker_type,
                    default_threshold=default_threshold,
                    decision_tuning=decision_tuning,
                )
                revision_ids = self.color_override_loader.last_active_revision_ids
                if revision_ids:
                    run_logger.info(
                        "Applied color calibration revisions: %s",
                        ", ".join(revision_ids),
                    )
                run_logger.info(f"Color checker loaded ({checker_type})")
            except (OSError, RuntimeError, TypeError, ValueError, KeyError) as e:
                run_logger.error(f"Color checker init failed: {e}")
                if getattr(self.config, "color_fail_closed", True):
                    raise RuntimeError(
                        "Color checker configuration could not be applied"
                    ) from e

    def _acquire_frame(self, frame: np.ndarray | None, run_logger) -> np.ndarray:
        """Capture a frame from the camera if not provided.

        Raises:
            RuntimeError: If the camera returns None or no camera/frame is
                available.  The caller (``detect``) catches this and maps
                it to ``DetectionResult(status='ERROR')``.
        """
        if frame is not None:
            return frame
        if self.camera:
            frame = self.camera.capture_frame()
            if frame is None:
                run_logger.critical(
                    "Hardware IO Error: camera.capture_frame() returned None. "
                    "Possible cable disconnect or driver failure."
                )
                raise RuntimeError(
                    "Failed to read frame from camera. Hardware failure."
                )
            return frame
        # No camera instance and no frame supplied
        run_logger.critical(
            "Hardware IO Error: no camera available and no frame provided."
        )
        raise RuntimeError("No camera available and no frame provided.")

    def run_inference(
        self,
        frame: np.ndarray,
        product: str,
        area: str,
        inference_type: str,
        run_logger,
    ) -> dict[str, Any]:
        """Perform model inference and post-processing on the frame.

        Public entry point for pipeline workers (see
        ``core.workers.DetectionPipelineHost``).
        """
        wait_started = time.monotonic()
        with self._inference_lock:
            waited = time.monotonic() - wait_started
            if waited >= 0.05:
                run_logger.warning(
                    "Inference waited %.3fs for the single-flight guard", waited
                )
            return self._run_inference_locked(
                frame, product, area, inference_type, run_logger
            )

    def _run_inference_locked(
        self,
        frame: np.ndarray,
        product: str,
        area: str,
        inference_type: str,
        run_logger,
    ) -> dict[str, Any]:
        """Run one backend call while ``_inference_lock`` is held."""
        if inference_type.lower() == "fusion":
            return FusionInferenceRunner(
                self.model_manager, self.config, self.result_sink
            ).run(
                frame,
                product,
                area,
                run_logger,
                adjust_anomalib_output_path=self._adjust_anomalib_output_path,
            )

        if not self.inference_engine:
            return {"status": "INFERENCE_ERROR", "error": "Model not loaded"}

        inference_type_name = inference_type.lower()
        output_path = None
        if inference_type_name == "anomalib":
            output_path = self.result_sink.get_annotated_path(
                status="TEMP", detector=inference_type, product=product, area=area
            )

        raw_result = self.inference_engine.infer(
            frame, product, area, InferenceTypeToken(inference_type_name), output_path
        )
        result = normalize_result(raw_result, inference_type_name, frame)
        if result.get("status") == "FAIL":
            result["status"] = "DETECTION_FAIL"

        if result.get("status") in {"ERROR", "INFERENCE_ERROR"}:
            run_logger.error(f"Inference failed: {result.get('error')}")
            return result

        if inference_type_name == "anomalib" and output_path:
            self._adjust_anomalib_output_path(result, output_path, run_logger)

        return result

    def _execute_pipeline(self, ctx: DetectionContext, run_logger, cancel_cb=None):
        """Build and run the post-processing pipeline."""
        verdict_steps, save_steps = self._build_pipeline_steps(ctx, run_logger)
        self._run_verdict_steps(ctx, verdict_steps, cancel_cb=cancel_cb)
        if ctx.status == "CANCELED":
            return
        self._run_save_steps(ctx, save_steps, cancel_cb=cancel_cb)

    def _build_pipeline_steps(self, ctx: DetectionContext, run_logger):
        """Build verdict and durability steps from the active config."""
        env = PipelineEnv(
            color_service=self.color_service,
            result_sink=self.result_sink,
            logger=run_logger,
            product=ctx.product,
            area=ctx.area,
            config=self.config,
        )

        pipe_cfg = getattr(self.config, "pipeline", None)
        step_opts_raw = getattr(self.config, "steps", {}) or {}
        step_opts = {str(k).lower(): v for k, v in step_opts_raw.items()}

        step_names = (
            [str(name) for name in pipe_cfg]
            if isinstance(pipe_cfg, list) and pipe_cfg
            else default_pipeline(env)
        )

        steps = build_pipeline(step_names, env, step_opts)
        if not steps:
            run_logger.warning(
                "Pipeline produced no executable steps; enforcing save_results fallback"
            )
            steps = build_pipeline(["save_results"], env, step_opts)

        # Run verdict steps first, finalize the status from color-corrected
        # signals, then persist — so saved artifacts/Excel reflect the final
        # verdict and a YOLO misclassification the color checker fixed no longer
        # leaves a stale FAIL on a good board.
        save_steps = [s for s in steps if isinstance(s, SaveResultsStep)]
        verdict_steps = [s for s in steps if not isinstance(s, SaveResultsStep)]
        return verdict_steps, save_steps

    def _run_verdict_steps(self, ctx, verdict_steps, *, cancel_cb=None) -> None:
        """Run CPU post-processing and compute the final operator verdict."""
        for step in verdict_steps:
            if self._is_canceled(cancel_cb):
                ctx.status = "CANCELED"
                return
            step.run(ctx)

        if self._is_canceled(cancel_cb):
            ctx.status = "CANCELED"
            return
        finalize_status(
            ctx, fail_on_unexpected=getattr(self.config, "fail_on_unexpected", True)
        )

    def _run_save_steps(self, ctx, save_steps, *, cancel_cb=None) -> None:
        """Persist an already-finalized verdict without recomputing it."""
        for step in save_steps:
            if self._is_canceled(cancel_cb):
                ctx.status = "CANCELED"
                return
            step.run(ctx)

    def finalize_detection(self, ctx: DetectionContext, run_logger) -> None:
        """Finalize color/count/sequence checks before notifying the GUI."""
        verdict_steps, _ = self._build_pipeline_steps(ctx, run_logger)
        self._run_verdict_steps(ctx, verdict_steps)

    def _adjust_anomalib_output_path(self, result: dict, temp_path: str, run_logger):
        """Move anomalib output from TEMP to a final PASS/FAIL directory."""
        correct_path = self.result_sink.get_annotated_path(
            status=result["status"],
            detector="anomalib",
            product=result.get("product"),
            area=result.get("area"),
            anomaly_score=result.get("anomaly_score"),
        )
        result["output_path"] = correct_path
        if temp_path != correct_path and os.path.exists(temp_path):
            ensure_subpath(temp_path, self.config.output_dir, must_exist=True)
            ensure_subpath(correct_path, self.config.output_dir, must_exist=False)
            os.makedirs(os.path.dirname(correct_path), exist_ok=True)
            try:
                shutil.move(temp_path, correct_path)
                run_logger.info(f"Moved output: {temp_path} -> {correct_path}")
            except Exception as e:
                run_logger.error(f"Move output failed: {e}")
            try:
                old_dir = os.path.dirname(temp_path)
                if os.path.isdir(old_dir) and not os.listdir(old_dir):
                    os.rmdir(old_dir)
            except Exception as cleanup_err:
                run_logger.warning(f"Cleanup temp dir failed: {cleanup_err}")

    def persist_detection(self, ctx: DetectionContext, run_logger) -> None:
        """Persist an already-finalized pipeline result and log its summary.

        Public entry point for the storage stage of the async pipeline
        (see ``core.workers.DetectionPipelineHost``).
        """
        _, save_steps = self._build_pipeline_steps(ctx, run_logger)
        self._run_save_steps(ctx, save_steps)
        self._log_summary(ctx, run_logger)

    def _log_summary(self, ctx: DetectionContext, run_logger):
        """Log a summary of the detection results and failure reasons."""
        if ctx.inference_type.lower() == "anomalib":
            self.logger.log_anomaly(ctx.status, ctx.result.get("anomaly_score", 0.0))
        else:
            self.logger.log_detection(ctx.status, ctx.result.get("detections", []))

        if str(ctx.status).upper() not in {"FAIL", "DETECTION_FAIL"}:
            return
        try:
            reasons = []
            decision = ctx.result.get("decision")
            if isinstance(decision, dict) and decision.get("reasons"):
                reasons.append(f"decision reasons: {decision['reasons']}")
            if ctx.result.get("missing_items"):
                reasons.append(f"missing items: {ctx.result['missing_items']}")
            if ctx.result.get("over_items"):
                reasons.append(f"extra items: {ctx.result['over_items']}")
            if color_res := ctx.color_result:
                if not color_res.get("is_ok", True):
                    items = color_res.get("items", []) or []
                    fails = sum(1 for i in items if not i.get("is_ok"))
                    reasons.append(f"color mismatch: {fails}/{len(items)} failed")
            seq_res = ctx.result.get("sequence_check")
            if seq_res and not seq_res.get("is_ok", True):
                expected_seq = seq_res.get("expected")
                observed_seq = seq_res.get("observed")
                reasons.append(
                    f"sequence mismatch: expected={expected_seq}, observed={observed_seq}"
                )
            if ctx.result.get("unexpected_items"):
                reasons.append(f"unexpected items: {ctx.result['unexpected_items']}")
            duplicate_filter = ctx.result.get("duplicate_filter")
            if isinstance(duplicate_filter, dict):
                proposed_count = int(
                    duplicate_filter.get("would_suppress_count", 0) or 0
                )
                suppressed_count = int(
                    duplicate_filter.get("suppressed_count", 0) or 0
                )
                if suppressed_count:
                    reasons.append(
                        f"cross-class duplicates suppressed: {suppressed_count}"
                    )
                elif proposed_count:
                    reasons.append(
                        f"cross-class duplicate candidates: {proposed_count}"
                    )
            pos_wrong = [
                d.get("class")
                for d in (ctx.result.get("detections", []) or [])
                if d.get("position_status") == "WRONG"
            ]
            if pos_wrong:
                reasons.append(f"position wrong: {pos_wrong}")
            if reasons:
                run_logger.info(f"Fail reasons: {'; '.join(reasons)}")
        except Exception:
            pass  # Avoid logging failures to interfere with main flow

    def detect(
        self,
        product: str,
        area: str,
        inference_type: str,
        frame: np.ndarray | None = None,
        cancel_cb=None,
        *,
        persist: bool = True,
    ) -> DetectionResult:
        """Run one complete inspection without overlapping shared backends.

        Args:
            product: Product identifier used to select the model bundle.
            area: Inspection area used to select the model bundle.
            inference_type: Runtime backend such as ``yolo``.
            frame: Optional pre-acquired BGR image.
            cancel_cb: Optional cooperative cancellation callback.
            persist: When ``False``, run the production verdict pipeline without
                its storage steps. This is intended for read-only acceptance
                and diagnostic tools that must not write production results.
        """
        if self.pipeline_running:
            return self._build_result(
                product,
                area,
                inference_type,
                "ERROR",
                error="Detection pipeline is still running or stopping",
            )
        if not self._inference_lock.acquire(blocking=False):
            return self._build_result(
                product,
                area,
                inference_type,
                "ERROR",
                error="Detection backend is busy finishing a previous inspection",
            )
        try:
            return self._detect_locked(
                product,
                area,
                inference_type,
                frame=frame,
                cancel_cb=cancel_cb,
                persist=persist,
            )
        finally:
            self._inference_lock.release()

    def _detect_locked(
        self,
        product: str,
        area: str,
        inference_type: str,
        frame: np.ndarray | None = None,
        cancel_cb=None,
        *,
        persist: bool = True,
    ) -> DetectionResult:
        """Runs the complete detection pipeline for a specific product and area.

        This includes image acquisition, model inference, post-processing steps
        (position check, color check), and results saving.

        Args:
            product: Name of the product (e.g., 'LED').
            area: Name of the station or area (e.g., 'A').
            inference_type: Type of model to run ('yolo' or 'anomalib').
            frame: Optional pre-acquired image frame. If None, captures from camera.
            cancel_cb: Optional callback that returns True to abort execution.
            persist: Whether to execute production storage steps.

        Returns:
            DetectionResult: Strongly-typed result containing status, detections,
            and artifact paths.
        """
        import time as _time
        t0 = _time.time()
        run_logger = context_adapter(self.logger.logger, product, area, inference_type)
        run_logger.info("Start detection")

        try:
            if self._is_canceled(cancel_cb):
                return self._build_result(
                    product, area, inference_type, "CANCELED"
                )

            self._prepare_resources(product, area, inference_type, run_logger)
            if frame is None:
                self._ensure_camera_settings_ready_for_capture()
            frame = self._acquire_frame(frame, run_logger)

            if self._is_canceled(cancel_cb):
                return self._build_result(
                    product, area, inference_type, "CANCELED"
                )

            result = self.run_inference(frame, product, area, inference_type, run_logger)
            if result.get("status") in {"ERROR", "INFERENCE_ERROR"}:
                error_msg = result.get("error", "Inference failed")
                return self._build_result(
                    product, area, inference_type, "INFERENCE_ERROR", error=error_msg
                )

            ctx = DetectionContext(
                product=product,
                area=area,
                inference_type=inference_type,
                frame=frame,
                processed_image=result.get("processed_image", frame),
                result=result,
                status=result["status"],
                config=self.config,
            )

            if persist:
                self._execute_pipeline(ctx, run_logger, cancel_cb)
            else:
                self.finalize_detection(ctx, run_logger)
            if ctx.status == "CANCELED":
                return self._build_result(
                    product, area, inference_type, "CANCELED"
                )

            self._log_summary(ctx, run_logger)

            # --- Build typed DetectionResult ---
            save_res = ctx.save_result or {}
            raw_dets = result.get("detections", [])
            items: list[DetectionItem] = []
            for d in raw_dets:
                bbox = d.get("bbox", (0.0, 0.0, 0.0, 0.0))
                if isinstance(bbox, list):
                    bbox = tuple(bbox)
                items.append(DetectionItem(
                    label=d.get("class", "unknown"),
                    confidence=float(d.get("confidence", 0.0)),
                    bbox_xyxy=bbox,
                    metadata={k: v for k, v in d.items()
                              if k not in ("class", "confidence", "bbox")},
                ))

            latency = _time.time() - t0
            return DetectionResult(
                status=ctx.status,  # type: ignore[arg-type]
                items=items,
                latency=latency,
                product=product,
                area=area,
                inference_type=inference_type,
                error=result.get("error", "") or None,
                ckpt_path=result.get("ckpt_path", ""),
                anomaly_score=result.get("anomaly_score"),
                missing_items=result.get("missing_items", []),
                missing_locations=save_res.get("missing_locations", []),
                over_items=result.get("over_items", []),
                unexpected_items=result.get("unexpected_items", []),
                original_image_path=save_res.get("original_path", ""),
                preprocessed_image_path=save_res.get("preprocessed_path", ""),
                annotated_path=save_res.get("annotated_path", ""),
                heatmap_path=save_res.get("heatmap_path", ""),
                cropped_paths=save_res.get("cropped_paths", []),
                color_check=ctx.color_result,
                sequence_check=ctx.result.get("sequence_check"),
                result_frame=result.get("result_frame"),
                processed_image=ctx.processed_image,
                metadata={
                    "decision": result.get("decision"),
                    "model_info": result.get("model_info"),
                    "inference_time": result.get("inference_time"),
                    "slot_check": result.get("slot_check"),
                    "slot_mismatches": result.get("slot_mismatches", []),
                    "over_items": result.get("over_items", []),
                    "layout_alignment": result.get("layout_alignment"),
                    "alignment_quality": result.get("alignment_quality"),
                    "aligned_expected_boxes": result.get("aligned_expected_boxes", {}),
                    "duplicate_filter": result.get("duplicate_filter"),
                    "raw_detection_count": len(
                        result.get("raw_detections")
                        or result.get("detections")
                        or []
                    ),
                    "inspection_release_id": (
                        self._active_inspection_release.release_id
                        if self._active_inspection_release is not None
                        else ""
                    ),
                    "inspection_release_version": (
                        self._active_inspection_release.display_version
                        if self._active_inspection_release is not None
                        else ""
                    ),
                },
            )

        except Exception as e:
            run_logger.error(f"Detection failed: {e}", exc_info=True)
            return self._build_result(
                product, area, inference_type, "ERROR", error=str(e)
            )

