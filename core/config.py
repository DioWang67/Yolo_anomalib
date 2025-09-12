import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)
try:  # optional pydantic validation
    from .config_schema import GlobalConfigSchema, _to_dict  # type: ignore
except Exception:  # pragma: no cover
    GlobalConfigSchema = None  # type: ignore
    _to_dict = None  # type: ignore

@dataclass
class DetectionConfig:
    """Shared runtime configuration.

    Loaded from global config.yaml and then overridden by per-model configs
    (models/<product>/<area>/<type>/config.yaml). This instance is mutated
    in-place by ModelManager.switch() to reflect the active model settings.
    """
    weights: str
    device: str = 'cpu'
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    imgsz: Tuple[int, int] = (640, 640)
    timeout: int = 2
    exposure_time: str = "1000"
    gain: str = "1.0"
    width: int = 3072
    height: int = 2048
    MV_CC_GetImageBuffer_nMsec: int = 10000
    current_product: Optional[str] = None
    current_area: Optional[str] = None
    expected_items: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    enable_yolo: bool = True
    enable_anomalib: bool = False
    enable_color_check: bool = False
    color_model_path: str | None = None
    color_threshold_overrides: Optional[Dict[str, float]] = None
    # Optional per-color rules overrides: { ColorName: { s_p90_max, s_p10_min, v_p50_min, v_p95_max } }
    color_rules_overrides: Optional[Dict[str, Dict[str, Optional[float]]]] = None
    output_dir: str = "Result"
    anomalib_config: Optional[Dict] = None
    position_config: Dict[str, Dict[str, Dict]] = field(default_factory=dict)
    max_cache_size: int = 3
    buffer_limit: int = 1
    flush_interval: float | None = None
    pipeline: Optional[List[str]] = None
    steps: Dict[str, Any] = field(default_factory=dict)
    backends: Optional[Dict[str, Dict[str, Any]]] = None  # extra/custom backends
    # Avoid duplicating cache with YOLO internal cache (default: disable)
    disable_internal_cache: bool = True
    # Saving controls
    save_original: bool = True
    save_processed: bool = True
    save_annotated: bool = True
    save_crops: bool = True
    save_fail_only: bool = False
    jpeg_quality: int = 95
    png_compression: int = 3
    max_crops_per_frame: Optional[int] = None
    fail_on_unexpected: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> 'DetectionConfig':
        """Load global config from YAML file (with optional schema normalization)."""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            logger.debug("Loaded YAML: %s", config_dict)
        # Validate/normalize via pydantic if available
        if GlobalConfigSchema is not None:
            try:
                model = GlobalConfigSchema(**(config_dict or {}))
                config_dict = _to_dict(model)  # type: ignore
            except Exception as e:
                logger.warning("Global config validation failed, using raw values: %s", e)
        return cls(
            weights=config_dict.get('weights'),
            device=config_dict.get('device', 'cpu'),
            conf_thres=config_dict.get('conf_thres', 0.25),
            iou_thres=config_dict.get('iou_thres', 0.45),
            imgsz=tuple(config_dict.get('imgsz', (640, 640))),
            timeout=config_dict.get('timeout', 2),
            exposure_time=config_dict.get('exposure_time', "1000"),
            gain=config_dict.get('gain', "1.0"),
            width=config_dict.get('width', 640),
            height=config_dict.get('height', 640),
            MV_CC_GetImageBuffer_nMsec=config_dict.get('MV_CC_GetImageBuffer_nMsec', 10000),
            current_product=config_dict.get('current_product'),
            current_area=config_dict.get('current_area'),
            expected_items=config_dict.get('expected_items', {}),
            enable_yolo=config_dict.get('enable_yolo', True),
            enable_anomalib=config_dict.get('enable_anomalib', False),
            enable_color_check=config_dict.get('enable_color_check', False),
            color_model_path=config_dict.get('color_model_path'),
            color_threshold_overrides=config_dict.get('color_threshold_overrides'),
            color_rules_overrides=config_dict.get('color_rules_overrides'),
            output_dir=config_dict.get('output_dir', 'Result'),
            anomalib_config=config_dict.get('anomalib_config'),
            position_config=config_dict.get('position_config', {}),
            max_cache_size=config_dict.get('max_cache_size', 3),
            buffer_limit=config_dict.get('buffer_limit', 10),
            flush_interval=config_dict.get('flush_interval', None),
            pipeline=config_dict.get('pipeline'),
            steps=config_dict.get('steps', {}),
            backends=config_dict.get('backends'),
            save_original=config_dict.get('save_original', True),
            save_processed=config_dict.get('save_processed', True),
            save_annotated=config_dict.get('save_annotated', True),
            save_crops=config_dict.get('save_crops', True),
            save_fail_only=config_dict.get('save_fail_only', False),
            jpeg_quality=int(config_dict.get('jpeg_quality', 95)),
            png_compression=int(config_dict.get('png_compression', 3)),
            max_crops_per_frame=config_dict.get('max_crops_per_frame'),
            fail_on_unexpected=bool(config_dict.get('fail_on_unexpected', True)),
            disable_internal_cache=config_dict.get('disable_internal_cache', True)
        )

    def get_items_by_area(self, product: str, area: str) -> Optional[List[str]]:
        return self.expected_items.get(product, {}).get(area)

    def get_position_config(self, product: str, area: str) -> Optional[Dict]:
        return self.position_config.get(product, {}).get(area, None)

    def is_position_check_enabled(self, product: str, area: str) -> bool:
        config = self.get_position_config(product, area)
        return config is not None and config.get("enabled", False)
    
    def get_tolerance_ratio(self, product: str, area: str) -> float:
        """取得指定區域的容忍比例 (0.05 表示 5%)"""
        config = self.get_position_config(product, area)
        if config is None:
            return 0.0

        val = config.get("tolerance", 0)
        if val <= 0:
            return 0.0
        if val <= 100:
            return val / 100.0  # 百分比轉小數
        return 0.0
