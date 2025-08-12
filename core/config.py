import os
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

@dataclass
class DetectionConfig:
    weights: Optional[str] = None
    device: str = 'cpu'
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    imgsz: Tuple[int, int] = (640, 640)
    timeout: int = 2
    exposure_time: str = "1000"
    gain: str = "1.0"
    width: int = 640
    height: int = 640
    MV_CC_GetImageBuffer_nMsec: int = 10000
    current_product: Optional[str] = None
    expected_items: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    enable_yolo: bool = False
    enable_anomalib: bool = False
    output_dir: str = "Result"
    anomalib_config: Optional[Dict[str, Any]] = None
    position_config: Dict[str, Dict[str, Dict]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> 'DetectionConfig':
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f) or {}
        cfg = cls()
        cfg._apply_dict(config_dict)
        return cfg

    def update_from_yaml(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f) or {}
        self._apply_dict(config_dict)

    def _apply_dict(self, config_dict: Dict[str, Any]) -> None:
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

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
