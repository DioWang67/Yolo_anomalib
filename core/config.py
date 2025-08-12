import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

@dataclass
class DetectionConfig:
    weights: str
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
    current_area: Optional[str] = None
    expected_items: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    enable_yolo: bool = True
    enable_anomalib: bool = False
    output_dir: str = "Result"
    anomalib_config: Optional[Dict] = None
    position_config: Dict[str, Dict[str, Dict]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> 'DetectionConfig':
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            print("Loaded YAML:", config_dict)
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
            output_dir=config_dict.get('output_dir', 'Result'),
            anomalib_config=config_dict.get('anomalib_config'),
            position_config=config_dict.get('position_config', {})
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
