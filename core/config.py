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
    expected_items: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    enable_yolo: bool = True
    enable_anomalib: bool = False
    output_dir: str = "Result"  # 新增 output_dir
    anomalib_config: Optional[Dict] = None

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
            expected_items=config_dict.get('expected_items', {}),
            enable_yolo=config_dict.get('enable_yolo', True),
            enable_anomalib=config_dict.get('enable_anomalib', False),
            output_dir=config_dict.get('output_dir', 'Result'),  # 新增 output_dir 解析
            anomalib_config=config_dict.get('anomalib_config')
        )

    def get_items_by_area(self, product: str, area: str) -> Optional[List[str]]:
        return self.expected_items.get(product, {}).get(area)