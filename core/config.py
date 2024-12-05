# config.py
import yaml
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class DetectionConfig:
    weights: str
    device: str = 'cpu'
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    imgsz: Tuple[int, int] = (1280, 1280)
    timeout: int = 2
    expected_items: List[str] = None
    anomalib_model_list: List[str] = None
    @classmethod
    def from_yaml(cls, path: str) -> 'DetectionConfig':
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            print("Loaded YAML:", config_dict)  # 調試輸出
        return cls(**config_dict)

