# -*- coding: utf-8 -*-
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

@dataclass
class DetectionConfig:
    weights: str
    device: str = 'cpu'
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    imgsz: Tuple[int, int] = (1280, 1280)
    timeout: int = 2
    exposure_time: str = "1000"
    gain: str = "1.0"
    width: int = 640
    height: int = 640
    MV_CC_GetImageBuffer_nMsec: int = 10000
    current_product: Optional[str] = None  # 新增 current_product
    expected_items: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)  # 更新為產品-區域結構

    @classmethod
    def from_yaml(cls, path: str) -> 'DetectionConfig':
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            print("Loaded YAML:", config_dict)  # 調試輸出
        return cls(**config_dict)

    def get_items_by_area(self, product: str, area: str) -> Optional[List[str]]:
        """
        根據產品和區域名稱獲取對應的項目列表。
        :param product: 產品名稱，例如 "PCBA1"
        :param area: 區域名稱，例如 "A"
        :return: 該區域的項目列表，若無則返回 None。
        """
        return self.expected_items.get(product, {}).get(area)

# 使用範例
if __name__ == "__main__":
    yaml_path = r"D:\Git\robotlearning\yolo11_inference_test\config.yaml"
    config = DetectionConfig.from_yaml(yaml_path)
    items = config.get_items_by_area("PCBA1", "A")
    print("PCBA1 A區域的項目列表:", items)
    items = config.get_items_by_area("PCBA1", "X")
    print("PCBA1 未知區域的項目列表:", items)
