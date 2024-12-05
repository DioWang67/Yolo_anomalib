import torch
from pathlib import Path
import numpy as np
import cv2
from anomalib.models.fastflow.torch_model import FastflowModel
from torchvision import transforms
import traceback
from typing import Dict, Union, Optional, List
import os

class FastFlowDetector:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用設備: {self.device}")
        self.model = self._initialize_model()
        self._load_weights(model_path)
        self.transform = self._setup_transforms()
        
    def _initialize_model(self) -> FastflowModel:
        model = FastflowModel(
            backbone="resnet18",
            pre_trained=True,
            flow_steps=8,
            conv3x3_only=False,
            hidden_ratio=1.0,
            input_size=(256, 256)
        ).to(self.device)
        return model
        
    def _load_weights(self, model_path: str) -> None:
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()
        
    def _setup_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_image(self, image: np.ndarray, percentile: int = 50, threshold: float = -0.25) -> Dict[str, Union[bool, float, np.ndarray]]:
        try:
            if image is None or not isinstance(image, np.ndarray):
                raise ValueError("輸入圖片無效")
                
            # 確保圖片是三通道的
            if len(image.shape) != 3:
                raise ValueError("圖片必須是三通道的")
                
            # 轉換 BGR 到 RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 預處理圖片
            input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # 進行預測
            with torch.no_grad():
                outputs = self.model(input_tensor)
                
            # 後處理結果
            anomaly_map = outputs.squeeze().cpu().numpy()
            anomaly_score = float(np.max(anomaly_map))
            
            # 調整異常圖大小
            anomaly_map_resized = cv2.resize(
                anomaly_map, 
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

            # 過濾和正規化異常圖
            threshold_value = np.percentile(anomaly_map_resized, percentile)
            anomaly_map_filtered = np.clip(
                anomaly_map_resized,
                threshold_value,
                anomaly_map_resized.max()
            )

            normalized_anomaly_map = cv2.normalize(
                anomaly_map_filtered,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U
            )

            # 生成熱力圖
            heatmap = cv2.applyColorMap(normalized_anomaly_map, cv2.COLORMAP_HOT)
            overlay = cv2.addWeighted(image, 0.4, heatmap, 0.6, 0)

            # 判斷是否為異常
            is_anomaly = anomaly_score > threshold

            # 返回結果，包含 normalized_anomaly_map
            return {
                "is_anomaly": is_anomaly,
                "score": anomaly_score,
                "anomaly_map": normalized_anomaly_map,  # 修改這裡
                "visualization": overlay
            }
        except Exception as e:
            print(f"圖片處理過程發生錯誤: {str(e)}")
            traceback.print_exc()
            return None

def detect_anomaly(cropped_images: List[np.ndarray], model_path: str) -> List[Dict[str, Union[bool, float, np.ndarray]]]:
    """
    處理多個裁剪圖像的異常檢測。

    Args:
        cropped_images: 裁剪圖像列表
        model_path: FastFlow模型路徑

    Returns:
        檢測結果列表
    """
    try:
        detector = FastFlowDetector(model_path)
        results = []
        
        for img in cropped_images:
            if img is not None and isinstance(img, np.ndarray):
                result = detector.process_image(img)
                if result is not None:
                    results.append(result)
                    print(f"\n檢測結果: {'異常' if result['is_anomaly'] else '正常'}")
                    print(f"異常分數: {result['score']:.4f}")
        
        return results
    except Exception as e:
        print(f"檢測過程發生錯誤: {str(e)}")
        traceback.print_exc()
        return []

def save_detection_results(anomalib_dir: str, image_name: str, results: List[Dict[str, Union[bool, float, np.ndarray]]] = None) -> None:
    """
    保存檢測結果，包括異常圖和疊加圖。

    Args:
        status: 檢測狀態 (PASS/FAIL)
        anomalib_dir: 保存目錄
        image_name: 圖像名稱
        results: 異常檢測結果列表
    """
    try:
        os.makedirs(anomalib_dir, exist_ok=True)
        
        # 如果有異常檢測結果，保存詳細資訊
        if results:
            for i, result in enumerate(results):
                # 保存正規化的異常圖
                anomaly_map_path = os.path.join(anomalib_dir, f"{image_name}_anomaly_map_{i+1}.png")
                cv2.imwrite(anomaly_map_path, result['anomaly_map'])  # 這裡的 anomaly_map 已經是正規化的

                # 保存疊加圖
                overlay_path = os.path.join(anomalib_dir, f"{image_name}_overlay_{i+1}.png")
                cv2.imwrite(overlay_path, result['visualization'])
        print(f"檢測結果已保存到 {anomalib_dir}")

    except Exception as e:
        print(f"保存檢測結果時發生錯誤: {str(e)}")
        traceback.print_exc()
