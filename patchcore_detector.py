import torch
from pathlib import Path
import numpy as np
import cv2
from anomalib.models.patchcore.torch_model import PatchcoreModel
from torchvision import transforms
import traceback
from typing import Dict, Union, Optional, List
import os

class PatchCoreDetector:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用設備: {self.device}")
        self.model = self._initialize_model()
        self._load_weights(model_path)
        self.transform = self._setup_transforms()

    def _initialize_model(self) -> PatchcoreModel:
        model = PatchcoreModel(
            input_size=(256, 256),
            backbone="wide_resnet50_2",
            pre_trained=True,
            layers=['layer2', 'layer3'],
            num_neighbors=9
        ).to(self.device)
        return model
        
    def _setup_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_weights(self, model_path: str) -> None:
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.eval()
            print(f"模型載入成功: {model_path}")
        except Exception as e:
            print(f"模型載入失敗: {str(e)}")
            traceback.print_exc()
            raise

    def normalize_anomaly_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        """更精確的異常圖正規化方法"""
        # 使用 Min-Max 正規化，保持異常分數的相對關係
        min_val = anomaly_map.min()
        max_val = anomaly_map.max()
        if max_val - min_val > 0:
            normalized_map = (anomaly_map - min_val) / (max_val - min_val)
        else:
            normalized_map = anomaly_map
        return (normalized_map * 255).astype(np.uint8)
    
    def calculate_anomaly_score(self, distances: np.ndarray) -> float:
        """基於 k-最近鄰距離計算異常分數"""
        # 取最大距離作為異常分數
        max_distance = np.max(distances)
        # 使用 softmax 正規化
        score = 1 / (1 + np.exp(-max_distance))
        return float(score)
    
    def generate_heatmap(self, anomaly_map: np.ndarray, image: np.ndarray) -> np.ndarray:
        """生成改進版熱力圖"""
        # 先做高斯模糊減少噪聲
        smoothed_map = cv2.GaussianBlur(anomaly_map, (3, 3), 0)
        
        # 應用不同的色彩映射
        heatmap = cv2.applyColorMap(smoothed_map, cv2.COLORMAP_JET)  # 或使用其他色彩映射
        
        # 調整透明度
        alpha = 0.5  # 可調整的透明度
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay
    def process_image(self, image: np.ndarray, percentile: int = 15, threshold: float = 40) -> Dict[str, Union[bool, float, np.ndarray]]:
        try:     
            # 檢查輸入是否有效
            if image is None or not isinstance(image, np.ndarray):
                raise ValueError("輸入圖片無效")
            if len(image.shape) != 3:
                raise ValueError(f"圖片必須是三通道的，當前形狀為 {image.shape}")

            # 圖像預處理
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

            # 模型預測
            with torch.no_grad():
                predictions = self.model(input_tensor)
            
            # 提取異常圖與分數
            anomaly_map = predictions[0].squeeze().cpu().numpy()
            anomaly_score = float(predictions[1].cpu().numpy())

            # 生成異常熱圖
            anomaly_map_resized = cv2.resize(
                anomaly_map, 
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            normalized_map = self.normalize_anomaly_map(anomaly_map_resized)
            visualization = self.generate_heatmap(normalized_map, image)

            # 計算動態閾值
            dynamic_threshold = max(
                np.percentile(normalized_map, percentile),
                np.max(normalized_map) * 0.8  # 確保閾值不超過最大值的80%
            )

            print(f"異常圖最大值: {np.max(normalized_map)}")
            print(f"動態閾值: {dynamic_threshold}")
            print(f"異常分數: {anomaly_score}")

            # 判定是否為異常
            is_anomaly = anomaly_score > dynamic_threshold or anomaly_score > threshold
            
            # 返回結果
            return {
                "is_anomaly": is_anomaly,
                "score": anomaly_score,
                "anomaly_map": normalized_map,
                "visualization": visualization,
                "threshold": dynamic_threshold
            }
        except Exception as e:
            print(f"處理圖片時發生錯誤: {str(e)}")
            traceback.print_exc()
            return None


def detect_anomaly(cropped_images: List[np.ndarray], model: PatchCoreDetector) -> List[Dict[str, Union[bool, float, np.ndarray]]]:
    """
    使用已載入好的 PatchCoreDetector 模型對多個裁剪圖像進行異常檢測。

    Args:
        cropped_images: 裁剪圖像列表
        model: 已初始化好的 PatchCoreDetector 實例

    Returns:
        檢測結果列表
    """
    results = []
    for img in cropped_images:
        if img is not None and isinstance(img, np.ndarray):
            result = model.process_image(img)
            if result is not None:
                results.append(result)
                print(f"\n檢測結果: {'異常' if result['is_anomaly'] else '正常'}")
                print(f"異常分數: {result['score']:.4f}")
    return results

def save_detection_results(anomalib_dir: str, image_name: str, results: List[Dict[str, Union[bool, float, np.ndarray]]] = None) -> None:
    """
    保存檢測結果，包括異常圖和疊加圖。

    Args:
        anomalib_dir: 保存目錄
        image_name: 圖像名稱
        results: 異常檢測結果列表
    """
    try:
        os.makedirs(anomalib_dir, exist_ok=True)
        
        if results:
            for i, result in enumerate(results):
                # 保存正規化的異常圖
                anomaly_map_path = os.path.join(anomalib_dir, f"{image_name}_anomaly_map_{i+1}.png")
                cv2.imwrite(anomaly_map_path, result['anomaly_map'])

                # 保存疊加圖
                overlay_path = os.path.join(anomalib_dir, f"{image_name}_overlay_{i+1}.png")
                cv2.imwrite(overlay_path, result['visualization'])
        print(f"檢測結果已保存到 {anomalib_dir}")

    except Exception as e:
        print(f"保存檢測結果時發生錯誤: {str(e)}")
        traceback.print_exc()


def load_images_from_directory(directory: str) -> List[np.ndarray]:
    """
    從指定目錄載入所有圖片。

    Args:
        directory: 圖片資料夾的路徑

    Returns:
        讀取到的圖片列表 (np.ndarray)
    """
    image_list = []
    image_paths = Path(directory).rglob("*.[pjP][nN][gG]")  # 匹配 .jpg 和 .png
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is not None:
            image_list.append(img)
            print(f"成功載入圖片: {path}")
        else:
            print(f"無法讀取圖片: {path}")
    return image_list

if __name__ == "__main__":
    # 設定模型和圖片的路徑
    MODEL_PATH = r"D:\Git\robotlearning\yolo_inference_test\anomalib_model\con1_model.ckpt"  # 修改為你的模型路徑
    IMAGE_DIR = r"S:\DioWang\robotlearning\img\con_for_anomalib\good"

    # 初始化 PatchCoreDetector 模型
    try:
        patchcore_detector = PatchCoreDetector(MODEL_PATH)
        print("模型初始化完成")
    except Exception as e:
        print(f"模型初始化失敗: {str(e)}")
        exit()

    # 載入圖片
    cropped_images = load_images_from_directory(IMAGE_DIR)
    if not cropped_images:
        print("未找到任何圖片，請檢查圖片資料夾路徑。")
        exit()

    # 進行異常檢測
    results = detect_anomaly(cropped_images, patchcore_detector)

    # 保存檢測結果
    SAVE_DIR = "output_results"  # 保存結果的資料夾
    os.makedirs(SAVE_DIR, exist_ok=True)
    for idx, result in enumerate(results):
        anomaly_map_path = os.path.join(SAVE_DIR, f"image_{idx+1}_anomaly_map.png")
        overlay_path = os.path.join(SAVE_DIR, f"image_{idx+1}_overlay.png")
        cv2.imwrite(anomaly_map_path, result['anomaly_map'])
        cv2.imwrite(overlay_path, result['visualization'])
        print(f"結果保存完成: {anomaly_map_path}, {overlay_path}")

    print("所有檢測完成，結果已保存！")



# import torch
# from pathlib import Path
# import numpy as np
# import cv2
# from anomalib.models.patchcore.torch_model import PatchcoreModel
# from torchvision import transforms
# import traceback
# from sklearn.utils.extmath import randomized_svd
# from typing import Dict, Union, Optional, List
# import os

# class PatchCoreDetector:
#     def __init__(self, model_path: str, coreset_size: float = 0.01):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"使用設備: {self.device}")
#         self.model = self._initialize_model()
#         self._load_weights(model_path)
#         self.transform = self._setup_transforms()
#         self.coreset_size = coreset_size
#         self.memory_bank = None  # 記憶庫初始化

#     def _initialize_model(self) -> PatchcoreModel:
#         model = PatchcoreModel(
#             input_size=(256, 256),
#             backbone="wide_resnet50_2",
#             pre_trained=True,
#             layers=['layer2', 'layer3'],
#             num_neighbors=9
#         ).to(self.device)
#         return model

#     def _setup_transforms(self) -> transforms.Compose:
#         return transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     def _load_weights(self, model_path: str) -> None:
#         try:
#             checkpoint = torch.load(model_path, map_location=self.device)
#             state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
#             new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
#             self.model.load_state_dict(new_state_dict, strict=False)
#             self.model.eval()
#             print(f"模型載入成功: {model_path}")
#         except Exception as e:
#             print(f"模型載入失敗: {str(e)}")
#             traceback.print_exc()
#             raise

#     def build_memory_bank(self, features: np.ndarray) -> None:
#         """基於 Coreset Subsampling 建立記憶庫"""
#         try:
#             num_samples = int(features.shape[0] * self.coreset_size)
#             _, _, Vt = randomized_svd(features, n_components=num_samples)
#             self.memory_bank = Vt
#             print(f"記憶庫建構完成，大小: {self.memory_bank.shape}")
#         except Exception as e:
#             print(f"記憶庫建構失敗: {str(e)}")

#     def normalize_anomaly_map(self, anomaly_map: np.ndarray) -> np.ndarray:
#         min_val, max_val = anomaly_map.min(), anomaly_map.max()
#         if max_val - min_val > 0:
#             normalized_map = (anomaly_map - min_val) / (max_val - min_val)
#         else:
#             normalized_map = anomaly_map
#         return (normalized_map * 255).astype(np.uint8)

#     def calculate_anomaly_score(self, distances: np.ndarray) -> float:
#         max_distance = np.max(distances)
#         return 1 / (1 + np.exp(-max_distance))

#     def generate_heatmap(self, anomaly_map: np.ndarray, image: np.ndarray) -> np.ndarray:
#         smoothed_map = cv2.GaussianBlur(anomaly_map, (3, 3), 0)
#         heatmap = cv2.applyColorMap(smoothed_map, cv2.COLORMAP_JET)
#         alpha = 0.5
#         overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
#         return overlay

#     def process_image(self, image: np.ndarray, percentile: int = 3) -> Dict[str, Union[bool, float, np.ndarray]]:
#         try:
#             if image is None or not isinstance(image, np.ndarray):
#                 raise ValueError("輸入圖片無效")
#             if len(image.shape) != 3:
#                 raise ValueError(f"圖片必須是三通道的，當前形狀為 {image.shape}")

#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

#             with torch.no_grad():
#                 predictions = self.model(input_tensor)

#             anomaly_map = predictions[0].squeeze().cpu().numpy()
#             anomaly_score = float(predictions[1].cpu().numpy())

#             anomaly_map_resized = cv2.resize(anomaly_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
#             normalized_map = self.normalize_anomaly_map(anomaly_map_resized)
#             visualization = self.generate_heatmap(normalized_map, image)

#             dynamic_threshold = np.percentile(normalized_map, percentile)
#             is_anomaly = anomaly_score > dynamic_threshold

#             return {
#                 "is_anomaly": is_anomaly,
#                 "score": float(anomaly_score),
#                 "anomaly_map": normalized_map,
#                 "visualization": visualization,
#                 "threshold": dynamic_threshold
#             }
#         except Exception as e:
#             print(f"處理圖片時發生錯誤: {str(e)}")
#             traceback.print_exc()
#             return None


# def detect_anomaly(cropped_images: List[np.ndarray], model: PatchCoreDetector) -> List[Dict[str, Union[bool, float, np.ndarray]]]:
#     results = []
#     for img in cropped_images:
#         if img is not None and isinstance(img, np.ndarray):
#             result = model.process_image(img)
#             if result is not None:
#                 results.append(result)
#                 print(f"\n檢測結果: {'異常' if result['is_anomaly'] else '正常'}")
#                 print(f"異常分數: {result['score']:.4f}")
#     return results


# def load_images_from_directory(directory: str) -> List[np.ndarray]:
#     image_list = []
#     image_paths = Path(directory).rglob("*.[pjP][nN][gG]")
#     for path in image_paths:
#         img = cv2.imread(str(path))
#         if img is not None:
#             image_list.append(img)
#             print(f"成功載入圖片: {path}")
#         else:
#             print(f"無法讀取圖片: {path}")
#     return image_list


# if __name__ == "__main__":
#     MODEL_PATH = r"D:\Git\robotlearning\yolo_inference_test\anomalib_model\con1_model.ckpt"
#     IMAGE_DIR = r"S:\DioWang\robotlearning\img\con_for_anomalib\ground_truth"

#     try:
#         patchcore_detector = PatchCoreDetector(MODEL_PATH)
#         print("模型初始化完成")
#     except Exception as e:
#         print(f"模型初始化失敗: {str(e)}")
#         exit()

#     cropped_images = load_images_from_directory(IMAGE_DIR)
#     if not cropped_images:
#         print("未找到任何圖片，請檢查圖片資料夾路徑。")
#         exit()

#     results = detect_anomaly(cropped_images, patchcore_detector)

#     SAVE_DIR = "output_results"
#     os.makedirs(SAVE_DIR, exist_ok=True)
#     for idx, result in enumerate(results):
#         anomaly_map_path = os.path.join(SAVE_DIR, f"image_{idx+1}_anomaly_map.png")
#         overlay_path = os.path.join(SAVE_DIR, f"image_{idx+1}_overlay.png")
#         cv2.imwrite(anomaly_map_path, result['anomaly_map'])
#         cv2.imwrite(overlay_path, result['visualization'])
#         print(f"結果保存完成: {anomaly_map_path}, {overlay_path}")

#     print("所有檢測完成，結果已保存！")
