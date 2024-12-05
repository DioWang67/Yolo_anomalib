# import torch
# import cv2
# import numpy as np
# from typing import Tuple, List, Dict
# from yolov5.utils.general import non_max_suppression, scale_boxes
# from yolov5.utils.plots import Annotator, colors
# from .utils import ImageUtils

# class YOLODetector:
#     def __init__(self, model, config):
#         self.model = model
#         self.config = config
#         self.colors = colors
#         self.image_utils = ImageUtils()
#         self.scale_boxes = scale_boxes
#         self.non_max_suppression = non_max_suppression
#         self.Annotator = Annotator

#     def preprocess_image(self, frame):
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         im = torch.from_numpy(frame_rgb).to(self.config.device).permute(2, 0, 1).float().unsqueeze(0)
#         im /= 255.0
#         return im
    
#     @staticmethod
#     def iou(box1, box2):
#         """计算两个框的 IoU（交并比）"""
#         x1 = max(box1[0], box2[0])
#         y1 = max(box1[1], box2[1])
#         x2 = min(box1[2], box2[2])
#         y2 = min(box1[3], box2[3])
        
#         intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
#         box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#         box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
#         return intersection_area / float(box1_area + box2_area - intersection_area)
    
#     def process_detections(self, pred, im, frame):
#         """处理检测结果"""
#         pred = self.non_max_suppression(pred, self.config.conf_thres, self.config.iou_thres)
#         detections = []
#         annotator = self.Annotator(frame, line_width=3, example=str(self.model.names))
#         expected_items = self.config.expected_items
#         overlapping_threshold = 0.3

#         for det in pred:
#             if len(det):
#                 det[:, :4] = self.scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
#                 for *xyxy, conf, cls in reversed(det):
#                     class_id = int(cls)
#                     class_name = self.model.names[class_id]
#                     print(class_name)
#                     box = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
#                     # 检查是否与已存在的检测框重叠
#                     if any(self.iou(box, det_box['box']) > overlapping_threshold for det_box in detections):
#                         continue

#                     detections.append({
#                         'label': class_name,
#                         'confidence': conf.item(),
#                         'box': box,
#                         'class_id': class_id  # 新增此行
#                     })

#                     color = self.colors(class_id, True)
#                     annotator.box_label(xyxy, f"{class_name} {conf:.2f}", color=color)

#         return annotator.result(), detections

#     def draw_results(self, frame, result_text, detections):
#         """绘制检测结果"""
#         result_frame = frame.copy()
#         # 绘制检测框和标签
#         for det in detections:
#             box = det['box']
#             label = f"{det['label']} {det['confidence']:.2f}"
#             class_id = det['class_id']  # 修改此行
#             color = self.colors(class_id, True)
#             cv2.rectangle(result_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
#             cv2.putText(result_frame, label, (box[0], box[1] - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#         # 绘制结果文本（PASS/FAIL）
#         cv2.putText(result_frame, result_text, (230, 230),
#                     cv2.FONT_HERSHEY_SIMPLEX, 3,
#                     (0, 255, 0) if result_text == "PASS" else (0, 0, 255), 3)
        
#         return result_frame


import torch
import cv2
import numpy as np
from typing import Tuple, List, Dict
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.plots import Annotator, colors
from .utils import ImageUtils

class YOLODetector:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.colors = colors
        self.image_utils = ImageUtils()
        self.scale_boxes = scale_boxes
        self.non_max_suppression = non_max_suppression
        self.Annotator = Annotator

    def preprocess_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = torch.from_numpy(frame_rgb).to(self.config.device).permute(2, 0, 1).float().unsqueeze(0)
        im /= 255.0
        return im

    @staticmethod
    def iou(box1, box2):
        """計算兩個框的 IoU（交並比）"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection_area / float(box1_area + box2_area - intersection_area)

    @staticmethod
    def check_missing_items(expected_items, detected_class_names):
        """
        檢查是否有缺少的項目。

        Args:
            expected_items (List[str]): 預期的項目列表。
            detected_class_names (Set[str]): 實際偵測到的類別名稱集合。

        Returns:
            Set[str]: 缺少的項目集合。
        """
        missing_items = set(expected_items) - detected_class_names
        return missing_items

    def process_detections(self, pred, im, frame):
        """处理检测结果"""
        pred = self.non_max_suppression(pred, self.config.conf_thres, self.config.iou_thres)
        detections = []
        annotator = self.Annotator(frame, line_width=3, example=str(self.model.names))
        expected_items = self.config.expected_items
        overlapping_threshold = 0.3
        detected_class_names = set()  # 用于记录已检测的类别名称

        for det in pred:
            if len(det):
                det[:, :4] = self.scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    class_id = int(cls)
                    class_name = self.model.names[class_id]
                    detected_class_names.add(class_name)  # 记录类别名称
                    box = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    # 检查是否与已存在的检测框重叠
                    if any(self.iou(box, det_box['box']) > overlapping_threshold for det_box in detections):
                        continue

                    detections.append({
                        'label': class_name,
                        'confidence': conf.item(),
                        'box': box,
                        'class_id': class_id
                    })

                    color = self.colors(class_id, True)
                    annotator.box_label(xyxy, f"{class_name} {conf:.2f}", color=color)

        # 使用新的函数检查缺少的项目
        missing_items = self.check_missing_items(expected_items, detected_class_names)
        if missing_items:
            print(f"缺少的项目: {missing_items}")
        else:
            print("所有预期项目均已检测到。")

        # 返回结果时，包含缺少的项目信息
        return annotator.result(), detections, missing_items


    def draw_results(self, frame, result_text, detections):
        """繪製偵測結果"""
        result_frame = frame.copy()
        # 繪製偵測框和標籤
        for det in detections:
            box = det['box']
            label = f"{det['label']} {det['confidence']:.2f}"
            class_id = det['class_id']
            color = self.colors(class_id, True)
            cv2.rectangle(result_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(result_frame, label, (box[0], box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # 繪製結果文本（PASS/FAIL）
        cv2.putText(result_frame, result_text, (230, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (0, 255, 0) if result_text == "PASS" else (0, 0, 255), 3)
        
        return result_frame
