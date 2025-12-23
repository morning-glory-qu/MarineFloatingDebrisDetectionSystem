"""
YOLO 模型检测器模块
处理模型的加载和推理
"""
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from ultralytics import YOLO

from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class YOLODetector:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.class_names = None

    def load_model(self):
        """
        加载预训练的 YOLO11 模型
        """
        logger.info(f"开始加载模型: {self.model_path}")
        logger.info(f"使用设备: {self.device}")

        # 验证模型文件是否存在（如果是自定义模型）
        model_file = Path(self.model_path)
        if not model_file.exists():
            logger.error(f"模型文件不存在: {self.model_path}")
            raise RuntimeError("模型文件不存在")

        # 加载模型
        self.model = YOLO(self.model_path)

        # 移动模型到指定设备
        self.model.to(self.device)

        # 设置为评估模式
        self.model.eval()

        # 获取类别名称
        if hasattr(self.model, 'names') and self.model.names:
            self.class_names = self.model.names
            logger.info(f"模型加载成功，包含 {len(self.class_names)} 个检测类别")

            # 记录类别信息用于调试
            debris_classes = {k: v for k, v in self.class_names.items()
                              if any(
                    keyword in v.lower() for keyword in ['can', 'foam', 'plastic', 'plastic bottle', 'unknow'])}
            if debris_classes:
                logger.info(f"检测到的海洋垃圾相关类别: {list(debris_classes.values())}")
        else:
            logger.warning("无法获取模型类别名称")
            self.class_names = {}

        logger.info("YOLO11模型加载完成")

    def detect(self, tensor: Tensor):
        """使用YOLO模型进行目标检测"""
        # 检查模型是否已加载，如未加载则先加载模型
        if self.model is None:
            self.load_model()

        # 迁移到模型所在设备
        try:
            model_device = next(self.model.model.parameters()).device
            if tensor.device != model_device:
                tensor = tensor.to(model_device)
        except Exception as e:
            print(f"设备迁移失败: {e}")

        # 执行推理，返回检测结果
        return self.model(tensor, verbose=False)

    def postprocess(self, raw_output, conf_threshold=0.25):
        """
        对YOLO模型的原始输出进行后处理

        Args:
            raw_output: YOLO模型的原始输出结果
            conf_threshold: 置信度阈值，默认0.25

        Returns:
            list: 标准化后的检测结果列表，每个元素为
                  [class_id, class_name, confidence, x_min, y_min, x_max, y_max]
        """
        results = []

        # 检查输入是否有效
        if raw_output is None:
            logger.warning("原始输出为空")
            return results

        # 处理单个或多个检测结果
        raw_outputs = raw_output if isinstance(raw_output, list) else [raw_output]

        for output in raw_outputs:
            # 检查是否有检测结果
            if output.boxes is None or len(output.boxes) == 0:
                continue

            # 获取检测框信息
            boxes = output.boxes

            # 应用置信度阈值过滤
            conf_mask = boxes.conf >= conf_threshold
            filtered_boxes = boxes[conf_mask]

            if len(filtered_boxes) == 0:
                continue

            # 提取检测信息
            for box in filtered_boxes:
                # 获取类别ID和置信度
                class_id = int(box.cls.item())
                confidence = box.conf.item()

                # 获取类别名称
                class_name = self.class_names.get(class_id, f"class_{class_id}")

                # 获取边界框坐标 (x_min, y_min, x_max, y_max)
                bbox = box.xyxy[0].cpu().numpy()  # 转换为numpy数组

                # 标准化结果格式
                result = [
                    class_id,
                    class_name,
                    confidence,
                    float(bbox[0]),  # x_min
                    float(bbox[1]),  # y_min
                    float(bbox[2]),  # x_max
                    float(bbox[3])  # y_max
                ]

                results.append(result)

        # 按置信度从高到低排序
        results.sort(key=lambda x: x[2], reverse=True)
        logger.debug(f"后处理完成，共检测到 {len(results)} 个目标")
        return results
