"""
YOLO 模型检测器模块
处理模型的加载和推理
"""
from pathlib import Path
from typing import Any
from typing import Optional

import torch
from ultralytics import YOLO

from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class YOLODetector:
    def __init__(self, model_path: str, device: str = 'cude:0'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.is_loaded = False
        self.class_names = None

    def load_model(self,
                   model_path: Optional[str] = None,
                   device: Optional[str] = None) -> tuple[bool, None] | tuple[bool, YOLO]:
        """
        加载预训练的 YOLO11 模型

        Args:
            :param device: 模型文件路径。
            :param model_path: 设备类型，如 'cuda'、'cpu' 等。
            :param self:类实例自身

        Returns:
            Tuple[bool, YOLO]: 返回元组，包含两个元素：
                - bool: 模型加载是否成功（True/False）
                - YOLO: 加载成功的 YOLO 模型实例，如果加载失败则返回 None

        """
        # 更新参数
        if model_path is not None:
            self.model_path = model_path
        if device is not None:
            self.device = device

        logger.info(f"开始加载模型: {self.model_path}")
        logger.info(f"使用设备: {self.device}")

        # 验证模型文件是否存在（如果是自定义模型）
        if not self.model_path.endswith(('.pt', '.pth')):
            # 可能是预训练模型名称
            pass
        else:
            model_file = Path(self.model_path)
            if not model_file.exists():
                logger.error(f"模型文件不存在: {self.model_path}")
                return False, None

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
                    keyword in v.lower() for keyword in ['plastic', 'bag', 'bottle', 'debris', 'trash', 'foam', 'oil'])}
            if debris_classes:
                logger.info(f"检测到的海洋垃圾相关类别: {list(debris_classes.values())}")
        else:
            logger.warning("无法获取模型类别名称")
            self.class_names = {}

        self.is_loaded = True

        logger.info("YOLO11模型加载完成")
        return True, self.model

    def detect(self, tensor: Any):
        """使用YOLO模型进行目标检测"""
        # 检查模型是否已加载，如未加载则先加载模型
        if not self.is_loaded or self.model is None:
            succ, model = self.load_model()
            if not succ:
                raise RuntimeError("模型加载失败")
            self.model = model
            self.is_loaded = True

        # 如果是torch张量，自动迁移到模型所在设备
        if torch is not None and isinstance(tensor, torch.Tensor):
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
