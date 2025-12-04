import cv2
import numpy as np
import torch
from typing import Union, Tuple, List


class ImageProcessor:
    """
    深度学习图像处理器类
    封装了图像加载、预处理和后处理功能
    """

    def __init__(self, input_size: int = 640):
        """
        初始化图像处理器

        Args:
            input_size: 模型输入的图像尺寸，默认640x640
        """
        self.input_size = input_size

    @staticmethod
    def load_image(image_input: Union[str, np.ndarray]) -> np.ndarray:
        """
        加载图像

        Args:
            image_input: 图像路径或numpy数组

        Returns:
            image: 加载的图像数组
        """
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"无法读取图像: {image_input}")
            return image
        elif isinstance(image_input, np.ndarray):
            return image_input.copy()
        else:
            raise TypeError("输入必须是图像路径或numpy数组")

    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple]:
        """
        图像预处理：尺寸缩放、归一化、通道转换等

        Args:
            image: 输入图像数组

        Returns:
            tensor: 预处理后的PyTorch张量
            original_shape: 原始图像尺寸 (h, w, c)
        """
        original_shape = image.shape

        # BGR to RGB转换
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调整尺寸到模型输入大小
        resized = cv2.resize(image_rgb, (self.input_size, self.input_size))

        # 像素值归一化 [0,255] -> [0,1]
        normalized = resized / 255.0

        # 维度转换: HWC -> CHW
        tensor = normalized.transpose(2, 0, 1)

        # 添加batch维度: CHW -> BCHW
        tensor = np.expand_dims(tensor, axis=0)

        # 转换为PyTorch张量
        tensor = torch.from_numpy(tensor).float()

        return tensor, original_shape

    def resize_to_original(self, bbox: List[int], original_shape: Tuple[int, int, int]) -> List[int]:
        """
        将检测框坐标调整回原始图像尺寸

        Args:
            bbox: 缩放后图像上的边界框坐标 [x1, y1, x2, y2]
            original_shape: 原始图像尺寸 (h, w, c)

        Returns:
            bbox_original: 原始图像上的边界框坐标
        """
        h_orig, w_orig = original_shape[:2]

        # 计算缩放比例
        scale_x = w_orig / self.input_size
        scale_y = h_orig / self.input_size

        x1, y1, x2, y2 = bbox
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        return [x1, y1, x2, y2]

    def process_pipeline(self, image_input: Union[str, np.ndarray]) -> Tuple[torch.Tensor, Tuple]:
        """
        完整的图像处理流水线：加载+预处理

        Args:
            image_input: 图像路径或numpy数组

        Returns:
            tensor: 预处理后的张量
            original_shape: 原始图像尺寸
        """
        image = self.load_image(image_input)
        return self.preprocess(image)