import cv2
import numpy as np
import torch
from typing import Union, Tuple


def load_image(image_input: Union[str, np.ndarray]) -> np.ndarray:
    """
    加载图像
    Args:
        image_input: 图像路径或numpy数组
    Returns:
        image: 加载的图像
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


def resize_to_original(bbox: list, original_shape: tuple, input_size: int) -> list:
    """
    将检测框坐标调整回原始图像尺寸
    """
    h_orig, w_orig = original_shape[:2]

    # 计算缩放比例
    scale_x = w_orig / input_size
    scale_y = h_orig / input_size

    x1, y1, x2, y2 = bbox
    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)

    return [x1, y1, x2, y2]


class ImageProcessor:
    """图像预处理处理器"""

    def __init__(self, input_size: int = 640):
        self.input_size = input_size

    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple]:
        """
        图像预处理
        Args:
            image: 输入图像
        Returns:
            tensor: 预处理后的张量
            original_shape: 原始图像尺寸 (h, w, c)
        """
        # 保存原始尺寸
        original_shape = image.shape

        # BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调整尺寸
        resized = cv2.resize(image_rgb, (self.input_size, self.input_size))

        # 归一化 [0,255] -> [0,1]
        normalized = resized / 255.0

        # 转换维度: HWC -> CHW
        tensor = normalized.transpose(2, 0, 1)

        # 添加batch维度: CHW -> BCHW
        tensor = np.expand_dims(tensor, axis=0)

        # 转换为PyTorch张量
        tensor = torch.from_numpy(tensor).float()

        return tensor, original_shape

