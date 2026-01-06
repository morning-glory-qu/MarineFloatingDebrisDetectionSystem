import cv2
import numpy as np
import torch
from typing import Union, List


class ImageProcessor:
    def __init__(self, image_input: Union[str, np.ndarray], input_size: int = 640, keep_ratio: bool = True):
        self.input_size = input_size
        self.keep_ratio = keep_ratio

        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"无法读取图像: {image_input}")
            self.original = img
        elif isinstance(image_input, np.ndarray):
            self.original = image_input.copy()
        else:
            raise TypeError("image_input 必须是图像路径或 numpy 数组")

        # 初始化属性
        self.rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.resized = None
        self.tensor = None
        self.scale = None
        self.pad = None

    # ------------------------------------------------------------------
    def preprocess(self):
        """缩放、归一化、BCHW 转换，结果保存到对象内部"""
        img = self.rgb
        h, w = img.shape[:2]

        if self.keep_ratio:
            # YOLO-style letterbox
            scale = min(self.input_size / w, self.input_size / h)
            new_w, new_h = int(w * scale), int(h * scale)

            resized = cv2.resize(img, (new_w, new_h))

            pad_w = self.input_size - new_w
            pad_h = self.input_size - new_h

            pad_left = pad_w // 2
            pad_top = pad_h // 2

            padded = cv2.copyMakeBorder(
                resized,
                pad_top, pad_h - pad_top,
                pad_left, pad_w - pad_left,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )
            self.resized = padded
            self.scale = (scale, scale)
            self.pad = (pad_left, pad_top)

        else:
            # 不保持比例，直接缩放
            resized = cv2.resize(img, (self.input_size, self.input_size))
            self.resized = resized
            self.scale = (
                self.input_size / w,
                self.input_size / h,
            )
            self.pad = (0, 0)

        # 归一化 + CHW + BCHW
        x = self.resized.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)
        self.tensor = torch.from_numpy(x).unsqueeze(0)

        return self.tensor

    # ------------------------------------------------------------------
    def map_bbox(self, bboxes: List[List[float]]) -> List[List[int]]:
        """将推理结果 bbox 映射回原图"""
        if self.scale is None:
            raise RuntimeError("请先调用 preprocess() 生成 scale 与 pad")

        h0, w0 = self.original.shape[:2]
        sx, sy = self.scale
        px, py = self.pad

        mapped = []
        for x1, y1, x2, y2 in bboxes:
            x1 = (x1 - px) / sx
            y1 = (y1 - py) / sy
            x2 = (x2 - px) / sx
            y2 = (y2 - py) / sy

            mapped.append([
                int(np.clip(x1, 0, w0)),
                int(np.clip(y1, 0, h0)),
                int(np.clip(x2, 0, w0)),
                int(np.clip(y2, 0, h0)),
            ])
        return mapped
