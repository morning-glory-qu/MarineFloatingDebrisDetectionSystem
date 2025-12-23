"""
水面垃圾检测 FastAPI 服务入口
"""
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.detector import YOLODetector
from api.image_processor import ImageProcessor
from utils.image_base64 import decode_base64_image, encode_image_to_base64
from utils.logging_utils import setup_logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

logger = setup_logger(__name__)

detector : YOLODetector = None


def draw_detections(
        image: np.ndarray,
        detections: List[Tuple],
        box_color: Tuple[int, int, int] = (0, 0, 255),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        box_thickness: int = 2,
        font_scale: float = 0.5
) -> np.ndarray:
    annotated_img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for class_name, confidence, x1, y1, x2, y2 in detections:
        # 转换为整数坐标
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 绘制边界框
        cv2.rectangle(
            annotated_img,
            (x1, y1),
            (x2, y2),
            box_color,
            box_thickness
        )

        # 准备标签文本
        label = f"{class_name}: {confidence:.2f}"

        # 计算文本大小
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, box_thickness
        )

        # 绘制标签背景框
        cv2.rectangle(
            annotated_img,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            box_color,
            cv2.FILLED
        )

        # 绘制标签文本
        cv2.putText(
            annotated_img,
            label,
            (x1, y1 - baseline - 5),
            font,
            font_scale,
            text_color,
            box_thickness
        )

    return annotated_img


def process_detection_results(
        img_cv: np.ndarray,
        conf_threshold: float = 0.25
) -> Tuple[List[Dict], np.ndarray]:
    # 使用ImageProcessor处理图片
    processor = ImageProcessor(img_cv, input_size=640, keep_ratio=True)
    input_tensor = processor.preprocess()

    # 执行目标检测
    raw_results = detector.detect(input_tensor)

    # 后处理
    processed_results = detector.postprocess(raw_results, conf_threshold)

    # 检测结果
    detections = []

    for res in processed_results:
        class_id, class_name, confidence, x1, y1, x2, y2 = res

        # 使用ImageProcessor映射边界框到原图坐标
        bboxes_mapped = processor.map_bbox([[x1, y1, x2, y2]])
        x1_mapped, y1_mapped, x2_mapped, y2_mapped = bboxes_mapped[0]

        detections.append({
            "class_id": int(class_id),  # 类别ID
            "class_name": class_name,  # 类别名称
            "confidence": round(confidence, 2),  # 置信度
            "bbox": [
                int(x1_mapped),  # x_min
                int(y1_mapped),  # y_min
                int(x2_mapped),  # x_max
                int(y2_mapped)  # y_max
            ]
        })

    # 绘制数据
    draw_data = []
    for det in detections:
        class_name = det["class_name"]
        confidence = det["confidence"]
        bbox = det["bbox"]  # [x1, y1, x2, y2]
        draw_data.append((class_name, confidence, *bbox))

    # 5. 绘制检测框
    annotated_img = draw_detections(img_cv, draw_data)

    return detections, annotated_img


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    logger.info("服务正在启动")

    use_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model_path = os.path.join(BASE_DIR, "weights", "yolo11s.pt")
        detector = YOLODetector(model_path=model_path, device=use_device)
        detector.load_model()
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败：{e}")
        sys.exit(1)

    yield


app = FastAPI(
    title="水面垃圾检测系统",
    version="1.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {
        "service": "水面垃圾检测系统",
        "status": "running",
        "api_docs": "/docs",
        "version": "1.0"
    }


@app.post("/detect")
# 处理目标检测请求
async def detect_objects(request_data: dict):
    try:
        image_base64 = request_data.get("image")
        if not image_base64:
            raise HTTPException(status_code=400, detail="未提供图片数据")

        img_cv = decode_base64_image(image_base64)

        detections, annotated_img = process_detection_results(
            img_cv, conf_threshold=0.25
        )

        img_base64 = encode_image_to_base64(annotated_img, "jpg")

        response_data = {
            "success": True,
            "data": {
                "image": f"data:image/jpeg;base64,{img_base64}",
                "detections": detections,
            }
        }

        return JSONResponse(response_data)

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"检测出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8000, reload=False)
