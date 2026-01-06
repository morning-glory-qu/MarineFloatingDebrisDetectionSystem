import base64

import cv2
import numpy as np
from fastapi import HTTPException


def decode_base64_image(image_base64: str) -> np.ndarray:
    if "base64," in image_base64:
        image_base64 = image_base64.split("base64,")[1]

    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base64解码失败: {str(e)}")

    # 转换为numpy数组并解码
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="无法解码图片数据")

    return img


def encode_image_to_base64(image: np.ndarray, fmt: str = "jpg") -> str:
    if fmt.lower() == "jpg":
        ext = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    else:
        ext = ".png"
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    success, encoded_img = cv2.imencode(ext, image, params)

    if not success:
        raise ValueError(f"图片编码失败: {fmt}")

    img_base64 = base64.b64encode(encoded_img).decode('utf-8')
    return img_base64

