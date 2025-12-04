"""
项目配置文件
"""

# 模型配置
MODEL_CONFIG = {
    'default_model': 'yolo11n.pt',
    'confidence_threshold': 0.25,
    'iou_threshold': 0.45,
    'image_size': 640
}

# 设备配置
DEVICE_CONFIG = {
    'inference_device': 'auto',  # auto, cuda, cpu, mps
    'half_precision': True  # 是否使用半精度推理
}

# 海洋垃圾类别映射（根据实际模型调整）
DEBRIS_CLASSES = {
    'plastic_bag': 0,
    'plastic_bottle': 1,
    'fishing_net': 2,
    'foam': 3,
    'oil': 4,
    'other_debris': 5
}