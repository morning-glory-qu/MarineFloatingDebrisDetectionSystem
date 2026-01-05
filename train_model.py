import os

import numpy as np
from ultralytics import YOLO


def train_model():
    data_yaml = os.path.join('datasets', 'dataset.yaml')

    model = YOLO('yolov11n.pt')

    model.train(
        data=data_yaml,  # 数据集配置文件路径
        epochs=100,  # 训练轮数
        imgsz=640,  # 输入图像大小
        batch=16,  # 批量大小
        workers=4,  # 数据加载线程数
        device=0,  # 使用第一个GPU (device=0)
        seed=42,  # 随机种子
        patience=30,  # 早停耐心值
        project='runs',  # 输出目录
        name='detector',  # 实验名称
        exist_ok=True,  # 允许覆盖现有目录
        pretrained=True,  # 使用预训练权重
        optimizer='auto',  # 优化器
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率系数
    )

    print("训练完成！")
    print("模型保存在: runs/detect/detector/weights/best.pt")


def validate_model(model_path="runs/detect/detector/weights/best.pt"):
    """验证模型"""
    model = YOLO(model_path)

    val_args = {
        "data": "datasets/dataset.yaml",
        "imgsz": 640,
        "batch": 16,
        "conf": 0.25,
        "iou": 0.6,
        "device": 0,
        "split": "val",
    }

    print("在验证集上评估模型...")

    try:
        metrics = model.val(**val_args)

        # 打印评估结果
        print("\n验证结果:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")

        # 处理可能是数组的精确率和召回率
        if isinstance(metrics.box.p, (np.ndarray, list)):
            precision = metrics.box.p.mean() if len(metrics.box.p) > 0 else 0.0
        else:
            precision = metrics.box.p
        print(f"精确率: {precision:.4f}")

        if isinstance(metrics.box.r, (np.ndarray, list)):
            recall = metrics.box.r.mean() if len(metrics.box.r) > 0 else 0.0
        else:
            recall = metrics.box.r
        print(f"召回率: {recall:.4f}")

        return metrics
    except Exception as e:
        print(f"验证时出错: {e}")
        print("检查datasets/dataset.yaml中验证集的配置是否正确")
        return None


def test_model(model_path="runs/detect/detector/weights/best.pt"):
    """在测试集上测试模型"""
    model = YOLO(model_path)

    test_args = {
        "data": "datasets/dataset.yaml",
        "imgsz": 640,
        "batch": 16,
        "conf": 0.25,
        "iou": 0.6,
        "device": 0,
        "split": "test",
    }

    print("在测试集上测试模型...")

    try:
        metrics = model.val(**test_args)

        print("\n测试结果:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")

        # 处理可能是数组的精确率和召回率
        if isinstance(metrics.box.p, (np.ndarray, list)):
            precision = metrics.box.p.mean() if len(metrics.box.p) > 0 else 0.0
        else:
            precision = metrics.box.p
        print(f"精确率: {precision:.4f}")

        if isinstance(metrics.box.r, (np.ndarray, list)):
            recall = metrics.box.r.mean() if len(metrics.box.r) > 0 else 0.0
        else:
            recall = metrics.box.r
        print(f"召回率: {recall:.4f}")

        return metrics
    except Exception as e:
        print(f"测试时出错: {e}")
        return None


if __name__ == "__main__":
    print("=" * 50)
    print("开始训练模型...")
    print("=" * 50)
    train_model()

    print("\n" + "=" * 50)
    print("开始验证模型...")
    print("=" * 50)
    val_metrics = validate_model()

    print("\n" + "=" * 50)
    print("开始测试模型...")
    print("=" * 50)
    test_metrics = test_model()

    if val_metrics and test_metrics:
        print("\n" + "=" * 50)
        print("模型性能对比:")
        print("=" * 50)
        print(
            f"验证集 mAP50: {val_metrics.box.map50:.4f} | 测试集 mAP50: {test_metrics.box.map50:.4f}"
        )
        print(
            f"验证集 mAP50-95: {val_metrics.box.map:.4f} | 测试集 mAP50-95: {test_metrics.box.map:.4f}"
        )
