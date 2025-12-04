from pathlib import Path
from typing import Optional

from ultralytics import YOLO

# 导入项目配置
from ..utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def load_model(self,
               model_path: Optional[str] = None,
               device: Optional[str] = None) -> tuple[bool, None] | tuple[bool, YOLO]:
    """
    加载预训练的YOLO11模型

    Args:
        :param device:模型文件路径。
        :param model_path:设备类型，如'cuda'、'cpu'等。
        :param self:类实例自身

    Returns:
        Tuple[bool, YOLO]: 返回元组，包含两个元素：
            - bool: 模型加载是否成功（True/False）
            - YOLO: 加载成功的YOLO模型实例，如果加载失败则返回None

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

    # 预热模型
    self._warmup_model()

    logger.info("YOLO11模型加载完成")
    return True, self.model
