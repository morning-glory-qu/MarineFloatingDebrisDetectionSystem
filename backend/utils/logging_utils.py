# 日志配置
import logging
import sys


def setup_logger(name: str) -> logging.Logger:
    """
    设置并返回配置好的logger

    Args:
        name: logger名称，通常使用 __name__

    Returns:
        logging.Logger: 配置好的logger实例
    """
    # 创建logger
    logger = logging.getLogger(name)

    # 避免重复添加handler（如果logger已经配置过）
    if logger.handlers:
        return logger

    # 设置日志级别
    logger.setLevel(logging.INFO)

    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 避免日志传递给root logger
    logger.propagate = False

    return logger