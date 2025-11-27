import torch


def test_cuda():
    print("=== PyTorch CUDA 测试 ===")

    # 基本检查
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 可用: {cuda_available}")

    if not cuda_available:
        print("CUDA 不可用，请检查安装")
        return False

    # GPU 信息
    device_count = torch.cuda.device_count()
    print(f"GPU 数量: {device_count}")

    for i in range(device_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.1f} GB")

    # 简单的张量运算测试
    try:
        # 创建张量
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()

        # GPU 运算
        z = torch.matmul(x, y)

        # 检查结果
        print(f"GPU 运算测试成功，结果形状: {z.shape}")
        print(f"张量所在设备: {z.device}")

        return True

    except Exception as e:
        print(f"GPU 测试失败: {e}")
        return False


if __name__ == "__main__":
    test_cuda()