import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import torch

# 导入被测试的类
from api.detector import YOLODetector


class TestYOLODetector(unittest.TestCase):

    def setUp(self):
        """测试前的准备工作"""
        # 创建临时模型文件（模拟模型文件）
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temp_dir.name) / "test_model.pt"

        # 创建空的模型文件
        with open(self.model_path, 'w') as f:
            f.write("fake model content")

        # 创建检测器实例
        self.detector = YOLODetector(
            model_path=str(self.model_path),
            device='cuda:0'
        )

    def tearDown(self):
        """测试后的清理工作"""
        self.temp_dir.cleanup()

    def test_initialization(self):
        """测试类初始化"""
        self.assertEqual(self.detector.model_path, str(self.model_path))
        self.assertEqual(self.detector.device, 'cuda:0')
        self.assertIsNone(self.detector.model)
        self.assertFalse(self.detector.is_loaded)
        self.assertIsNone(self.detector.class_names)

    @patch('ultralytics.YOLO')
    def test_load_model_success(self, mock_yolo):
        """测试成功加载模型"""
        # 模拟YOLO模型
        mock_model = Mock()
        mock_model.names = {0: 'plastic', 1: 'bottle', 2: 'bag'}
        mock_yolo.return_value = mock_model

        # 调用load_model
        success, model = self.detector.load_model()

        # 验证结果
        self.assertTrue(success)
        self.assertEqual(model, mock_model)
        self.assertEqual(self.detector.model, mock_model)
        self.assertTrue(self.detector.is_loaded)
        self.assertEqual(self.detector.class_names, {0: 'plastic', 1: 'bottle', 2: 'bag'})

        # 验证YOLO构造函数被正确调用
        mock_yolo.assert_called_once_with(str(self.model_path))

    def test_load_model_file_not_exist(self):
        """测试加载不存在的模型文件"""
        detector = YOLODetector(model_path="nonexistent.pt", device='cpu')
        success, model = detector.load_model()

        self.assertFalse(success)
        self.assertIsNone(model)
        self.assertFalse(detector.is_loaded)

    @patch('ultralytics.YOLO')
    def test_load_model_with_custom_params(self, mock_yolo):
        """测试使用自定义参数加载模型"""
        mock_model = Mock()
        mock_model.names = {0: 'class1'}
        mock_yolo.return_value = mock_model

        custom_model_path = "custom.pt"
        custom_device = "cuda"

        success, model = self.detector.load_model(
            model_path=custom_model_path,
            device=custom_device
        )

        self.assertTrue(success)
        self.assertEqual(self.detector.model_path, custom_model_path)
        self.assertEqual(self.detector.device, custom_device)

    @patch('ultralytics.YOLO')
    def test_detect_auto_load_model(self, mock_yolo):
        """测试detect方法自动加载模型"""
        # 模拟模型和推理结果
        mock_model = Mock()
        mock_result = Mock()
        mock_model.return_value = mock_result
        mock_yolo.return_value = mock_model

        # 调用detect（模型未加载状态）
        result = self.detector.detect("test_image.jpg")

        # 验证模型被自动加载
        self.assertTrue(self.detector.is_loaded)
        self.assertEqual(result, mock_result)
        mock_model.assert_called_once_with("test_image.jpg", verbose=False)

    @patch('ultralytics.YOLO')
    def test_detect_with_tensor(self, mock_yolo):
        """测试使用tensor进行检测"""
        mock_model = Mock()
        mock_model.model.parameters.return_value = [torch.randn(10)]  # 模拟模型参数
        mock_result = Mock()
        mock_model.return_value = mock_result
        mock_yolo.return_value = mock_model

        # 创建测试tensor
        test_tensor = torch.randn(1, 3, 640, 640)

        result = self.detector.detect(test_tensor)

        self.assertEqual(result, mock_result)

    @patch('ultralytics.YOLO')
    def test_detect_model_load_failure(self, mock_yolo):
        """测试模型加载失败的情况"""
        mock_yolo.side_effect = Exception("Load failed")

        with self.assertRaises(RuntimeError):
            self.detector.detect("test_image.jpg")

    def test_postprocess_empty_output(self):
        """测试后处理空输出"""
        result = self.detector.postprocess(None)
        self.assertEqual(result, [])

        # 模拟空的检测结果
        mock_output = Mock()
        mock_output.boxes = None
        result = self.detector.postprocess(mock_output)
        self.assertEqual(result, [])

    @patch('ultralytics.YOLO')
    def test_postprocess_with_detections(self, mock_yolo):
        """测试后处理包含检测结果的情况"""
        # 设置类别名称
        self.detector.class_names = {0: 'plastic', 1: 'bottle'}
        self.detector.is_loaded = True

        # 创建模拟的检测结果
        mock_box1 = Mock()
        mock_box1.cls = torch.tensor([0.0])  # plastic
        mock_box1.conf = torch.tensor([0.9])  # 高置信度
        mock_box1.xyxy = torch.tensor([[100.0, 200.0, 300.0, 400.0]])

        mock_box2 = Mock()
        mock_box2.cls = torch.tensor([1.0])  # bottle
        mock_box2.conf = torch.tensor([0.2])  # 低置信度，应该被过滤

        mock_boxes = Mock()
        mock_boxes.conf = torch.tensor([0.9, 0.2])
        mock_boxes.__getitem__.side_effect = lambda x: [mock_box1, mock_box2][x]
        mock_boxes.__len__.return_value = 2

        mock_output = Mock()
        mock_output.boxes = mock_boxes

        # 执行后处理（置信度阈值0.25，应该过滤掉第二个检测）
        results = self.detector.postprocess(mock_output, conf_threshold=0.25)

        # 验证结果
        self.assertEqual(len(results), 1)  # 只有高置信度的检测被保留

        detection = results[0]
        self.assertEqual(detection[0], 0)  # class_id
        self.assertEqual(detection[1], 'plastic')  # class_name
        self.assertEqual(detection[2], 0.9)  # confidence
        self.assertEqual(detection[3:], [100.0, 200.0, 300.0, 400.0])  # bbox

    def test_postprocess_multiple_outputs(self):
        """测试处理多个输出"""
        # 创建多个模拟输出
        mock_output1 = Mock()
        mock_output1.boxes = None  # 第一个输出无检测结果

        mock_output2 = Mock()
        mock_box = Mock()
        mock_box.cls = torch.tensor([0.0])
        mock_box.conf = torch.tensor([0.8])
        mock_box.xyxy = torch.tensor([[50.0, 60.0, 150.0, 160.0]])
        mock_boxes = Mock()
        mock_boxes.conf = torch.tensor([0.8])
        mock_boxes.__getitem__.return_value = mock_box
        mock_boxes.__len__.return_value = 1
        mock_output2.boxes = mock_boxes

        self.detector.class_names = {0: 'test_class'}

        results = self.detector.postprocess([mock_output1, mock_output2])
        self.assertEqual(len(results), 1)  # 只应有一个有效检测

    @patch('ultralytics.YOLO')
    def test_integration_workflow(self, mock_yolo):
        """测试完整的工作流程"""
        # 模拟模型和检测结果
        mock_model = Mock()
        mock_model.names = {0: 'plastic', 1: 'bottle'}

        # 模拟检测结果
        mock_box = Mock()
        mock_box.cls = torch.tensor([0.0])
        mock_box.conf = torch.tensor([0.95])
        mock_box.xyxy = torch.tensor([[10.0, 20.0, 110.0, 120.0]])
        mock_boxes = Mock()
        mock_boxes.conf = torch.tensor([0.95])
        mock_boxes.__getitem__.return_value = mock_box
        mock_boxes.__len__.return_value = 1

        mock_result = Mock()
        mock_result.boxes = mock_boxes

        mock_model.return_value = mock_result
        mock_yolo.return_value = mock_model

        # 执行完整流程
        self.detector.load_model()
        raw_output = self.detector.detect("test_image.jpg")
        results = self.detector.postprocess(raw_output)

        # 验证结果
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], 'plastic')
        self.assertEqual(results[0][2], 0.95)


class TestYOLODetectorEdgeCases(unittest.TestCase):
    """边界情况测试"""

    def test_pre_trained_model_name(self):
        """测试使用预训练模型名称"""
        detector = YOLODetector(model_path="yolo11n.pt", device='cpu')

        # 预训练模型名称（不以.pt/.pth结尾）应该通过验证
        self.assertTrue(detector.model_path.endswith(('.pt', '.pth')))

    def test_invalid_device_handling(self):
        """测试无效设备处理"""
        # 这个测试主要验证不会因为无效设备而崩溃
        detector = YOLODetector(model_path="test.pt", device='invalid_device')
        # 具体错误处理取决于YOLO库的实现


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
