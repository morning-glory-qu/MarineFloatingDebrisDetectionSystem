import React, { useState, useRef, useCallback } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Navigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Home as HomeIcon,
  Upload,
  Image as ImageIcon,
  BarChart3,
  Trash2,
  Download,
  AlertCircle,
  RefreshCw,
  Database,
  Settings,
  Search,
  Filter,
  Calendar,
  FileText,
  Eye,
  MoreVertical,
  ChevronLeft,
  Globe,
  Camera,
  Shield,
  Users,
  TrendingUp,
  TrendingDown
} from 'lucide-react';
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line
} from 'recharts';

// 你的原始接口定义
interface DetectionResult {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: {
    x_min: number;
    y_min: number;
    x_max: number;
    y_max: number;
  };
}

interface ApiResponse {
  success: boolean;
  data: {
    image: string;  // 完整的data URL: "data:image/jpeg;base64,..."
    detections: DetectionResult[];
  };
}

// 主页组件
const HomePage = () => {
  const features = [
    {
      icon: <Camera className="w-6 h-6" />,
      title: "智能检测",
      description: "基于深度学习模型，精准识别多种海洋垃圾类型"
    },
    {
      icon: <BarChart3 className="w-6 h-6" />,
      title: "数据分析",
      description: "可视化分析垃圾分布和统计数据"
    },
    {
      icon: <Database className="w-6 h-6" />,
      title: "数据管理",
      description: "历史记录存储和管理功能"
    },
    {
      icon: <Globe className="w-6 h-6" />,
      title: "环保报告",
      description: "生成详细的环保报告和统计信息"
    },
    {
      icon: <Shield className="w-6 h-6" />,
      title: "高精度识别",
      description: "多种垃圾类型高精度检测，置信度高达95%以上"
    },
    {
      icon: <FileText className="w-6 h-6" />,
      title: "报告导出",
      description: "支持检测结果导出为多种格式"
    }
  ];

  const quickActions = [
    {
      title: "开始检测",
      description: "上传图片进行海洋垃圾检测",
      icon: <Upload className="w-8 h-8" />,
      link: "/detect",
      color: "bg-blue-500 hover:bg-blue-600"
    },
    {
      title: "查看统计",
      description: "查看历史检测数据和统计信息",
      icon: <BarChart3 className="w-8 h-8" />,
      link: "/statistics",
      color: "bg-green-500 hover:bg-green-600"
    },
    {
      title: "数据管理",
      description: "管理已检测的图片和历史记录",
      icon: <Database className="w-8 h-8" />,
      link: "/data",
      color: "bg-purple-500 hover:bg-purple-600"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      {/* Hero Section */}
      <div className="relative overflow-hidden bg-gradient-to-r from-blue-600 to-teal-600 text-white">
        <div className="absolute inset-0 bg-black opacity-20"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className="text-center">
            <h1 className="text-4xl md:text-6xl font-bold mb-6">
              海洋漂浮垃圾智能检测系统
            </h1>
            <p className="text-xl md:text-2xl mb-8 text-blue-100">
              利用人工智能技术保护海洋生态环境
            </p>
            <p className="text-lg mb-12 max-w-3xl mx-auto text-blue-50">
              基于深度学习模型，实时检测、分类和分析海洋中的漂浮垃圾，为海洋环境保护提供科学数据支持
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link to="/detect">
                <Button size="lg" className="bg-white text-blue-600 hover:bg-blue-50 px-8">
                  <Upload className="w-5 h-5 mr-2" />
                  开始检测
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">快速开始</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
          {quickActions.map((action, index) => (
            <Link key={index} to={action.link}>
              <Card className="h-full hover:shadow-lg transition-shadow cursor-pointer border-2 border-transparent hover:border-blue-200">
                <CardContent className="p-8 text-center">
                  <div className={`${action.color} w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-6 text-white transition-transform hover:scale-110`}>
                    {action.icon}
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-3">{action.title}</h3>
                  <p className="text-gray-600">{action.description}</p>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>

        {/* Features */}
        <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">核心功能</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-16">
          {features.map((feature, index) => (
            <Card key={index} className="hover:shadow-md transition-shadow">
              <CardContent className="p-6">
                <div className="w-12 h-12 rounded-lg bg-blue-100 text-blue-600 flex items-center justify-center mb-4">
                  {feature.icon}
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">{feature.title}</h3>
                <p className="text-gray-600 text-sm">{feature.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Stats */}
        <div className="bg-gradient-to-r from-blue-50 to-teal-50 rounded-2xl p-8 mb-16">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            <div className="text-center">
              <div className="text-4xl font-bold text-blue-600 mb-2">10,000+</div>
              <div className="text-gray-600">已检测图片</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-green-600 mb-2">95%</div>
              <div className="text-gray-600">识别准确率</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-purple-600 mb-2">8+</div>
              <div className="text-gray-600">垃圾类型识别</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-teal-600 mb-2">24/7</div>
              <div className="text-gray-600">全天候服务</div>
            </div>
          </div>
        </div>

        {/* CTA */}
        <div className="text-center">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">准备好保护我们的海洋了吗？</h2>
          <p className="text-gray-600 mb-8 max-w-2xl mx-auto">
            加入我们，利用人工智能技术为海洋环境保护贡献一份力量
          </p>
          <Link to="/detect">
            <Button size="lg" className="bg-gradient-to-r from-blue-600 to-teal-600 hover:from-blue-700 hover:to-teal-700 px-12">
              开始使用
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
};

// 检测页面组件（你的原始代码，稍作修改）
const DetectPage = () => {
  const [uploading, setUploading] = useState(false);
  const [detectionResults, setDetectionResults] = useState<DetectionResult[]>([]);
  const [originalImage, setOriginalImage] = useState<string>('');
  const [processedImage, setProcessedImage] = useState<string>('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);

  // API基础URL
  const API_BASE_URL = 'http://localhost:8000';

  // 清理所有数据
  const clearAllData = useCallback(() => {
    setDetectionResults([]);
    setOriginalImage('');
    setProcessedImage('');
    setSelectedFile(null);
    setError(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  // 清除错误信息
  const clearError = () => {
    setError(null);
  };

  // 文件验证
  const validateFile = (file: File): void => {
    const MAX_FILE_SIZE = 10 * 1024 * 1024;
    if (file.size > MAX_FILE_SIZE) {
      throw new Error(`文件大小不能超过 ${MAX_FILE_SIZE / 1024 / 1024}MB`);
    }

    const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
      throw new Error('仅支持 JPG、PNG、WebP 格式的图片');
    }
  };

  // 上传图片到后端API
  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    clearAllData();

    setSelectedFile(file);
    setUploading(true);

    try {
      validateFile(file);

      // 创建文件预览
      const previewUrl = URL.createObjectURL(file);
      setOriginalImage(previewUrl);

      // 读取文件为Base64（完整格式）
      const base64Image = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = error => reject(error);
        reader.readAsDataURL(file);
      });

      console.log('发送数据格式:', {
        fileType: file.type,
        fileSize: file.size,
        base64Prefix: base64Image.substring(0, 30)
      });

      // 发送完整格式的Base64给后端
      const response = await fetch(`${API_BASE_URL}/detect`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: base64Image  // 发送完整格式：data:image/jpeg;base64,xxx
        }),
      });

      console.log('响应状态:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`检测失败: ${response.status} - ${errorText}`);
      }

      const data: ApiResponse = await response.json();

      // 根据后端响应结构处理
      if (data.success && data.data) {
        setDetectionResults(data.data.detections || []);

        // 后端已经返回完整格式，直接使用
        if (data.data.image) {
          setProcessedImage(data.data.image);
          console.log('接收到的图片数据格式:', data.data.image.substring(0, 30));
        }
      } else {
        throw new Error('后端返回数据格式错误');
      }

    } catch (error) {
      console.error('检测失败详情:', error);
      const errorMessage = error instanceof Error ? error.message : '未知错误';
      setError(`处理失败: ${errorMessage}`);
    } finally {
      setUploading(false);
    }
  };

  // 重新检测图片
  const reDetectImage = async () => {
    if (!selectedFile) return;

    setUploading(true);

    try {
      // 读取文件为Base64（完整格式）
      const base64Image = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = error => reject(error);
        reader.readAsDataURL(selectedFile);
      });

      const response = await fetch(`${API_BASE_URL}/detect`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: base64Image
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`重新检测失败: ${response.status} - ${errorText}`);
      }

      const data: ApiResponse = await response.json();

      if (data.success && data.data) {
        setDetectionResults(data.data.detections || []);

        if (data.data.image) {
          setProcessedImage(data.data.image);
        }
      } else {
        throw new Error('后端返回数据格式错误');
      }

    } catch (error) {
      console.error('重新检测失败:', error);
      setError('重新检测失败，请检查后端服务');
    } finally {
      setUploading(false);
    }
  };

  // 下载检测报告
  const downloadReport = async () => {
    if (!selectedFile || detectionResults.length === 0) return;

    const reportData = {
      文件名: selectedFile.name,
      上传时间: new Date().toLocaleTimeString(),
      检测总数: detectionResults.length,
      垃圾类型分布: getStats().byClass,
      平均置信度: (getStats().avgConfidence * 100).toFixed(1) + '%',
      检测详情: detectionResults,
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `海洋垃圾检测报告_${selectedFile.name.split('.')[0]}_${new Date().getTime()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // 计算统计数据
  const getStats = () => {
    const total = detectionResults.length;
    const byClass = detectionResults.reduce((acc, result) => {
      acc[result.class_name] = (acc[result.class_name] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const avgConfidence = detectionResults.reduce((sum, r) => sum + r.confidence, 0) / total || 0;

    return { total, byClass, avgConfidence };
  };

  const stats = getStats();

  // 图表数据
  const pieData = Object.entries(stats.byClass).map(([name, value]) => ({ name, value }));
  const barData = detectionResults.map((result, index) => ({
    name: `检测${index + 1}`,
    置信度: result.confidence * 100,
  }));

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-4 md:p-6">
      <div className="max-w-7xl mx-auto">
        {/* 返回首页按钮 */}
        <Link to="/">
          <Button variant="ghost" className="mb-4">
            <ChevronLeft className="w-4 h-4 mr-2" />
            返回首页
          </Button>
        </Link>

        {error && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>错误</AlertTitle>
            <AlertDescription className="flex justify-between items-center">
              <span>{error}</span>
              <Button variant="outline" size="sm" onClick={clearError}>
                关闭
              </Button>
            </AlertDescription>
          </Alert>
        )}

        <header className="mb-8">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <ImageIcon className="w-8 h-8 text-blue-600" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900">海洋漂浮垃圾检测</h1>
                <p className="text-gray-600">自动识别和分类漂浮垃圾</p>
              </div>
            </div>
          </div>
          <Separator className="my-4" />
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-6">
            <Card
              className="border-2 border-dashed border-gray-200 hover:border-blue-300 transition-colors">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="w-5 h-5" />
                  上传海洋图片
                </CardTitle>
                <CardDescription>
                  支持 JPG、PNG、WebP 格式，最大支持 10MB
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-center w-full">
                    <label
                      className="flex flex-col items-center justify-center w-full h-64 rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 transition-colors">
                      <div className="flex flex-col items-center justify-center pt-5 pb-6">
                        {uploading ? (
                          <>
                            <div
                              className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
                            <p className="mb-2 text-sm text-gray-500">正在处理图片...</p>
                          </>
                        ) : originalImage ? (
                          <>
                            <img
                              src={originalImage}
                              alt="预览"
                              className="h-40 object-cover rounded-lg mb-2 max-w-full"
                            />
                            <p className="text-sm text-gray-500">点击更换图片</p>
                          </>
                        ) : (
                          <>
                            <Upload className="w-12 h-12 text-gray-400 mb-4" />
                            <p className="mb-2 text-sm text-gray-500">
                              <span className="font-semibold">点击上传</span> 或拖拽图片
                            </p>
                            <p className="text-xs text-gray-400">JPG, PNG, WebP
                              格式，最大10MB</p>
                          </>
                        )}
                      </div>
                      <input
                        ref={fileInputRef}
                        type="file"
                        className="hidden"
                        accept="image/jpeg,image/png,image/jpg,image/webp"
                        onChange={handleImageUpload}
                        disabled={uploading}
                      />
                    </label>
                  </div>

                  {detectionResults.length > 0 && (
                    <div className="grid grid-cols-4 gap-4">
                      <div className="bg-blue-50 p-4 rounded-lg">
                        <p className="text-sm text-blue-600">检测总数</p>
                        <p className="text-2xl font-bold">{stats.total}</p>
                      </div>
                      <div className="bg-green-50 p-4 rounded-lg">
                        <p className="text-sm text-green-600">平均置信度</p>
                        <p className="text-2xl font-bold">{(stats.avgConfidence * 100).toFixed(1)}%</p>
                      </div>
                      <div className="bg-purple-50 p-4 rounded-lg">
                        <p className="text-sm text-purple-600">分类数量</p>
                        <p className="text-2xl font-bold">{Object.keys(stats.byClass).length}</p>
                      </div>
                    </div>
                  )}

                  <div className="flex gap-2 flex-wrap">
                    <Button
                      onClick={() => fileInputRef.current?.click()}
                      disabled={uploading}
                      className="flex-1 min-w-[120px]"
                    >
                      {uploading ? '处理中...' : '上传图片'}
                    </Button>
                    {detectionResults.length > 0 && (
                      <>
                        <Button
                          variant="outline"
                          onClick={reDetectImage}
                          disabled={uploading}
                          className="flex-1 min-w-[120px]"
                        >
                          <RefreshCw className="w-4 h-4 mr-2" />
                          重新检测
                        </Button>
                        <Button
                          variant="outline"
                          onClick={downloadReport}
                          className="flex-1 min-w-[120px]"
                        >
                          <Download className="w-4 h-4 mr-2" />
                          下载报告
                        </Button>
                        <Button
                          variant="destructive"
                          onClick={clearAllData}
                          className="flex-1 min-w-[120px]"
                        >
                          <Trash2 className="w-4 h-4 mr-2" />
                          清除所有
                        </Button>
                      </>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>

                        {detectionResults.length > 0 && (
                            <Card>
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2">
                                        <BarChart3 className="w-5 h-5"/>
                                        检测结果详情
                                    </CardTitle>
                                    <CardDescription>检测到 {detectionResults.length} 个垃圾对象</CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <Tabs defaultValue="table">
                                        <TabsList className="grid w-full grid-cols-3">
                                            <TabsTrigger value="table">列表视图</TabsTrigger>
                                            <TabsTrigger value="chart">类型分布</TabsTrigger>
                                            <TabsTrigger value="confidence">置信度</TabsTrigger>
                                        </TabsList>
                                        <TabsContent value="table" className="space-y-4">
                                            <div className="max-h-96 overflow-auto">
                                                <Table>
                                                    <TableHeader>
                                                        <TableRow>
                                                            <TableHead>垃圾类型</TableHead>
                                                            <TableHead>置信度</TableHead>
                                                            <TableHead>位置</TableHead>
                                                            <TableHead>面积</TableHead>
                                                        </TableRow>
                                                    </TableHeader>
                                                    <TableBody>
                                                        {detectionResults.map((result) => (
                                                            <TableRow key={result.class_id}>
                                                                <TableCell>
                                                                    <Badge className={
                                                                        result.class_name === '塑料' ? 'bg-blue-100 text-blue-800' :
                                                                            result.class_name === '金属' ? 'bg-yellow-100 text-yellow-800' :
                                                                                result.class_name === '玻璃' ? 'bg-green-100 text-green-800' :
                                                                                    'bg-gray-100 text-gray-800'
                                                                    }>
                                                                        {result.class_name}
                                                                    </Badge>
                                                                </TableCell>
                                                                <TableCell>
                                                                    <div className="space-y-1">
                                                                        <div className="flex justify-between text-sm">
                                                                            <span>{(result.confidence * 100).toFixed(1)}%</span>
                                                                        </div>
                                                                        <Progress value={result.confidence * 100}
                                                                                  className="h-2"/>
                                                                    </div>
                                                                </TableCell>
                                                                <TableCell className="text-sm">
                                                                    ({result.bbox.x_min}, {result.bbox.y_min})
                                                                </TableCell>
                                                            </TableRow>
                                                        ))}
                                                    </TableBody>
                                                </Table>
                                            </div>
                                        </TabsContent>
                                        <TabsContent value="chart">
                                            <div className="h-80">
                                                <ResponsiveContainer width="100%" height="100%">
                                                    <PieChart>
                                                        <Pie
                                                            data={pieData}
                                                            cx="50%"
                                                            cy="50%"
                                                            labelLine={false}
                                                            label={({
                                                                        name,
                                                                        percent
                                                                    }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                                                            outerRadius={80}
                                                            fill="#8884d8"
                                                            dataKey="value"
                                                        >
                                                            {pieData.map((_entry, index) => (
                                                                <Cell key={`cell-${index}`}
                                                                      fill={COLORS[index % COLORS.length]}/>
                                                            ))}
                                                        </Pie>
                                                        <Tooltip/>
                                                        <Legend/>
                                                    </PieChart>
                                                </ResponsiveContainer>
                                            </div>
                                        </TabsContent>
                                        <TabsContent value="confidence">
                                            <div className="h-80">
                                                <ResponsiveContainer width="100%" height="100%">
                                                    <BarChart data={barData}>
                                                        <CartesianGrid strokeDasharray="3 3"/>
                                                        <XAxis dataKey="name"/>
                                                        <YAxis label={{
                                                            value: '置信度 (%)',
                                                            angle: -90,
                                                            position: 'insideLeft'
                                                        }}/>
                                                        <Tooltip/>
                                                        <Bar dataKey="置信度" fill="#3b82f6"/>
                                                    </BarChart>
                                                </ResponsiveContainer>
                                            </div>
                                        </TabsContent>
                                    </Tabs>
                                </CardContent>
                            </Card>
                        )}
                    </div>

          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <ImageIcon className="w-5 h-5" />
                  图片对比
                </CardTitle>
                <CardDescription>原始图片与检测结果对比</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <h3 className="text-sm font-medium">原始图片</h3>
                    <div
                      className="border rounded-lg overflow-hidden bg-gray-50 h-64 flex items-center justify-center">
                      {originalImage ? (
                        <img
                          src={originalImage}
                          alt="原始图片"
                          className="w-full h-full object-contain"
                        />
                      ) : (
                        <div className="text-gray-400 text-center p-4">
                          <ImageIcon className="w-12 h-12 mx-auto mb-2" />
                          <p>未上传图片</p>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="space-y-2">
                    <h3 className="text-sm font-medium">检测结果</h3>
                    <div
                      className="border rounded-lg overflow-hidden bg-gray-50 h-64 flex items-center justify-center">
                      {processedImage ? (
                        <img
                          src={processedImage}
                          alt="检测结果"
                          className="w-full h-full object-contain"
                        />
                      ) : uploading ? (
                        <div className="text-center p-4">
                          <div
                            className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                          <p className="text-gray-500">正在分析图片...</p>
                        </div>
                      ) : (
                        <div className="text-gray-400 text-center p-4">
                          <div
                            className="w-12 h-12 mx-auto mb-2 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center">
                            <AlertCircle className="w-6 h-6" />
                          </div>
                          <p>等待检测结果</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

// 统计数据页面组件
const StatisticsPage = () => {
  const [timeRange, setTimeRange] = useState('week');

  // 模拟数据
  const detectionStats = {
    total: 1245,
    today: 42,
    weeklyAvg: 178,
    monthlyAvg: 720
  };

  const classDistribution = [
    { name: '塑料', value: 45, color: '#3b82f6' },
    { name: '金属', value: 20, color: '#f59e0b' },
    { name: '玻璃', value: 15, color: '#10b981' },
    { name: '纸类', value: 12, color: '#8b5cf6' },
    { name: '其他', value: 8, color: '#ef4444' }
  ];

  const dailyData = [
    { date: '01-01', 塑料: 24, 金属: 12, 玻璃: 8, 纸类: 6 },
    { date: '01-02', 塑料: 28, 金属: 14, 玻璃: 9, 纸类: 7 },
    { date: '01-03', 塑料: 32, 金属: 16, 玻璃: 10, 纸类: 8 },
    { date: '01-04', 塑料: 26, 金属: 13, 玻璃: 8, 纸类: 6 },
    { date: '01-05', 塑料: 30, 金属: 15, 玻璃: 9, 纸类: 7 },
  ];

  const confidenceTrend = [
    { date: '01-01', 平均置信度: 92, 最高置信度: 98 },
    { date: '01-02', 平均置信度: 93, 最高置信度: 99 },
    { date: '01-03', 平均置信度: 91, 最高置信度: 97 },
    { date: '01-04', 平均置信度: 94, 最高置信度: 99 },
    { date: '01-05', 平均置信度: 95, 最高置信度: 100 },
  ];

  return (
    <div className="min-h-screen bg-gray-50 p-4 md:p-6">
      <div className="max-w-7xl mx-auto">
        {/* 头部 */}
        <div className="mb-8">
          <Link to="/">
            <Button variant="ghost" className="mb-4">
              <ChevronLeft className="w-4 h-4 mr-2" />
              返回首页
            </Button>
          </Link>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">数据统计</h1>
          <p className="text-gray-600">查看海洋垃圾检测的统计数据和趋势分析</p>
        </div>

        {/* 统计卡片 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">总检测次数</p>
                  <p className="text-3xl font-bold">{detectionStats.total.toLocaleString()}</p>
                </div>
                <div className="p-3 bg-blue-100 rounded-lg">
                  <TrendingUp className="w-6 h-6 text-blue-600" />
                </div>
              </div>
              <div className="mt-4">
                <div className="flex items-center justify-between text-sm">
                  <span>较上周增长</span>
                  <span className="text-green-600 flex items-center">
                    <TrendingUp className="w-4 h-4 mr-1" />
                    12.5%
                  </span>
                </div>
                <Progress value={65} className="h-2 mt-2" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">今日检测</p>
                  <p className="text-3xl font-bold">{detectionStats.today}</p>
                </div>
                <div className="p-3 bg-green-100 rounded-lg">
                  <Calendar className="w-6 h-6 text-green-600" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">周平均检测</p>
                  <p className="text-3xl font-bold">{detectionStats.weeklyAvg}</p>
                </div>
                <div className="p-3 bg-purple-100 rounded-lg">
                  <TrendingUp className="w-6 h-6 text-purple-600" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">平均置信度</p>
                  <p className="text-3xl font-bold">94.2%</p>
                </div>
                <div className="p-3 bg-orange-100 rounded-lg">
                  <TrendingUp className="w-6 h-6 text-orange-600" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* 图表区域 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>垃圾类别分布</CardTitle>
                  <CardDescription>各类垃圾的检测数量占比</CardDescription>
                </div>
                <Select value={timeRange} onValueChange={setTimeRange}>
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="选择时间范围" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="week">最近一周</SelectItem>
                    <SelectItem value="month">最近一月</SelectItem>
                    <SelectItem value="quarter">最近三月</SelectItem>
                    <SelectItem value="year">最近一年</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardHeader>
            <CardContent>
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={classDistribution}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {classDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>每日检测趋势</CardTitle>
              <CardDescription>近期垃圾检测数量变化</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={dailyData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="塑料" fill="#3b82f6" />
                    <Bar dataKey="金属" fill="#f59e0b" />
                    <Bar dataKey="玻璃" fill="#10b981" />
                    <Bar dataKey="纸类" fill="#8b5cf6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </div>

        <Card className="mb-8">
          <CardHeader>
            <CardTitle>置信度趋势</CardTitle>
            <CardDescription>检测置信度随时间变化</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={confidenceTrend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis label={{ value: '置信度 (%)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="平均置信度" stroke="#3b82f6" strokeWidth={2} activeDot={{ r: 8 }} />
                  <Line type="monotone" dataKey="最高置信度" stroke="#10b981" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

// 数据管理页面组件
const DataManagementPage = () => {
  const [selectedRows, setSelectedRows] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState('');

  // 模拟数据
  const data = [
    {
      id: '1',
      filename: 'ocean_001.jpg',
      date: '2024-01-05 14:30',
      type: '塑料',
      confidence: 96,
      count: 8,
      size: '2.4 MB',
      status: '已处理'
    },
    {
      id: '2',
      filename: 'ocean_002.png',
      date: '2024-01-05 10:15',
      type: '金属',
      confidence: 89,
      count: 3,
      size: '3.1 MB',
      status: '已处理'
    },
    {
      id: '3',
      filename: 'ocean_003.jpg',
      date: '2024-01-04 16:45',
      type: '玻璃',
      confidence: 92,
      count: 5,
      size: '1.8 MB',
      status: '已处理'
    },
    {
      id: '4',
      filename: 'ocean_004.webp',
      date: '2024-01-04 09:20',
      type: '塑料',
      confidence: 94,
      count: 12,
      size: '4.2 MB',
      status: '已处理'
    },
    {
      id: '5',
      filename: 'ocean_005.jpg',
      date: '2024-01-03 11:10',
      type: '其他',
      confidence: 87,
      count: 2,
      size: '2.9 MB',
      status: '已处理'
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50 p-4 md:p-6">
      <div className="max-w-7xl mx-auto">
        {/* 头部 */}
        <div className="mb-8">
          <Link to="/">
            <Button variant="ghost" className="mb-4">
              <ChevronLeft className="w-4 h-4 mr-2" />
              返回首页
            </Button>
          </Link>
          <div className="flex flex-col md:flex-row md:items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">数据管理</h1>
              <p className="text-gray-600">管理已检测的图片和检测记录</p>
            </div>
            <div className="flex items-center space-x-2 mt-4 md:mt-0">
              <Button variant="outline">
                <Download className="w-4 h-4 mr-2" />
                导出数据
              </Button>
              <Button variant="outline" className="text-red-600 hover:text-red-700 hover:bg-red-50">
                <Trash2 className="w-4 h-4 mr-2" />
                批量删除
              </Button>
            </div>
          </div>
        </div>

        {/* 搜索和过滤 */}
        <Card className="mb-6">
          <CardContent className="p-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="flex-1">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                  <Input
                    placeholder="搜索文件名或类型..."
                    className="pl-10"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <Select defaultValue="all">
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="文件类型" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">所有类型</SelectItem>
                    <SelectItem value="plastic">塑料</SelectItem>
                    <SelectItem value="metal">金属</SelectItem>
                    <SelectItem value="glass">玻璃</SelectItem>
                    <SelectItem value="paper">纸类</SelectItem>
                  </SelectContent>
                </Select>
                <Select defaultValue="latest">
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="排序方式" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="latest">最新上传</SelectItem>
                    <SelectItem value="oldest">最早上传</SelectItem>
                    <SelectItem value="name">文件名</SelectItem>
                    <SelectItem value="count">检测数量</SelectItem>
                  </SelectContent>
                </Select>
                <Button variant="outline">
                  <Filter className="w-4 h-4 mr-2" />
                  更多筛选
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 数据表格 */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>检测记录</CardTitle>
                <CardDescription>共 {data.length} 条记录</CardDescription>
              </div>
              <div className="flex items-center space-x-2">
                <Checkbox
                  checked={selectedRows.length === data.length}
                  onCheckedChange={(checked) => {
                    if (checked) {
                      setSelectedRows(data.map(item => item.id));
                    } else {
                      setSelectedRows([]);
                    }
                  }}
                />
                <span className="text-sm text-gray-600">全选</span>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[50px]">选择</TableHead>
                    <TableHead>文件名</TableHead>
                    <TableHead>上传时间</TableHead>
                    <TableHead>主要类型</TableHead>
                    <TableHead>置信度</TableHead>
                    <TableHead>检测数量</TableHead>
                    <TableHead>文件大小</TableHead>
                    <TableHead>状态</TableHead>
                    <TableHead className="text-right">操作</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.map((item) => (
                    <TableRow key={item.id}>
                      <TableCell>
                        <Checkbox
                          checked={selectedRows.includes(item.id)}
                          onCheckedChange={(checked) => {
                            if (checked) {
                              setSelectedRows([...selectedRows, item.id]);
                            } else {
                              setSelectedRows(selectedRows.filter(id => id !== item.id));
                            }
                          }}
                        />
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center">
                          <ImageIcon className="w-5 h-5 text-gray-400 mr-2" />
                          <span className="font-medium">{item.filename}</span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center">
                          <Calendar className="w-4 h-4 text-gray-400 mr-2" />
                          {item.date}
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge className={
                          item.type === '塑料' ? 'bg-blue-100 text-blue-800' :
                            item.type === '金属' ? 'bg-yellow-100 text-yellow-800' :
                              item.type === '玻璃' ? 'bg-green-100 text-green-800' :
                                'bg-gray-100 text-gray-800'
                        }>
                          {item.type}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center">
                          <span className="font-medium">{item.confidence}%</span>
                          <div className="ml-2 w-20">
                            <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                              <div
                                className={`h-full ${item.confidence >= 90 ? 'bg-green-500' :
                                    item.confidence >= 80 ? 'bg-yellow-500' : 'bg-red-500'
                                  }`}
                                style={{ width: `${item.confidence}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        <span className="font-medium">{item.count} 个</span>
                      </TableCell>
                      <TableCell className="text-gray-500">{item.size}</TableCell>
                      <TableCell>
                        <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                          {item.status}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex justify-end space-x-2">
                          <Button size="sm" variant="ghost">
                            <Eye className="w-4 h-4" />
                          </Button>
                          <Button size="sm" variant="ghost">
                            <FileText className="w-4 h-4" />
                          </Button>
                          <Button size="sm" variant="ghost" className="text-red-600 hover:text-red-700">
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

// 设置页面组件
const SettingsPage = () => {
  return (
    <div className="min-h-screen bg-gray-50 p-4 md:p-6">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <Link to="/">
            <Button variant="ghost" className="mb-4">
              <ChevronLeft className="w-4 h-4 mr-2" />
              返回首页
            </Button>
          </Link>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">系统设置</h1>
          <p className="text-gray-600">配置系统参数和个性化选项</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-2 space-y-6">
            {/* 通用设置 */}
            <Card>
              <CardHeader>
                <CardTitle>通用设置</CardTitle>
                <CardDescription>系统通用配置选项</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="api-endpoint">API 端点</Label>
                  <Input
                    id="api-endpoint"
                    placeholder="http://localhost:8000"
                    defaultValue="http://localhost:8000"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="language">语言设置</Label>
                  <Select defaultValue="zh">
                    <SelectTrigger>
                      <SelectValue placeholder="选择语言" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="zh">简体中文</SelectItem>
                      <SelectItem value="en">English</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="theme">主题模式</Label>
                  <Select defaultValue="light">
                    <SelectTrigger>
                      <SelectValue placeholder="选择主题" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="light">浅色模式</SelectItem>
                      <SelectItem value="dark">深色模式</SelectItem>
                      <SelectItem value="auto">跟随系统</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>

            {/* 检测设置 */}
            <Card>
              <CardHeader>
                <CardTitle>检测设置</CardTitle>
                <CardDescription>模型和检测参数配置</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>自动保存检测结果</Label>
                    <p className="text-sm text-gray-500">检测完成后自动保存结果到本地</p>
                  </div>
                  <Switch defaultChecked />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>显示置信度阈值</Label>
                    <p className="text-sm text-gray-500">低于阈值的检测结果将被隐藏</p>
                  </div>
                  <div className="w-32">
                    <Input type="number" defaultValue="0.5" min="0" max="1" step="0.1" />
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>使用GPU加速</Label>
                    <p className="text-sm text-gray-500">启用GPU加速提高检测速度</p>
                  </div>
                  <Switch />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* 侧边栏 */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>系统信息</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="text-sm text-gray-500">版本</div>
                  <div className="font-medium">v1.0.0</div>
                </div>
                <Separator />
                <div className="space-y-2">
                  <div className="text-sm text-gray-500">最后更新</div>
                  <div className="font-medium">2024-01-05</div>
                </div>
                <Separator />
                <div className="space-y-2">
                  <div className="text-sm text-gray-500">支持格式</div>
                  <div className="font-medium">JPG, PNG, WebP</div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>快速操作</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button variant="outline" className="w-full justify-start">
                  导出所有设置
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  导入设置
                </Button>
                <Separator />
                <Button variant="destructive" className="w-full justify-start">
                  恢复默认设置
                </Button>
                <Button className="w-full justify-start">
                  保存设置
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

// 主应用组件
const App = () => {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        {/* 导航栏 */}
        <nav className="bg-white shadow-sm border-b sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center">
                <Link to="/" className="flex items-center space-x-2">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-r from-blue-600 to-teal-600"></div>
                  <span className="text-xl font-bold text-gray-900">海洋垃圾检测</span>
                </Link>

                <div className="hidden md:block ml-10">
                  <div className="flex items-baseline space-x-4">
                    <Link to="/">
                      <Button variant="ghost" className="flex items-center">
                        <HomeIcon className="w-4 h-4 mr-2" />
                        首页
                      </Button>
                    </Link>
                    <Link to="/detect">
                      <Button variant="ghost" className="flex items-center">
                        <Upload className="w-4 h-4 mr-2" />
                        检测
                      </Button>
                    </Link>
                    <Link to="/statistics">
                      <Button variant="ghost" className="flex items-center">
                        <BarChart3 className="w-4 h-4 mr-2" />
                        统计
                      </Button>
                    </Link>
                    <Link to="/data">
                      <Button variant="ghost" className="flex items-center">
                        <Database className="w-4 h-4 mr-2" />
                        数据
                      </Button>
                    </Link>
                  </div>
                </div>
              </div>

              <div className="flex items-center space-x-4">
                <Link to="/settings">
                  <Button variant="ghost" size="icon">
                    <Settings className="w-5 h-5" />
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        </nav>

        {/* 路由内容 */}
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/detect" element={<DetectPage />} />
          <Route path="/statistics" element={<StatisticsPage />} />
          <Route path="/data" element={<DataManagementPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;