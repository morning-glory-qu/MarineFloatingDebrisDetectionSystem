import React, {useState, useRef, useCallback} from 'react';
import {Link} from 'react-router-dom';
import {Card, CardContent, CardDescription, CardHeader, CardTitle} from '@/components/ui/card';
import {Button} from '@/components/ui/button';
import {Tabs, TabsContent, TabsList, TabsTrigger} from '@/components/ui/tabs';
import {Table, TableBody, TableCell, TableHead, TableHeader, TableRow} from '@/components/ui/table';
import {Progress} from '@/components/ui/progress';
import {Badge} from '@/components/ui/badge';
import {Alert, AlertDescription, AlertTitle} from '@/components/ui/alert';
import {Separator} from '@/components/ui/separator';
import {
  Upload,
  Image as ImageIcon,
  BarChart3,
  Trash2,
  Download,
  AlertCircle,
  RefreshCw,
  ChevronLeft
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
  ResponsiveContainer
} from 'recharts';

interface DetectionResult {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
}

interface ApiResponse {
  success: boolean;
  data: {
    image: string; // 完整的data URL: "data:image/jpeg;base64,..."
    detections: DetectionResult[];
  };
}

const Detector = () => {
  const [uploading, setUploading] = useState(false);
  const [detectionResults, setDetectionResults] = useState<DetectionResult[]>([]);
  const [originalImage, setOriginalImage] = useState<string>('');
  const [processedImage, setProcessedImage] = useState<string>('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);

  // API基础URL
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
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

    const allowedTypes = ["image/jpeg", "image/png", "image/jpg", "image/webp"];
    if (!allowedTypes.includes(file.type)) {
      throw new Error("仅支持 JPG、PNG、WebP 格式的图片");
    }
  };

  // 上传图片到后端API
  const handleImageUpload = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
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
        reader.onerror = (error) => reject(error);
        reader.readAsDataURL(file);
      });

      console.log("发送数据格式:", {
        fileType: file.type,
        fileSize: file.size,
        base64Prefix: base64Image.substring(0, 30),
      });

      // 发送完整格式的Base64给后端
      const response = await fetch(`${API_BASE_URL}/detect`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: base64Image, // 发送完整格式：data:image/jpeg;base64,xxx
        }),
      });

      console.log("响应状态:", response.status);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`检测失败: ${response.status} - ${errorText}`);
      }

      const data: ApiResponse = await response.json();

      // 根据后端响应结构处理
      if (data.success && data.data) {
        setDetectionResults(data.data.detections || []);
        console.log(data.data.detections)

        // 后端已经返回完整格式，直接使用
        if (data.data.image) {
          setProcessedImage(data.data.image);
          console.log(
            "接收到的图片数据格式:",
            data.data.image.substring(0, 30),
          );
        }
      } else {
        throw new Error("后端返回数据格式错误");
      }
    } catch (error) {
      console.error("检测失败详情:", error);
      const errorMessage = error instanceof Error ? error.message : "未知错误";
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
        reader.onerror = (error) => reject(error);
        reader.readAsDataURL(selectedFile);
      });

      const response = await fetch(`${API_BASE_URL}/detect`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: base64Image,
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
        throw new Error("后端返回数据格式错误");
      }
    } catch (error) {
      console.error("重新检测失败:", error);
      setError("重新检测失败，请检查后端服务");
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
      平均置信度: (getStats().avgConfidence * 100).toFixed(1) + "%",
      检测详情: detectionResults,
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `海洋垃圾检测报告_${
      selectedFile.name.split(".")[0]
    }_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  // 计算统计数据
  const getStats = () => {
    const total = detectionResults.length;
    const byClass = detectionResults.reduce((acc, result) => {
      acc[result.class_name] = (acc[result.class_name] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const avgConfidence =
      detectionResults.reduce((sum, r) => sum + r.confidence, 0) / total || 0;

    return {total, byClass, avgConfidence};
  };

  const stats = getStats();

  // 图表数据
  const pieData = Object.entries(stats.byClass).map(([name, value]) => ({
    name,
    value,
  }));
  const barData = detectionResults.map((result, index) => ({
    name: `检测${index + 1}`,
    置信度: result.confidence * 100,
  }));

  const COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"];

  return (
    <div className="min-h-screen bg-linear-to-br from-gray-50 to-gray-100 p-4 md:p-6">
      <div className="max-w-7xl mx-auto">
        {/* 返回首页按钮 */}
        <Link to="/">
          <Button variant="ghost" className="mb-4">
            <ChevronLeft className="w-4 h-4 mr-2"/>
            返回首页
          </Button>
        </Link>

        {error && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4"/>
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
                <ImageIcon className="w-8 h-8 text-blue-600"/>
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900">
                  海洋漂浮垃圾检测
                </h1>
                <p className="text-gray-600">自动识别和分类漂浮垃圾</p>
              </div>
            </div>
          </div>
          <Separator className="my-4"/>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-6">
            <Card className="border-2 border-dashed border-gray-200 hover:border-blue-300 transition-colors">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="w-5 h-5"/>
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
                            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
                            <p className="mb-2 text-sm text-gray-500">
                              正在处理图片...
                            </p>
                          </>
                        ) : originalImage ? (
                          <>
                            <img
                              src={originalImage}
                              alt="预览"
                              className="h-40 object-cover rounded-lg mb-2 max-w-full"
                            />
                            <p className="text-sm text-gray-500">
                              点击更换图片
                            </p>
                          </>
                        ) : (
                          <>
                            <Upload className="w-12 h-12 text-gray-400 mb-4"/>
                            <p className="mb-2 text-sm text-gray-500">
                              <span className="font-semibold">点击上传</span>{" "}
                              或拖拽图片
                            </p>
                            <p className="text-xs text-gray-400">
                              JPG, PNG, WebP 格式，最大10MB
                            </p>
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
                        <p className="text-2xl font-bold">
                          {(stats.avgConfidence * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div className="bg-purple-50 p-4 rounded-lg">
                        <p className="text-sm text-purple-600">分类数量</p>
                        <p className="text-2xl font-bold">
                          {Object.keys(stats.byClass).length}
                        </p>
                      </div>
                    </div>
                  )}

                  <div className="flex gap-2 flex-wrap">
                    <Button
                      onClick={() => fileInputRef.current?.click()}
                      disabled={uploading}
                      className="flex-1 min-w-30"
                    >
                      {uploading ? "处理中..." : "上传图片"}
                    </Button>
                    {detectionResults.length > 0 && (
                      <>
                        <Button
                          variant="outline"
                          onClick={reDetectImage}
                          disabled={uploading}
                          className="flex-1 min-w-30"
                        >
                          <RefreshCw className="w-4 h-4 mr-2"/>
                          重新检测
                        </Button>
                        <Button
                          variant="outline"
                          onClick={downloadReport}
                          className="flex-1 min-w-30"
                        >
                          <Download className="w-4 h-4 mr-2"/>
                          下载报告
                        </Button>
                        <Button
                          variant="destructive"
                          onClick={clearAllData}
                          className="flex-1 min-w-30"
                        >
                          <Trash2 className="w-4 h-4 mr-2"/>
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
                  <CardDescription>
                    检测到 {detectionResults.length} 个垃圾对象
                  </CardDescription>
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
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {detectionResults.map((result) => (
                              <TableRow key={result.class_id}>
                                <TableCell>
                                  <Badge
                                    className={
                                      result.class_name === "塑料"
                                        ? "bg-blue-100 text-blue-800"
                                        : result.class_name === "金属"
                                          ? "bg-yellow-100 text-yellow-800"
                                          : result.class_name === "玻璃"
                                            ? "bg-green-100 text-green-800"
                                            : "bg-gray-100 text-gray-800"
                                    }
                                  >
                                    {result.class_name}
                                  </Badge>
                                </TableCell>
                                <TableCell>
                                  <div className="space-y-1">
                                    <div className="flex justify-between text-sm">
                                      <span>
                                        {(result.confidence * 100).toFixed(1)}%
                                      </span>
                                    </div>
                                    <Progress
                                      value={result.confidence * 100}
                                      className="h-2"
                                    />
                                  </div>
                                </TableCell>
                                <TableCell className="text-sm">
                                  ({Number(result.bbox[0])}, {Number(result.bbox[1])}) -
                                  ({Number(result.bbox[2])}, {Number(result.bbox[3])})
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
                              label={({name, percent}) =>
                                `${name}: ${((percent || 0) * 100).toFixed(0)}%`
                              }
                              outerRadius={80}
                              fill="#8884d8"
                              dataKey="value"
                            >
                              {pieData.map((entry, index) => (
                                <Cell
                                  key={`cell-${entry.name}-${entry.value}`}
                                  fill={COLORS[index % COLORS.length]}
                                />
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
                            <YAxis
                              label={{
                                value: "置信度 (%)",
                                angle: -90,
                                position: "insideLeft",
                              }}
                            />
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
                  <ImageIcon className="w-5 h-5"/>
                  图片对比
                </CardTitle>
                <CardDescription>原始图片与检测结果对比</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <h3 className="text-sm font-medium">原始图片</h3>
                    <div className="border rounded-lg overflow-hidden bg-gray-50 h-64 flex items-center justify-center">
                      {originalImage ? (
                        <img
                          src={originalImage}
                          alt="原始图片"
                          className="w-full h-full object-contain"
                        />
                      ) : (
                        <div className="text-gray-400 text-center p-4">
                          <ImageIcon className="w-12 h-12 mx-auto mb-2"/>
                          <p>未上传图片</p>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="space-y-2">
                    <h3 className="text-sm font-medium">检测结果</h3>
                    <div className="border rounded-lg overflow-hidden bg-gray-50 h-64 flex items-center justify-center">
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
                            <AlertCircle className="w-6 h-6"/>
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

export default Detector;
