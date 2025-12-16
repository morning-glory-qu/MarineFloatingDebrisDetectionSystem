// frontend/water-garbage-detector/src/App.tsx
import React, {useState} from 'react';
import {Card, CardContent, CardDescription, CardHeader, CardTitle} from '@/components/ui/card';
import {Button} from '@/components/ui/button';
import {Tabs, TabsContent, TabsList, TabsTrigger} from '@/components/ui/tabs';
import {Table, TableBody, TableCell, TableHead, TableHeader, TableRow} from '@/components/ui/table';
import {Progress} from '@/components/ui/progress';
import {Badge} from '@/components/ui/badge';
import {Alert, AlertDescription, AlertTitle} from '@/components/ui/alert';
import {Separator} from '@/components/ui/separator';
import {Skeleton} from '@/components/ui/skeleton';
import {Upload, Image as ImageIcon, BarChart3, Trash2, Download, AlertCircle} from 'lucide-react';
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

// 类型定义
interface DetectionResult {
    id: number;
    className: string;
    confidence: number;
    bbox: {
        x_min: number;
        y_min: number;
        x_max: number;
        y_max: number;
    };
    area: number;
}

interface StatisticsData {
    date: string;
    plastic: number;
    metal: number;
    glass: number;
    paper: number;
    other: number;
}

const App = () => {
    const [uploading, setUploading] = useState(false);
    const [detectionResults, setDetectionResults] = useState<DetectionResult[]>([]);
    const [originalImage, setOriginalImage] = useState<string | null>(null);
    const [processedImage, setProcessedImage] = useState<string | null>(null);
    const [statistics, setStatistics] = useState<StatisticsData[]>([]);

    // 模拟上传图片
    const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setUploading(true);
        setOriginalImage(URL.createObjectURL(file));

        // 模拟API调用延迟
        setTimeout(() => {
            // 模拟检测结果
            const mockResults: DetectionResult[] = [
                {
                    id: 1,
                    className: '塑料',
                    confidence: 0.95,
                    bbox: {x_min: 100, y_min: 150, x_max: 200, y_max: 250},
                    area: 10000
                },
                {
                    id: 2,
                    className: '金属',
                    confidence: 0.87,
                    bbox: {x_min: 300, y_min: 200, x_max: 350, y_max: 280},
                    area: 5000
                },
                {
                    id: 3,
                    className: '塑料',
                    confidence: 0.92,
                    bbox: {x_min: 400, y_min: 100, x_max: 450, y_max: 180},
                    area: 4000
                },
                {
                    id: 4,
                    className: '玻璃',
                    confidence: 0.78,
                    bbox: {x_min: 200, y_min: 300, x_max: 250, y_max: 350},
                    area: 2500
                },
            ];

            // 模拟统计数据
            const mockStats: StatisticsData[] = [
                {date: '01-01', plastic: 12, metal: 5, glass: 3, paper: 2, other: 1},
                {date: '01-02', plastic: 15, metal: 6, glass: 4, paper: 3, other: 2},
                {date: '01-03', plastic: 18, metal: 7, glass: 5, paper: 2, other: 3},
                {date: '01-04', plastic: 20, metal: 8, glass: 6, paper: 4, other: 2},
                {date: '01-05', plastic: 22, metal: 9, glass: 7, paper: 3, other: 4},
            ];

            setDetectionResults(mockResults);
            setProcessedImage('https://via.placeholder.com/600x400/3b82f6/ffffff?text=Processed+Image+with+Detections');
            setStatistics(mockStats);
            setUploading(false);
        }, 1500);
    };

    // 计算统计数据
    const getStats = () => {
        const total = detectionResults.length;
        const byClass = detectionResults.reduce((acc, result) => {
            acc[result.className] = (acc[result.className] || 0) + 1;
            return acc;
        }, {} as Record<string, number>);

        const avgConfidence = detectionResults.reduce((sum, r) => sum + r.confidence, 0) / total || 0;
        const totalArea = detectionResults.reduce((sum, r) => sum + r.area, 0);

        return {total, byClass, avgConfidence, totalArea};
    };

    const stats = getStats();

    // 图表数据
    const pieData = Object.entries(stats.byClass).map(([name, value]) => ({name, value}));
    const barData = detectionResults.map((result, index) => ({
        name: `检测${index + 1}`,
        置信度: result.confidence * 100,
    }));

    const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

    // @ts-ignore
    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-4 md:p-6">
            <div className="max-w-7xl mx-auto">
                {/* 头部 */}
                <header className="mb-8">
                    <div className="flex items-center gap-3 mb-2">
                        <div className="p-2 bg-blue-100 rounded-lg">
                            <ImageIcon className="w-8 h-8 text-blue-600"/>
                        </div>
                        <div>
                            <h1 className="text-3xl font-bold text-gray-900">海洋漂浮垃圾检测系统</h1>
                            <p className="text-gray-600">上传海洋图像，自动识别和分类漂浮垃圾，助力海洋环境保护</p>
                        </div>
                    </div>
                    <Separator className="my-4"/>
                </header>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* 左侧 - 上传和检测区 */}
                    <div className="space-y-6">
                        {/* 上传卡片 */}
                        <Card
                            className="border-2 border-dashed border-gray-200 hover:border-blue-300 transition-colors">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Upload className="w-5 h-5"/>
                                    上传海洋图片
                                </CardTitle>
                                <CardDescription>
                                    支持 JPG、PNG 格式，图片大小不超过 10MB
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
                                                        <p className="mb-2 text-sm text-gray-500">分析中...</p>
                                                    </>
                                                ) : originalImage ? (
                                                    <>
                                                        <img src={originalImage} alt="预览"
                                                             className="h-40 object-cover rounded-lg mb-2"/>
                                                        <p className="text-sm text-gray-500">点击更换图片</p>
                                                    </>
                                                ) : (
                                                    <>
                                                        <Upload className="w-12 h-12 text-gray-400 mb-4"/>
                                                        <p className="mb-2 text-sm text-gray-500">
                                                            <span className="font-semibold">点击上传</span> 或拖拽图片
                                                        </p>
                                                        <p className="text-xs text-gray-400">PNG, JPG, GIF 格式</p>
                                                    </>
                                                )}
                                            </div>
                                            <input
                                                type="file"
                                                className="hidden"
                                                accept="image/*"
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
                                            <div className="bg-yellow-50 p-4 rounded-lg">
                                                <p className="text-sm text-yellow-600">垃圾面积</p>
                                                <p className="text-2xl font-bold">{(stats.totalArea / 1000).toFixed(1)}k</p>
                                            </div>
                                            <div className="bg-purple-50 p-4 rounded-lg">
                                                <p className="text-sm text-purple-600">分类数量</p>
                                                <p className="text-2xl font-bold">{Object.keys(stats.byClass).length}</p>
                                            </div>
                                        </div>
                                    )}

                                    <div className="flex gap-2">
                                        <Button
                                            onClick={() => (document.querySelector('input[type="file"]') as HTMLInputElement)?.click()}>
                                            disabled={uploading}
                                            {uploading ? '分析中...' : '上传图片'}
                                        </Button>
                                        {detectionResults.length > 0 && (
                                            <Button variant="outline">
                                                <Download className="w-4 h-4 mr-2"/>
                                                下载报告
                                            </Button>
                                        )}
                                    </div>
                                </div>
                            </CardContent>
                        </Card>

                        {/* 结果展示卡片 */}
                        {detectionResults.length > 0 && (
                            <Card>
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2">
                                        <BarChart3 className="w-5 h-5"/>
                                        检测结果详情
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <Tabs defaultValue="table">
                                        <TabsList className="grid w-full grid-cols-3">
                                            <TabsTrigger value="table">列表视图</TabsTrigger>
                                            <TabsTrigger value="chart">类型分布</TabsTrigger>
                                            <TabsTrigger value="confidence">置信度</TabsTrigger>
                                        </TabsList>
                                        <TabsContent value="table" className="space-y-4">
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
                                                        <TableRow key={result.id}>
                                                            <TableCell>
                                                                <Badge className={
                                                                    result.className === '塑料' ? 'bg-blue-100 text-blue-800' :
                                                                        result.className === '金属' ? 'bg-yellow-100 text-yellow-800' :
                                                                            result.className === '玻璃' ? 'bg-green-100 text-green-800' :
                                                                                'bg-gray-100 text-gray-800'
                                                                }>
                                                                    {result.className}
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
                                                            <TableCell>{result.area} px²</TableCell>
                                                        </TableRow>
                                                    ))}
                                                </TableBody>
                                            </Table>
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

                    {/* 右侧 - 图片和统计区 */}
                    <div className="space-y-6">
                        {/* 图片对比卡片 */}
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
                                        <div
                                            className="border rounded-lg overflow-hidden bg-gray-50 h-64 flex items-center justify-center">
                                            {originalImage ? (
                                                <img src={originalImage} alt="原始图片"
                                                     className="w-full h-full object-contain"/>
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
                                        <div
                                            className="border rounded-lg overflow-hidden bg-gray-50 h-64 flex items-center justify-center">
                                            {processedImage ? (
                                                <img src={processedImage} alt="检测结果"
                                                     className="w-full h-full object-contain"/>
                                            ) : (
                                                <div className="text-gray-400 text-center p-4">
                                                    <div
                                                        className="w-12 h-12 mx-auto mb-2 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center">
                                                        <AlertCircle className="w-6 h-6"/>
                                                    </div>
                                                    <p>等待分析结果</p>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>

                        {/* 统计趋势卡片 */}
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <BarChart3 className="w-5 h-5"/>
                                    检测趋势统计
                                </CardTitle>
                                <CardDescription>最近检测数据趋势分析</CardDescription>
                            </CardHeader>
                            <CardContent>
                                {statistics.length > 0 ? (
                                    <div className="h-80">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <LineChart data={statistics}>
                                                <CartesianGrid strokeDasharray="3 3"/>
                                                <XAxis dataKey="date"/>
                                                <YAxis/>
                                                <Tooltip/>
                                                <Legend/>
                                                <Line type="monotone" dataKey="plastic" stroke="#3b82f6" name="塑料"
                                                      strokeWidth={2}/>
                                                <Line type="monotone" dataKey="metal" stroke="#f59e0b" name="金属"
                                                      strokeWidth={2}/>
                                                <Line type="monotone" dataKey="glass" stroke="#10b981" name="玻璃"
                                                      strokeWidth={2}/>
                                                <Line type="monotone" dataKey="paper" stroke="#8b5cf6" name="纸张"
                                                      strokeWidth={2}/>
                                            </LineChart>
                                        </ResponsiveContainer>
                                    </div>
                                ) : (
                                    <div className="space-y-4">
                                        <Skeleton className="h-60 w-full"/>
                                        <div className="space-y-2">
                                            <Skeleton className="h-4 w-full"/>
                                            <Skeleton className="h-4 w-3/4"/>
                                        </div>
                                    </div>
                                )}
                            </CardContent>
                        </Card>

                        {/* 环保提示卡片 */}
                        <Alert>
                            <Trash2 className="h-4 w-4"/>
                            <AlertTitle>环保小贴士</AlertTitle>
                            <AlertDescription className="space-y-2">
                                <p>检测到的海洋垃圾中，塑料制品占比最高，占所有海洋垃圾的80%以上。</p>
                                <p className="text-sm text-gray-600">保护海洋环境，从减少使用一次性塑料制品开始。</p>
                            </AlertDescription>
                        </Alert>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default App;