import {BrowserRouter as Router, Routes, Route, Link, Navigate} from 'react-router-dom';
import {Button} from '@/components/ui/button';
import {HomeIcon, Upload, BarChart3, Database, Settings} from 'lucide-react';

import Detector from './Detector';

// 主页组件
const HomePage = () => {
  const features = [
    {
      icon: <Upload className="w-6 h-6"/>,
      title: "智能检测",
      description: "基于深度学习模型，精准识别多种海洋垃圾类型"
    },
    {
      icon: <BarChart3 className="w-6 h-6"/>,
      title: "数据分析",
      description: "可视化分析垃圾分布和统计数据"
    },
    {
      icon: <Database className="w-6 h-6"/>,
      title: "数据管理",
      description: "历史记录存储和管理功能"
    }
  ];

  return (
    <div className="min-h-screen bg-linear-to-br from-gray-50 to-blue-50">
      {/* Hero Section */}
      <div className="relative overflow-hidden bg-linear-to-r from-blue-600 to-teal-600 text-white">
        <div className="absolute inset-0 bg-black opacity-20"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className="text-center">
            <h1 className="text-4xl md:text-6xl font-bold mb-6">
              海洋漂浮垃圾智能检测系统
            </h1>
            <p className="text-xl md:text-2xl mb-8 text-blue-100">
              利用人工智能技术保护海洋生态环境
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link to="/detect">
                <Button size="lg" className="bg-white text-blue-600 hover:bg-blue-50 px-8">
                  <Upload className="w-5 h-5 mr-2"/>
                  开始检测
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Features */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">核心功能</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
          {features.map((feature, index) => (
            <div key={index} className="bg-white rounded-xl shadow-md p-6 hover:shadow-lg transition-shadow">
              <div className="w-12 h-12 rounded-lg bg-blue-100 text-blue-600 flex items-center justify-center mb-4">
                {feature.icon}
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">{feature.title}</h3>
              <p className="text-gray-600 text-sm">{feature.description}</p>
            </div>
          ))}
        </div>

        {/* Stats */}
        <div className="bg-linear-to-r from-blue-50 to-teal-50 rounded-2xl p-8 mb-16">
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
            <Button size="lg"
                    className="bg-linear-to-r from-blue-600 to-teal-600 hover:from-blue-700 hover:to-teal-700 px-12">
              开始使用
            </Button>
          </Link>
        </div>
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
              返回首页
            </Button>
          </Link>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">系统设置</h1>
          <p className="text-gray-600">配置系统参数和个性化选项</p>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="space-y-4">
            <div>
              <h3 className="text-lg font-medium mb-2">API 端点设置</h3>
              <p className="text-sm text-gray-500 mb-2">配置后端服务地址</p>
              <input
                type="text"
                defaultValue="http://localhost:8000"
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
              />
            </div>

            <div className="pt-4 border-t">
              <h3 className="text-lg font-medium mb-2">检测参数</h3>
              <div className="space-y-2">
                <label className="flex items-center space-x-2">
                  <input type="checkbox" defaultChecked className="rounded"/>
                  <span>自动保存检测结果</span>
                </label>
                <label className="flex items-center space-x-2">
                  <input type="checkbox" className="rounded"/>
                  <span>使用GPU加速</span>
                </label>
              </div>
            </div>

            <div className="pt-4 border-t">
              <Button className="w-full">保存设置</Button>
            </div>
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
                  <div className="w-8 h-8 rounded-lg bg-linear-to-r from-blue-600 to-teal-600"></div>
                  <span className="text-xl font-bold text-gray-900">海洋垃圾检测</span>
                </Link>

                <div className="hidden md:block ml-10">
                  <div className="flex items-baseline space-x-4">
                    <Link to="/">
                      <Button variant="ghost" className="flex items-center">
                        <HomeIcon className="w-4 h-4 mr-2"/>
                        首页
                      </Button>
                    </Link>
                    <Link to="/detect">
                      <Button variant="ghost" className="flex items-center">
                        <Upload className="w-4 h-4 mr-2"/>
                        检测
                      </Button>
                    </Link>
                    <Link to="/settings">
                      <Button variant="ghost" className="flex items-center">
                        <Settings className="w-4 h-4 mr-2"/>
                        设置
                      </Button>
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </nav>

        {/* 路由内容 */}
        <Routes>
          <Route path="/" element={<HomePage/>}/>
          <Route path="/detect" element={<Detector/>}/>
          <Route path="/settings" element={<SettingsPage/>}/>
          <Route path="*" element={<Navigate to="/" replace/>}/>
        </Routes>
      </div>
    </Router>
  );
};

export default App;