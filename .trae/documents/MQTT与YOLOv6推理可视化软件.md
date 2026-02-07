## 项目规划

### 技术栈
- **GUI框架**: PyQt6 - 功能强大、界面美观、支持多线程
- **MQTT**: paho-mqtt (客户端) + hbmqtt (服务端)
- **推理引擎**: YOLOv6 (通过torch加载)
- **图像处理**: OpenCV, Pillow
- **配置管理**: JSON + pydantic验证

### 项目结构
```
MQTT与推理可用面板/
├── main.py                 # 程序入口
├── config.json            # 配置文件
├── requirements.txt       # 依赖列表
├── src/
│   ├── core/              # 核心功能
│   │   ├── mqtt_server.py     # MQTT服务端
│   │   ├── mqtt_client.py     # MQTT客户端
│   │   ├── yolo_inference.py  # YOLOv6推理
│   │   └── config_manager.py  # 配置管理
│   └── ui/                # 界面模块
│       ├── main_window.py     # 主窗口
│       ├── mqtt_widget.py     # MQTT面板
│       ├── inference_widget.py # 推理面板
│       └── settings_dialog.py # 设置对话框
├── models/                # 模型存放目录
└── logs/                  # 日志目录
```

### 功能模块

#### 1. MQTT服务端功能
- 基于 hbmqtt 实现MQTT broker
- 连接管理：显示在线客户端列表
- 消息管理：消息历史记录、主题筛选、搜索
- 主题管理：CRUD操作、通配符支持

#### 2. 推理UI功能
- **数据源选择**: 摄像头、本地文件、HTTP推流、MQTT Base64图像
- **推理模式**: 目标检测、实例分割、关键点检测、图像分类
- **模型管理**: 自动模式匹配、手动路径设置

#### 3. 配置管理
- JSON配置文件结构
- 配置验证 (pydantic)
- 导入/导出功能
- 设置UI同步

### 实现步骤
1. 项目初始化 - 创建目录结构、requirements.txt
2. 配置管理模块 - 实现ConfigManager类
3. MQTT服务端 - 实现MQTT broker功能
4. YOLOv6推理 - 封装推理接口
5. 主界面框架 - PyQt6主窗口、布局
6. MQTT面板 - 连接、主题、消息管理UI
7. 推理面板 - 视频显示、控制按钮
8. 设置对话框 - 配置编辑界面
9. 集成测试 - 功能联调、错误处理

确认后我将开始逐步实现。