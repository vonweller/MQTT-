# MQTT与YOLOv6推理可视化软件

一个集成 MQTT 通信与 YOLOv6 推理功能的可视化桌面应用程序，支持实时目标检测、实例分割、关键点检测和图像分类。

## 功能特性

### MQTT 通信
- **MQTT 服务端**：内置 MQTT Broker，支持多客户端连接
- **MQTT 客户端**：可连接外部 MQTT 服务器
- **WebSocket 支持**：提供网页端 MQTT 客户端访问
- **消息管理**：消息历史记录、主题筛选、搜索功能
- **主题管理**：支持通配符订阅

### AI 推理功能
- **多任务支持**：
  - 目标检测 (Detection)
  - 实例分割 (Segmentation)
  - 关键点检测 (Pose)
  - 图像分类 (Classification)
- **多种数据源**：
  - 本地摄像头
  - 视频文件
  - HTTP 视频流
  - MQTT Base64 图像流
- **模型管理**：支持自动模式匹配和手动路径设置

### 用户界面
- **双面板布局**：左侧 MQTT 面板，右侧推理面板
- **深色/浅色主题**：可切换的主题风格
- **实时可视化**：推理结果实时显示，支持 FPS 显示
- **快捷键支持**：
  - `F5`：开始/停止推理
  - `F6`：截图
  - `Ctrl+P`：首选项
  - `Ctrl+Q`：退出

## 技术栈

- **GUI 框架**：PyQt6
- **MQTT**：paho-mqtt + asyncio-mqtt
- **AI 推理**：PyTorch + Ultralytics YOLO
- **图像处理**：OpenCV + Pillow
- **配置管理**：Pydantic
- **日志**：Loguru

## 安装

### 环境要求
- Python 3.10+
- CUDA (可选，用于 GPU 加速)

### 安装依赖

```bash
# 克隆仓库
git clone <repository-url>
cd MQTT与推理可用面板

# 安装依赖
pip install -r requirements.txt
```

### 主要依赖
```
PyQt6>=6.4.0
paho-mqtt>=1.6.1
torch>=2.0.0
opencv-python>=4.7.0
mediapipe>=0.10.0
ultralytics>=8.0.0
```

## 使用方法

### 启动应用

```bash
python main.py
```

### 基本操作流程

1. **启动 MQTT 服务器**
   - 在左侧面板点击"启动服务器"
   - 查看弹出的 IP 地址信息，用于客户端连接

2. **连接 MQTT 客户端**
   - 配置 Broker 地址和端口
   - 点击"连接"按钮

3. **加载 AI 模型**
   - 在右侧面板选择模型路径
   - 选择任务类型（检测/分割/关键点/分类）
   - 点击"加载模型"

4. **开始推理**
   - 选择数据源（摄像头/文件/HTTP/MQTT）
   - 点击"开始推理"或按 `F5`

5. **发布/接收消息**
   - 在 MQTT 面板订阅主题
   - 发布消息到指定主题
   - 查看消息历史记录

## 项目结构

```
MQTT与推理可用面板/
├── main.py                 # 程序入口
├── requirements.txt        # 依赖列表
├── config.json            # 配置文件（运行时生成）
├── src/
│   ├── core/              # 核心功能模块
│   │   ├── config_manager.py   # 配置管理
│   │   ├── mqtt_server.py      # MQTT 服务端
│   │   ├── mqtt_client.py      # MQTT 客户端
│   │   ├── mqtt_ws_server.py   # WebSocket 服务器
│   │   └── yolo_inference.py   # YOLO 推理引擎
│   ├── ui/                # 用户界面模块
│   │   ├── main_window.py      # 主窗口
│   │   ├── mqtt_widget.py      # MQTT 面板
│   │   ├── inference_widget.py # 推理面板
│   │   └── settings_dialog.py  # 设置对话框
│   └── web/               # Web 资源
│       └── index.html     # WebSocket 客户端页面
├── models/                # 模型文件目录
└── logs/                  # 日志文件目录
```

## 配置说明

配置文件 `config.json` 包含以下设置：

```json
{
  "app_name": "MQTT与YOLOv6推理可视化软件",
  "version": "1.0.0",
  "theme": "dark",
  "window_width": 1400,
  "window_height": 900,
  "log_level": "INFO",
  "mqtt_server": {
    "host": "0.0.0.0",
    "port": 1883
  },
  "mqtt_client": {
    "broker_host": "localhost",
    "broker_port": 1883,
    "client_id": "",
    "username": "",
    "password": "",
    "keepalive": 60
  },
  "inference": {
    "device": "auto",
    "conf_threshold": 0.25,
    "iou_threshold": 0.45,
    "img_size": 640,
    "half_precision": false
  }
}
```

## 模型支持

### YOLO 模型格式
- `.pt` - PyTorch 模型文件
- 支持 YOLOv5、YOLOv8、YOLO11 等 Ultralytics 系列模型

### MediaPipe 模型
- 手势识别 (Gesture Recognition)
- 手部关键点 (Hand Landmarker)
- 姿态估计 (Pose Landmarker)

## 开发计划

- [x] 基础 MQTT 服务端/客户端
- [x] YOLO 推理功能
- [x] 多数据源支持
- [x] WebSocket 支持
- [ ] 模型量化加速
- [ ] 多语言支持
- [ ] 插件系统

## 注意事项

1. **模型文件**：大模型文件建议放在 `models/` 目录，并使用 `.gitignore` 忽略
2. **配置文件**：`config.json` 包含敏感信息时不建议提交到版本控制
3. **日志文件**：日志自动保存在 `logs/` 目录，按日期轮转

## 许可证

MIT License

## 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/)
- [paho-mqtt](https://github.com/eclipse/paho.mqtt.python)
