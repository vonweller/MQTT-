"""
主窗口界面
"""

import sys
import os
import asyncio
import socket
import time
import hashlib
from pathlib import Path
from typing import Optional, List

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QStatusBar, QMenuBar, QMenu,
    QFileDialog, QMessageBox, QLabel, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut, QAction
from loguru import logger

from .mqtt_widget import MQTTWidget
from .inference_widget import InferenceWidget
from .settings_dialog import SettingsDialog
from ..core.config_manager import ConfigManager
from ..core.mqtt_server import MQTTServer
from ..core.mqtt_client import MQTTClient
from ..core.yolo_inference import YOLOInference
from ..core.mqtt_ws_server import MQTTWebSocketServer
from ..core.inference_result_publisher import (
    InferenceResultPublisher,
    PublisherConfig
)


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        
        # 核心组件
        self.mqtt_server: Optional[MQTTServer] = None
        self.mqtt_client: Optional[MQTTClient] = None
        self.yolo_inference: Optional[YOLOInference] = None
        self.ws_server: Optional[MQTTWebSocketServer] = None
        
        # 跟踪最近发布的消息哈希，用于过滤重复
        self._published_message_hashes = set()
        self._cleanup_timer = QTimer()
        self._cleanup_timer.timeout.connect(self._cleanup_published_hashes)
        self._cleanup_timer.start(5000)  # 每5秒清理一次
        
        # 初始化UI
        self._init_ui()
        self._init_menu()
        self._init_statusbar()
        self._init_shortcuts()
        
        # 定时更新状态
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(1000)  # 每秒更新
        
        # 设置窗口属性
        self.setWindowTitle(self.config.app_name)
        self.resize(self.config.window_width, self.config.window_height)
        
        # 初始化推理结果发布器
        self._init_result_publisher()
        
        logger.info("主窗口初始化完成")
    
    def _init_result_publisher(self):
        """初始化推理结果发布器"""
        logger.info("[结果发布] 初始化推理结果发布器")

        # 创建配置
        config = PublisherConfig(
            topic="siot/推理结果",
            qos=0,
            enabled=True,
            include_timestamp=True,
            include_fps=True
        )

        # 创建发布器
        self.result_publisher = InferenceResultPublisher(config)

        # 设置UI回调，让消息显示在消息查看面板
        self.result_publisher.set_ui_callback(self._add_inference_result_to_ui)

        # 延迟设置，等待组件初始化完成
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(200, self._setup_publisher_integration)

    def _add_inference_result_to_ui(self, topic: str, payload: str):
        """
        将推理结果消息添加到UI的消息查看面板

        Args:
            topic: 主题
            payload: 消息内容
        """
        try:
            # 使用与_main_window._add_published_message_to_ui相同的方法
            self._add_published_message_to_ui(topic, payload, 0, False)
        except Exception as e:
            logger.warning(f"[结果发布] 添加消息到UI失败: {e}")

    def _setup_publisher_integration(self):
        """设置发布器的集成（延迟执行）"""
        if self.mqtt_client and self.mqtt_client.is_connected():
            self.result_publisher.set_mqtt_client(self.mqtt_client)
            logger.info("[结果发布] MQTT客户端已设置")
            self._add_inference_callbacks()

    def _add_inference_callbacks(self):
        """添加推理回调"""
        if not self.result_publisher.is_available():
            return

        if self.yolo_inference:
            yolo_callback = self.result_publisher.get_yolo_callback()
            self.yolo_inference.add_inference_callback(yolo_callback)
            logger.info("[结果发布] YOLO推理回调已添加")

    def _setup_publisher_after_connect(self):
        """MQTT客户端连接后设置发布器"""
        if hasattr(self, 'result_publisher') and self.mqtt_client:
            self.result_publisher.set_mqtt_client(self.mqtt_client)
            logger.info("[结果发布] MQTT客户端连接成功，发布器已就绪")
            self._add_inference_callbacks()

    def _setup_publisher_for_new_inference(self):
        """新推理器加载后设置发布器"""
        if hasattr(self, 'result_publisher') and self.result_publisher.is_available():
            if self.yolo_inference:
                yolo_callback = self.result_publisher.get_yolo_callback()
                self.yolo_inference.add_inference_callback(yolo_callback)
                logger.info("[结果发布] 新YOLO推理器回调已添加")
    
    def _get_local_ipv4_addresses(self) -> List[str]:
        """获取本机所有可用的IPv4地址（排除回环地址）"""
        ip_addresses = []
        try:
            # 获取所有网络接口
            hostname = socket.gethostname()
            # 获取所有IP地址
            all_ips = socket.getaddrinfo(hostname, None, socket.AF_INET)
            
            for ip_info in all_ips:
                ip = ip_info[4][0]
                # 排除回环地址
                if not ip.startswith('127.'):
                    ip_addresses.append(ip)
            
            # 去重并保持顺序
            ip_addresses = list(dict.fromkeys(ip_addresses))
            
            # 如果没有找到非回环地址，尝试另一种方法
            if not ip_addresses:
                # 创建一个UDP连接来获取本机IP
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(('8.8.8.8', 80))
                    ip = s.getsockname()[0]
                    if not ip.startswith('127.'):
                        ip_addresses.append(ip)
                    s.close()
                except Exception:
                    pass
            
        except Exception as e:
            logger.error(f"获取本机IP地址失败: {e}")
        
        return ip_addresses
    
    def _show_server_ip_addresses(self, port: int):
        """显示服务器可连接的IP地址"""
        ip_addresses = self._get_local_ipv4_addresses()
        
        if ip_addresses:
            # 构建IP地址显示文本
            ip_list_text = "\n".join([f"  • {ip}:{port}" for ip in ip_addresses])
            message = (
                f"🟢 MQTT服务器已启动！\n\n"
                f"可通过以下地址连接：\n"
                f"{ip_list_text}\n\n"
                f"客户端配置示例：\n"
                f"  • Broker主机: {ip_addresses[0]}\n"
                f"  • Broker端口: {port}"
            )
            
            # 在控制台输出
            logger.info(f"服务器已启动，可连接地址：")
            for ip in ip_addresses:
                logger.info(f"  - {ip}:{port}")
            
            # 显示对话框
            QMessageBox.information(self, "服务器启动成功", message)
        else:
            # 如果没有找到非回环地址，提示用户使用localhost
            message = (
                f"🟢 MQTT服务器已启动！\n\n"
                f"监听端口: {port}\n\n"
                f"未检测到外部网络接口，\n"
                f"请使用 localhost:{port} 或 127.0.0.1:{port} 进行本地连接。"
            )
            logger.info(f"服务器已启动，监听端口: {port}")
            QMessageBox.information(self, "服务器启动成功", message)
    
    def _init_ui(self):
        """初始化UI"""
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # MQTT面板
        self.mqtt_widget = MQTTWidget(self.config_manager)
        self.mqtt_widget.server_start_requested.connect(self._start_mqtt_server)
        self.mqtt_widget.server_stop_requested.connect(self._stop_mqtt_server)
        self.mqtt_widget.client_connect_requested.connect(self._connect_mqtt_client)
        self.mqtt_widget.client_disconnect_requested.connect(self._disconnect_mqtt_client)
        self.mqtt_widget.message_publish_requested.connect(self._publish_message)
        # 设置获取服务端主题列表的回调
        self.mqtt_widget.set_get_server_topics_callback(self._get_server_topics)
        splitter.addWidget(self.mqtt_widget)
        
        # 推理面板
        self.inference_widget = InferenceWidget(self.config_manager)
        self.inference_widget.inference_start_requested.connect(self._start_inference)
        self.inference_widget.inference_stop_requested.connect(self._stop_inference)
        self.inference_widget.model_load_requested.connect(self._load_model)
        splitter.addWidget(self.inference_widget)
        
        # 设置分割比例
        splitter.setSizes([400, 1000])
        
        # 应用主题
        self._apply_theme()
    
    def _init_menu(self):
        """初始化菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        
        # 导入配置
        import_action = QAction("导入配置", self)
        import_action.setShortcut("Ctrl+I")
        import_action.triggered.connect(self._import_config)
        file_menu.addAction(import_action)
        
        # 导出配置
        export_action = QAction("导出配置", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._export_config)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        # 退出
        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 设置菜单
        settings_menu = menubar.addMenu("设置(&S)")
        
        # 首选项
        pref_action = QAction("首选项", self)
        pref_action.setShortcut("Ctrl+P")
        pref_action.triggered.connect(self._show_settings)
        settings_menu.addAction(pref_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图(&V)")
        
        # 主题切换
        theme_menu = view_menu.addMenu("主题")
        
        dark_action = QAction("深色", self)
        dark_action.setCheckable(True)
        dark_action.setChecked(self.config.theme == "dark")
        dark_action.triggered.connect(lambda: self._set_theme("dark"))
        theme_menu.addAction(dark_action)
        
        light_action = QAction("浅色", self)
        light_action.setCheckable(True)
        light_action.setChecked(self.config.theme == "light")
        light_action.triggered.connect(lambda: self._set_theme("light"))
        theme_menu.addAction(light_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _init_statusbar(self):
        """初始化状态栏"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # MQTT服务器状态
        self.server_status_label = QLabel("MQTT服务器: 停止")
        self.statusbar.addWidget(self.server_status_label)
        
        # 添加分隔符
        separator1 = QLabel(" | ")
        self.statusbar.addWidget(separator1)
        
        # MQTT客户端状态
        self.client_status_label = QLabel("MQTT客户端: 断开")
        self.statusbar.addWidget(self.client_status_label)
        
        # 添加分隔符
        separator2 = QLabel(" | ")
        self.statusbar.addWidget(separator2)
        
        # 推理状态
        self.inference_status_label = QLabel("推理: 停止")
        self.statusbar.addWidget(self.inference_status_label)
        
        # 添加分隔符
        separator3 = QLabel(" | ")
        self.statusbar.addWidget(separator3)
        
        # FPS显示
        self.fps_label = QLabel("FPS: 0")
        self.statusbar.addWidget(self.fps_label)
        
        # 进度条（右侧）
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(150)
        self.progress_bar.setVisible(False)
        self.statusbar.addPermanentWidget(self.progress_bar)
    
    def _init_shortcuts(self):
        """初始化快捷键"""
        # F5 - 开始/停止推理
        self.shortcut_inference = QShortcut(QKeySequence("F5"), self)
        self.shortcut_inference.activated.connect(self._toggle_inference)
        
        # F6 - 截图
        self.shortcut_screenshot = QShortcut(QKeySequence("F6"), self)
        self.shortcut_screenshot.activated.connect(self._take_screenshot)
    
    def _apply_theme(self):
        """应用主题"""
        if self.config.theme == "dark":
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #2b2b2b;
                }
                QWidget {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QMenuBar {
                    background-color: #3c3f41;
                    color: #ffffff;
                }
                QMenuBar::item:selected {
                    background-color: #4b6eaf;
                }
                QMenu {
                    background-color: #3c3f41;
                    color: #ffffff;
                    border: 1px solid #555555;
                }
                QMenu::item:selected {
                    background-color: #4b6eaf;
                }
                QStatusBar {
                    background-color: #3c3f41;
                    color: #ffffff;
                }
                QGroupBox {
                    border: 1px solid #555555;
                    border-radius: 5px;
                    margin-top: 10px;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                }
                QPushButton {
                    background-color: #4b6eaf;
                    color: white;
                    border: none;
                    padding: 5px 15px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #5a7ec0;
                }
                QPushButton:pressed {
                    background-color: #3d5a8a;
                }
                QPushButton:disabled {
                    background-color: #555555;
                    color: #888888;
                }
                QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                    background-color: #45494a;
                    border: 1px solid #646464;
                    color: #ffffff;
                    padding: 3px;
                }
                QTableWidget {
                    background-color: #2b2b2b;
                    border: 1px solid #555555;
                    color: #ffffff;
                    gridline-color: #555555;
                }
                QTableWidget::item:selected {
                    background-color: #4b6eaf;
                }
                QHeaderView::section {
                    background-color: #3c3f41;
                    color: #ffffff;
                    padding: 5px;
                    border: 1px solid #555555;
                }
                QTabWidget::pane {
                    border: 1px solid #555555;
                }
                QTabBar::tab {
                    background-color: #3c3f41;
                    color: #ffffff;
                    padding: 8px 15px;
                    border: 1px solid #555555;
                }
                QTabBar::tab:selected {
                    background-color: #4b6eaf;
                }
                QScrollBar:vertical {
                    background-color: #2b2b2b;
                    width: 12px;
                }
                QScrollBar::handle:vertical {
                    background-color: #555555;
                    min-height: 20px;
                }
                QScrollBar::handle:vertical:hover {
                    background-color: #666666;
                }
                QLabel {
                    color: #ffffff;
                }
            """)
        else:
            self.setStyleSheet("")  # 使用默认浅色主题
    
    def _set_theme(self, theme: str):
        """设置主题"""
        self.config.theme = theme
        self.config_manager.save()
        self._apply_theme()
        
        # 通知子部件
        self.mqtt_widget.apply_theme(theme)
        self.inference_widget.apply_theme(theme)
    
    def _start_mqtt_server(self):
        """启动MQTT服务器"""
        logger.debug(f"[MainWindow] 开始启动MQTT服务器: {self.config.mqtt_server.host}:{self.config.mqtt_server.port}")
        
        try:
            # 确保推理结果主题在允许列表中
            inference_topic = "siot/推理结果"
            if hasattr(self.config.mqtt_server, 'allowed_topics'):
                if inference_topic not in self.config.mqtt_server.allowed_topics:
                    self.config.mqtt_server.allowed_topics.append(inference_topic)
                    logger.info(f"[结果发布] 已添加主题到允许列表: {inference_topic}")
            
            if self.mqtt_server is None:
                logger.debug("[MainWindow] 创建MQTTServer实例...")
                self.mqtt_server = MQTTServer(
                    host=self.config.mqtt_server.host,
                    port=self.config.mqtt_server.port,
                    strict_topic_mode=self.config.mqtt_server.strict_topic_mode
                )
                
                # 设置允许的主题列表
                if self.config.mqtt_server.strict_topic_mode and self.config.mqtt_server.allowed_topics:
                    self.mqtt_server.set_allowed_topics(self.config.mqtt_server.allowed_topics)
                    logger.info(f"[MainWindow] 严格模式已启用，允许的主题: {self.config.mqtt_server.allowed_topics}")
                
                # 添加回调 - 使用信号确保在主线程执行UI更新
                from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
                
                def on_server_connect_safe(client_id):
                    QMetaObject.invokeMethod(
                        self.mqtt_widget, "on_client_connected",
                        Qt.ConnectionType.QueuedConnection,
                        Q_ARG(str, client_id)
                    )
                
                def on_server_disconnect_safe(client_id):
                    QMetaObject.invokeMethod(
                        self.mqtt_widget, "on_client_disconnected",
                        Qt.ConnectionType.QueuedConnection,
                        Q_ARG(str, client_id)
                    )
                
                def on_server_message_safe(msg):
                    # 只广播给WebSocket客户端，不添加到UI
                    # UI由MQTT客户端回调处理（避免重复）
                    if self.ws_server:
                        logger.info(f"[MainWindow] 广播消息到WebSocket: {msg.topic}")
                        asyncio.create_task(self.ws_server.broadcast_message(
                            msg.topic, msg.payload, msg.qos
                        ))
                    else:
                        logger.debug(f"[MainWindow] WebSocket服务器未启动，跳过广播")
                
                self.mqtt_server.add_connect_callback(on_server_connect_safe)
                self.mqtt_server.add_disconnect_callback(on_server_disconnect_safe)
                self.mqtt_server.add_message_callback(on_server_message_safe)
                logger.debug("[MainWindow] MQTTServer实例创建完成，回调已注册")
            
            # 在异步循环中启动
            logger.debug("[MainWindow] 创建异步任务启动服务器...")
            task = asyncio.create_task(self.mqtt_server.start())
            logger.debug(f"[MainWindow] 异步任务已创建: {task}")
            
            self.server_status_label.setText("MQTT服务器: 运行中")
            self.server_status_label.setStyleSheet("color: #4caf50;")
            
            logger.info("[MainWindow] MQTT服务器启动任务已提交")
            
            # 启动WebSocket服务（如果启用）
            self._start_ws_server()
            
            # 延迟显示IP地址信息，等待服务器完全启动
            port = self.config.mqtt_server.port
            QTimer.singleShot(500, lambda: self._show_server_ip_addresses(port))
            
        except Exception as e:
            logger.error(f"[MainWindow] 启动MQTT服务器失败: {e}")
            logger.exception("[MainWindow] 详细错误:")
            QMessageBox.critical(self, "错误", f"启动MQTT服务器失败:\n{e}")
    
    def _start_ws_server(self):
        """启动WebSocket服务器"""
        try:
            # 检查是否启用Web服务
            if not hasattr(self.mqtt_widget, 'ws_enabled_check') or not self.mqtt_widget.ws_enabled_check.isChecked():
                logger.info("[MainWindow] Web服务未启用，跳过启动")
                return
            
            ws_port = self.mqtt_widget.ws_port_input.value()
            
            # 创建WebSocket服务器
            self.ws_server = MQTTWebSocketServer(
                host="0.0.0.0",
                port=ws_port,
                mqtt_server=self.mqtt_server,
                config=self.config
            )
            
            # 添加回调
            from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
            
            def on_ws_connect_safe(client_id):
                QMetaObject.invokeMethod(
                    self, "_on_ws_client_connect",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(str, client_id)
                )
            
            def on_ws_disconnect_safe(client_id):
                QMetaObject.invokeMethod(
                    self, "_on_ws_client_disconnect",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(str, client_id)
                )
            
            self.ws_server.add_connect_callback(on_ws_connect_safe)
            self.ws_server.add_disconnect_callback(on_ws_disconnect_safe)
            
            # 添加WebSocket消息发布回调 - 让网页版发布的消息显示在桌面客户端
            def on_ws_publish_safe(topic, payload, qos):
                from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
                QMetaObject.invokeMethod(
                    self.mqtt_widget, "on_message_received",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(object, type('Message', (), {
                        'topic': topic,
                        'payload': payload.encode() if isinstance(payload, str) else payload,
                        'qos': qos,
                        'timestamp': time.time(),
                        'payload_text': payload if isinstance(payload, str) else payload.decode('utf-8', errors='replace')
                    })())
                )
            
            self.ws_server.add_publish_callback(on_ws_publish_safe)
            
            # 启动服务
            asyncio.create_task(self.ws_server.start())
            
            # 更新UI
            self.mqtt_widget.ws_status_label.setText("状态: 运行中")
            self.mqtt_widget.ws_status_label.setStyleSheet("color: #4caf50;")
            
            # 获取本机IP地址
            import socket
            try:
                # 尝试获取局域网IP
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
            except Exception:
                # 备用方案
                hostname = socket.gethostname()
                local_ip = socket.getaddrinfo(hostname, None, socket.AF_INET)[0][4][0]
            
            # WebSocket地址（用于连接）
            ws_url = f"ws://{local_ip}:{ws_port}"
            # HTTP地址（用于访问网页）
            http_url = f"http://{local_ip}:{ws_port}"
            
            self.mqtt_widget.ws_url_label.setText(f"网页: {http_url}\nWS: {ws_url}")
            self.mqtt_widget.ws_url_label.setStyleSheet("color: #4fc3f7;")
            
            logger.info(f"[MainWindow] WebSocket服务器启动: {ws_url}")
            
        except Exception as e:
            logger.error(f"[MainWindow] 启动WebSocket服务器失败: {e}")
            self.mqtt_widget.ws_status_label.setText(f"状态: 启动失败")
            self.mqtt_widget.ws_status_label.setStyleSheet("color: #f44336;")
    
    def _stop_ws_server(self):
        """停止WebSocket服务器"""
        if self.ws_server:
            asyncio.create_task(self.ws_server.stop())
            self.ws_server = None
            
            self.mqtt_widget.ws_status_label.setText("状态: 已停止")
            self.mqtt_widget.ws_status_label.setStyleSheet("")
            self.mqtt_widget.ws_url_label.setText("")
            
            logger.info("[MainWindow] WebSocket服务器已停止")
    
    def _on_ws_client_connect(self, client_id: str):
        """WebSocket客户端连接"""
        logger.info(f"[WebSocket] 客户端连接: {client_id}")
    
    def _on_ws_client_disconnect(self, client_id: str):
        """WebSocket客户端断开"""
        logger.info(f"[WebSocket] 客户端断开: {client_id}")
    
    def _stop_mqtt_server(self):
        """停止MQTT服务器"""
        # 先停止WebSocket服务
        self._stop_ws_server()
        
        if self.mqtt_server:
            asyncio.create_task(self.mqtt_server.stop())
            self.mqtt_server = None
        
        self.server_status_label.setText("MQTT服务器: 停止")
        self.server_status_label.setStyleSheet("")
        
        logger.info("MQTT服务器已停止")
    
    def _get_server_topics(self) -> list:
        """获取MQTT服务端已订阅的主题列表"""
        if self.mqtt_server is None:
            return []
        try:
            return self.mqtt_server.get_topics()
        except Exception as e:
            logger.debug(f"获取服务端主题列表失败: {e}")
            return []
    
    def _connect_mqtt_client(self):
        """连接MQTT客户端"""
        try:
            if self.mqtt_client is None:
                self.mqtt_client = MQTTClient(
                    broker_host=self.config.mqtt_client.broker_host,
                    broker_port=self.config.mqtt_client.broker_port,
                    client_id=self.config.mqtt_client.client_id,
                    username=self.config.mqtt_client.username,
                    password=self.config.mqtt_client.password,
                    keepalive=self.config.mqtt_client.keepalive
                )
                
                # 添加回调 - 使用信号确保在主线程执行UI更新
                from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
                
                def on_connect_safe(success):
                    QMetaObject.invokeMethod(
                        self, "_on_client_connect",
                        Qt.ConnectionType.QueuedConnection,
                        Q_ARG(bool, success)
                    )
                
                def on_disconnect_safe():
                    QMetaObject.invokeMethod(
                        self, "_on_client_disconnect",
                        Qt.ConnectionType.QueuedConnection
                    )
                
                def on_message_safe(msg):
                    # 检查消息是否已经存在（避免重复添加自己发布的消息）
                    msg_hash = hashlib.md5(f"{msg.topic}:{msg.payload_text}:{msg.qos}:{msg.retain}".encode()).hexdigest()
                    logger.info(f"[MainWindow] MQTT客户端收到消息: {msg.topic}, hash={msg_hash}")
                    logger.info(f"[MainWindow] 已发布消息哈希: {list(self._published_message_hashes)[:5]}...")
                    if msg_hash not in self._published_message_hashes:
                        # 不是自己发布的消息，添加到UI
                        logger.info(f"[MainWindow] 添加消息到UI: {msg.topic}")
                        QMetaObject.invokeMethod(
                            self.mqtt_widget, "on_client_message",
                            Qt.ConnectionType.QueuedConnection,
                            Q_ARG(object, msg)
                        )
                    else:
                        logger.info(f"[MainWindow] 跳过自己发布的消息: {msg.topic}")
                
                self.mqtt_client.add_connect_callback(on_connect_safe)
                self.mqtt_client.add_disconnect_callback(on_disconnect_safe)
                self.mqtt_client.add_message_callback(on_message_safe)
            
            self.mqtt_client.connect()
            
        except Exception as e:
            logger.error(f"连接MQTT客户端失败: {e}")
            QMessageBox.critical(self, "错误", f"连接MQTT客户端失败:\n{e}")
    
    def _disconnect_mqtt_client(self):
        """断开MQTT客户端"""
        if self.mqtt_client:
            self.mqtt_client.disconnect()
            self.mqtt_client = None
        
        self.client_status_label.setText("MQTT客户端: 断开")
        self.client_status_label.setStyleSheet("")
    
    from PyQt6.QtCore import pyqtSlot
    
    @pyqtSlot(bool)
    def _on_client_connect(self, success: bool):
        """客户端连接回调"""
        logger.debug(f"[MainWindow] 连接回调执行: success={success}")
        if success:
            self.client_status_label.setText("MQTT客户端: 已连接")
            self.client_status_label.setStyleSheet("color: #4caf50;")
            self.mqtt_widget.on_client_connect_success()
            # 设置推理结果发布器
            self._setup_publisher_after_connect()
        else:
            self.client_status_label.setText("MQTT客户端: 连接失败")
            self.client_status_label.setStyleSheet("color: #f44336;")
            # 通知UI连接失败
            self.mqtt_widget.on_client_connect_failed()
    
    @pyqtSlot()
    def _on_client_disconnect(self):
        """客户端断开回调"""
        logger.debug("[MainWindow] 断开回调执行")
        self.client_status_label.setText("MQTT客户端: 断开")
        self.client_status_label.setStyleSheet("")
        self.mqtt_widget.on_client_disconnect()
    
    def _publish_message(self, topic: str, payload: str, qos: int, retain: bool):
        """发布消息到MQTT服务器"""
        logger.info(f"[MainWindow] 发布消息: {topic}")
        
        try:
            if self.mqtt_client and self.mqtt_client.is_connected():
                # 使用MQTT客户端发布消息
                success = self.mqtt_client.publish(topic, payload, qos, retain)
                if success:
                    logger.info(f"[MainWindow] 消息发布成功: {topic}")
                    # 直接添加消息到UI，避免通过回调重复添加
                    self._add_published_message_to_ui(topic, payload, qos, retain)
                else:
                    logger.error(f"[MainWindow] 消息发布失败: {topic}")
                    QMessageBox.warning(self, "警告", "消息发布失败")
            elif self.mqtt_server and self.mqtt_server.running:
                # 如果客户端未连接但服务器在运行，直接通过服务器发布
                import asyncio
                asyncio.create_task(self.mqtt_server.publish(topic, payload.encode(), qos, retain))
                logger.info(f"[MainWindow] 通过服务器发布消息: {topic}")
                # 直接添加消息到UI
                self._add_published_message_to_ui(topic, payload, qos, retain)
            else:
                logger.error("[MainWindow] MQTT客户端未连接，无法发布消息")
                QMessageBox.warning(self, "警告", "MQTT客户端未连接，无法发布消息")
        except Exception as e:
            logger.error(f"[MainWindow] 发布消息失败: {e}")
            QMessageBox.critical(self, "错误", f"发布消息失败:\n{e}")
    
    def _add_published_message_to_ui(self, topic: str, payload: str, qos: int, retain: bool):
        """将发布的消息直接添加到UI（避免通过回调重复添加）"""
        import uuid
        import hashlib
        msg_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        msg_data = {
            'id': msg_id,
            'topic': topic,
            'payload': payload,
            'qos': qos,
            'retain': retain,
            'timestamp': time.time()
        }
        # 计算消息哈希值（不包含时间戳，用于检测重复）
        msg_hash = hashlib.md5(f"{topic}:{payload}:{qos}:{retain}".encode()).hexdigest()
        logger.info(f"[MainWindow] 发布消息并添加哈希: {topic}, hash={msg_hash}")
        # 添加到已发布消息哈希集合
        self._published_message_hashes.add(msg_hash)
        logger.info(f"[MainWindow] 已发布消息哈希集合大小: {len(self._published_message_hashes)}")
        self.mqtt_widget._add_message_to_table(msg_data)
    
    def _cleanup_published_hashes(self):
        """清理过期的已发布消息哈希"""
        # 简单地清空集合，保留最近的消息
        if len(self._published_message_hashes) > 50:
            self._published_message_hashes.clear()
    
    def _load_model(self, model_path: str, task_type: int = 0):
        """加载模型"""
        logger.info(f"[MainWindow] 开始加载模型: {model_path}, 任务类型索引: {task_type}")
        
        try:
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                logger.error(f"[MainWindow] 模型文件不存在: {model_path}")
                QMessageBox.critical(self, "错误", f"模型文件不存在:\n{model_path}")
                self.inference_widget.on_model_loaded(False)
                return
            
            # 使用传入的任务类型参数
            task_types = ["detection", "segmentation", "pose", "obb", "classification"]
            model_type = task_types[task_type] if task_type < len(task_types) else "detection"
            logger.info(f"[MainWindow] 任务类型: {model_type}")
            
            # 创建推理器实例
            logger.info("[MainWindow] 创建YOLOInference实例...")
            self.yolo_inference = YOLOInference(
                model_path="",  # 先不加载模型
                model_type=model_type,  # 使用界面选择的任务类型
                device=self.config.inference.device,
                conf_threshold=self.config.inference.conf_threshold,
                iou_threshold=self.config.inference.iou_threshold,
                img_size=self.config.inference.img_size,
                half_precision=self.config.inference.half_precision
            )
            
            # 加载模型
            logger.info("[MainWindow] 调用load_model加载模型...")
            success = self.yolo_inference.load_model(model_path)
            
            if not success:
                logger.error("[MainWindow] 模型加载失败")
                self.inference_widget.on_model_loaded(False)
                QMessageBox.critical(self, "错误", "模型加载失败，请检查模型文件是否正确")
                self.yolo_inference = None
                return
            
            # 检查模型是否真的加载了
            if self.yolo_inference.model is None:
                logger.error("[MainWindow] 模型加载后仍为None")
                self.inference_widget.on_model_loaded(False)
                QMessageBox.critical(self, "错误", "模型加载失败，模型对象为None")
                self.yolo_inference = None
                return
            
            logger.info("[MainWindow] 模型加载成功，传递给inference_widget...")
            # 将推理引擎传递给 inference_widget
            self.inference_widget.set_inference_engine(self.yolo_inference)
            self.inference_widget.on_model_loaded(True)
            # 设置推理结果发布器的回调
            self._setup_publisher_for_new_inference()
            logger.info(f"[MainWindow] 模型加载完成: {model_path}")
            
        except Exception as e:
            logger.exception(f"[MainWindow] 加载模型时发生异常: {e}")
            self.inference_widget.on_model_loaded(False)
            QMessageBox.critical(self, "错误", f"加载模型失败:\n{e}")
            self.yolo_inference = None
    
    def _start_inference(self):
        """开始推理"""
        if self.yolo_inference is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        self.inference_status_label.setText("推理: 运行中")
        self.inference_status_label.setStyleSheet("color: #4caf50;")
        
        # 将推理器传递给inference_widget
        self.inference_widget.set_inference_engine(self.yolo_inference)
        self.inference_widget.start_inference()
    
    def _stop_inference(self):
        """停止推理"""
        self.inference_status_label.setText("推理: 停止")
        self.inference_status_label.setStyleSheet("")
        self.inference_widget.stop_inference()
    
    def _toggle_inference(self):
        """切换推理状态"""
        if self.inference_widget.is_running:
            self._stop_inference()
        else:
            self._start_inference()
    
    def _take_screenshot(self):
        """截图"""
        self.inference_widget.take_screenshot()
    
    def _update_status(self):
        """更新状态栏"""
        # 更新FPS
        if self.inference_widget and hasattr(self.inference_widget, 'current_fps'):
            fps = self.inference_widget.current_fps
            self.fps_label.setText(f"FPS: {fps:.1f}")
        
        # 更新服务器状态
        if self.mqtt_server and self.mqtt_server.running:
            stats = self.mqtt_server.get_stats()
            self.server_status_label.setText(
                f"MQTT服务器: 运行中 ({stats['connected_clients']} 客户端)"
            )
    
    def _show_settings(self):
        """显示设置对话框"""
        dialog = SettingsDialog(self.config_manager, self)
        if dialog.exec() == SettingsDialog.DialogCode.Accepted:
            # 重新加载配置
            self.config = self.config_manager.get_config()
            self._apply_theme()
    
    def _import_config(self):
        """导入配置"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "导入配置", "", "JSON文件 (*.json)"
        )
        if file_path:
            if self.config_manager.import_config(file_path):
                QMessageBox.information(self, "成功", "配置已导入")
                # 重新加载
                self.config = self.config_manager.get_config()
                self._apply_theme()
            else:
                QMessageBox.critical(self, "错误", "导入配置失败")
    
    def _export_config(self):
        """导出配置"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出配置", "config_backup.json", "JSON文件 (*.json)"
        )
        if file_path:
            if self.config_manager.export_config(file_path):
                QMessageBox.information(self, "成功", "配置已导出")
            else:
                QMessageBox.critical(self, "错误", "导出配置失败")
    
    def _show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self,
            "关于",
            f"""<h2>{self.config.app_name}</h2>
            <p>版本: {self.config.version}</p>
            <p>一个集成MQTT通信与YOLOv6推理功能的可视化软件</p>
            <p>支持功能:</p>
            <ul>
                <li>MQTT服务端与客户端</li>
                <li>YOLOv6目标检测、分割、关键点检测、分类</li>
                <li>多种推理源: 摄像头、文件、HTTP流、MQTT</li>
                <li>实时可视化与配置管理</li>
            </ul>
            """
        )
    
    def closeEvent(self, event):
        """关闭事件"""
        # 停止所有服务
        self._stop_inference()
        self._disconnect_mqtt_client()
        self._stop_mqtt_server()
        
        # 保存配置
        self.config_manager.save()
        
        logger.info("应用程序关闭")
        event.accept()
