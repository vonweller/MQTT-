"""
MQTT面板UI
提供MQTT服务端和客户端的可视化界面
"""

import json
import time
import uuid
from typing import Optional
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QSpinBox, QTableWidget, QTableWidgetItem,
    QTextEdit, QGroupBox, QTabWidget, QComboBox, QCheckBox,
    QMessageBox, QHeaderView, QSplitter, QFormLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from loguru import logger

from ..core.config_manager import ConfigManager


class MQTTWidget(QWidget):
    """MQTT面板"""
    
    # 信号
    server_start_requested = pyqtSignal()
    server_stop_requested = pyqtSignal()
    client_connect_requested = pyqtSignal()
    client_disconnect_requested = pyqtSignal()
    message_publish_requested = pyqtSignal(str, str, int, bool)  # topic, payload, qos, retain
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        
        # 跟踪已添加的消息ID，避免重复
        self._added_message_ids = set()
        
        # 回调函数，用于获取服务端主题列表
        self.get_server_topics_callback = None
        
        # 消息存储（用于筛选）
        self.all_messages = []  # 存储所有消息的完整数据
        self.current_filter = ""  # 当前筛选条件
        
        self._init_ui()
        self._load_config()
        
        # 定时刷新
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._refresh_data)
        self.refresh_timer.start(2000)  # 每2秒刷新
    
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # 标题
        title_label = QLabel("MQTT 管理面板")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Tab控件
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # 服务端Tab
        self.server_tab = self._create_server_tab()
        self.tab_widget.addTab(self.server_tab, "服务端")
        
        # 客户端Tab
        self.client_tab = self._create_client_tab()
        self.tab_widget.addTab(self.client_tab, "客户端")
        
        # 主题管理Tab
        self.topic_tab = self._create_topic_tab()
        self.tab_widget.addTab(self.topic_tab, "主题管理")
        
        # 消息查看Tab
        self.message_tab = self._create_message_tab()
        self.tab_widget.addTab(self.message_tab, "消息查看")
    
    def _create_server_tab(self) -> QWidget:
        """创建服务端Tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        
        # 服务器配置组
        config_group = QGroupBox("服务器配置")
        config_layout = QFormLayout(config_group)
        
        self.server_host_input = QLineEdit("0.0.0.0")
        config_layout.addRow("主机:", self.server_host_input)
        
        self.server_port_input = QSpinBox()
        self.server_port_input.setRange(1, 65535)
        self.server_port_input.setValue(1883)
        config_layout.addRow("端口:", self.server_port_input)
        
        self.server_auth_check = QCheckBox("启用认证")
        config_layout.addRow("认证:", self.server_auth_check)
        
        self.server_username_input = QLineEdit()
        self.server_username_input.setEnabled(False)
        config_layout.addRow("用户名:", self.server_username_input)
        
        self.server_password_input = QLineEdit()
        self.server_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.server_password_input.setEnabled(False)
        config_layout.addRow("密码:", self.server_password_input)
        
        self.server_auth_check.toggled.connect(
            lambda checked: (
                self.server_username_input.setEnabled(checked),
                self.server_password_input.setEnabled(checked)
            )
        )
        
        layout.addWidget(config_group)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        
        self.server_start_btn = QPushButton("启动服务器")
        self.server_start_btn.setStyleSheet("background-color: #4caf50;")
        self.server_start_btn.clicked.connect(self._on_server_start)
        btn_layout.addWidget(self.server_start_btn)
        
        self.server_stop_btn = QPushButton("停止服务器")
        self.server_stop_btn.setStyleSheet("background-color: #f44336;")
        self.server_stop_btn.setEnabled(False)
        self.server_stop_btn.clicked.connect(self._on_server_stop)
        btn_layout.addWidget(self.server_stop_btn)
        
        layout.addLayout(btn_layout)
        
        # Web服务配置组
        ws_group = QGroupBox("Web服务配置")
        ws_layout = QFormLayout(ws_group)
        
        self.ws_enabled_check = QCheckBox("启用Web服务")
        ws_layout.addRow("Web服务:", self.ws_enabled_check)
        
        self.ws_port_input = QSpinBox()
        self.ws_port_input.setRange(1, 65535)
        self.ws_port_input.setValue(8765)
        ws_layout.addRow("Web端口:", self.ws_port_input)
        
        self.ws_status_label = QLabel("状态: 未启动")
        ws_layout.addRow("Web状态:", self.ws_status_label)
        
        self.ws_url_label = QLabel("")
        ws_layout.addRow("访问地址:", self.ws_url_label)
        
        layout.addWidget(ws_group)
        
        # 主题权限配置组
        topic_auth_group = QGroupBox("主题权限控制")
        topic_auth_layout = QVBoxLayout(topic_auth_group)
        
        self.strict_mode_check = QCheckBox("启用严格模式（只允许预定义主题）")
        self.strict_mode_check.setToolTip("启用后，客户端只能发布/订阅预定义的主题")
        topic_auth_layout.addWidget(self.strict_mode_check)
        
        # 允许的主题列表
        topics_label = QLabel("允许的主题（每行一个，支持通配符 + 和 #）：")
        topic_auth_layout.addWidget(topics_label)
        
        self.allowed_topics_input = QTextEdit()
        self.allowed_topics_input.setPlaceholderText("sensor/#\ncontrol/+\ninference/result")
        self.allowed_topics_input.setMaximumHeight(80)
        topic_auth_layout.addWidget(self.allowed_topics_input)
        
        # 快速添加常用主题
        quick_topics_layout = QHBoxLayout()
        quick_topics_label = QLabel("快速添加:")
        quick_topics_layout.addWidget(quick_topics_label)
        
        self.add_topic_sensor_btn = QPushButton("sensor/#")
        self.add_topic_sensor_btn.setMaximumWidth(80)
        self.add_topic_sensor_btn.clicked.connect(lambda: self._add_quick_topic("sensor/#"))
        quick_topics_layout.addWidget(self.add_topic_sensor_btn)
        
        self.add_topic_control_btn = QPushButton("control/#")
        self.add_topic_control_btn.setMaximumWidth(80)
        self.add_topic_control_btn.clicked.connect(lambda: self._add_quick_topic("control/#"))
        quick_topics_layout.addWidget(self.add_topic_control_btn)
        
        self.add_topic_inference_btn = QPushButton("inference/#")
        self.add_topic_inference_btn.setMaximumWidth(80)
        self.add_topic_inference_btn.clicked.connect(lambda: self._add_quick_topic("inference/#"))
        quick_topics_layout.addWidget(self.add_topic_inference_btn)
        
        quick_topics_layout.addStretch()
        topic_auth_layout.addLayout(quick_topics_layout)
        
        layout.addWidget(topic_auth_group)
        
        # 状态显示
        status_group = QGroupBox("服务器状态")
        status_layout = QVBoxLayout(status_group)
        
        self.server_status_label = QLabel("状态: 停止")
        status_layout.addWidget(self.server_status_label)
        
        self.server_clients_label = QLabel("连接客户端: 0")
        status_layout.addWidget(self.server_clients_label)
        
        self.server_messages_label = QLabel("接收消息: 0")
        status_layout.addWidget(self.server_messages_label)
        
        layout.addWidget(status_group)
        
        # 客户端列表
        clients_group = QGroupBox("已连接客户端")
        clients_layout = QVBoxLayout(clients_group)
        
        self.clients_table = QTableWidget()
        self.clients_table.setColumnCount(4)
        self.clients_table.setHorizontalHeaderLabels(["客户端ID", "用户名", "连接时间", "订阅数"])
        self.clients_table.horizontalHeader().setStretchLastSection(True)
        self.clients_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.clients_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        clients_layout.addWidget(self.clients_table)
        
        layout.addWidget(clients_group)
        
        layout.addStretch()
        return tab
    
    def _create_client_tab(self) -> QWidget:
        """创建客户端Tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        
        # 连接配置组
        config_group = QGroupBox("连接配置")
        config_layout = QFormLayout(config_group)
        
        self.client_host_input = QLineEdit("localhost")
        config_layout.addRow("Broker主机:", self.client_host_input)
        
        self.client_port_input = QSpinBox()
        self.client_port_input.setRange(1, 65535)
        self.client_port_input.setValue(1883)
        config_layout.addRow("Broker端口:", self.client_port_input)
        
        self.client_id_input = QLineEdit()
        config_layout.addRow("客户端ID:", self.client_id_input)
        
        self.client_auth_check = QCheckBox("启用认证")
        config_layout.addRow("认证:", self.client_auth_check)
        
        self.client_username_input = QLineEdit()
        self.client_username_input.setEnabled(False)
        config_layout.addRow("用户名:", self.client_username_input)
        
        self.client_password_input = QLineEdit()
        self.client_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.client_password_input.setEnabled(False)
        config_layout.addRow("密码:", self.client_password_input)
        
        self.client_auth_check.toggled.connect(
            lambda checked: (
                self.client_username_input.setEnabled(checked),
                self.client_password_input.setEnabled(checked)
            )
        )
        
        layout.addWidget(config_group)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        
        self.client_connect_btn = QPushButton("连接")
        self.client_connect_btn.setStyleSheet("background-color: #4caf50;")
        self.client_connect_btn.clicked.connect(self._on_client_connect)
        btn_layout.addWidget(self.client_connect_btn)
        
        self.client_disconnect_btn = QPushButton("断开")
        self.client_disconnect_btn.setStyleSheet("background-color: #f44336;")
        self.client_disconnect_btn.setEnabled(False)
        self.client_disconnect_btn.clicked.connect(self._on_client_disconnect)
        btn_layout.addWidget(self.client_disconnect_btn)
        
        layout.addLayout(btn_layout)
        
        # 连接状态显示
        self.client_connection_status = QLabel("连接状态: 未连接")
        self.client_connection_status.setStyleSheet("color: #ff9800; font-weight: bold;")
        layout.addWidget(self.client_connection_status)
        
        # 连接信息显示
        self.client_info_label = QLabel("")
        self.client_info_label.setWordWrap(True)
        layout.addWidget(self.client_info_label)
        
        # 订阅组
        sub_group = QGroupBox("订阅管理")
        sub_layout = QVBoxLayout(sub_group)
        
        # 订阅输入
        sub_input_layout = QHBoxLayout()
        
        self.sub_topic_combo = QComboBox()
        self.sub_topic_combo.setEditable(True)
        self.sub_topic_combo.setPlaceholderText("选择或输入主题...")
        sub_input_layout.addWidget(self.sub_topic_combo)
        
        self.sub_qos_combo = QComboBox()
        self.sub_qos_combo.addItems(["QoS 0", "QoS 1", "QoS 2"])
        sub_input_layout.addWidget(self.sub_qos_combo)
        
        self.sub_btn = QPushButton("订阅")
        self.sub_btn.clicked.connect(self._on_subscribe)
        sub_input_layout.addWidget(self.sub_btn)
        
        sub_layout.addLayout(sub_input_layout)
        
        # 订阅列表
        self.sub_list = QTableWidget()
        self.sub_list.setColumnCount(3)
        self.sub_list.setHorizontalHeaderLabels(["主题", "QoS", "操作"])
        self.sub_list.horizontalHeader().setStretchLastSection(True)
        sub_layout.addWidget(self.sub_list)
        
        layout.addWidget(sub_group)
        
        layout.addStretch()
        return tab
    
    def _create_topic_tab(self) -> QWidget:
        """创建主题管理Tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 添加主题
        add_layout = QHBoxLayout()
        
        self.new_topic_input = QLineEdit()
        self.new_topic_input.setPlaceholderText("新主题...")
        add_layout.addWidget(self.new_topic_input)
        
        self.new_topic_desc_input = QLineEdit()
        self.new_topic_desc_input.setPlaceholderText("描述...")
        add_layout.addWidget(self.new_topic_desc_input)
        
        self.add_topic_btn = QPushButton("添加")
        self.add_topic_btn.clicked.connect(self._on_add_topic)
        add_layout.addWidget(self.add_topic_btn)
        
        layout.addLayout(add_layout)
        
        # 主题列表
        self.topic_table = QTableWidget()
        self.topic_table.setColumnCount(4)
        self.topic_table.setHorizontalHeaderLabels(["主题", "描述", "QoS", "操作"])
        self.topic_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.topic_table)
        
        return tab
    
    def _create_message_tab(self) -> QWidget:
        """创建消息查看Tab - 包含消息筛选和发布功能"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # ===== 消息发布区域 =====
        pub_group = QGroupBox("发布消息")
        pub_layout = QVBoxLayout(pub_group)
        
        # 主题选择
        pub_topic_layout = QHBoxLayout()
        pub_topic_layout.addWidget(QLabel("发布主题:"))
        self.msg_pub_topic_combo = QComboBox()
        self.msg_pub_topic_combo.setEditable(True)
        self.msg_pub_topic_combo.setPlaceholderText("选择或输入主题...")
        pub_topic_layout.addWidget(self.msg_pub_topic_combo)
        pub_layout.addLayout(pub_topic_layout)
        
        # 消息内容
        self.msg_pub_content = QTextEdit()
        self.msg_pub_content.setPlaceholderText("输入消息内容...")
        self.msg_pub_content.setMaximumHeight(80)
        pub_layout.addWidget(self.msg_pub_content)
        
        # 发布选项
        pub_options_layout = QHBoxLayout()
        self.msg_pub_qos = QComboBox()
        self.msg_pub_qos.addItems(["QoS 0", "QoS 1", "QoS 2"])
        pub_options_layout.addWidget(self.msg_pub_qos)
        
        self.msg_pub_retain = QCheckBox("保留")
        pub_options_layout.addWidget(self.msg_pub_retain)
        
        pub_options_layout.addStretch()
        
        self.msg_pub_btn = QPushButton("发布")
        self.msg_pub_btn.setStyleSheet("background-color: #2196f3; color: white;")
        self.msg_pub_btn.clicked.connect(self._on_publish_from_message_tab)
        pub_options_layout.addWidget(self.msg_pub_btn)
        
        pub_layout.addLayout(pub_options_layout)
        layout.addWidget(pub_group)
        
        # ===== 消息筛选区域 =====
        filter_group = QGroupBox("消息筛选")
        filter_layout = QHBoxLayout(filter_group)
        
        filter_layout.addWidget(QLabel("主题筛选:"))
        
        self.message_filter_combo = QComboBox()
        self.message_filter_combo.setEditable(True)
        self.message_filter_combo.setPlaceholderText("选择主题筛选...")
        filter_layout.addWidget(self.message_filter_combo)
        
        self.message_filter_btn = QPushButton("筛选")
        self.message_filter_btn.clicked.connect(self._on_filter_messages)
        filter_layout.addWidget(self.message_filter_btn)
        
        self.message_show_all_btn = QPushButton("查看全部")
        self.message_show_all_btn.clicked.connect(self._on_show_all_messages)
        filter_layout.addWidget(self.message_show_all_btn)
        
        self.message_clear_btn = QPushButton("清空")
        self.message_clear_btn.clicked.connect(self._on_clear_messages)
        filter_layout.addWidget(self.message_clear_btn)
        
        layout.addWidget(filter_group)
        
        # ===== 消息列表 =====
        self.message_table = QTableWidget()
        self.message_table.setColumnCount(6)
        self.message_table.setHorizontalHeaderLabels(["时间", "主题", "QoS", "大小", "消息ID", "内容预览"])
        self.message_table.horizontalHeader().setStretchLastSection(True)
        self.message_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.message_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.message_table)
        
        # 消息详情
        self.message_detail = QTextEdit()
        self.message_detail.setPlaceholderText("选择消息查看详情...")
        self.message_detail.setMaximumHeight(150)
        self.message_detail.setReadOnly(True)
        layout.addWidget(self.message_detail)
        
        self.message_table.itemClicked.connect(self._on_message_selected)
        
        return tab
    
    def _load_config(self):
        """加载配置到UI"""
        # 服务端配置
        self.server_host_input.setText(self.config.mqtt_server.host)
        self.server_port_input.setValue(self.config.mqtt_server.port)
        self.server_auth_check.setChecked(self.config.mqtt_server.enable_auth)
        self.server_username_input.setText(self.config.mqtt_server.username)
        self.server_password_input.setText(self.config.mqtt_server.password)
        
        # 加载严格模式配置
        self.strict_mode_check.setChecked(self.config.mqtt_server.strict_topic_mode)
        if self.config.mqtt_server.allowed_topics:
            self.allowed_topics_input.setPlainText('\n'.join(self.config.mqtt_server.allowed_topics))
        
        # 客户端配置
        self.client_host_input.setText(self.config.mqtt_client.broker_host)
        self.client_port_input.setValue(self.config.mqtt_client.broker_port)
        self.client_id_input.setText(self.config.mqtt_client.client_id)
        self.client_auth_check.setChecked(
            bool(self.config.mqtt_client.username)
        )
        self.client_username_input.setText(self.config.mqtt_client.username)
        self.client_password_input.setText(self.config.mqtt_client.password)
        
        # 加载主题列表
        self._load_topic_table()
    
    def _add_quick_topic(self, topic: str):
        """快速添加主题"""
        current_text = self.allowed_topics_input.toPlainText()
        topics = [t.strip() for t in current_text.split('\n') if t.strip()]
        
        if topic not in topics:
            topics.append(topic)
            self.allowed_topics_input.setPlainText('\n'.join(topics))
    
    def _load_topic_table(self):
        """加载主题列表"""
        self.topic_table.setRowCount(len(self.config.mqtt_topics))
        
        for i, topic_item in enumerate(self.config.mqtt_topics):
            self.topic_table.setItem(i, 0, QTableWidgetItem(topic_item.get('topic', '')))
            self.topic_table.setItem(i, 1, QTableWidgetItem(topic_item.get('description', '')))
            self.topic_table.setItem(i, 2, QTableWidgetItem(str(topic_item.get('qos', 0))))
            
            # 删除按钮
            delete_btn = QPushButton("删除")
            delete_btn.clicked.connect(lambda checked, t=topic_item.get('topic'): self._on_delete_topic(t))
            self.topic_table.setCellWidget(i, 3, delete_btn)
    
    def _on_server_start(self):
        """启动服务器"""
        # 保存配置
        self.config.mqtt_server.host = self.server_host_input.text()
        self.config.mqtt_server.port = self.server_port_input.value()
        self.config.mqtt_server.enable_auth = self.server_auth_check.isChecked()
        self.config.mqtt_server.username = self.server_username_input.text()
        self.config.mqtt_server.password = self.server_password_input.text()
        
        # 保存严格模式配置
        self.config.mqtt_server.strict_topic_mode = self.strict_mode_check.isChecked()
        allowed_topics_text = self.allowed_topics_input.toPlainText().strip()
        self.config.mqtt_server.allowed_topics = [
            t.strip() for t in allowed_topics_text.split('\n') if t.strip()
        ]
        
        self.config_manager.save()
        
        self.server_start_requested.emit()
        
        self.server_start_btn.setEnabled(False)
        self.server_stop_btn.setEnabled(True)
        self.server_status_label.setText("状态: 运行中")
        self.server_status_label.setStyleSheet("color: #4caf50;")
    
    def _on_server_stop(self):
        """停止服务器"""
        self.server_stop_requested.emit()
        
        self.server_start_btn.setEnabled(True)
        self.server_stop_btn.setEnabled(False)
        self.server_status_label.setText("状态: 停止")
        self.server_status_label.setStyleSheet("")
    
    def _on_client_connect(self):
        """连接客户端"""
        # 防重复点击检查
        if hasattr(self, '_is_connecting') and self._is_connecting:
            logger.warning("[MQTT Client] 连接正在进行中，忽略重复点击")
            return
        
        # 保存配置
        self.config.mqtt_client.broker_host = self.client_host_input.text()
        self.config.mqtt_client.broker_port = self.client_port_input.value()
        self.config.mqtt_client.client_id = self.client_id_input.text()
        
        if self.client_auth_check.isChecked():
            self.config.mqtt_client.username = self.client_username_input.text()
            self.config.mqtt_client.password = self.client_password_input.text()
        else:
            self.config.mqtt_client.username = ""
            self.config.mqtt_client.password = ""
        
        self.config_manager.save()
        
        # 设置连接中标志
        self._is_connecting = True
        
        # 禁用所有相关按钮，防止重复操作
        self.client_connect_btn.setEnabled(False)
        self.client_connect_btn.setText("正在连接中...")
        self.client_connect_btn.setStyleSheet("background-color: #ff9800;")  # 橙色表示进行中
        self.client_disconnect_btn.setEnabled(False)
        
        # 禁用输入框，防止连接过程中修改配置
        self.client_host_input.setEnabled(False)
        self.client_port_input.setEnabled(False)
        self.client_id_input.setEnabled(False)
        self.client_auth_check.setEnabled(False)
        self.client_username_input.setEnabled(False)
        self.client_password_input.setEnabled(False)
        
        # 显示正在连接的状态
        self.client_connection_status.setText("连接状态: 正在连接... ⏳")
        self.client_connection_status.setStyleSheet("color: #ff9800; font-weight: bold;")
        
        logger.info(f"[MQTT Client] 正在连接到 {self.config.mqtt_client.broker_host}:{self.config.mqtt_client.broker_port}")
        
        self.client_connect_requested.emit()
    
    def _on_client_disconnect(self):
        """断开客户端"""
        self.client_disconnect_requested.emit()
    
    def on_client_connect_success(self):
        """客户端连接成功"""
        from PyQt6.QtCore import QTimer
        
        # 重置连接中标志
        self._is_connecting = False
        
        # 恢复按钮状态
        self.client_connect_btn.setEnabled(False)
        self.client_connect_btn.setText("已连接")
        self.client_connect_btn.setStyleSheet("background-color: #4caf50;")  # 绿色表示已连接
        self.client_disconnect_btn.setEnabled(True)
        self.sub_btn.setEnabled(True)
        
        # 更新连接状态显示
        self.client_connection_status.setText("连接状态: 已连接 ✓")
        self.client_connection_status.setStyleSheet("color: #4caf50; font-weight: bold;")
        
        # 显示连接信息
        broker = self.config.mqtt_client.broker_host
        port = self.config.mqtt_client.broker_port
        client_id = self.config.mqtt_client.client_id or "自动分配"
        self.client_info_label.setText(f"Broker: {broker}:{port}\n客户端ID: {client_id}")
        
        logger.info(f"[MQTT Client] 界面更新: 客户端已连接到 {broker}:{port}")
        
        # 使用定时器延迟显示成功提示，避免阻塞UI
        QTimer.singleShot(100, lambda: QMessageBox.information(self, "连接成功", f"已成功连接到 MQTT Broker:\n{broker}:{port}"))
    
    def on_client_connect_failed(self):
        """客户端连接失败"""
        from PyQt6.QtCore import QTimer
        
        # 重置连接中标志
        self._is_connecting = False
        
        # 恢复按钮和输入框状态
        self.client_connect_btn.setEnabled(True)
        self.client_connect_btn.setText("连接")
        self.client_connect_btn.setStyleSheet("background-color: #4caf50;")  # 恢复绿色
        self.client_disconnect_btn.setEnabled(False)
        
        # 恢复输入框
        self.client_host_input.setEnabled(True)
        self.client_port_input.setEnabled(True)
        self.client_id_input.setEnabled(True)
        self.client_auth_check.setEnabled(True)
        # 根据认证状态恢复用户名密码输入框
        self.client_username_input.setEnabled(self.client_auth_check.isChecked())
        self.client_password_input.setEnabled(self.client_auth_check.isChecked())
        
        # 更新连接状态显示
        self.client_connection_status.setText("连接状态: 连接失败 ✗")
        self.client_connection_status.setStyleSheet("color: #f44336; font-weight: bold;")
        self.client_info_label.setText("")
        
        logger.error("[MQTT Client] 界面更新: 客户端连接失败")
        
        # 使用定时器延迟显示失败提示，避免阻塞UI
        QTimer.singleShot(100, lambda: QMessageBox.critical(self, "连接失败", "无法连接到 MQTT Broker，请检查:\n1. Broker地址和端口是否正确\n2. 网络连接是否正常\n3. Broker是否已启动"))
    
    def on_client_disconnect(self):
        """客户端断开"""
        # 重置连接中标志
        self._is_connecting = False
        
        # 恢复按钮状态
        self.client_connect_btn.setEnabled(True)
        self.client_connect_btn.setText("连接")
        self.client_connect_btn.setStyleSheet("background-color: #4caf50;")  # 恢复绿色
        self.client_disconnect_btn.setEnabled(False)
        self.sub_btn.setEnabled(False)
        self.sub_list.setRowCount(0)
        
        # 恢复输入框
        self.client_host_input.setEnabled(True)
        self.client_port_input.setEnabled(True)
        self.client_id_input.setEnabled(True)
        self.client_auth_check.setEnabled(True)
        self.client_username_input.setEnabled(self.client_auth_check.isChecked())
        self.client_password_input.setEnabled(self.client_auth_check.isChecked())
        
        # 更新连接状态显示
        self.client_connection_status.setText("连接状态: 已断开")
        self.client_connection_status.setStyleSheet("color: #f44336; font-weight: bold;")
        self.client_info_label.setText("")
        
        logger.info("[MQTT Client] 界面更新: 客户端已断开连接")
    
    def _on_subscribe(self):
        """订阅主题"""
        topic = self.sub_topic_combo.currentText().strip()
        if not topic:
            QMessageBox.warning(self, "警告", "请选择或输入主题")
            return
        
        qos = self.sub_qos_combo.currentIndex()
        
        # 添加到订阅列表
        row = self.sub_list.rowCount()
        self.sub_list.insertRow(row)
        self.sub_list.setItem(row, 0, QTableWidgetItem(topic))
        self.sub_list.setItem(row, 1, QTableWidgetItem(str(qos)))
        
        # 取消订阅按钮
        unsub_btn = QPushButton("取消")
        unsub_btn.clicked.connect(lambda: self._on_unsubscribe(row, topic))
        self.sub_list.setCellWidget(row, 2, unsub_btn)
        
        # 清空选择
        self.sub_topic_combo.setCurrentIndex(-1)
        
        logger.info(f"订阅主题: {topic} (QoS: {qos})")
    
    def _on_unsubscribe(self, row: int, topic: str):
        """取消订阅"""
        self.sub_list.removeRow(row)
        logger.info(f"取消订阅: {topic}")
    
    def _on_publish(self):
        """发布消息"""
        topic = self.pub_topic_combo.currentText().strip()
        message = self.pub_message_input.toPlainText()
        
        if not topic:
            QMessageBox.warning(self, "警告", "请选择或输入主题")
            return
        
        if not message:
            QMessageBox.warning(self, "警告", "请输入消息内容")
            return
        
        # 触发发布信号（主窗口会处理实际的MQTT发布）
        logger.info(f"发布消息到 {topic}: {message[:50]}...")
        self.message_publish_requested.emit(topic, message, 0, False)
        
        # 清空输入
        self.pub_message_input.clear()
    
    def _on_add_topic(self):
        """添加主题"""
        topic = self.new_topic_input.text().strip()
        description = self.new_topic_desc_input.text().strip()
        
        if not topic:
            QMessageBox.warning(self, "警告", "请输入主题")
            return
        
        self.config_manager.add_mqtt_topic(topic, description)
        self._load_topic_table()
        
        self.new_topic_input.clear()
        self.new_topic_desc_input.clear()
    
    def _on_delete_topic(self, topic: str):
        """删除主题"""
        self.config_manager.remove_mqtt_topic(topic)
        self._load_topic_table()
    
    def _on_filter_messages(self):
        """筛选消息 - 根据主题过滤消息列表"""
        filter_topic = self.message_filter_combo.currentText().strip()
        self.current_filter = filter_topic
        
        logger.info(f"筛选消息，主题: {filter_topic}")
        
        # 清空当前显示
        self.message_table.setRowCount(0)
        
        # 根据筛选条件重新填充
        for msg_data in self.all_messages:
            if not filter_topic or filter_topic in msg_data.get('topic', ''):
                self._add_message_to_table_filtered(msg_data)
    
    def _add_message_to_table_filtered(self, msg_data: dict):
        """添加消息到消息列表（不检查筛选条件，用于重新填充）"""
        row = self.message_table.rowCount()
        self.message_table.insertRow(row)
        
        timestamp = datetime.fromtimestamp(msg_data.get('timestamp', time.time())).strftime('%H:%M:%S.%f')[:-3]
        
        self.message_table.setItem(row, 0, QTableWidgetItem(timestamp))
        self.message_table.setItem(row, 1, QTableWidgetItem(msg_data.get('topic', '')))
        self.message_table.setItem(row, 2, QTableWidgetItem(str(msg_data.get('qos', 0))))
        
        payload = msg_data.get('payload', '')
        size = len(payload.encode('utf-8')) if isinstance(payload, str) else len(payload)
        self.message_table.setItem(row, 3, QTableWidgetItem(f"{size} bytes"))
        self.message_table.setItem(row, 4, QTableWidgetItem(msg_data.get('id', '')))
        
        # 内容预览
        preview = payload[:50] + "..." if len(str(payload)) > 50 else str(payload)
        self.message_table.setItem(row, 5, QTableWidgetItem(preview))
    
    def _on_show_all_messages(self):
        """查看全部消息 - 取消筛选"""
        self.current_filter = ""  # 清空筛选条件
        self.message_filter_combo.setCurrentIndex(-1)  # 重置筛选下拉框
        
        # 清空当前显示
        self.message_table.setRowCount(0)
        
        # 显示所有消息
        for msg_data in self.all_messages:
            self._add_message_to_table_filtered(msg_data)
        
        logger.info(f"查看全部消息，共 {len(self.all_messages)} 条")
    
    def _on_clear_messages(self):
        """清空消息"""
        self.all_messages.clear()  # 清空存储的所有消息
        self.current_filter = ""  # 重置筛选条件
        self.message_table.setRowCount(0)
        self.message_detail.clear()
        self.message_filter_combo.setCurrentIndex(-1)  # 重置筛选下拉框
    
    def _on_publish_from_message_tab(self):
        """从消息查看页面发布消息"""
        topic = self.msg_pub_topic_combo.currentText().strip()
        message = self.msg_pub_content.toPlainText()
        qos = self.msg_pub_qos.currentIndex()
        retain = self.msg_pub_retain.isChecked()
        
        if not topic:
            QMessageBox.warning(self, "警告", "请选择或输入发布主题")
            return
        
        if not message:
            QMessageBox.warning(self, "警告", "请输入消息内容")
            return
        
        # 生成消息ID
        msg_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        
        # 构建消息数据
        msg_data = {
            'id': msg_id,
            'topic': topic,
            'payload': message,
            'qos': qos,
            'retain': retain,
            'timestamp': time.time()
        }
        
        logger.info(f"发布消息: {topic} (ID: {msg_id})")
        
        # 清空输入
        self.msg_pub_content.clear()
        
        # 触发发布信号（主窗口会处理实际的MQTT发布）
        # 消息会通过MQTT服务器的回调自动添加到列表，不需要手动添加
        self.message_publish_requested.emit(topic, message, qos, retain)
        
    def _add_message_to_table(self, msg_data: dict):
        """添加消息到消息列表"""
        logger.info(f"[MQTTWidget] _add_message_to_table 被调用: {msg_data.get('topic', 'unknown')}, id={msg_data.get('id', 'no-id')}")
        
        # 保存到完整消息列表
        self.all_messages.append(msg_data)
        
        # 检查是否符合当前筛选条件
        if self.current_filter and self.current_filter not in msg_data.get('topic', ''):
            return  # 不符合筛选条件，不显示
        
        row = self.message_table.rowCount()
        self.message_table.insertRow(row)
        logger.info(f"[MQTTWidget] 消息已添加到表格第 {row} 行")
        
        timestamp = datetime.fromtimestamp(msg_data.get('timestamp', time.time())).strftime('%H:%M:%S.%f')[:-3]
        
        self.message_table.setItem(row, 0, QTableWidgetItem(timestamp))
        self.message_table.setItem(row, 1, QTableWidgetItem(msg_data.get('topic', '')))
        self.message_table.setItem(row, 2, QTableWidgetItem(str(msg_data.get('qos', 0))))
        
        payload = msg_data.get('payload', '')
        size = len(payload.encode('utf-8')) if isinstance(payload, str) else len(payload)
        self.message_table.setItem(row, 3, QTableWidgetItem(f"{size} bytes"))
        self.message_table.setItem(row, 4, QTableWidgetItem(msg_data.get('id', '')))
        
        # 内容预览
        preview = payload[:50] + "..." if len(str(payload)) > 50 else str(payload)
        self.message_table.setItem(row, 5, QTableWidgetItem(preview))
    
    def _on_message_selected(self):
        """消息被选中 - 显示消息详情"""
        row = self.message_table.currentRow()
        if row < 0:
            return
        
        try:
            # 获取消息各列数据
            time_item = self.message_table.item(row, 0)
            topic_item = self.message_table.item(row, 1)
            qos_item = self.message_table.item(row, 2)
            size_item = self.message_table.item(row, 3)
            msg_id_item = self.message_table.item(row, 4)
            preview_item = self.message_table.item(row, 5)
            
            # 构建详情文本
            details = []
            details.append(f"时间: {time_item.text() if time_item else '-'}")
            details.append(f"主题: {topic_item.text() if topic_item else '-'}")
            details.append(f"QoS: {qos_item.text() if qos_item else '-'}")
            details.append(f"大小: {size_item.text() if size_item else '-'}")
            details.append(f"消息ID: {msg_id_item.text() if msg_id_item else '-'}")
            details.append("-" * 40)
            details.append("内容:")
            details.append(preview_item.text() if preview_item else '-')
            
            self.message_detail.setText("\n".join(details))
            
        except Exception as e:
            logger.error(f"显示消息详情失败: {e}")
            self.message_detail.setText(f"无法显示详情: {e}")
    
    from PyQt6.QtCore import pyqtSlot
    
    @pyqtSlot(str)
    def on_client_connected(self, client_id: str):
        """有客户端连接"""
        logger.info(f"服务端 - 客户端连接: {client_id}")
        # 更新连接客户端数显示
        try:
            current_text = self.server_clients_label.text()
            current_count = int(current_text.split(":")[1].strip())
            self.server_clients_label.setText(f"连接客户端: {current_count + 1}")
            # 添加到已连接客户端列表
            self._add_client_to_list(client_id)
        except Exception as e:
            logger.error(f"更新客户端连接状态失败: {e}")
    
    @pyqtSlot(str)
    def on_client_disconnected(self, client_id: str):
        """有客户端断开"""
        logger.info(f"服务端 - 客户端断开: {client_id}")
        # 更新连接客户端数显示
        try:
            current_text = self.server_clients_label.text()
            current_count = int(current_text.split(":")[1].strip())
            if current_count > 0:
                self.server_clients_label.setText(f"连接客户端: {current_count - 1}")
            # 从已连接客户端列表移除
            self._remove_client_from_list(client_id)
        except Exception as e:
            logger.error(f"更新客户端断开状态失败: {e}")
    
    def _add_client_to_list(self, client_id: str):
        """添加客户端到列表"""
        from datetime import datetime
        from PyQt6.QtWidgets import QTableWidgetItem
        
        row = self.clients_table.rowCount()
        self.clients_table.insertRow(row)
        self.clients_table.setItem(row, 0, QTableWidgetItem(client_id))
        self.clients_table.setItem(row, 1, QTableWidgetItem("-"))
        self.clients_table.setItem(row, 2, QTableWidgetItem(datetime.now().strftime("%H:%M:%S")))
        self.clients_table.setItem(row, 3, QTableWidgetItem("0"))
        
        # 更新日志
        self._log(f"客户端 [{client_id}] 已连接")
    
    def _remove_client_from_list(self, client_id: str):
        """从列表移除客户端"""
        for row in range(self.clients_table.rowCount()):
            if self.clients_table.item(row, 0) and self.clients_table.item(row, 0).text() == client_id:
                self.clients_table.removeRow(row)
                break
        
        # 更新日志
        self._log(f"客户端 [{client_id}] 已断开")
    
    def _log(self, message: str):
        """添加日志"""
        logger.info(f"[MQTTWidget] {message}")
    
    @pyqtSlot(object)
    def on_message_received(self, message):
        """收到消息"""
        # 构建消息数据
        import uuid
        msg_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        msg_data = {
            'id': msg_id,
            'topic': message.topic,
            'payload': message.payload_text,
            'qos': message.qos,
            'retain': getattr(message, 'retain', False),
            'timestamp': message.timestamp,
            'size': getattr(message, 'size', len(message.payload_text))
        }
        
        # 检查是否已经添加过（避免重复）
        if msg_id in self._added_message_ids:
            logger.debug(f"[MQTTWidget] 跳过重复消息: {message.topic}")
            return
        
        # 标记为已添加
        self._added_message_ids.add(msg_id)
        
        # 限制集合大小
        if len(self._added_message_ids) > 1000:
            self._added_message_ids = set(list(self._added_message_ids)[-500:])
        
        # 保存到完整消息列表并显示
        self._add_message_to_table(msg_data)
        
        # 限制总消息数
        if len(self.all_messages) > 1000:
            self.all_messages.pop(0)
    
    from PyQt6.QtCore import pyqtSlot
    
    @pyqtSlot(object)
    def on_client_message(self, message):
        """客户端收到消息"""
        self.on_message_received(message)
    
    def _refresh_data(self):
        """刷新数据"""
        # 更新客户端Tab的订阅主题下拉框
        self._update_sub_topic_combo()
        # 更新消息查看页面的主题下拉框
        self._update_message_tab_topic_combos()
    
    def _update_sub_topic_combo(self):
        """更新客户端Tab的订阅主题下拉框，显示服务端已订阅的主题和配置中的主题"""
        try:
            # 获取当前输入的文本
            current_text = self.sub_topic_combo.currentText()
            
            # 获取服务端动态主题
            server_topics = []
            if self.get_server_topics_callback:
                server_topics = self.get_server_topics_callback()
            
            # 获取配置中的静态主题
            config_topics = [item.get('topic', '') for item in self.config.mqtt_topics if item.get('topic')]
            
            # 合并主题列表（去重）
            all_topics = list(set(server_topics + config_topics))
            
            # 保存当前列表中的自定义输入（非服务端/配置主题）
            custom_items = []
            for i in range(self.sub_topic_combo.count()):
                item_text = self.sub_topic_combo.itemText(i)
                if item_text and item_text not in all_topics:
                    custom_items.append(item_text)
            
            # 清空并重新填充
            self.sub_topic_combo.clear()
            
            # 添加服务端和配置主题
            for topic in sorted(all_topics):
                self.sub_topic_combo.addItem(topic)
            
            # 添加自定义主题（之前用户输入的）
            for custom in custom_items:
                if custom not in all_topics:
                    self.sub_topic_combo.addItem(custom)
            
            # 恢复之前的选择
            if current_text:
                index = self.sub_topic_combo.findText(current_text)
                if index >= 0:
                    self.sub_topic_combo.setCurrentIndex(index)
                else:
                    self.sub_topic_combo.setEditText(current_text)
                    
        except Exception as e:
            logger.debug(f"更新订阅主题列表失败: {e}")
    
    def _update_message_tab_topic_combos(self):
        """更新消息查看页面的主题下拉框"""
        try:
            # 获取服务端动态主题
            server_topics = []
            if self.get_server_topics_callback:
                server_topics = self.get_server_topics_callback()
            
            # 获取配置中的静态主题
            config_topics = [item.get('topic', '') for item in self.config.mqtt_topics if item.get('topic')]
            
            # 合并主题列表（去重）
            all_topics = list(set(server_topics + config_topics))
            
            # 更新发布主题下拉框
            current_pub_text = self.msg_pub_topic_combo.currentText()
            self._update_combo_with_topics(self.msg_pub_topic_combo, all_topics, current_pub_text)
            
            # 更新筛选主题下拉框
            current_filter_text = self.message_filter_combo.currentText()
            self._update_combo_with_topics(self.message_filter_combo, all_topics, current_filter_text)
            
        except Exception as e:
            logger.debug(f"更新消息页面主题列表失败: {e}")
    
    def _update_combo_with_topics(self, combo: QComboBox, topics: list, current_text: str):
        """辅助方法：用主题列表更新下拉框"""
        # 保存当前列表中的自定义输入
        custom_items = []
        for i in range(combo.count()):
            item_text = combo.itemText(i)
            if item_text and item_text not in topics:
                custom_items.append(item_text)
        
        # 清空并重新填充
        combo.clear()
        
        # 添加服务端主题
        for topic in sorted(topics):
            combo.addItem(topic)
        
        # 添加自定义主题
        for custom in custom_items:
            if custom not in topics:
                combo.addItem(custom)
        
        # 恢复之前的选择
        if current_text:
            index = combo.findText(current_text)
            if index >= 0:
                combo.setCurrentIndex(index)
            else:
                combo.setEditText(current_text)
    
    def set_get_server_topics_callback(self, callback):
        """设置获取服务端主题列表的回调函数"""
        self.get_server_topics_callback = callback
    
    def apply_theme(self, theme: str):
        """应用主题"""
        # 主题已在主窗口中设置，这里可以添加额外的主题相关调整
        pass
