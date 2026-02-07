"""
设置对话框
提供配置编辑界面
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QTabWidget, QWidget, QFormLayout, QDialogButtonBox, QFileDialog,
    QMessageBox, QGroupBox
)
from PyQt6.QtCore import Qt
from loguru import logger

from ..core.config_manager import ConfigManager


class SettingsDialog(QDialog):
    """设置对话框"""
    
    def __init__(self, config_manager: ConfigManager, parent=None):
        super().__init__(parent)
        
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        
        self.setWindowTitle("设置")
        self.setMinimumSize(600, 500)
        
        self._init_ui()
        self._load_config()
    
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Tab控件
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # 常规设置Tab
        self.general_tab = self._create_general_tab()
        self.tab_widget.addTab(self.general_tab, "常规")
        
        # MQTT服务端设置Tab
        self.mqtt_server_tab = self._create_mqtt_server_tab()
        self.tab_widget.addTab(self.mqtt_server_tab, "MQTT服务端")
        
        # MQTT客户端设置Tab
        self.mqtt_client_tab = self._create_mqtt_client_tab()
        self.tab_widget.addTab(self.mqtt_client_tab, "MQTT客户端")
        
        # 推理设置Tab
        self.inference_tab = self._create_inference_tab()
        self.tab_widget.addTab(self.inference_tab, "推理")
        
        # 按钮框
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_save)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _create_general_tab(self) -> QWidget:
        """创建常规设置Tab"""
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setSpacing(10)
        
        # 应用名称
        self.app_name_input = QLineEdit()
        layout.addRow("应用名称:", self.app_name_input)
        
        # 窗口大小
        size_layout = QHBoxLayout()
        self.window_width_spin = QSpinBox()
        self.window_width_spin.setRange(800, 3840)
        self.window_width_spin.setSingleStep(100)
        size_layout.addWidget(self.window_width_spin)
        size_layout.addWidget(QLabel("x"))
        self.window_height_spin = QSpinBox()
        self.window_height_spin.setRange(600, 2160)
        self.window_height_spin.setSingleStep(100)
        size_layout.addWidget(self.window_height_spin)
        size_layout.addStretch()
        layout.addRow("窗口大小:", size_layout)
        
        # 主题
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["深色", "浅色"])
        layout.addRow("主题:", self.theme_combo)
        
        # 语言
        self.language_combo = QComboBox()
        self.language_combo.addItems(["简体中文", "English"])
        layout.addRow("语言:", self.language_combo)
        
        # 日志级别
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        layout.addRow("日志级别:", self.log_level_combo)
        
        return tab
    
    def _create_mqtt_server_tab(self) -> QWidget:
        """创建MQTT服务端设置Tab"""
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setSpacing(10)
        
        # 主机
        self.mqtt_server_host_input = QLineEdit()
        layout.addRow("主机:", self.mqtt_server_host_input)
        
        # 端口
        self.mqtt_server_port_spin = QSpinBox()
        self.mqtt_server_port_spin.setRange(1, 65535)
        layout.addRow("端口:", self.mqtt_server_port_spin)
        
        # WebSocket端口
        self.mqtt_ws_port_spin = QSpinBox()
        self.mqtt_ws_port_spin.setRange(1, 65535)
        layout.addRow("WebSocket端口:", self.mqtt_ws_port_spin)
        
        # 认证
        self.mqtt_server_auth_check = QCheckBox("启用认证")
        layout.addRow("认证:", self.mqtt_server_auth_check)
        
        # 用户名
        self.mqtt_server_username_input = QLineEdit()
        layout.addRow("用户名:", self.mqtt_server_username_input)
        
        # 密码
        self.mqtt_server_password_input = QLineEdit()
        self.mqtt_server_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addRow("密码:", self.mqtt_server_password_input)
        
        # 最大连接数
        self.mqtt_max_connections_spin = QSpinBox()
        self.mqtt_max_connections_spin.setRange(1, 10000)
        layout.addRow("最大连接数:", self.mqtt_max_connections_spin)
        
        return tab
    
    def _create_mqtt_client_tab(self) -> QWidget:
        """创建MQTT客户端设置Tab"""
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setSpacing(10)
        
        # Broker主机
        self.mqtt_client_host_input = QLineEdit()
        layout.addRow("Broker主机:", self.mqtt_client_host_input)
        
        # Broker端口
        self.mqtt_client_port_spin = QSpinBox()
        self.mqtt_client_port_spin.setRange(1, 65535)
        layout.addRow("Broker端口:", self.mqtt_client_port_spin)
        
        # 客户端ID
        self.mqtt_client_id_input = QLineEdit()
        layout.addRow("客户端ID:", self.mqtt_client_id_input)
        
        # 认证
        self.mqtt_client_auth_check = QCheckBox("启用认证")
        layout.addRow("认证:", self.mqtt_client_auth_check)
        
        # 用户名
        self.mqtt_client_username_input = QLineEdit()
        layout.addRow("用户名:", self.mqtt_client_username_input)
        
        # 密码
        self.mqtt_client_password_input = QLineEdit()
        self.mqtt_client_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addRow("密码:", self.mqtt_client_password_input)
        
        # KeepAlive
        self.mqtt_keepalive_spin = QSpinBox()
        self.mqtt_keepalive_spin.setRange(5, 3600)
        layout.addRow("KeepAlive:", self.mqtt_keepalive_spin)
        
        # 重连延迟
        self.mqtt_reconnect_spin = QSpinBox()
        self.mqtt_reconnect_spin.setRange(1, 300)
        layout.addRow("重连延迟(秒):", self.mqtt_reconnect_spin)
        
        return tab
    
    def _create_inference_tab(self) -> QWidget:
        """创建推理设置Tab"""
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setSpacing(10)
        
        # 模型路径
        model_path_layout = QHBoxLayout()
        self.model_path_input = QLineEdit()
        model_path_layout.addWidget(self.model_path_input)
        self.browse_model_btn = QPushButton("浏览")
        self.browse_model_btn.clicked.connect(self._browse_model)
        model_path_layout.addWidget(self.browse_model_btn)
        layout.addRow("默认模型路径:", model_path_layout)
        
        # 模型类型
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["目标检测", "实例分割", "关键点检测", "图像分类"])
        layout.addRow("默认模型类型:", self.model_type_combo)
        
        # 设备
        self.device_combo = QComboBox()
        self.device_combo.addItems(["自动", "CPU", "CUDA"])
        layout.addRow("推理设备:", self.device_combo)
        
        # 置信度阈值
        self.conf_threshold_spin = QDoubleSpinBox()
        self.conf_threshold_spin.setRange(0.0, 1.0)
        self.conf_threshold_spin.setSingleStep(0.05)
        layout.addRow("置信度阈值:", self.conf_threshold_spin)
        
        # IOU阈值
        self.iou_threshold_spin = QDoubleSpinBox()
        self.iou_threshold_spin.setRange(0.0, 1.0)
        self.iou_threshold_spin.setSingleStep(0.05)
        layout.addRow("IOU阈值:", self.iou_threshold_spin)
        
        # 图像尺寸
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(32, 1280)
        self.img_size_spin.setSingleStep(32)
        layout.addRow("图像尺寸:", self.img_size_spin)
        
        # 半精度
        self.half_precision_check = QCheckBox("使用半精度(FP16)")
        layout.addRow("半精度:", self.half_precision_check)
        
        return tab
    
    def _load_config(self):
        """加载配置到UI"""
        # 常规设置
        self.app_name_input.setText(self.config.app_name)
        self.window_width_spin.setValue(self.config.window_width)
        self.window_height_spin.setValue(self.config.window_height)
        self.theme_combo.setCurrentIndex(0 if self.config.theme == "dark" else 1)
        self.language_combo.setCurrentIndex(0 if self.config.language == "zh_CN" else 1)
        
        log_level_index = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}.get(
            self.config.log_level, 1
        )
        self.log_level_combo.setCurrentIndex(log_level_index)
        
        # MQTT服务端设置
        self.mqtt_server_host_input.setText(self.config.mqtt_server.host)
        self.mqtt_server_port_spin.setValue(self.config.mqtt_server.port)
        self.mqtt_ws_port_spin.setValue(self.config.mqtt_server.websocket_port)
        self.mqtt_server_auth_check.setChecked(self.config.mqtt_server.enable_auth)
        self.mqtt_server_username_input.setText(self.config.mqtt_server.username)
        self.mqtt_server_password_input.setText(self.config.mqtt_server.password)
        self.mqtt_max_connections_spin.setValue(self.config.mqtt_server.max_connections)
        
        # MQTT客户端设置
        self.mqtt_client_host_input.setText(self.config.mqtt_client.broker_host)
        self.mqtt_client_port_spin.setValue(self.config.mqtt_client.broker_port)
        self.mqtt_client_id_input.setText(self.config.mqtt_client.client_id)
        self.mqtt_client_auth_check.setChecked(bool(self.config.mqtt_client.username))
        self.mqtt_client_username_input.setText(self.config.mqtt_client.username)
        self.mqtt_client_password_input.setText(self.config.mqtt_client.password)
        self.mqtt_keepalive_spin.setValue(self.config.mqtt_client.keepalive)
        self.mqtt_reconnect_spin.setValue(self.config.mqtt_client.reconnect_delay)
        
        # 推理设置
        self.model_path_input.setText(self.config.inference.model_path)
        
        model_type_index = {
            "detection": 0, "segmentation": 1, "pose": 2, "classification": 3
        }.get(self.config.inference.model_type, 0)
        self.model_type_combo.setCurrentIndex(model_type_index)
        
        device_index = {"auto": 0, "cpu": 1, "cuda": 2}.get(self.config.inference.device, 0)
        self.device_combo.setCurrentIndex(device_index)
        
        self.conf_threshold_spin.setValue(self.config.inference.conf_threshold)
        self.iou_threshold_spin.setValue(self.config.inference.iou_threshold)
        self.img_size_spin.setValue(self.config.inference.img_size)
        self.half_precision_check.setChecked(self.config.inference.half_precision)
    
    def _browse_model(self):
        """浏览模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.pt *.pth *.onnx);;所有文件 (*.*)"
        )
        if file_path:
            self.model_path_input.setText(file_path)
    
    def _on_save(self):
        """保存配置"""
        # 常规设置
        self.config.app_name = self.app_name_input.text()
        self.config.window_width = self.window_width_spin.value()
        self.config.window_height = self.window_height_spin.value()
        self.config.theme = "dark" if self.theme_combo.currentIndex() == 0 else "light"
        self.config.language = "zh_CN" if self.language_combo.currentIndex() == 0 else "en_US"
        self.config.log_level = self.log_level_combo.currentText()
        
        # MQTT服务端设置
        self.config.mqtt_server.host = self.mqtt_server_host_input.text()
        self.config.mqtt_server.port = self.mqtt_server_port_spin.value()
        self.config.mqtt_server.websocket_port = self.mqtt_ws_port_spin.value()
        self.config.mqtt_server.enable_auth = self.mqtt_server_auth_check.isChecked()
        self.config.mqtt_server.username = self.mqtt_server_username_input.text()
        self.config.mqtt_server.password = self.mqtt_server_password_input.text()
        self.config.mqtt_server.max_connections = self.mqtt_max_connections_spin.value()
        
        # MQTT客户端设置
        self.config.mqtt_client.broker_host = self.mqtt_client_host_input.text()
        self.config.mqtt_client.broker_port = self.mqtt_client_port_spin.value()
        self.config.mqtt_client.client_id = self.mqtt_client_id_input.text()
        
        if self.mqtt_client_auth_check.isChecked():
            self.config.mqtt_client.username = self.mqtt_client_username_input.text()
            self.config.mqtt_client.password = self.mqtt_client_password_input.text()
        else:
            self.config.mqtt_client.username = ""
            self.config.mqtt_client.password = ""
        
        self.config.mqtt_client.keepalive = self.mqtt_keepalive_spin.value()
        self.config.mqtt_client.reconnect_delay = self.mqtt_reconnect_spin.value()
        
        # 推理设置
        self.config.inference.model_path = self.model_path_input.text()
        self.config.inference.model_type = ["detection", "segmentation", "pose", "classification"][
            self.model_type_combo.currentIndex()
        ]
        self.config.inference.device = ["auto", "cpu", "cuda"][self.device_combo.currentIndex()]
        self.config.inference.conf_threshold = self.conf_threshold_spin.value()
        self.config.inference.iou_threshold = self.iou_threshold_spin.value()
        self.config.inference.img_size = self.img_size_spin.value()
        self.config.inference.half_precision = self.half_precision_check.isChecked()
        
        # 验证配置
        valid, errors = self.config_manager.validate_config()
        if not valid:
            error_msg = "\n".join(errors)
            QMessageBox.warning(self, "配置验证失败", f"以下配置项存在问题:\n\n{error_msg}")
            return
        
        # 保存配置
        if self.config_manager.save():
            logger.info("配置已保存")
            self.accept()
        else:
            QMessageBox.critical(self, "错误", "保存配置失败")
