"""
推理结果MQTT发布模块 - 集成补丁
可以直接复制到main.py中使用
"""

# ============================================
# 在main.py的导入区域添加以下代码
# ============================================

# 添加这行导入
from src.core.inference_result_publisher import (
    InferenceResultPublisher,
    PublisherConfig
)

# ============================================
# 在MainWindow.__init__的末尾添加以下代码
# ============================================

def _init_result_publisher(self):
    """初始化推理结果发布器"""
    from loguru import logger

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

    # 延迟设置，等待组件初始化完成
    from PyQt6.QtCore import QTimer
    QTimer.singleShot(200, self._setup_publisher_integration)

def _setup_publisher_integration(self):
    """设置发布器的集成（延迟执行）"""
    from loguru import logger

    # 1. 如果已有MQTT客户端，设置它
    if self.mqtt_client and self.mqtt_client.is_connected():
        self.result_publisher.set_mqtt_client(self.mqtt_client)
        logger.info("[结果发布] MQTT客户端已设置")
        self._add_inference_callbacks()

    # 2. 监听MQTT客户端连接事件（如果客户端还没连接）
    # 这里我们可以通过猴子补丁或其他方式监听连接状态

def _add_inference_callbacks(self):
    """添加推理回调"""
    from loguru import logger

    if not self.result_publisher.is_available():
        return

    # 检查yolo_inference
    if hasattr(self, 'yolo_inference') and self.yolo_inference:
        yolo_callback = self.result_publisher.get_yolo_callback()
        self.yolo_inference.add_inference_callback(yolo_callback)
        logger.info("[结果发布] YOLO推理回调已添加")

    # 检查inference_widget中的推理器
    if hasattr(self, 'inference_widget'):
        # 如果inference_widget里也保存了推理器引用
        if hasattr(self.inference_widget, 'yolo_inference') and self.inference_widget.yolo_inference:
            yolo_callback = self.result_publisher.get_yolo_callback()
            self.inference_widget.yolo_inference.add_inference_callback(yolo_callback)
            logger.info("[结果发布] YOLO推理回调已添加（通过inference_widget）")

# ============================================
# 修改 _on_client_connect 方法，在连接成功时设置发布器
# ============================================

# 在 _on_client_connect 方法中，success == True 分支里添加：
# self._setup_publisher_after_connect()

def _setup_publisher_after_connect(self):
    """MQTT客户端连接后设置发布器"""
    from loguru import logger

    if hasattr(self, 'result_publisher') and self.mqtt_client:
        self.result_publisher.set_mqtt_client(self.mqtt_client)
        logger.info("[结果发布] MQTT客户端连接成功，发布器已就绪")
        self._add_inference_callbacks()

# ============================================
# 修改 _load_model 方法，在模型加载成功后添加回调
# ============================================

# 在 _load_model 方法中，模型加载成功后（大约第810行后）添加：
# self._setup_publisher_for_new_inference()

def _setup_publisher_for_new_inference(self):
    """新推理器加载后设置发布器"""
    from loguru import logger

    if hasattr(self, 'result_publisher') and self.result_publisher.is_available():
        if self.yolo_inference:
            yolo_callback = self.result_publisher.get_yolo_callback()
            self.yolo_inference.add_inference_callback(yolo_callback)
            logger.info("[结果发布] 新YOLO推理器回调已添加")


# ============================================
# 完整修改版的main.py代码片段
# ============================================
def get_modified_main_code():
    """获取修改后的main.py关键部分代码"""
    return '''
# 在导入部分添加：
from src.core.inference_result_publisher import (
    InferenceResultPublisher,
    PublisherConfig
)

# 在MainWindow.__init__方法末尾添加（第69行后）：
        self._init_result_publisher()

# 在MainWindow类中添加以下方法：

    def _init_result_publisher(self):
        """初始化推理结果发布器"""
        from loguru import logger

        logger.info("[结果发布] 初始化推理结果发布器")

        config = PublisherConfig(
            topic="siot/推理结果",
            qos=0,
            enabled=True,
            include_timestamp=True,
            include_fps=True
        )

        self.result_publisher = InferenceResultPublisher(config)

        from PyQt6.QtCore import QTimer
        QTimer.singleShot(200, self._setup_publisher_integration)

    def _setup_publisher_integration(self):
        """设置发布器集成"""
        from loguru import logger

        if self.mqtt_client and self.mqtt_client.is_connected():
            self.result_publisher.set_mqtt_client(self.mqtt_client)
            logger.info("[结果发布] MQTT客户端已设置")
            self._add_inference_callbacks()

    def _add_inference_callbacks(self):
        """添加推理回调"""
        from loguru import logger

        if not self.result_publisher.is_available():
            return

        if self.yolo_inference:
            yolo_callback = self.result_publisher.get_yolo_callback()
            self.yolo_inference.add_inference_callback(yolo_callback)
            logger.info("[结果发布] YOLO推理回调已添加")

    def _setup_publisher_after_connect(self):
        """MQTT连接后设置发布器"""
        from loguru import logger

        if hasattr(self, 'result_publisher') and self.mqtt_client:
            self.result_publisher.set_mqtt_client(self.mqtt_client)
            logger.info("[结果发布] 发布器已就绪")
            self._add_inference_callbacks()

    def _setup_publisher_for_new_inference(self):
        """新推理器加载后设置"""
        from loguru import logger

        if hasattr(self, 'result_publisher') and self.result_publisher.is_available():
            if self.yolo_inference:
                yolo_callback = self.result_publisher.get_yolo_callback()
                self.yolo_inference.add_inference_callback(yolo_callback)
                logger.info("[结果发布] 新YOLO推理器回调已添加")

# 修改 _on_client_connect 方法（第683行）：
    @pyqtSlot(bool)
    def _on_client_connect(self, success: bool):
        """客户端连接回调"""
        logger.debug(f"[MainWindow] 连接回调执行: success={success}")
        if success:
            self.client_status_label.setText("MQTT客户端: 已连接")
            self.client_status_label.setStyleSheet("color: #4caf50;")
            self.mqtt_widget.on_client_connect_success()
            # 添加这行
            self._setup_publisher_after_connect()
        else:
            self.client_status_label.setText("MQTT客户端: 连接失败")
            self.client_status_label.setStyleSheet("color: #f44336;")
            self.mqtt_widget.on_client_connect_failed()

# 修改 _load_model 方法（第810行后，模型加载成功后）：
            logger.info("[MainWindow] 模型加载成功，传递给inference_widget...")
            self.inference_widget.set_inference_engine(self.yolo_inference)
            self.inference_widget.on_model_loaded(True)
            # 添加这行
            self._setup_publisher_for_new_inference()
            logger.info(f"[MainWindow] 模型加载完成: {model_path}")
'''


if __name__ == "__main__":
    print("推理结果MQTT发布模块 - 集成补丁")
    print("=" * 60)
    print("请按照以下步骤集成：")
    print("1. 查看 patch_integration.py 中的代码示例")
    print("2. 根据说明修改 main.py")
    print("3. 或者查看 integration_example.py 了解更多方式")
    print("=" * 60)
    print("\nget_modified_main_code() 返回的代码可参考：")
    print(get_modified_main_code())
