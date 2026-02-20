"""
推理结果MQTT发布模块 - 快速集成脚本

使用方法：
1. 确保已理解integration_example.py中的内容
2. 根据你的具体需求，选择一种集成方式
3. 本文件提供了可以直接使用的集成函数
"""

from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.core.inference_result_publisher import (
    InferenceResultPublisher,
    PublisherConfig
)


def integrate_with_main_window(main_window):
    """
    在MainWindow中快速集成推理结果发布器

    Args:
        main_window: MainWindow实例

    使用方法：
        在main.py中找到MainWindow的初始化代码，在window.show()之前调用：
            integrate_with_main_window(window)
    """
    logger.info("[快速集成] 开始集成推理结果发布器...")

    # 1. 创建发布器
    config = PublisherConfig(
        topic="siot/推理结果",
        qos=0,
        retain=False,
        enabled=True,
        include_timestamp=True,
        include_fps=True
    )
    publisher = InferenceResultPublisher(config)

    # 2. 保存到main_window
    main_window.result_publisher = publisher

    # 3. 尝试设置MQTT客户端
    if hasattr(main_window, 'mqtt_client') and main_window.mqtt_client:
        publisher.set_mqtt_client(main_window.mqtt_client)
        logger.info("[快速集成] MQTT客户端已设置")

        # 4. 尝试设置推理回调
        _setup_inference_callbacks(main_window, publisher)
    else:
        logger.warning("[快速集成] MQTT客户端尚未连接，将在连接后设置")
        # 这里可以添加连接后的回调设置逻辑

    logger.info("[快速集成] 集成完成！")
    return publisher


def _setup_inference_callbacks(main_window, publisher):
    """
    设置推理回调（内部函数）
    """
    # 检查inference_widget
    if not hasattr(main_window, 'inference_widget'):
        logger.warning("[快速集成] 未找到inference_widget")
        return

    inference_widget = main_window.inference_widget

    # 尝试设置YOLO回调
    if hasattr(inference_widget, 'yolo_inference') and inference_widget.yolo_inference:
        yolo_callback = publisher.get_yolo_callback()
        inference_widget.yolo_inference.add_inference_callback(yolo_callback)
        logger.info("[快速集成] YOLO推理回调已添加")

    # 尝试设置MediaPipe回调（如果有）
    # 注意：MediaPipe的回调可能需要根据实际代码结构调整
    if hasattr(inference_widget, 'mediapipe_inference') and inference_widget.mediapipe_inference:
        # 如果有MediaPipe推理器，也可以添加相应的回调
        logger.info("[快速集成] 检测到MediaPipe推理器（需手动添加回调）")


def create_integrated_main():
    """
    创建一个修改版的main函数（可直接复制到main.py末尾）

    使用方法：
        1. 备份原main.py
        2. 将此函数的内容复制到main.py中
        3. 替换原来的main()函数
    """
    code = '''
def main():
    """应用程序入口 - 集成了推理结果发布器"""
    from loguru import logger

    # 启用高DPI支持
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("MQTT与推理可视化面板")

    # 设置事件循环
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    # 初始化配置管理器
    config_manager = ConfigManager()

    # 创建主窗口
    window = MainWindow(config_manager)

    # ===== 集成推理结果发布器 =====
    from src.core.inference_result_publisher import (
        InferenceResultPublisher,
        PublisherConfig
    )

    logger.info("[集成] 初始化推理结果发布器")
    publisher_config = PublisherConfig(
        topic="siot/推理结果",
        qos=0,
        enabled=True,
        include_timestamp=True,
        include_fps=True
    )
    result_publisher = InferenceResultPublisher(publisher_config)
    window.result_publisher = result_publisher

    # 延迟设置MQTT客户端和回调（等待主窗口初始化完成）
    def setup_after_init():
        # 设置MQTT客户端
        if hasattr(window, 'mqtt_client') and window.mqtt_client:
            result_publisher.set_mqtt_client(window.mqtt_client)
            logger.info("[集成] MQTT客户端已设置")

        # 设置推理回调
        if hasattr(window, 'inference_widget'):
            inference_widget = window.inference_widget
            if hasattr(inference_widget, 'yolo_inference') and inference_widget.yolo_inference:
                yolo_callback = result_publisher.get_yolo_callback()
                inference_widget.yolo_inference.add_inference_callback(yolo_callback)
                logger.info("[集成] YOLO推理回调已添加")

    # 使用QTimer延迟执行
    from PyQt6.QtCore import QTimer
    QTimer.singleShot(100, setup_after_init)
    # ===============================

    window.show()

    logger.info("应用程序已启动")

    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()
'''
    return code


if __name__ == "__main__":
    logger.info("推理结果MQTT发布模块 - 快速集成工具")
    logger.info("")
    logger.info("可选的集成方式：")
    logger.info("1. 在main.py的MainWindow初始化后调用 integrate_with_main_window(window)")
    logger.info("2. 使用create_integrated_main()的代码替换原main.py的main函数")
    logger.info("3. 参考integration_example.py了解更多细节")
    logger.info("")
    logger.info("请阅读integration_example.py获取完整说明！")
