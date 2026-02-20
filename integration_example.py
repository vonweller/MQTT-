"""
推理结果MQTT发布模块 - 集成示例
展示如何在不修改现有代码的情况下集成该模块

使用方法：
1. 将此文件放在项目根目录
2. 根据需要修改main.py或使用独立的启动脚本
"""

from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.core.inference_result_publisher import (
    InferenceResultPublisher,
    PublisherConfig,
    get_global_publisher
)


def example_basic_usage():
    """
    示例1: 基础使用方法
    """
    logger.info("=" * 50)
    logger.info("示例1: 基础使用方法")
    logger.info("=" * 50)

    # 创建发布器配置
    config = PublisherConfig(
        topic="siot/推理结果",
        qos=0,
        retain=False,
        enabled=True,
        include_timestamp=True,
        include_fps=True
    )

    # 创建发布器
    publisher = InferenceResultPublisher(config)

    logger.info(f"发布器已创建，主题: {publisher.config.topic}")
    logger.info(f"可用状态: {publisher.is_available()}")

    return publisher


def example_with_mqtt_client(mqtt_client):
    """
    示例2: 与项目现有MQTTClient集成

    Args:
        mqtt_client: 项目中的MQTTClient实例
    """
    logger.info("=" * 50)
    logger.info("示例2: 与现有MQTTClient集成")
    logger.info("=" * 50)

    # 创建发布器
    publisher = InferenceResultPublisher()

    # 设置MQTT客户端
    publisher.set_mqtt_client(mqtt_client)

    logger.info(f"MQTT客户端已连接: {publisher._connected}")
    logger.info(f"可用状态: {publisher.is_available()}")

    return publisher


def example_with_publish_function():
    """
    示例3: 使用自定义发布函数（最灵活的方式）
    可以适配任何MQTT库
    """
    logger.info("=" * 50)
    logger.info("示例3: 使用自定义发布函数")
    logger.info("=" * 50)

    # 定义自定义发布函数
    def my_publish_function(topic: str, payload, qos: int = 0, retain: bool = False) -> bool:
        """
        自定义发布函数
        这里可以调用任何MQTT库的发布方法
        """
        logger.info(f"[自定义发布] 主题: {topic}")
        logger.info(f"[自定义发布] 内容: {str(payload)[:100]}...")
        logger.info(f"[自定义发布] QoS: {qos}, Retain: {retain}")
        return True  # 返回是否成功

    # 创建发布器
    publisher = InferenceResultPublisher()

    # 设置自定义发布函数
    publisher.set_mqtt_publish_function(my_publish_function)

    logger.info(f"可用状态: {publisher.is_available()}")

    return publisher


def example_integrate_with_yolo(yolo_inference):
    """
    示例4: 与YOLO推理器集成

    Args:
        yolo_inference: YOLOInference实例
    """
    logger.info("=" * 50)
    logger.info("示例4: 与YOLO推理器集成")
    logger.info("=" * 50)

    # 创建发布器
    publisher = InferenceResultPublisher()

    # 获取YOLO回调函数
    yolo_callback = publisher.get_yolo_callback()

    # 添加回调到YOLO推理器
    yolo_inference.add_inference_callback(yolo_callback)

    logger.info("YOLO推理回调已添加")
    logger.info("推理结果将自动发布到MQTT")

    return publisher


def example_using_global_publisher():
    """
    示例5: 使用全局单例发布器
    适合在整个应用中共享一个发布器实例
    """
    logger.info("=" * 50)
    logger.info("示例5: 使用全局单例发布器")
    logger.info("=" * 50)

    # 首次调用时创建并配置
    config = PublisherConfig(topic="siot/推理结果")
    publisher = get_global_publisher(config)

    # 之后在任何地方都可以获取同一个实例
    publisher2 = get_global_publisher()

    logger.info(f"是否为同一实例: {publisher is publisher2}")

    return publisher


# ============================================
# 以下是如何在main.py中集成的示例代码
# 将这些代码添加到main.py的适当位置
# ============================================

def integrate_in_main_window_example(main_window):
    """
    在MainWindow中集成的示例代码

    建议在MainWindow.__init__的最后调用此函数
    或直接将代码嵌入到__init__中

    Args:
        main_window: MainWindow实例
    """
    from src.core.inference_result_publisher import (
        InferenceResultPublisher,
        PublisherConfig
    )

    logger.info("[集成示例] 初始化推理结果发布器")

    # 1. 创建发布器
    config = PublisherConfig(
        topic="siot/推理结果",
        qos=0,
        enabled=True
    )
    publisher = InferenceResultPublisher(config)

    # 2. 保存到main_window（可选，方便后续访问）
    main_window.result_publisher = publisher

    # 3. 等待MQTT客户端连接后设置
    # 方式A: 直接使用主窗口的mqtt_client（需要先连接）
    def setup_after_client_connect(success):
        if success and hasattr(main_window, 'mqtt_client'):
            publisher.set_mqtt_client(main_window.mqtt_client)
            logger.info("[集成示例] MQTT客户端已设置到发布器")

            # 同时设置到推理面板
            if hasattr(main_window, 'inference_widget'):
                _setup_inference_widget_callbacks(main_window, publisher)

    # 方式B: 如果已有MQTT客户端，直接设置
    if hasattr(main_window, 'mqtt_client') and main_window.mqtt_client:
        publisher.set_mqtt_client(main_window.mqtt_client)
        _setup_inference_widget_callbacks(main_window, publisher)

    logger.info("[集成示例] 推理结果发布器初始化完成")


def _setup_inference_widget_callbacks(main_window, publisher):
    """
    设置推理面板的回调（内部辅助函数）
    """
    # 访问inference_widget
    if not hasattr(main_window, 'inference_widget'):
        return

    inference_widget = main_window.inference_widget

    # 方式1: 如果inference_widget保存了inference_engine引用
    if hasattr(inference_widget, 'inference_engine') and inference_widget.inference_engine:
        yolo_callback = publisher.get_yolo_callback()
        inference_widget.inference_engine.add_inference_callback(yolo_callback)
        logger.info("[集成示例] YOLO推理回调已添加")


# ============================================
# 独立启动脚本示例
# ============================================

def create_standalone_launch_script():
    """
    创建独立的启动脚本内容
    可以保存为 start_with_publisher.py
    """
    script_content = '''"""
带推理结果MQTT发布功能的启动脚本
不修改原main.py，使用装饰器模式
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from loguru import logger

# 导入原main模块
import main as original_main

# 导入发布模块
from src.core.inference_result_publisher import (
    InferenceResultPublisher,
    PublisherConfig
)


def main():
    """包装主函数"""
    logger.info("=" * 60)
    logger.info("启动应用 - 带推理结果MQTT发布功能")
    logger.info("=" * 60)

    # 创建发布器（先创建，但暂不设置MQTT客户端）
    publisher = InferenceResultPublisher(
        PublisherConfig(topic="siot/推理结果")
    )

    # 保存为全局变量（可以在其他地方访问）
    import builtins
    builtins._result_publisher = publisher

    # 启动原应用
    # 注意：需要在应用启动后找到合适的时机设置MQTT客户端和回调
    # 可以通过猴子补丁或其他方式实现

    logger.info("启动原应用...")
    original_main.main()


if __name__ == "__main__":
    main()
'''
    return script_content


if __name__ == "__main__":
    logger.info("推理结果MQTT发布模块 - 集成示例")
    logger.info("此文件展示了如何使用该模块")
    logger.info("")

    # 运行示例
    example_basic_usage()
    print()

    example_with_publish_function()
    print()

    example_using_global_publisher()
    print()

    logger.info("=" * 50)
    logger.info("集成说明:")
    logger.info("1. 最简单的方式: 修改main.py，在适当位置添加集成代码")
    logger.info("2. 零侵入方式: 创建独立启动脚本，使用猴子补丁")
    logger.info("3. 参考integration_example.py中的详细示例")
    logger.info("=" * 50)
