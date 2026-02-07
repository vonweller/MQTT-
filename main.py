"""
MQTT与YOLOv6推理可视化软件
主程序入口
"""

import sys
import os
import asyncio
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from loguru import logger

# 使用 qasync 来集成 asyncio 和 PyQt6
try:
    import qasync
    from qasync import QEventLoop
    QASYNC_AVAILABLE = True
except ImportError:
    QASYNC_AVAILABLE = False
    logger.warning("qasync 未安装，MQTT功能可能无法正常工作")

from src.core.config_manager import ConfigManager
from src.ui.main_window import MainWindow


def setup_logging(log_dir: Path, log_level: str = "INFO"):
    """配置日志"""
    log_dir.mkdir(exist_ok=True)
    
    # 移除默认处理器
    logger.remove()
    
    # 添加控制台输出
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # 添加文件输出
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="7 days",
        level=log_level,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )


def main():
    """主函数"""
    # 启用高DPI支持
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("MQTT与YOLOv6推理可视化软件")
    app.setApplicationVersion("1.0.0")
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load()
    
    # 设置日志
    log_dir = project_root / "logs"
    setup_logging(log_dir, config.log_level)
    
    logger.info("=" * 50)
    logger.info(f"启动 {config.app_name} v{config.version}")
    logger.info("=" * 50)
    
    # 验证配置
    valid, errors = config_manager.validate_config()
    if not valid:
        logger.warning(f"配置验证失败: {errors}")
    
    # 创建并显示主窗口
    try:
        window = MainWindow(config_manager)
        window.show()
        
        logger.info("主窗口已显示")
        
        # 使用 qasync 运行应用
        if QASYNC_AVAILABLE:
            logger.debug("使用 qasync 创建事件循环")
            loop = qasync.QEventLoop(app)
            asyncio.set_event_loop(loop)
            
            with loop:
                loop.run_forever()
        else:
            # 如果没有 qasync，使用标准方式
            logger.warning("未使用 qasync，MQTT功能可能受限")
            sys.exit(app.exec())
        
    except Exception as e:
        logger.exception("应用程序启动失败")
        raise


if __name__ == "__main__":
    main()
