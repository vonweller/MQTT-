"""
配置管理模块
提供配置的加载、保存、验证功能
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from loguru import logger


@dataclass
class MQTTServerConfig:
    """MQTT服务端配置"""
    host: str = "0.0.0.0"
    port: int = 1883
    websocket_port: int = 8083
    username: str = ""
    password: str = ""
    enable_auth: bool = False
    max_connections: int = 100


@dataclass
class MQTTClientConfig:
    """MQTT客户端配置"""
    broker_host: str = "localhost"
    broker_port: int = 1883
    client_id: str = ""
    username: str = ""
    password: str = ""
    keepalive: int = 60
    reconnect_delay: int = 5


@dataclass
class InferenceConfig:
    """推理配置"""
    model_path: str = ""
    model_type: str = "detection"  # detection, segmentation, pose, classification
    device: str = "auto"  # auto, cpu, cuda
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45
    img_size: int = 640
    half_precision: bool = False


@dataclass
class SourceConfig:
    """数据源配置"""
    source_type: str = "camera"  # camera, file, http, mqtt
    camera_id: int = 0
    file_path: str = ""
    http_url: str = ""
    mqtt_topic: str = "inference/image"
    fps_limit: int = 30


@dataclass
class AppConfig:
    """应用主配置"""
    app_name: str = "MQTT与YOLOv6推理可视化软件"
    version: str = "1.0.0"
    window_width: int = 1400
    window_height: int = 900
    theme: str = "dark"  # dark, light
    language: str = "zh_CN"
    log_level: str = "INFO"
    log_max_size: int = 10  # MB
    log_backup_count: int = 5
    
    mqtt_server: MQTTServerConfig = field(default_factory=MQTTServerConfig)
    mqtt_client: MQTTClientConfig = field(default_factory=MQTTClientConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    
    # 自定义主题列表
    mqtt_topics: List[Dict[str, Any]] = field(default_factory=list)
    
    # 最近使用的文件
    recent_files: List[str] = field(default_factory=list)
    max_recent_files: int = 10


class ConfigManager:
    """配置管理器"""
    
    DEFAULT_CONFIG_FILE = "config.json"
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = AppConfig()
        self._ensure_directories()
        
    def _get_default_config_path(self) -> str:
        """获取默认配置文件路径"""
        base_dir = Path(__file__).parent.parent.parent
        return str(base_dir / self.DEFAULT_CONFIG_FILE)
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        base_dir = Path(self.config_path).parent
        
        # 创建必要的子目录
        for subdir in ['models', 'logs', 'exports']:
            (base_dir / subdir).mkdir(exist_ok=True)
    
    def load(self) -> AppConfig:
        """从文件加载配置"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 解析配置
                self.config = self._parse_config(data)
                logger.info(f"配置已加载: {self.config_path}")
            else:
                logger.info("配置文件不存在，使用默认配置")
                self.save()  # 创建默认配置文件
                
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            logger.info("使用默认配置")
            
        return self.config
    
    def _parse_config(self, data: Dict) -> AppConfig:
        """解析配置数据"""
        config = AppConfig()
        
        # 基础配置
        if 'app_name' in data:
            config.app_name = data['app_name']
        if 'version' in data:
            config.version = data['version']
        if 'window_width' in data:
            config.window_width = data['window_width']
        if 'window_height' in data:
            config.window_height = data['window_height']
        if 'theme' in data:
            config.theme = data['theme']
        if 'language' in data:
            config.language = data['language']
        if 'log_level' in data:
            config.log_level = data['log_level']
        
        # MQTT服务端配置
        if 'mqtt_server' in data:
            ms = data['mqtt_server']
            config.mqtt_server = MQTTServerConfig(
                host=ms.get('host', '0.0.0.0'),
                port=ms.get('port', 1883),
                websocket_port=ms.get('websocket_port', 8083),
                username=ms.get('username', ''),
                password=ms.get('password', ''),
                enable_auth=ms.get('enable_auth', False),
                max_connections=ms.get('max_connections', 100)
            )
        
        # MQTT客户端配置
        if 'mqtt_client' in data:
            mc = data['mqtt_client']
            config.mqtt_client = MQTTClientConfig(
                broker_host=mc.get('broker_host', 'localhost'),
                broker_port=mc.get('broker_port', 1883),
                client_id=mc.get('client_id', ''),
                username=mc.get('username', ''),
                password=mc.get('password', ''),
                keepalive=mc.get('keepalive', 60),
                reconnect_delay=mc.get('reconnect_delay', 5)
            )
        
        # 推理配置
        if 'inference' in data:
            inf = data['inference']
            config.inference = InferenceConfig(
                model_path=inf.get('model_path', ''),
                model_type=inf.get('model_type', 'detection'),
                device=inf.get('device', 'auto'),
                conf_threshold=inf.get('conf_threshold', 0.5),
                iou_threshold=inf.get('iou_threshold', 0.45),
                img_size=inf.get('img_size', 640),
                half_precision=inf.get('half_precision', False)
            )
        
        # 数据源配置
        if 'source' in data:
            src = data['source']
            config.source = SourceConfig(
                source_type=src.get('source_type', 'camera'),
                camera_id=src.get('camera_id', 0),
                file_path=src.get('file_path', ''),
                http_url=src.get('http_url', ''),
                mqtt_topic=src.get('mqtt_topic', 'inference/image'),
                fps_limit=src.get('fps_limit', 30)
            )
        
        # 主题列表
        if 'mqtt_topics' in data:
            config.mqtt_topics = data['mqtt_topics']
        
        # 最近文件
        if 'recent_files' in data:
            config.recent_files = data['recent_files']
        
        return config
    
    def save(self) -> bool:
        """保存配置到文件"""
        try:
            data = self._config_to_dict(self.config)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"配置已保存: {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False
    
    def _config_to_dict(self, config: AppConfig) -> Dict:
        """将配置对象转换为字典"""
        return {
            'app_name': config.app_name,
            'version': config.version,
            'window_width': config.window_width,
            'window_height': config.window_height,
            'theme': config.theme,
            'language': config.language,
            'log_level': config.log_level,
            'log_max_size': config.log_max_size,
            'log_backup_count': config.log_backup_count,
            'mqtt_server': {
                'host': config.mqtt_server.host,
                'port': config.mqtt_server.port,
                'websocket_port': config.mqtt_server.websocket_port,
                'username': config.mqtt_server.username,
                'password': config.mqtt_server.password,
                'enable_auth': config.mqtt_server.enable_auth,
                'max_connections': config.mqtt_server.max_connections
            },
            'mqtt_client': {
                'broker_host': config.mqtt_client.broker_host,
                'broker_port': config.mqtt_client.broker_port,
                'client_id': config.mqtt_client.client_id,
                'username': config.mqtt_client.username,
                'password': config.mqtt_client.password,
                'keepalive': config.mqtt_client.keepalive,
                'reconnect_delay': config.mqtt_client.reconnect_delay
            },
            'inference': {
                'model_path': config.inference.model_path,
                'model_type': config.inference.model_type,
                'device': config.inference.device,
                'conf_threshold': config.inference.conf_threshold,
                'iou_threshold': config.inference.iou_threshold,
                'img_size': config.inference.img_size,
                'half_precision': config.inference.half_precision
            },
            'source': {
                'source_type': config.source.source_type,
                'camera_id': config.source.camera_id,
                'file_path': config.source.file_path,
                'http_url': config.source.http_url,
                'mqtt_topic': config.source.mqtt_topic,
                'fps_limit': config.source.fps_limit
            },
            'mqtt_topics': config.mqtt_topics,
            'recent_files': config.recent_files,
            'max_recent_files': config.max_recent_files
        }
    
    def export_config(self, export_path: str) -> bool:
        """导出配置到指定路径"""
        try:
            data = self._config_to_dict(self.config)
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(f"配置已导出: {export_path}")
            return True
        except Exception as e:
            logger.error(f"导出配置失败: {e}")
            return False
    
    def import_config(self, import_path: str) -> bool:
        """从指定路径导入配置"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.config = self._parse_config(data)
            self.save()
            logger.info(f"配置已导入: {import_path}")
            return True
        except Exception as e:
            logger.error(f"导入配置失败: {e}")
            return False
    
    def validate_config(self) -> tuple[bool, List[str]]:
        """验证配置有效性"""
        errors = []
        
        # 验证MQTT服务端端口
        if not (1 <= self.config.mqtt_server.port <= 65535):
            errors.append("MQTT服务端端口必须在1-65535之间")
        
        # 验证MQTT客户端端口
        if not (1 <= self.config.mqtt_client.broker_port <= 65535):
            errors.append("MQTT客户端端口必须在1-65535之间")
        
        # 验证推理阈值
        if not (0 <= self.config.inference.conf_threshold <= 1):
            errors.append("置信度阈值必须在0-1之间")
        
        if not (0 <= self.config.inference.iou_threshold <= 1):
            errors.append("IOU阈值必须在0-1之间")
        
        # 验证图像尺寸
        if self.config.inference.img_size < 32:
            errors.append("图像尺寸必须至少为32")
        
        # 验证模型路径
        if self.config.inference.model_path:
            if not os.path.exists(self.config.inference.model_path):
                errors.append(f"模型路径不存在: {self.config.inference.model_path}")
        
        return len(errors) == 0, errors
    
    def add_recent_file(self, file_path: str):
        """添加最近使用的文件"""
        if file_path in self.config.recent_files:
            self.config.recent_files.remove(file_path)
        
        self.config.recent_files.insert(0, file_path)
        
        # 限制数量
        if len(self.config.recent_files) > self.config.max_recent_files:
            self.config.recent_files = self.config.recent_files[:self.config.max_recent_files]
        
        self.save()
    
    def add_mqtt_topic(self, topic: str, description: str = "", qos: int = 0):
        """添加MQTT主题"""
        topic_item = {
            'topic': topic,
            'description': description,
            'qos': qos
        }
        
        # 检查是否已存在
        for i, t in enumerate(self.config.mqtt_topics):
            if t['topic'] == topic:
                self.config.mqtt_topics[i] = topic_item
                self.save()
                return
        
        self.config.mqtt_topics.append(topic_item)
        self.save()
    
    def remove_mqtt_topic(self, topic: str):
        """删除MQTT主题"""
        self.config.mqtt_topics = [t for t in self.config.mqtt_topics if t['topic'] != topic]
        self.save()
    
    def get_config(self) -> AppConfig:
        """获取当前配置"""
        return self.config
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.save()
