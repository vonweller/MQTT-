"""
MQTT客户端模块
提供MQTT客户端连接、订阅、发布功能
"""

import json
import base64
import time
import asyncio
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from loguru import logger

try:
    import paho.mqtt.client as mqtt
    PAHO_AVAILABLE = True
except ImportError:
    PAHO_AVAILABLE = False
    logger.warning("paho-mqtt未安装，MQTT客户端功能将不可用")


@dataclass
class MQTTMessage:
    """MQTT消息"""
    topic: str
    payload: bytes
    qos: int = 0
    retain: bool = False
    timestamp: float = field(default_factory=time.time)
    
    @property
    def payload_text(self) -> str:
        """获取文本格式的payload"""
        try:
            return self.payload.decode('utf-8')
        except:
            return str(self.payload)
    
    @property
    def payload_dict(self) -> Optional[Dict]:
        """尝试将payload解析为字典"""
        try:
            return json.loads(self.payload_text)
        except:
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'topic': self.topic,
            'payload': self.payload_text[:1000] if len(self.payload) > 1000 else self.payload_text,
            'qos': self.qos,
            'retain': self.retain,
            'timestamp': self.timestamp,
            'size': len(self.payload)
        }


class MQTTClient:
    """MQTT客户端"""
    
    def __init__(self, 
                 broker_host: str = "localhost",
                 broker_port: int = 1883,
                 client_id: str = "",
                 username: str = "",
                 password: str = "",
                 keepalive: int = 60):
        
        if not PAHO_AVAILABLE:
            raise ImportError("paho-mqtt未安装，请运行: pip install paho-mqtt")
        
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id or f"yolo_client_{int(time.time())}"
        self.username = username
        self.password = password
        self.keepalive = keepalive
        
        # MQTT客户端
        self.client = mqtt.Client(client_id=self.client_id)
        
        # 认证
        if self.username:
            self.client.username_pw_set(self.username, self.password)
        
        # 回调设置
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.on_subscribe = self._on_subscribe
        self.client.on_unsubscribe = self._on_unsubscribe
        
        # 状态
        self.connected = False
        self.connecting = False
        self.subscriptions: Dict[str, int] = {}  # topic -> qos
        
        # 消息历史
        self.message_history: List[MQTTMessage] = []
        self.max_history = 1000
        
        # 回调函数
        self.on_connect_callbacks: List[Callable[[bool], None]] = []
        self.on_disconnect_callbacks: List[Callable[[], None]] = []
        self.on_message_callbacks: List[Callable[[MQTTMessage], None]] = []
        
        # 统计
        self.stats = {
            'messages_received': 0,
            'messages_sent': 0,
            'connected_at': None,
            'disconnected_at': None
        }
        
        # 自动重连
        self.auto_reconnect = True
        self.reconnect_delay = 5
        self._reconnect_task = None
        
        logger.info(f"MQTT客户端初始化: {self.client_id}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """连接回调"""
        if rc == 0:
            self.connected = True
            self.connecting = False
            self.stats['connected_at'] = time.time()
            logger.info(f"MQTT连接成功: {self.broker_host}:{self.broker_port}")
            
            # 重新订阅
            for topic, qos in self.subscriptions.items():
                self.client.subscribe(topic, qos)
                logger.info(f"重新订阅: {topic}")
            
            # 触发回调
            for callback in self.on_connect_callbacks:
                try:
                    callback(True)
                except Exception as e:
                    logger.error(f"连接回调错误: {e}")
        else:
            self.connecting = False
            logger.error(f"MQTT连接失败，返回码: {rc}")
            
            # 触发回调
            for callback in self.on_connect_callbacks:
                try:
                    callback(False)
                except Exception as e:
                    logger.error(f"连接回调错误: {e}")
            
            # 自动重连
            if self.auto_reconnect:
                self._schedule_reconnect()
    
    def _on_disconnect(self, client, userdata, rc):
        """断开回调"""
        self.connected = False
        self.stats['disconnected_at'] = time.time()
        
        if rc != 0:
            logger.warning(f"MQTT异常断开，返回码: {rc}")
            # 自动重连
            if self.auto_reconnect:
                self._schedule_reconnect()
        else:
            logger.info("MQTT正常断开")
        
        # 触发回调
        for callback in self.on_disconnect_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"断开回调错误: {e}")
    
    def _on_message(self, client, userdata, msg):
        """消息回调"""
        message = MQTTMessage(
            topic=msg.topic,
            payload=msg.payload,
            qos=msg.qos,
            retain=msg.retain
        )
        
        # 添加到历史
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)
        
        self.stats['messages_received'] += 1
        
        # 触发回调
        for callback in self.on_message_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"消息回调错误: {e}")
    
    def _on_subscribe(self, client, userdata, mid, granted_qos):
        """订阅回调"""
        logger.info(f"订阅成功，消息ID: {mid}, QoS: {granted_qos}")
    
    def _on_unsubscribe(self, client, userdata, mid):
        """取消订阅回调"""
        logger.info(f"取消订阅成功，消息ID: {mid}")
    
    def _schedule_reconnect(self):
        """计划重连"""
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())
    
    async def _reconnect_loop(self):
        """重连循环"""
        while self.auto_reconnect and not self.connected:
            logger.info(f"{self.reconnect_delay}秒后尝试重连...")
            await asyncio.sleep(self.reconnect_delay)
            
            if not self.connected:
                try:
                    self.connect()
                except Exception as e:
                    logger.error(f"重连失败: {e}")
    
    def connect(self) -> bool:
        """连接到MQTT Broker"""
        if self.connected or self.connecting:
            return True
        
        try:
            self.connecting = True
            self.client.connect(self.broker_host, self.broker_port, self.keepalive)
            self.client.loop_start()
            return True
        except Exception as e:
            self.connecting = False
            logger.error(f"连接失败: {e}")
            if self.auto_reconnect:
                self._schedule_reconnect()
            return False
    
    def disconnect(self):
        """断开连接"""
        self.auto_reconnect = False
        self.client.loop_stop()
        self.client.disconnect()
        self.connected = False
        logger.info("已断开MQTT连接")
    
    def subscribe(self, topic: str, qos: int = 0) -> bool:
        """订阅主题"""
        try:
            result, mid = self.client.subscribe(topic, qos)
            if result == mqtt.MQTT_ERR_SUCCESS:
                self.subscriptions[topic] = qos
                logger.info(f"订阅主题: {topic} (QoS: {qos})")
                return True
            else:
                logger.error(f"订阅失败: {topic}, 错误码: {result}")
                return False
        except Exception as e:
            logger.error(f"订阅错误: {e}")
            return False
    
    def unsubscribe(self, topic: str) -> bool:
        """取消订阅主题"""
        try:
            result, mid = self.client.unsubscribe(topic)
            if result == mqtt.MQTT_ERR_SUCCESS:
                self.subscriptions.pop(topic, None)
                logger.info(f"取消订阅: {topic}")
                return True
            else:
                logger.error(f"取消订阅失败: {topic}")
                return False
        except Exception as e:
            logger.error(f"取消订阅错误: {e}")
            return False
    
    def publish(self, topic: str, payload, qos: int = 0, retain: bool = False) -> bool:
        """发布消息"""
        try:
            # 处理不同类型的payload
            if isinstance(payload, dict):
                payload = json.dumps(payload, ensure_ascii=False)
            
            if isinstance(payload, str):
                payload = payload.encode('utf-8')
            
            result = self.client.publish(topic, payload, qos, retain)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.stats['messages_sent'] += 1
                return True
            else:
                logger.error(f"发布失败: {topic}, 错误码: {result.rc}")
                return False
        except Exception as e:
            logger.error(f"发布错误: {e}")
            return False
    
    def publish_image(self, topic: str, image_data: bytes, qos: int = 0) -> bool:
        """发布图像数据（Base64编码）"""
        try:
            # Base64编码图像
            encoded = base64.b64encode(image_data).decode('utf-8')
            
            # 构建JSON消息
            message = {
                'type': 'image',
                'encoding': 'base64',
                'data': encoded,
                'timestamp': time.time()
            }
            
            return self.publish(topic, message, qos)
        except Exception as e:
            logger.error(f"发布图像失败: {e}")
            return False
    
    def add_connect_callback(self, callback: Callable[[bool], None]):
        """添加连接回调"""
        self.on_connect_callbacks.append(callback)
    
    def add_disconnect_callback(self, callback: Callable[[], None]):
        """添加断开回调"""
        self.on_disconnect_callbacks.append(callback)
    
    def add_message_callback(self, callback: Callable[[MQTTMessage], None]):
        """添加消息回调"""
        self.on_message_callbacks.append(callback)
    
    def remove_message_callback(self, callback: Callable[[MQTTMessage], None]):
        """移除消息回调"""
        if callback in self.on_message_callbacks:
            self.on_message_callbacks.remove(callback)
    
    def get_subscriptions(self) -> List[str]:
        """获取订阅列表"""
        return list(self.subscriptions.keys())
    
    def get_message_history(self, limit: int = 100, topic_filter: str = None) -> List[Dict[str, Any]]:
        """获取消息历史"""
        messages = self.message_history[-limit:]
        
        if topic_filter:
            messages = [m for m in messages if topic_filter in m.topic]
        
        return [m.to_dict() for m in messages]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        stats['connected'] = self.connected
        stats['subscriptions_count'] = len(self.subscriptions)
        stats['message_history_count'] = len(self.message_history)
        return stats
    
    def clear_message_history(self):
        """清空消息历史"""
        self.message_history.clear()
        logger.info("消息历史已清空")
