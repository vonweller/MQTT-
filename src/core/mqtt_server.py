"""
MQTT服务端模块
基于asyncio实现的轻量级MQTT Broker
"""

import asyncio
import json
import base64
import time
from typing import Dict, List, Set, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger


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


@dataclass
class MQTTClient:
    """MQTT客户端信息"""
    client_id: str
    connected_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    subscriptions: Set[str] = field(default_factory=set)
    username: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'client_id': self.client_id,
            'connected_at': self.connected_at,
            'last_activity': self.last_activity,
            'subscriptions': list(self.subscriptions),
            'username': self.username
        }


class MQTTServer:
    """MQTT服务端"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 1883, strict_topic_mode: bool = False):
        self.host = host
        self.port = port
        self.running = False
        self.server = None
        
        # 客户端管理
        self.clients: Dict[str, MQTTClient] = {}
        self.client_writers: Dict[str, asyncio.StreamWriter] = {}
        
        # 订阅管理: topic -> set of client_ids
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # 主题权限控制
        self.strict_topic_mode = strict_topic_mode  # 严格模式：只允许预定义的主题
        self.allowed_topics: Set[str] = set()  # 允许的主题列表
        
        # 消息历史
        self.message_history: List[MQTTMessage] = []
        self.max_history = 1000
        
        # 保留消息
        self.retained_messages: Dict[str, MQTTMessage] = {}
        
        # 回调函数
        self.on_message_callbacks: List[Callable[[MQTTMessage], None]] = []
        self.on_connect_callbacks: List[Callable[[str], None]] = []
        self.on_disconnect_callbacks: List[Callable[[str], None]] = []
        
        # 统计
        self.stats = {
            'messages_received': 0,
            'messages_sent': 0,
            'clients_connected': 0,
            'clients_disconnected': 0,
            'messages_rejected': 0,  # 被拒绝的消息数
            'subscriptions_rejected': 0,  # 被拒绝的订阅数
            'start_time': None
        }
        
        logger.info(f"MQTT服务端初始化: {host}:{port}, 严格模式: {strict_topic_mode}")
    
    async def start(self):
        """启动MQTT服务端"""
        logger.debug(f"[MQTT Server] 开始启动服务端: {self.host}:{self.port}")
        
        if self.running:
            logger.warning("[MQTT Server] 服务端已在运行")
            return
        
        try:
            logger.debug(f"[MQTT Server] 正在创建异步服务器...")
            self.server = await asyncio.start_server(
                self._handle_client,
                self.host,
                self.port
            )
            self.running = True
            self.stats['start_time'] = time.time()
            
            addr = self.server.sockets[0].getsockname()
            logger.info(f"[MQTT Server] 服务端已启动: {addr}")
            logger.debug(f"[MQTT Server] 服务器对象: {self.server}")
            
            async with self.server:
                logger.debug("[MQTT Server] 进入 serve_forever 循环")
                await self.server.serve_forever()
                
        except OSError as e:
            logger.error(f"[MQTT Server] 启动失败 - 操作系统错误: {e}")
            logger.error(f"[MQTT Server] 错误码: {e.errno if hasattr(e, 'errno') else 'N/A'}")
            self.running = False
            raise
        except Exception as e:
            logger.error(f"[MQTT Server] 启动失败: {e}")
            logger.exception("[MQTT Server] 详细错误信息:")
            self.running = False
            raise
    
    async def stop(self):
        """停止MQTT服务端"""
        if not self.running:
            return
        
        self.running = False
        
        # 断开所有客户端
        for client_id in list(self.clients.keys()):
            await self._disconnect_client(client_id)
        
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("MQTT服务端已停止")
    
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """处理客户端连接"""
        client_id = None
        
        try:
            while self.running:
                # 读取固定头部
                fixed_header = await reader.read(1)
                if not fixed_header:
                    break
                
                packet_type = (fixed_header[0] >> 4) & 0x0F
                
                # 读取剩余长度
                remaining_length = await self._read_remaining_length(reader)
                
                # 读取包体
                packet_body = await reader.read(remaining_length)
                
                # 处理包
                if packet_type == 1:  # CONNECT
                    client_id = await self._handle_connect(packet_body, writer)
                    if not client_id:
                        break
                elif packet_type == 3:  # PUBLISH
                    await self._handle_publish(packet_body)
                elif packet_type == 8:  # SUBSCRIBE
                    await self._handle_subscribe(packet_body, writer, client_id)
                elif packet_type == 10:  # UNSUBSCRIBE
                    await self._handle_unsubscribe(packet_body, writer, client_id)
                elif packet_type == 12:  # PINGREQ
                    await self._handle_pingreq(writer)
                elif packet_type == 14:  # DISCONNECT
                    break
                
                # 更新客户端活动时间
                if client_id and client_id in self.clients:
                    self.clients[client_id].last_activity = time.time()
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"客户端处理错误: {e}")
        finally:
            if client_id:
                await self._disconnect_client(client_id)
    
    async def _read_remaining_length(self, reader: asyncio.StreamReader) -> int:
        """读取MQTT包的剩余长度"""
        multiplier = 1
        value = 0
        while True:
            byte = await reader.read(1)
            if not byte:
                raise ConnectionError("连接断开")
            encoded_byte = byte[0]
            value += (encoded_byte & 127) * multiplier
            multiplier *= 128
            if multiplier > 128 * 128 * 128:
                raise ValueError("剩余长度格式错误")
            if (encoded_byte & 128) == 0:
                break
        return value
    
    async def _handle_connect(self, packet_body: bytes, writer: asyncio.StreamWriter) -> Optional[str]:
        """处理CONNECT包"""
        try:
            # 解析协议名和级别
            idx = 0
            proto_len = int.from_bytes(packet_body[idx:idx+2], 'big')
            idx += 2
            proto_name = packet_body[idx:idx+proto_len].decode('utf-8')
            idx += proto_len
            proto_level = packet_body[idx]
            idx += 1
            
            # 连接标志
            connect_flags = packet_body[idx]
            idx += 1
            clean_session = (connect_flags >> 1) & 1
            will_flag = (connect_flags >> 2) & 1
            will_qos = (connect_flags >> 3) & 3
            will_retain = (connect_flags >> 5) & 1
            password_flag = (connect_flags >> 6) & 1
            username_flag = (connect_flags >> 7) & 1
            
            # 保持连接
            keep_alive = int.from_bytes(packet_body[idx:idx+2], 'big')
            idx += 2
            
            # 客户端ID
            client_id_len = int.from_bytes(packet_body[idx:idx+2], 'big')
            idx += 2
            client_id = packet_body[idx:idx+client_id_len].decode('utf-8')
            idx += client_id_len
            
            # 用户名和密码
            username = ""
            password = ""
            if will_flag:
                # 跳过遗嘱主题和消息
                will_topic_len = int.from_bytes(packet_body[idx:idx+2], 'big')
                idx += 2 + will_topic_len
                will_msg_len = int.from_bytes(packet_body[idx:idx+2], 'big')
                idx += 2 + will_msg_len
            
            if username_flag:
                username_len = int.from_bytes(packet_body[idx:idx+2], 'big')
                idx += 2
                username = packet_body[idx:idx+username_len].decode('utf-8')
                idx += username_len
            
            if password_flag:
                password_len = int.from_bytes(packet_body[idx:idx+2], 'big')
                idx += 2
                password = packet_body[idx:idx+password_len].decode('utf-8')
            
            # 检查客户端ID是否已存在
            if client_id in self.clients:
                # 断开旧连接
                await self._disconnect_client(client_id)
            
            # 创建客户端记录
            client = MQTTClient(
                client_id=client_id,
                username=username
            )
            self.clients[client_id] = client
            self.client_writers[client_id] = writer
            self.stats['clients_connected'] += 1
            
            # 发送CONNACK
            connack = bytes([0x20, 0x02, 0x00, 0x00])  # Session Present = 0, Return Code = 0
            writer.write(connack)
            await writer.drain()
            
            logger.info(f"客户端连接: {client_id} (用户名: {username})")
            
            # 触发回调
            for callback in self.on_connect_callbacks:
                try:
                    callback(client_id)
                except Exception as e:
                    logger.error(f"连接回调错误: {e}")
            
            return client_id
            
        except Exception as e:
            logger.error(f"CONNECT处理错误: {e}")
            # 发送CONNACK失败
            connack = bytes([0x20, 0x02, 0x00, 0x02])
            writer.write(connack)
            await writer.drain()
            return None
    
    async def _handle_publish(self, packet_body: bytes):
        """处理PUBLISH包"""
        try:
            idx = 0
            
            # 解析固定头部标志
            # 这里假设已经在handle_client中解析了固定头部
            # 实际需要根据完整的MQTT协议实现
            
            # 主题名
            topic_len = int.from_bytes(packet_body[idx:idx+2], 'big')
            idx += 2
            topic = packet_body[idx:idx+topic_len].decode('utf-8')
            idx += topic_len
            
            # 检查主题权限
            if not self._is_topic_allowed(topic):
                logger.warning(f"[MQTT Server] 拒绝发布到未授权主题: {topic}")
                self.stats['messages_rejected'] += 1
                return
            
            # 消息ID (QoS > 0)
            # 简化处理，假设QoS = 0
            
            # Payload
            payload = packet_body[idx:]
            
            # 创建消息
            message = MQTTMessage(
                topic=topic,
                payload=payload,
                qos=0,
                retain=False
            )
            
            # 添加到历史
            self.message_history.append(message)
            if len(self.message_history) > self.max_history:
                self.message_history.pop(0)
            
            self.stats['messages_received'] += 1
            
            logger.info(f"[MQTT Server] 收到消息: {topic}, 大小: {len(payload)} bytes")
            
            # 触发回调
            for callback in self.on_message_callbacks:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"消息回调错误: {e}")
            
            # 转发给订阅者
            await self._forward_message(message)
            
        except Exception as e:
            logger.error(f"PUBLISH处理错误: {e}")
    
    async def _handle_subscribe(self, packet_body: bytes, writer: asyncio.StreamWriter, client_id: str):
        """处理SUBSCRIBE包"""
        try:
            idx = 0
            
            # 消息ID
            message_id = int.from_bytes(packet_body[idx:idx+2], 'big')
            idx += 2
            
            # 订阅列表
            return_codes = []
            while idx < len(packet_body):
                # 主题过滤器
                topic_len = int.from_bytes(packet_body[idx:idx+2], 'big')
                idx += 2
                topic = packet_body[idx:idx+topic_len].decode('utf-8')
                idx += topic_len
                
                # QoS
                qos = packet_body[idx]
                idx += 1
                
                # 检查主题权限
                if not self._is_topic_allowed(topic):
                    logger.warning(f"[MQTT Server] 拒绝订阅未授权主题: {topic} (客户端: {client_id})")
                    self.stats['subscriptions_rejected'] += 1
                    return_codes.append(0x80)  # 订阅失败
                    continue
                
                # 添加订阅
                self.subscriptions[topic].add(client_id)
                if client_id in self.clients:
                    self.clients[client_id].subscriptions.add(topic)
                
                return_codes.append(qos)  # 授予的QoS
                logger.info(f"客户端 {client_id} 订阅主题: {topic} (QoS: {qos})")
            
            # 发送SUBACK
            suback = bytes([0x90, 2 + len(return_codes)]) + message_id.to_bytes(2, 'big') + bytes(return_codes)
            writer.write(suback)
            await writer.drain()
            
        except Exception as e:
            logger.error(f"SUBSCRIBE处理错误: {e}")
    
    async def _handle_unsubscribe(self, packet_body: bytes, writer: asyncio.StreamWriter, client_id: str):
        """处理UNSUBSCRIBE包"""
        try:
            idx = 0
            
            # 消息ID
            message_id = int.from_bytes(packet_body[idx:idx+2], 'big')
            idx += 2
            
            # 取消订阅列表
            while idx < len(packet_body):
                topic_len = int.from_bytes(packet_body[idx:idx+2], 'big')
                idx += 2
                topic = packet_body[idx:idx+topic_len].decode('utf-8')
                idx += topic_len
                
                # 移除订阅
                if topic in self.subscriptions:
                    self.subscriptions[topic].discard(client_id)
                if client_id in self.clients:
                    self.clients[client_id].subscriptions.discard(topic)
                
                logger.info(f"客户端 {client_id} 取消订阅主题: {topic}")
            
            # 发送UNSUBACK
            unsuback = bytes([0xB0, 0x02]) + message_id.to_bytes(2, 'big')
            writer.write(unsuback)
            await writer.drain()
            
        except Exception as e:
            logger.error(f"UNSUBSCRIBE处理错误: {e}")
    
    async def _handle_pingreq(self, writer: asyncio.StreamWriter):
        """处理PINGREQ包"""
        # 发送PINGRESP
        pingresp = bytes([0xD0, 0x00])
        writer.write(pingresp)
        await writer.drain()
    
    async def _disconnect_client(self, client_id: str):
        """断开客户端连接"""
        if client_id not in self.clients:
            return
        
        # 移除订阅
        client = self.clients[client_id]
        for topic in client.subscriptions:
            if topic in self.subscriptions:
                self.subscriptions[topic].discard(client_id)
        
        # 关闭连接
        if client_id in self.client_writers:
            try:
                writer = self.client_writers[client_id]
                writer.close()
                await writer.wait_closed()
            except:
                pass
            del self.client_writers[client_id]
        
        # 移除客户端记录
        del self.clients[client_id]
        self.stats['clients_disconnected'] += 1
        
        logger.info(f"客户端断开: {client_id}")
        
        # 触发回调
        for callback in self.on_disconnect_callbacks:
            try:
                callback(client_id)
            except Exception as e:
                logger.error(f"断开回调错误: {e}")
    
    async def _forward_message(self, message: MQTTMessage):
        """转发消息给订阅者"""
        for topic_pattern, client_ids in self.subscriptions.items():
            if self._topic_match(topic_pattern, message.topic):
                for client_id in client_ids:
                    if client_id in self.client_writers:
                        try:
                            await self._send_publish(
                                self.client_writers[client_id],
                                message.topic,
                                message.payload,
                                message.qos
                            )
                            self.stats['messages_sent'] += 1
                        except Exception as e:
                            logger.error(f"转发消息失败 {client_id}: {e}")
    
    async def _send_publish(self, writer: asyncio.StreamWriter, topic: str, payload: bytes, qos: int = 0):
        """发送PUBLISH包"""
        topic_bytes = topic.encode('utf-8')
        
        # 构建包
        variable_header = len(topic_bytes).to_bytes(2, 'big') + topic_bytes
        # QoS > 0时需要消息ID
        
        remaining = variable_header + payload
        fixed_header = bytes([0x30 | (qos << 1), len(remaining)])
        
        packet = fixed_header + remaining
        writer.write(packet)
        await writer.drain()
    
    def _topic_match(self, pattern: str, topic: str) -> bool:
        """检查主题是否匹配模式（支持通配符）"""
        pattern_parts = pattern.split('/')
        topic_parts = topic.split('/')
        
        idx = 0
        for pattern_part in pattern_parts:
            if pattern_part == '#':
                # 多级通配符
                return True
            elif pattern_part == '+':
                # 单级通配符
                idx += 1
            else:
                if idx >= len(topic_parts) or pattern_part != topic_parts[idx]:
                    return False
                idx += 1
        
        return idx == len(topic_parts)
    
    def _is_topic_allowed(self, topic: str) -> bool:
        """
        检查主题是否被允许
        在严格模式下，只有预定义的主题才能使用
        在非严格模式下，所有主题都被允许
        """
        if not self.strict_topic_mode:
            return True
        
        # 检查是否匹配允许的主题列表（支持通配符匹配）
        for allowed_topic in self.allowed_topics:
            if self._topic_match(allowed_topic, topic):
                return True
            # 反向匹配：如果允许的主题是具体主题，检查是否匹配通配符订阅
            if self._topic_match(topic, allowed_topic):
                return True
        
        return False
    
    def add_allowed_topic(self, topic: str):
        """添加允许的主题"""
        self.allowed_topics.add(topic)
        logger.info(f"[MQTT Server] 添加允许的主题: {topic}")
    
    def remove_allowed_topic(self, topic: str) -> bool:
        """移除允许的主题"""
        if topic in self.allowed_topics:
            self.allowed_topics.remove(topic)
            logger.info(f"[MQTT Server] 移除允许的主题: {topic}")
            return True
        return False
    
    def set_allowed_topics(self, topics: List[str]):
        """设置允许的主题列表"""
        self.allowed_topics = set(topics)
        logger.info(f"[MQTT Server] 设置允许的主题列表: {topics}")
    
    def get_allowed_topics(self) -> List[str]:
        """获取允许的主题列表"""
        return list(self.allowed_topics)
    
    def set_strict_mode(self, enabled: bool):
        """设置严格模式"""
        self.strict_topic_mode = enabled
        logger.info(f"[MQTT Server] 严格模式: {'启用' if enabled else '禁用'}")
    
    def add_message_callback(self, callback: Callable[[MQTTMessage], None]):
        """添加消息回调"""
        self.on_message_callbacks.append(callback)
    
    def remove_message_callback(self, callback: Callable[[MQTTMessage], None]):
        """移除消息回调"""
        if callback in self.on_message_callbacks:
            self.on_message_callbacks.remove(callback)
    
    def add_connect_callback(self, callback: Callable[[str], None]):
        """添加连接回调"""
        self.on_connect_callbacks.append(callback)
    
    def add_disconnect_callback(self, callback: Callable[[str], None]):
        """添加断开回调"""
        self.on_disconnect_callbacks.append(callback)
    
    def get_clients(self) -> List[Dict[str, Any]]:
        """获取所有客户端信息"""
        return [client.to_dict() for client in self.clients.values()]
    
    def get_topics(self) -> List[str]:
        """获取所有订阅主题"""
        return list(self.subscriptions.keys())
    
    def get_message_history(self, limit: int = 100, topic_filter: str = None) -> List[Dict[str, Any]]:
        """获取消息历史"""
        messages = self.message_history[-limit:]
        
        if topic_filter:
            messages = [m for m in messages if topic_filter in m.topic]
        
        return [m.to_dict() for m in messages]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        uptime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        return {
            **self.stats,
            'uptime': uptime,
            'connected_clients': len(self.clients),
            'total_subscriptions': sum(len(s) for s in self.subscriptions.values())
        }
    
    async def publish(self, topic: str, payload: bytes, qos: int = 0, retain: bool = False):
        """服务端发布消息"""
        message = MQTTMessage(
            topic=topic,
            payload=payload,
            qos=qos,
            retain=retain
        )
        
        if retain:
            self.retained_messages[topic] = message
        
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)
        
        # 触发消息回调（用于WebSocket广播）
        for callback in self.on_message_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"消息回调错误: {e}")
        
        # 转发给订阅者
        await self._forward_message(message)
