"""
MQTT WebSocket服务器模块
提供WebSocket接口和HTTP静态文件服务，允许网页客户端通过WebSocket连接MQTT服务
"""

import asyncio
import json
import time
import os
from typing import Dict, Set, Callable, Optional, Any
from dataclasses import dataclass, field
from loguru import logger

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("websockets未安装，WebSocket功能将不可用")

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp未安装，HTTP静态文件服务将不可用")


@dataclass
class WSClient:
    """WebSocket客户端"""
    websocket: WebSocketServerProtocol
    client_id: str
    connected_at: float = field(default_factory=time.time)
    subscriptions: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'client_id': self.client_id,
            'connected_at': self.connected_at,
            'subscriptions': list(self.subscriptions)
        }


class MQTTWebSocketServer:
    """MQTT WebSocket服务器 - 桥接WebSocket客户端和MQTT服务，同时提供HTTP服务"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765, mqtt_server=None, web_root: str = None, config=None):
        self.host = host
        self.port = port
        self.mqtt_server = mqtt_server  # 关联的MQTT服务器实例
        self.config = config  # 配置对象，用于获取主题列表
        
        # Web根目录
        if web_root is None:
            # 默认使用当前文件所在目录的web子目录
            self.web_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'web')
        else:
            self.web_root = web_root
        
        self.clients: Dict[str, WSClient] = {}
        self.ws_server = None
        self.http_runner = None
        self.http_site = None
        self.running = False
        
        # 回调函数
        self.on_connect_callbacks: List[Callable[[str], None]] = []
        self.on_disconnect_callbacks: List[Callable[[str], None]] = []
        self.on_message_callbacks: List[Callable[[Dict], None]] = []
        self.on_publish_callbacks: List[Callable[[str, str, int], None]] = []  # 消息发布回调
        
        # 统计
        self.stats = {
            'total_connections': 0,
            'messages_sent': 0,
            'messages_received': 0
        }
        
        logger.info(f"MQTT WebSocket服务器初始化: {host}:{port}, web_root={self.web_root}")
    
    async def start(self):
        """启动WebSocket服务器和HTTP服务器"""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("websockets库未安装，无法启动WebSocket服务器")
            return False
        
        try:
            # 启动HTTP服务器（包含WebSocket升级支持）
            await self._start_http_server()
            
            self.running = True
            logger.info(f"MQTT WebSocket服务器已启动: http://{self.host}:{self.port}")
            return True
        except Exception as e:
            logger.exception(f"启动WebSocket服务器失败: {e}")
            return False
    
    async def _start_http_server(self):
        """启动HTTP服务器，同时支持WebSocket"""
        if AIOHTTP_AVAILABLE:
            # 使用aiohttp提供HTTP服务和WebSocket
            app = web.Application()
            
            # 添加路由
            app.router.add_get('/', self._handle_index)
            app.router.add_get('/ws', self._handle_ws_aiohttp)
            app.router.add_static('/', path=self.web_root, show_index=True)
            
            self.http_runner = web.AppRunner(app)
            await self.http_runner.setup()
            
            self.http_site = web.TCPSite(self.http_runner, self.host, self.port)
            await self.http_site.start()
            
            logger.info(f"HTTP服务器已启动，提供静态文件服务: {self.web_root}")
        else:
            # 回退到纯WebSocket服务器
            self.ws_server = await websockets.serve(
                self._handle_client,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=10
            )
            logger.warning("aiohttp未安装，仅启动WebSocket服务，无法提供HTML页面")
    
    async def _handle_index(self, request):
        """处理首页请求"""
        index_path = os.path.join(self.web_root, 'index.html')
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return web.Response(text=content, content_type='text/html')
        else:
            return web.Response(text="index.html not found", status=404)
    
    async def _handle_ws_aiohttp(self, request):
        """使用aiohttp处理WebSocket连接"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        client_id = f"ws_{id(ws)}_{int(time.time())}"
        
        # 创建客户端对象（使用aiohttp的WebSocket）
        # 将aiohttp的ws对象存储在client中
        client = WSClient(websocket=ws, client_id=client_id)
        client._aiohttp_ws = ws  # 保存aiohttp的WebSocket对象
        self.clients[client_id] = client
        self.stats['total_connections'] += 1
        
        logger.info(f"WebSocket客户端连接: {client_id}")
        
        # 触发连接回调
        for callback in self.on_connect_callbacks:
            try:
                callback(client_id)
            except Exception as e:
                logger.error(f"连接回调错误: {e}")
        
        try:
            # 发送连接成功消息
            await ws.send_json({
                'type': 'connected',
                'client_id': client_id,
                'timestamp': time.time()
            })
            
            # 处理客户端消息
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_aiohttp_message(client, data, ws)
                    except json.JSONDecodeError:
                        await ws.send_json({'type': 'error', 'message': 'Invalid JSON format'})
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WebSocket错误: {ws.exception()}")
        except Exception as e:
            logger.exception(f"WebSocket客户端错误: {e}")
        finally:
            # 清理客户端
            if client_id in self.clients:
                del self.clients[client_id]
            
            # 触发断开回调
            for callback in self.on_disconnect_callbacks:
                try:
                    callback(client_id)
                except Exception as e:
                    logger.error(f"断开回调错误: {e}")
            
            logger.info(f"WebSocket客户端断开: {client_id}")
        
        return ws
    
    async def _handle_aiohttp_message(self, client: WSClient, data: Dict, ws):
        """处理aiohttp WebSocket客户端消息"""
        msg_type = data.get('type', '')
        
        if msg_type == 'publish':
            await self._handle_publish_aiohttp(client, data, ws)
        elif msg_type == 'subscribe':
            await self._handle_subscribe_aiohttp(client, data, ws)
        elif msg_type == 'unsubscribe':
            await self._handle_unsubscribe_aiohttp(client, data, ws)
        elif msg_type == 'get_topics':
            await self._handle_get_topics_aiohttp(client, ws)
        elif msg_type == 'ping':
            await ws.send_json({'type': 'pong', 'timestamp': time.time()})
        else:
            await ws.send_json({'type': 'error', 'message': f"Unknown message type: {msg_type}"})
    
    def _get_all_topics(self) -> list:
        """获取所有可用主题（配置中的主题 + MQTT服务器已订阅的主题）"""
        topics = set()
        
        # 从配置中获取主题
        if self.config and hasattr(self.config, 'mqtt_topics'):
            for topic_item in self.config.mqtt_topics:
                if isinstance(topic_item, dict) and 'topic' in topic_item:
                    topics.add(topic_item['topic'])
                elif isinstance(topic_item, str):
                    topics.add(topic_item)
        
        # 从MQTT服务器获取已订阅的主题
        if self.mqtt_server and hasattr(self.mqtt_server, 'get_topics'):
            try:
                mqtt_topics = self.mqtt_server.get_topics()
                topics.update(mqtt_topics)
            except Exception as e:
                logger.debug(f"从MQTT服务器获取主题失败: {e}")
        
        return sorted(list(topics))
    
    async def _handle_get_topics_aiohttp(self, client: WSClient, ws):
        """处理获取主题列表请求（aiohttp版本）"""
        try:
            topics = self._get_all_topics()
            
            await ws.send_json({
                'type': 'topics_list',
                'topics': topics,
                'timestamp': time.time()
            })
            
            logger.debug(f"发送主题列表给客户端 {client.client_id}: {len(topics)} 个主题")
        except Exception as e:
            logger.error(f"获取主题列表失败: {e}")
            await ws.send_json({'type': 'error', 'message': f"Failed to get topics: {str(e)}"})
    
    async def _handle_publish_aiohttp(self, client: WSClient, data: Dict, ws):
        """处理发布消息请求（aiohttp版本）"""
        topic = data.get('topic', '')
        payload = data.get('payload', '')
        qos = data.get('qos', 0)
        retain = data.get('retain', False)
        
        if not topic:
            await ws.send_json({'type': 'error', 'message': 'Topic is required'})
            return
        
        try:
            if self.mqtt_server:
                await self.mqtt_server.publish(topic, payload.encode(), qos, retain)
                self.stats['messages_received'] += 1
                
                # 触发发布回调，通知桌面客户端
                for callback in self.on_publish_callbacks:
                    try:
                        callback(topic, payload, qos)
                    except Exception as e:
                        logger.error(f"发布回调错误: {e}")
                
                await ws.send_json({
                    'type': 'publish_ack',
                    'topic': topic,
                    'timestamp': time.time()
                })
                
                logger.debug(f"WebSocket客户端 {client.client_id} 发布消息到 {topic}")
            else:
                await ws.send_json({'type': 'error', 'message': 'MQTT server not available'})
        except Exception as e:
            logger.exception(f"发布消息失败: {e}")
            await ws.send_json({'type': 'error', 'message': f"Publish failed: {str(e)}"})
    
    async def _handle_subscribe_aiohttp(self, client: WSClient, data: Dict, ws):
        """处理订阅请求（aiohttp版本）"""
        topic = data.get('topic', '')
        
        if not topic:
            await ws.send_json({'type': 'error', 'message': 'Topic is required'})
            return
        
        client.subscriptions.add(topic)
        
        await ws.send_json({
            'type': 'subscribe_ack',
            'topic': topic,
            'subscriptions': list(client.subscriptions)
        })
        
        logger.info(f"WebSocket客户端 {client.client_id} 订阅主题: {topic}")
    
    async def _handle_unsubscribe_aiohttp(self, client: WSClient, data: Dict, ws):
        """处理取消订阅请求（aiohttp版本）"""
        topic = data.get('topic', '')
        
        if topic in client.subscriptions:
            client.subscriptions.discard(topic)
        
        await ws.send_json({
            'type': 'unsubscribe_ack',
            'topic': topic,
            'subscriptions': list(client.subscriptions)
        })
        
        logger.info(f"WebSocket客户端 {client.client_id} 取消订阅主题: {topic}")
    
    async def stop(self):
        """停止WebSocket服务器和HTTP服务器"""
        self.running = False
        
        # 关闭所有客户端连接
        for client in list(self.clients.values()):
            try:
                if hasattr(client.websocket, 'close'):
                    await client.websocket.close()
            except Exception:
                pass
        self.clients.clear()
        
        # 关闭HTTP服务器
        if self.http_runner:
            await self.http_runner.cleanup()
            self.http_runner = None
            self.http_site = None
        
        # 关闭WebSocket服务器
        if self.ws_server:
            self.ws_server.close()
            await self.ws_server.wait_closed()
            self.ws_server = None
        
        logger.info("MQTT WebSocket服务器已停止")
    
    # 以下是websockets库的处理方法（备用）
    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """处理WebSocket客户端连接（websockets库版本）"""
        client_id = f"ws_{id(websocket)}_{int(time.time())}"
        client = WSClient(websocket=websocket, client_id=client_id)
        self.clients[client_id] = client
        self.stats['total_connections'] += 1
        
        logger.info(f"WebSocket客户端连接: {client_id}")
        
        # 触发连接回调
        for callback in self.on_connect_callbacks:
            try:
                callback(client_id)
            except Exception as e:
                logger.error(f"连接回调错误: {e}")
        
        try:
            # 发送连接成功消息
            await self._send_to_client(client, {
                'type': 'connected',
                'client_id': client_id,
                'timestamp': time.time()
            })
            
            # 处理客户端消息
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(client, data)
                except json.JSONDecodeError:
                    await self._send_error(client, "Invalid JSON format")
                except Exception as e:
                    logger.exception(f"处理客户端消息错误: {e}")
                    await self._send_error(client, str(e))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket客户端断开: {client_id}")
        except Exception as e:
            logger.exception(f"WebSocket客户端错误: {e}")
        finally:
            # 清理客户端
            if client_id in self.clients:
                del self.clients[client_id]
            
            # 触发断开回调
            for callback in self.on_disconnect_callbacks:
                try:
                    callback(client_id)
                except Exception as e:
                    logger.error(f"断开回调错误: {e}")
    
    async def _handle_message(self, client: WSClient, data: Dict):
        """处理客户端发送的消息（websockets库版本）"""
        msg_type = data.get('type', '')
        
        if msg_type == 'publish':
            await self._handle_publish(client, data)
        elif msg_type == 'subscribe':
            await self._handle_subscribe(client, data)
        elif msg_type == 'unsubscribe':
            await self._handle_unsubscribe(client, data)
        elif msg_type == 'get_topics':
            await self._handle_get_topics(client)
        elif msg_type == 'ping':
            await self._send_to_client(client, {'type': 'pong', 'timestamp': time.time()})
        else:
            await self._send_error(client, f"Unknown message type: {msg_type}")
    
    async def _handle_get_topics(self, client: WSClient):
        """处理获取主题列表请求"""
        try:
            topics = self._get_all_topics()
            
            await self._send_to_client(client, {
                'type': 'topics_list',
                'topics': topics,
                'timestamp': time.time()
            })
            
            logger.debug(f"发送主题列表给客户端 {client.client_id}: {len(topics)} 个主题")
        except Exception as e:
            logger.error(f"获取主题列表失败: {e}")
            await self._send_error(client, f"Failed to get topics: {str(e)}")
    
    async def _handle_publish(self, client: WSClient, data: Dict):
        """处理发布消息请求"""
        topic = data.get('topic', '')
        payload = data.get('payload', '')
        qos = data.get('qos', 0)
        retain = data.get('retain', False)
        
        if not topic:
            await self._send_error(client, "Topic is required")
            return
        
        try:
            if self.mqtt_server:
                await self.mqtt_server.publish(topic, payload.encode(), qos, retain)
                self.stats['messages_received'] += 1
                
                # 触发发布回调，通知桌面客户端
                for callback in self.on_publish_callbacks:
                    try:
                        callback(topic, payload, qos)
                    except Exception as e:
                        logger.error(f"发布回调错误: {e}")
                
                await self._send_to_client(client, {
                    'type': 'publish_ack',
                    'topic': topic,
                    'timestamp': time.time()
                })
                
                logger.debug(f"WebSocket客户端 {client.client_id} 发布消息到 {topic}")
            else:
                await self._send_error(client, "MQTT server not available")
        except Exception as e:
            logger.exception(f"发布消息失败: {e}")
            await self._send_error(client, f"Publish failed: {str(e)}")
    
    async def _handle_subscribe(self, client: WSClient, data: Dict):
        """处理订阅请求"""
        topic = data.get('topic', '')
        
        if not topic:
            await self._send_error(client, "Topic is required")
            return
        
        client.subscriptions.add(topic)
        
        await self._send_to_client(client, {
            'type': 'subscribe_ack',
            'topic': topic,
            'subscriptions': list(client.subscriptions)
        })
        
        logger.info(f"WebSocket客户端 {client.client_id} 订阅主题: {topic}")
    
    async def _handle_unsubscribe(self, client: WSClient, data: Dict):
        """处理取消订阅请求"""
        topic = data.get('topic', '')
        
        if topic in client.subscriptions:
            client.subscriptions.discard(topic)
        
        await self._send_to_client(client, {
            'type': 'unsubscribe_ack',
            'topic': topic,
            'subscriptions': list(client.subscriptions)
        })
        
        logger.info(f"WebSocket客户端 {client.client_id} 取消订阅主题: {topic}")
    
    async def _send_to_client(self, client: WSClient, data: Dict):
        """发送消息给指定客户端"""
        try:
            if client.websocket:
                await client.websocket.send(json.dumps(data))
        except Exception as e:
            logger.error(f"发送消息给客户端 {client.client_id} 失败: {e}")
    
    async def _send_error(self, client: WSClient, error_msg: str):
        """发送错误消息给客户端"""
        await self._send_to_client(client, {
            'type': 'error',
            'message': error_msg,
            'timestamp': time.time()
        })
    
    async def broadcast_message(self, topic: str, payload: bytes, qos: int = 0):
        """广播MQTT消息给所有WebSocket客户端"""
        logger.info(f"[BROADCAST] 开始广播消息: {topic}")
        
        if not self.clients:
            logger.info(f"[BROADCAST] 没有WebSocket客户端连接，跳过广播: {topic}")
            return
        
        message = {
            'type': 'message',
            'topic': topic,
            'payload': payload.decode('utf-8', errors='replace'),
            'qos': qos,
            'timestamp': time.time()
        }
        
        logger.info(f"[BROADCAST] 广播消息: {topic}, 客户端数: {len(self.clients)}")
        
        # 发送给所有连接的客户端
        tasks = []
        for client_id, client in self.clients.items():
            logger.info(f"[BROADCAST] 处理客户端 {client_id}")
            logger.info(f"[BROADCAST]   - 订阅: {list(client.subscriptions)}")
            logger.info(f"[BROADCAST]   - has _aiohttp_ws: {hasattr(client, '_aiohttp_ws')}")
            
            if hasattr(client, '_aiohttp_ws') and client._aiohttp_ws:
                logger.info(f"[BROADCAST]   - _aiohttp_ws type: {type(client._aiohttp_ws)}")
                logger.info(f"[BROADCAST]   - _aiohttp_ws: {client._aiohttp_ws}")
                try:
                    # 直接发送，不添加到tasks
                    await client._aiohttp_ws.send_json(message)
                    logger.info(f"[BROADCAST]   -> 直接发送成功")
                    self.stats['messages_sent'] += 1
                except Exception as e:
                    logger.error(f"[BROADCAST]   -> 直接发送失败: {e}")
            elif client.websocket:
                try:
                    await self._send_to_client(client, message)
                    logger.info(f"[BROADCAST]   -> websockets发送成功")
                    self.stats['messages_sent'] += 1
                except Exception as e:
                    logger.error(f"[BROADCAST]   -> websockets发送失败: {e}")
            else:
                logger.warning(f"[BROADCAST]   -> 客户端没有有效的WebSocket连接")
        
        logger.info(f"[BROADCAST] 广播完成: {topic}, 总发送: {self.stats['messages_sent']}")
    
    def _topic_matches_subscriptions(self, topic: str, subscriptions: Set[str]) -> bool:
        """检查主题是否匹配订阅列表（支持通配符）"""
        if not subscriptions:
            logger.info(f"客户端没有订阅任何主题")
            return False
        
        logger.info(f"检查主题 '{topic}' 是否匹配订阅: {list(subscriptions)}")
        for sub in subscriptions:
            match_result = self._topic_match(sub, topic)
            logger.info(f"  订阅 '{sub}' 匹配 '{topic}': {match_result}")
            if match_result:
                return True
        return False
    
    def _topic_match(self, pattern: str, topic: str) -> bool:
        """主题匹配（支持+和#通配符）"""
        if pattern == topic:
            return True
        
        pattern_parts = pattern.split('/')
        topic_parts = topic.split('/')
        
        for i, part in enumerate(pattern_parts):
            if part == '#':
                return True
            if part == '+':
                continue
            if i >= len(topic_parts) or part != topic_parts[i]:
                return False
        
        return len(pattern_parts) == len(topic_parts)
    
    def add_connect_callback(self, callback: Callable[[str], None]):
        """添加连接回调"""
        self.on_connect_callbacks.append(callback)
    
    def add_disconnect_callback(self, callback: Callable[[str], None]):
        """添加断开回调"""
        self.on_disconnect_callbacks.append(callback)
    
    def add_publish_callback(self, callback: Callable[[str, str, int], None]):
        """添加消息发布回调 - 用于通知桌面客户端"""
        self.on_publish_callbacks.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'connected_clients': len(self.clients),
            'running': self.running
        }
    
    def get_clients(self) -> list:
        """获取所有连接的客户端"""
        return [client.to_dict() for client in self.clients.values()]
