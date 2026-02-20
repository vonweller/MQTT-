"""
推理结果发布模块
将YOLO和MediaPipe的推理结果通过MQTT主题发送出去
完全独立，不侵入现有代码
"""

import json
import time
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class InferenceType(Enum):
    """推理类型枚举"""
    YOLO = "yolo"
    MEDIAPIPE = "mediapipe"


@dataclass
class PublisherConfig:
    """发布器配置"""
    topic: str = "siot/推理结果"
    qos: int = 0
    retain: bool = False
    enabled: bool = True
    include_timestamp: bool = True
    include_fps: bool = True
    max_payload_size: int = 1024 * 1024  # 1MB


class NumpyEncoder(json.JSONEncoder):
    """处理NumPy类型的JSON编码器"""
    def default(self, obj):
        if NUMPY_AVAILABLE:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
        return super().default(obj)


class InferenceResultPublisher:
    """
    推理结果发布器
    监听推理回调，将结果通过MQTT发送
    """

    def __init__(self, config: Optional[PublisherConfig] = None):
        """
        初始化发布器

        Args:
            config: 发布器配置，如果为None则使用默认配置
        """
        self.config = config or PublisherConfig()
        self._mqtt_client = None
        self._mqtt_publish_func = None
        self._ui_callback = None
        self._connected = False
        self._message_count = 0
        self._start_time = time.time()

        logger.info(f"[ResultPublisher] 初始化，主题: {self.config.topic}")

    def set_ui_callback(self, ui_callback):
        """
        设置UI回调函数，用于将消息显示在UI中

        Args:
            ui_callback: 签名为 def callback(topic: str, payload: str)
        """
        self._ui_callback = ui_callback
        logger.info("[ResultPublisher] UI回调已设置")

    def set_mqtt_client(self, mqtt_client):
        """
        设置MQTT客户端对象（使用项目现有的MQTTClient）

        Args:
            mqtt_client: 项目中的MQTTClient实例
        """
        self._mqtt_client = mqtt_client
        self._connected = mqtt_client.connected if hasattr(mqtt_client, 'connected') else False
        logger.info(f"[ResultPublisher] MQTT客户端已设置，连接状态: {self._connected}")

    def set_mqtt_publish_function(self, publish_func):
        """
        设置MQTT发布函数（灵活的方式，不依赖特定客户端）

        Args:
            publish_func: 签名为 def publish(topic: str, payload, qos: int = 0, retain: bool = False) -> bool
        """
        self._mqtt_publish_func = publish_func
        logger.info("[ResultPublisher] MQTT发布函数已设置")

    def is_available(self) -> bool:
        """检查发布器是否可用"""
        if not self.config.enabled:
            return False
        return self._mqtt_client is not None or self._mqtt_publish_func is not None

    def enable(self):
        """启用发布"""
        self.config.enabled = True
        logger.info("[ResultPublisher] 已启用")

    def disable(self):
        """禁用发布"""
        self.config.enabled = False
        logger.info("[ResultPublisher] 已禁用")

    def set_topic(self, topic: str):
        """设置发布主题"""
        self.config.topic = topic
        logger.info(f"[ResultPublisher] 主题已更新: {topic}")

    def _serialize_yolo_result(self, result) -> Dict[str, Any]:
        """
        序列化YOLO推理结果

        Args:
            result: YOLO的InferenceResult对象

        Returns:
            可序列化的字典
        """
        data = {
            "type": InferenceType.YOLO.value,
            "model_type": getattr(result, 'model_type', 'unknown'),
            "inference_time": getattr(result, 'inference_time', 0),
        }

        if self.config.include_fps:
            data["fps"] = getattr(result, 'fps', 0)

        if self.config.include_timestamp:
            data["timestamp"] = time.time()

        detections = []
        if hasattr(result, 'detections'):
            for det in result.detections:
                detections.append({
                    "class_name": getattr(det, 'class_name', ''),
                    "class_id": getattr(det, 'class_id', -1),
                    "confidence": getattr(det, 'confidence', 0),
                    "bbox": getattr(det, 'bbox', [])
                })
        data["detections"] = detections

        segmentations = []
        if hasattr(result, 'segmentations'):
            for seg in result.segmentations:
                segmentations.append({
                    "class_name": getattr(seg, 'class_name', ''),
                    "class_id": getattr(seg, 'class_id', -1),
                    "confidence": getattr(seg, 'confidence', 0),
                    "bbox": getattr(seg, 'bbox', [])
                })
        data["segmentations"] = segmentations

        keypoints = []
        if hasattr(result, 'keypoints'):
            for kp in result.keypoints:
                keypoints.append({
                    "keypoints": getattr(kp, 'keypoints', []),
                    "bbox": getattr(kp, 'bbox', []),
                    "confidence": getattr(kp, 'confidence', 0)
                })
        data["keypoints"] = keypoints

        classifications = []
        if hasattr(result, 'classifications'):
            for cls in result.classifications:
                classifications.append({
                    "class_name": getattr(cls, 'class_name', ''),
                    "class_id": getattr(cls, 'class_id', -1),
                    "confidence": getattr(cls, 'confidence', 0),
                    "top5": getattr(cls, 'top5', [])
                })
        data["classifications"] = classifications

        return data

    def _serialize_mediapipe_result(self, result) -> Dict[str, Any]:
        """
        序列化MediaPipe推理结果

        Args:
            result: MediaPipeResult对象

        Returns:
            可序列化的字典
        """
        data = {
            "type": InferenceType.MEDIAPIPE.value,
            "inference_time": getattr(result, 'inference_time', 0),
        }

        if self.config.include_fps:
            data["fps"] = getattr(result, 'fps', 0)

        if self.config.include_timestamp:
            data["timestamp"] = time.time()

        poses = []
        if hasattr(result, 'poses'):
            for pose in result.poses:
                pose_data = {
                    "keypoints": [],
                    "confidence": getattr(pose, 'confidence', 0)
                }
                if hasattr(pose, 'keypoints'):
                    for kp in pose.keypoints:
                        pose_data["keypoints"].append({
                            "x": getattr(kp, 'x', 0),
                            "y": getattr(kp, 'y', 0),
                            "z": getattr(kp, 'z', 0),
                            "visibility": getattr(kp, 'visibility', 1)
                        })
                poses.append(pose_data)
        data["poses"] = poses

        hands = []
        if hasattr(result, 'hands'):
            for hand in result.hands:
                hand_data = {
                    "keypoints": [],
                    "gesture": getattr(hand, 'gesture', ''),
                    "gesture_score": getattr(hand, 'gesture_score', 0)
                }
                if hasattr(hand, 'keypoints'):
                    for kp in hand.keypoints:
                        hand_data["keypoints"].append({
                            "x": getattr(kp, 'x', 0),
                            "y": getattr(kp, 'y', 0),
                            "z": getattr(kp, 'z', 0)
                        })
                hands.append(hand_data)
        data["hands"] = hands

        faces = []
        if hasattr(result, 'faces'):
            for face in result.faces:
                face_data = []
                for kp in face:
                    face_data.append({
                        "x": getattr(kp, 'x', 0),
                        "y": getattr(kp, 'y', 0),
                        "z": getattr(kp, 'z', 0)
                    })
                faces.append(face_data)
        data["faces"] = faces

        return data

    def _publish_payload(self, payload: Dict[str, Any]) -> bool:
        """
        发布消息到MQTT

        Args:
            payload: 要发布的数据字典

        Returns:
            是否成功
        """
        if not self.config.enabled:
            return False

        try:
            payload_json = json.dumps(payload, ensure_ascii=False, cls=NumpyEncoder)

            payload_size = len(payload_json.encode('utf-8'))
            if payload_size > self.config.max_payload_size:
                logger.warning(
                    f"[ResultPublisher] 消息过大 ({payload_size} bytes)，超过限制 {self.config.max_payload_size} bytes"
                )
                return False

            # 先调用UI回调，将消息显示在UI中
            if self._ui_callback:
                try:
                    self._ui_callback(self.config.topic, payload_json)
                    logger.debug("[ResultPublisher] 消息已添加到UI")
                except Exception as e:
                    logger.warning(f"[ResultPublisher] UI回调失败: {e}")

            if self._mqtt_publish_func:
                success = self._mqtt_publish_func(
                    self.config.topic,
                    payload_json,
                    self.config.qos,
                    self.config.retain
                )
            elif self._mqtt_client and hasattr(self._mqtt_client, 'publish'):
                success = self._mqtt_client.publish(
                    self.config.topic,
                    payload_json,
                    self.config.qos,
                    self.config.retain
                )
            else:
                logger.warning("[ResultPublisher] 没有可用的MQTT发布方式，仅显示在UI")
                # 即使没有MQTT，只要UI回调成功也返回True
                return self._ui_callback is not None

            if success:
                self._message_count += 1
                logger.debug(
                    f"[ResultPublisher] 消息已发布 #{self._message_count} "
                    f"到 {self.config.topic} ({payload_size} bytes)"
                )
            else:
                logger.warning("[ResultPublisher] 消息发布失败")

            return success

        except Exception as e:
            logger.exception(f"[ResultPublisher] 发布异常: {e}")
            return False

    def publish_yolo_result(self, result) -> bool:
        """
        发布YOLO推理结果

        Args:
            result: YOLO的InferenceResult对象

        Returns:
            是否成功
        """
        if not self.is_available():
            return False

        try:
            payload = self._serialize_yolo_result(result)
            return self._publish_payload(payload)
        except Exception as e:
            logger.exception(f"[ResultPublisher] 处理YOLO结果异常: {e}")
            return False

    def publish_mediapipe_result(self, result) -> bool:
        """
        发布MediaPipe推理结果

        Args:
            result: MediaPipeResult对象

        Returns:
            是否成功
        """
        if not self.is_available():
            return False

        try:
            payload = self._serialize_mediapipe_result(result)
            return self._publish_payload(payload)
        except Exception as e:
            logger.exception(f"[ResultPublisher] 处理MediaPipe结果异常: {e}")
            return False

    def get_yolo_callback(self):
        """
        获取YOLO推理回调函数
        可以直接传递给YOLOInference.add_inference_callback()

        Returns:
            回调函数
        """
        def callback(result):
            self.publish_yolo_result(result)
        return callback

    def get_stats(self) -> Dict[str, Any]:
        """
        获取发布统计信息

        Returns:
            统计字典
        """
        return {
            "enabled": self.config.enabled,
            "topic": self.config.topic,
            "message_count": self._message_count,
            "uptime": time.time() - self._start_time,
            "available": self.is_available()
        }

    def reset_stats(self):
        """重置统计信息"""
        self._message_count = 0
        self._start_time = time.time()
        logger.info("[ResultPublisher] 统计已重置")


# 全局单例实例（可选使用）
_global_publisher: Optional[InferenceResultPublisher] = None


def get_global_publisher(config: Optional[PublisherConfig] = None) -> InferenceResultPublisher:
    """
    获取全局发布器单例

    Args:
        config: 首次调用时的配置

    Returns:
        全局InferenceResultPublisher实例
    """
    global _global_publisher
    if _global_publisher is None:
        _global_publisher = InferenceResultPublisher(config)
    return _global_publisher
