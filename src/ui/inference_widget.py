"""
推理面板UI - 支持YOLO和MediaPipe双专栏
提供目标检测和关键点检测的可视化界面
"""

import os
import cv2
import numpy as np
import base64
import json
import csv
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageFont

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QFileDialog, QGroupBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QTextEdit, QSplitter, QMessageBox, QProgressBar, QTabWidget,
    QListWidget, QListWidgetItem, QFormLayout, QStackedWidget,
    QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from loguru import logger

from ..core.config_manager import ConfigManager
from ..core.yolo_inference import YOLOInference, InferenceResult


# ============== MediaPipe 数据类 (兼容Python 3.11) ==============
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Keypoint:
    """关键点数据"""
    x: float  # 归一化坐标 0-1
    y: float
    z: float = 0.0
    visibility: float = 1.0  # 可见度 0-1

@dataclass
class PoseResult:
    """姿态检测结果"""
    keypoints: List[Keypoint] = field(default_factory=list)
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2]
    confidence: float = 0.0

@dataclass
class HandResult:
    """手部检测结果"""
    keypoints: List[Keypoint] = field(default_factory=list)
    gesture: str = ""  # 手势类别
    gesture_score: float = 0.0  # 手势置信度

@dataclass
class MediaPipeResult:
    """MediaPipe推理结果"""
    poses: List[PoseResult] = field(default_factory=list)
    hands: List[HandResult] = field(default_factory=list)  # 改为HandResult包含手势信息
    faces: List[List[Keypoint]] = field(default_factory=list)
    fps: float = 0.0
    inference_time: float = 0.0


# ============== MediaPipe 关键点检测线程 (新版tasks API) ==============
class MediaPipeThread(QThread):
    """MediaPipe关键点检测线程 - 使用mediapipe.tasks API (v0.10.x)"""

    frame_ready = pyqtSignal(np.ndarray, MediaPipeResult)
    fps_updated = pyqtSignal(float)
    error_occurred = pyqtSignal(str)

    # 模型下载URL
    MODEL_URLS = {
        'pose_landmarker_lite.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        'pose_landmarker_full.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
        'pose_landmarker_heavy.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
        'hand_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        'face_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        'gesture_recognizer.task': 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'
    }

    # 模型复杂度映射: 0=轻量, 1=标准, 2=重型
    POSE_MODEL_MAP = {
        0: 'pose_landmarker_lite.task',
        1: 'pose_landmarker_full.task',
        2: 'pose_landmarker_heavy.task'
    }

    def __init__(self, source_type: str, source_config: dict,
                 enable_pose: bool = True, enable_hands: bool = False,
                 enable_face: bool = False, model_complexity: int = 1,
                 enable_gesture: bool = False):
        super().__init__()
        self.source_type = source_type
        self.source_config = source_config
        self.enable_pose = enable_pose
        self.enable_hands = enable_hands
        self.enable_face = enable_face
        self.enable_gesture = enable_gesture  # 启用手势识别
        self.model_complexity = model_complexity  # 0=轻量, 1=标准, 2=重型
        self.running = False
        self.cap = None

        # MediaPipe任务对象
        self.pose_landmarker = None
        self.hand_landmarker = None
        self.face_landmarker = None
        self.gesture_recognizer = None  # 手势识别器

        # 模型目录（原始位置，可能包含中文）
        self.models_dir = Path(__file__).parent.parent.parent / "models" / "mediapipe"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # MediaPipe兼容目录（纯英文路径，避免C++底层无法处理中文路径的问题）
        self.compat_models_dir = Path.home() / ".mediapipe_models"
        self.compat_models_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_path(self, model_name: str) -> str:
        """获取模型路径，确保是纯英文路径供MediaPipe使用"""
        # 源文件路径（可能包含中文）
        source_path = self.models_dir / model_name

        # 如果源文件不存在，需要下载
        if not source_path.exists():
            self._download_model(model_name)

        # 目标路径（纯英文路径）
        target_path = self.compat_models_dir / model_name

        # 如果目标文件不存在或源文件更新，则复制
        if not target_path.exists() or (
            source_path.exists() and
            source_path.stat().st_mtime > target_path.stat().st_mtime
        ):
            import shutil
            shutil.copy2(source_path, target_path)
            logger.info(f"[MediaPipeThread] 模型已复制到兼容路径: {target_path}")

        return str(target_path)

    def _download_model(self, model_name: str):
        """下载模型文件到原始目录"""
        model_path = self.models_dir / model_name

        url = self.MODEL_URLS.get(model_name)
        if not url:
            raise Exception(f"未知的模型: {model_name}")

        logger.info(f"[MediaPipeThread] 正在下载模型: {model_name}")

        try:
            import urllib.request
            import ssl

            # 创建SSL上下文
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            # 下载文件
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ssl_context)
            )
            urllib.request.install_opener(opener)

            urllib.request.urlretrieve(url, model_path)
            logger.info(f"[MediaPipeThread] 模型下载完成: {model_path}")
        except Exception as e:
            logger.error(f"[MediaPipeThread] 模型下载失败: {e}")
            raise

    def _create_pose_landmarker(self):
        """创建姿态检测器"""
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision

            # 根据模型复杂度选择对应的模型文件
            model_name = self.POSE_MODEL_MAP.get(self.model_complexity, 'pose_landmarker_full.task')
            complexity_names = {0: '轻量', 1: '标准', 2: '重型'}
            complexity_name = complexity_names.get(self.model_complexity, '标准')
            logger.info(f"[MediaPipeThread] 使用姿态检测模型: {complexity_name} ({model_name})")

            model_path = self._get_model_path(model_name)

            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
            logger.info("[MediaPipeThread] 姿态检测器创建成功")
            return True
        except Exception as e:
            logger.warning(f"[MediaPipeThread] 姿态检测器创建失败: {e}")
            return False

    def _create_hand_landmarker(self):
        """创建手部检测器"""
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision

            model_path = self._get_model_path('hand_landmarker.task')

            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
            logger.info("[MediaPipeThread] 手部检测器创建成功")
            return True
        except Exception as e:
            logger.warning(f"[MediaPipeThread] 手部检测器创建失败: {e}")
            return False

    def _create_gesture_recognizer(self):
        """创建手势识别器"""
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision

            model_path = self._get_model_path('gesture_recognizer.task')

            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.gesture_recognizer = vision.GestureRecognizer.create_from_options(options)
            logger.info("[MediaPipeThread] 手势识别器创建成功")
            return True
        except Exception as e:
            logger.warning(f"[MediaPipeThread] 手势识别器创建失败: {e}")
            return False

    def _create_face_landmarker(self):
        """创建面部检测器"""
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision

            model_path = self._get_model_path('face_landmarker.task')

            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
            logger.info("[MediaPipeThread] 面部检测器创建成功")
            return True
        except Exception as e:
            logger.warning(f"[MediaPipeThread] 面部检测器创建失败: {e}")
            return False

    def run(self):
        """运行推理"""
        self.running = True

        try:
            # 初始化MediaPipe任务
            if self.enable_pose:
                self._create_pose_landmarker()

            if self.enable_hands:
                self._create_hand_landmarker()

            if self.enable_gesture:
                self._create_gesture_recognizer()

            if self.enable_face:
                self._create_face_landmarker()

            # 检查是否至少有一个检测器成功创建
            if not any([self.pose_landmarker, self.hand_landmarker, self.face_landmarker, self.gesture_recognizer]):
                raise Exception("没有可用的MediaPipe检测器，请检查模型文件和网络连接")

            # 根据源类型运行推理
            if self.source_type == "camera":
                self._run_camera_inference()
            elif self.source_type == "file":
                self._run_file_inference()

        except Exception as e:
            logger.exception(f"[MediaPipeThread] 推理异常: {e}")
            self.error_occurred.emit(str(e))
        finally:
            self.running = False
            if self.cap:
                self.cap.release()
            # 释放MediaPipe资源
            if self.pose_landmarker:
                self.pose_landmarker.close()
            if self.hand_landmarker:
                self.hand_landmarker.close()
            if self.gesture_recognizer:
                self.gesture_recognizer.close()
            if self.face_landmarker:
                self.face_landmarker.close()

    def _process_frame(self, frame: np.ndarray, timestamp_ms: int) -> tuple:
        """处理单帧图像"""
        import mediapipe as mp

        result = MediaPipeResult()
        h, w = frame.shape[:2]

        # 转换为MediaPipe图像格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 姿态检测
        if self.pose_landmarker:
            try:
                pose_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                if pose_result.pose_landmarks:
                    for pose_landmarks in pose_result.pose_landmarks:
                        pose = PoseResult()
                        pose.confidence = 1.0
                        for landmark in pose_landmarks:
                            pose.keypoints.append(Keypoint(
                                x=landmark.x,
                                y=landmark.y,
                                z=landmark.z,
                                visibility=landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                            ))
                        result.poses.append(pose)
            except Exception as e:
                logger.error(f"姿态检测失败: {e}")

        # 手势识别（如果启用了手势识别，使用GestureRecognizer，它同时返回关键点和手势）
        if self.gesture_recognizer:
            try:
                gesture_result = self.gesture_recognizer.recognize_for_video(mp_image, timestamp_ms)
                if gesture_result.hand_landmarks:
                    for i, hand_landmarks in enumerate(gesture_result.hand_landmarks):
                        hand = HandResult()
                        for landmark in hand_landmarks:
                            hand.keypoints.append(Keypoint(
                                x=landmark.x,
                                y=landmark.y,
                                z=landmark.z
                            ))
                        # 获取手势类别
                        if gesture_result.gestures and i < len(gesture_result.gestures):
                            gesture = gesture_result.gestures[i]
                            if gesture:
                                # 取置信度最高的手势
                                best_gesture = max(gesture, key=lambda g: g.score)
                                hand.gesture = best_gesture.category_name
                                hand.gesture_score = best_gesture.score
                        result.hands.append(hand)
            except Exception as e:
                logger.error(f"手势识别失败: {e}")
        # 仅手部检测（不识别手势）
        elif self.hand_landmarker:
            try:
                hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                if hand_result.hand_landmarks:
                    for hand_landmarks in hand_result.hand_landmarks:
                        hand = HandResult()
                        for landmark in hand_landmarks:
                            hand.keypoints.append(Keypoint(
                                x=landmark.x,
                                y=landmark.y,
                                z=landmark.z
                            ))
                        result.hands.append(hand)
            except Exception as e:
                logger.error(f"手部检测失败: {e}")

        # 面部检测
        if self.face_landmarker:
            try:
                face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
                if face_result.face_landmarks:
                    for face_landmarks in face_result.face_landmarks:
                        face_keypoints = []
                        for landmark in face_landmarks:
                            face_keypoints.append(Keypoint(
                                x=landmark.x,
                                y=landmark.y,
                                z=landmark.z
                            ))
                        result.faces.append(face_keypoints)
            except Exception as e:
                logger.error(f"面部检测失败: {e}")

        # 绘制结果
        display_frame = self._draw_results(frame, result)

        return display_frame, result

    def _draw_results(self, frame: np.ndarray, result: MediaPipeResult) -> np.ndarray:
        """绘制关键点结果"""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        # 绘制姿态
        for pose in result.poses:
            # 绘制边界框
            if pose.bbox:
                x1, y1, x2, y2 = map(int, pose.bbox)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制关键点
            for i, kp in enumerate(pose.keypoints):
                x, y = int(kp.x * w), int(kp.y * h)
                color = (0, 255, 0) if kp.visibility > 0.5 else (128, 128, 128)
                cv2.circle(display_frame, (x, y), 4, color, -1)
                cv2.putText(display_frame, str(i), (x+5, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

            # 绘制骨骼连接 (简化的连接)
            connections = [(0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8),
                          (9,10), (11,12), (11,13), (13,15), (15,17), (12,14), (14,16), (16,18)]
            for start_idx, end_idx in connections:
                if len(pose.keypoints) > max(start_idx, end_idx):
                    kp1 = pose.keypoints[start_idx]
                    kp2 = pose.keypoints[end_idx]
                    if kp1.visibility > 0.5 and kp2.visibility > 0.5:
                        x1, y1 = int(kp1.x * w), int(kp1.y * h)
                        x2, y2 = int(kp2.x * w), int(kp2.y * h)
                        cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # 绘制手部
        for hand in result.hands:
            # 绘制关键点
            for kp in hand.keypoints:
                x, y = int(kp.x * w), int(kp.y * h)
                cv2.circle(display_frame, (x, y), 3, (255, 0, 0), -1)

            # 绘制手势标签
            if hand.gesture:
                # 计算手部中心位置用于显示标签
                if hand.keypoints:
                    center_x = int(sum(kp.x for kp in hand.keypoints) / len(hand.keypoints) * w)
                    center_y = int(sum(kp.y for kp in hand.keypoints) / len(hand.keypoints) * h)
                    # 手势名称映射（英文->中文/emoji）
                    gesture_map = {
                        'Thumb_Up': '👍 赞',
                        'Thumb_Down': '👎 踩',
                        'Open_Palm': '✋ 手掌',
                        'Closed_Fist': '✊ 拳头',
                        'Victory': '✌️ 胜利',
                        'Pointing_Up': '☝️ 指上',
                        'ILoveYou': '🤟 爱你',
                        'None': '无手势'
                    }
                    gesture_text = gesture_map.get(hand.gesture, hand.gesture)
                    # 计算文字大小
                    temp_pil = Image.new('RGB', (1, 1))
                    temp_draw = ImageDraw.Draw(temp_pil)
                    try:
                        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 20)
                    except:
                        font = ImageFont.load_default()
                    bbox = temp_draw.textbbox((0, 0), gesture_text, font=font)
                    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

                    # 绘制背景框
                    cv2.rectangle(display_frame,
                                 (center_x - text_w//2 - 5, center_y - text_h - 10),
                                 (center_x + text_w//2 + 5, center_y + 5),
                                 (255, 255, 0), -1)
                    # 使用PIL绘制中文文字
                    display_frame = self._draw_chinese_text(
                        display_frame, gesture_text,
                        (center_x - text_w//2, center_y - text_h - 5),
                        20, (0, 0, 0)
                    )

        # 绘制面部
        for face in result.faces:
            for kp in face:
                x, y = int(kp.x * w), int(kp.y * h)
                cv2.circle(display_frame, (x, y), 1, (0, 0, 255), -1)

        # 显示模式信息
        display_frame = self._draw_chinese_text(display_frame, "MediaPipe Tasks API", (10, 30), 20, (0, 255, 255))

        return display_frame

    def _draw_chinese_text(self, img: np.ndarray, text: str, position: tuple, font_size: int, color: tuple) -> np.ndarray:
        """使用PIL绘制中文文字"""
        # 转换OpenCV图像为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 尝试加载中文字体
        font = None
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
            "C:/Windows/Fonts/simkai.ttf",  # 楷体
        ]

        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue

        if font is None:
            # 如果找不到中文字体，使用默认字体
            font = ImageFont.load_default()

        # 绘制文字
        draw.text(position, text, font=font, fill=color[::-1])  # BGR to RGB

        # 转换回OpenCV图像
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def _run_camera_inference(self):
        """摄像头推理"""
        camera_id = self.source_config.get('camera_id', 0)
        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            raise Exception(f"无法打开摄像头 {camera_id}")

        frame_count = 0
        start_time = datetime.now()
        start_timestamp_ms = int(start_time.timestamp() * 1000)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # 计算当前时间戳
            current_timestamp_ms = int(datetime.now().timestamp() * 1000) - start_timestamp_ms

            display_frame, result = self._process_frame(frame, current_timestamp_ms)

            # 计算FPS
            frame_count += 1
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > 0:
                result.fps = frame_count / elapsed
                result.inference_time = elapsed / frame_count

            self.frame_ready.emit(display_frame, result)
            self.fps_updated.emit(result.fps)

    def _run_file_inference(self):
        """文件推理"""
        file_path = self.source_config.get('file_path', '')

        if os.path.isfile(file_path):
            self.cap = cv2.VideoCapture(file_path)
            frame_count = 0
            start_time = datetime.now()
            start_timestamp_ms = int(start_time.timestamp() * 1000)

            while self.running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                # 计算当前时间戳
                current_timestamp_ms = int(datetime.now().timestamp() * 1000) - start_timestamp_ms

                display_frame, result = self._process_frame(frame, current_timestamp_ms)

                frame_count += 1
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > 0:
                    result.fps = frame_count / elapsed

                self.frame_ready.emit(display_frame, result)
                self.fps_updated.emit(result.fps)

    def stop(self):
        """停止推理"""
        self.running = False
        self.wait(1000)


# ============== YOLO 推理线程 ==============
class InferenceThread(QThread):
    """YOLO推理线程"""
    
    frame_ready = pyqtSignal(np.ndarray, InferenceResult)
    fps_updated = pyqtSignal(float)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, inference_engine: YOLOInference, source_type: str, source_config: dict):
        super().__init__()
        self.inference_engine = inference_engine
        self.source_type = source_type
        self.source_config = source_config
        self.running = False
        self.cap = None
    
    def run(self):
        """运行推理"""
        logger.info(f"[InferenceThread] 线程开始运行，源类型: {self.source_type}")
        self.running = True
        
        try:
            if self.source_type == "camera":
                self._run_camera_inference()
            elif self.source_type == "file":
                self._run_file_inference()
            elif self.source_type == "http":
                self._run_http_inference()
            elif self.source_type == "mqtt":
                self._run_mqtt_inference()
        except Exception as e:
            logger.exception(f"[InferenceThread] 推理线程异常: {e}")
            self.error_occurred.emit(str(e))
        finally:
            self.running = False
            if self.cap:
                self.cap.release()
    
    def _run_camera_inference(self):
        """摄像头推理"""
        camera_id = self.source_config.get('camera_id', 0)
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            raise Exception(f"无法打开摄像头 {camera_id}")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            if self.inference_engine is None:
                break
            
            result = self.inference_engine.infer(frame)
            display_frame = self.inference_engine.draw_results(frame, result)
            
            self.frame_ready.emit(display_frame, result)
            self.fps_updated.emit(result.fps)
    
    def _run_file_inference(self):
        """文件推理"""
        file_path = self.source_config.get('file_path', '')
        
        if os.path.isfile(file_path):
            self.cap = cv2.VideoCapture(file_path)
            while self.running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                result = self.inference_engine.infer(frame)
                display_frame = self.inference_engine.draw_results(frame, result)
                
                self.frame_ready.emit(display_frame, result)
                self.fps_updated.emit(result.fps)
    
    def _run_http_inference(self):
        """HTTP流推理"""
        url = self.source_config.get('http_url', '')
        self.cap = cv2.VideoCapture(url)
        
        if not self.cap.isOpened():
            raise Exception(f"无法打开视频流 {url}")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            result = self.inference_engine.infer(frame)
            display_frame = self.inference_engine.draw_results(frame, result)
            
            self.frame_ready.emit(display_frame, result)
            self.fps_updated.emit(result.fps)
    
    def _run_mqtt_inference(self):
        """MQTT图像推理"""
        pass
    
    def stop(self):
        """停止推理"""
        self.running = False
        self.wait(1000)


# ============== YOLO 专栏 ==============
class YOLOPanel(QWidget):
    """YOLO目标检测专栏"""
    
    # 信号
    model_load_requested = pyqtSignal(str, int)  # 模型路径, 任务类型
    inference_start_requested = pyqtSignal()
    inference_stop_requested = pyqtSignal()
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        
        self.inference_engine: Optional[YOLOInference] = None
        self.inference_thread: Optional[InferenceThread] = None
        self.is_running = False
        self.current_fps = 0.0
        
        self._init_ui()
        self._load_config()
    
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # 标题
        title_label = QLabel("🎯 YOLO 目标检测")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #4caf50;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # 分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # 左侧控制面板
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # 右侧视频显示面板
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        # 设置分割比例
        splitter.setSizes([350, 850])
    
    def _create_left_panel(self) -> QWidget:
        """创建左侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # 模型设置组
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout(model_group)
        
        # 模型选择方式
        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("模型来源:"))
        
        self.model_source_combo = QComboBox()
        self.model_source_combo.addItems(["官方预训练模型", "自定义模型"])
        self.model_source_combo.currentIndexChanged.connect(self._on_model_source_changed)
        model_select_layout.addWidget(self.model_source_combo)
        
        model_layout.addLayout(model_select_layout)
        
        # 官方模型选择
        self.official_model_widget = QWidget()
        official_layout = QHBoxLayout(self.official_model_widget)
        official_layout.setContentsMargins(0, 0, 0, 0)
        official_layout.addWidget(QLabel("任务类型:"))
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems([
            "目标检测 (Detection)",
            "实例分割 (Segmentation)",
            "姿态估计 (Pose)",
            "定向检测 (OBB)",
            "图像分类 (Classification)"
        ])
        self.task_type_combo.currentIndexChanged.connect(self._on_task_type_changed)
        official_layout.addWidget(self.task_type_combo)
        
        official_layout.addWidget(QLabel("模型:"))
        self.model_size_combo = QComboBox()
        self._update_model_combo_items()
        self.model_size_combo.currentIndexChanged.connect(self._check_official_model_status)
        official_layout.addWidget(self.model_size_combo)
        
        self.download_model_btn = QPushButton("下载模型")
        self.download_model_btn.setStyleSheet("background-color: #ff9800;")
        self.download_model_btn.clicked.connect(self._download_official_model)
        official_layout.addWidget(self.download_model_btn)
        
        model_layout.addWidget(self.official_model_widget)
        
        # 自定义模型路径
        self.custom_model_widget = QWidget()
        custom_layout = QVBoxLayout(self.custom_model_widget)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        custom_layout.setSpacing(10)
        
        custom_task_layout = QHBoxLayout()
        custom_task_layout.addWidget(QLabel("模型类别:"))
        self.custom_task_type_combo = QComboBox()
        self.custom_task_type_combo.addItems([
            "目标检测 (Detection)",
            "实例分割 (Segmentation)",
            "姿态估计 (Pose)",
            "定向检测 (OBB)",
            "图像分类 (Classification)"
        ])
        custom_task_layout.addWidget(self.custom_task_type_combo)
        custom_layout.addLayout(custom_task_layout)
        
        path_layout = QHBoxLayout()
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("选择模型文件...")
        path_layout.addWidget(self.model_path_input)
        
        self.browse_model_btn = QPushButton("浏览")
        self.browse_model_btn.clicked.connect(self._browse_model)
        path_layout.addWidget(self.browse_model_btn)
        custom_layout.addLayout(path_layout)
        
        model_layout.addWidget(self.custom_model_widget)
        self.custom_model_widget.setVisible(False)
        
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["目标检测", "实例分割", "姿态估计", "定向检测", "图像分类"])
        self.model_type_combo.setVisible(False)
        self.task_type_combo.currentIndexChanged.connect(self._sync_model_type)
        
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.setStyleSheet("background-color: #2196f3;")
        self.load_model_btn.clicked.connect(self._load_model)
        model_layout.addWidget(self.load_model_btn)
        
        self.model_status_label = QLabel("模型状态: 未加载")
        model_layout.addWidget(self.model_status_label)
        
        layout.addWidget(model_group)
        
        # 推理源设置组
        source_group = QGroupBox("推理源设置")
        source_layout = QVBoxLayout(source_group)
        
        self.source_type_combo = QComboBox()
        self.source_type_combo.addItems(["摄像头", "本地文件", "HTTP推流", "MQTT图像"])
        self.source_type_combo.currentIndexChanged.connect(self._on_source_type_changed)
        source_layout.addWidget(self.source_type_combo)
        
        self.source_config_widget = QWidget()
        self.source_config_layout = QVBoxLayout(self.source_config_widget)
        self.source_config_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.addWidget(self.source_config_widget)
        
        # 摄像头配置
        self.camera_config = QWidget()
        camera_layout = QHBoxLayout(self.camera_config)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        camera_layout.addWidget(QLabel("摄像头ID:"))
        self.camera_id_spin = QSpinBox()
        self.camera_id_spin.setRange(0, 10)
        camera_layout.addWidget(self.camera_id_spin)
        camera_layout.addStretch()
        self.source_config_layout.addWidget(self.camera_config)
        
        # 文件配置
        self.file_config = QWidget()
        file_layout = QHBoxLayout(self.file_config)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.file_path_input = QLineEdit()
        self.file_path_input.setPlaceholderText("选择文件或文件夹...")
        file_layout.addWidget(self.file_path_input)
        self.browse_file_btn = QPushButton("浏览")
        self.browse_file_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(self.browse_file_btn)
        self.source_config_layout.addWidget(self.file_config)
        self.file_config.setVisible(False)
        
        # HTTP配置
        self.http_config = QWidget()
        http_layout = QHBoxLayout(self.http_config)
        http_layout.setContentsMargins(0, 0, 0, 0)
        http_layout.addWidget(QLabel("URL:"))
        self.http_url_input = QLineEdit()
        self.http_url_input.setPlaceholderText("rtsp://... 或 http://...")
        http_layout.addWidget(self.http_url_input)
        self.source_config_layout.addWidget(self.http_config)
        self.http_config.setVisible(False)
        
        # MQTT配置
        self.mqtt_config = QWidget()
        mqtt_layout = QHBoxLayout(self.mqtt_config)
        mqtt_layout.setContentsMargins(0, 0, 0, 0)
        mqtt_layout.addWidget(QLabel("主题:"))
        self.mqtt_topic_input = QLineEdit()
        self.mqtt_topic_input.setPlaceholderText("inference/image")
        mqtt_layout.addWidget(self.mqtt_topic_input)
        self.source_config_layout.addWidget(self.mqtt_config)
        self.mqtt_config.setVisible(False)
        
        layout.addWidget(source_group)
        
        # 推理参数组
        params_group = QGroupBox("推理参数")
        params_layout = QFormLayout(params_group)
        
        self.conf_threshold_spin = QDoubleSpinBox()
        self.conf_threshold_spin.setRange(0.0, 1.0)
        self.conf_threshold_spin.setSingleStep(0.05)
        self.conf_threshold_spin.setValue(0.5)
        params_layout.addRow("置信度阈值:", self.conf_threshold_spin)
        
        self.iou_threshold_spin = QDoubleSpinBox()
        self.iou_threshold_spin.setRange(0.0, 1.0)
        self.iou_threshold_spin.setSingleStep(0.05)
        self.iou_threshold_spin.setValue(0.45)
        params_layout.addRow("IOU阈值:", self.iou_threshold_spin)
        
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(32, 1280)
        self.img_size_spin.setSingleStep(32)
        self.img_size_spin.setValue(640)
        params_layout.addRow("图像尺寸:", self.img_size_spin)
        
        self.half_precision_check = QCheckBox("使用半精度(FP16)")
        params_layout.addRow("半精度:", self.half_precision_check)
        
        layout.addWidget(params_group)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("开始推理 (F5)")
        self.start_btn.setStyleSheet("background-color: #4caf50; font-size: 14px; padding: 10px;")
        self.start_btn.clicked.connect(self._start_inference)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setStyleSheet("background-color: #f44336; font-size: 14px; padding: 10px;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_inference)
        btn_layout.addWidget(self.stop_btn)
        
        layout.addLayout(btn_layout)
        
        # 截图按钮
        self.screenshot_btn = QPushButton("截图 (F6)")
        self.screenshot_btn.clicked.connect(self.take_screenshot)
        layout.addWidget(self.screenshot_btn)
        
        # 推理统计
        stats_group = QGroupBox("推理统计")
        stats_layout = QVBoxLayout(stats_group)
        
        self.fps_label = QLabel("FPS: 0")
        stats_layout.addWidget(self.fps_label)
        
        self.inference_time_label = QLabel("推理时间: 0 ms")
        stats_layout.addWidget(self.inference_time_label)
        
        self.detection_count_label = QLabel("检测数量: 0")
        stats_layout.addWidget(self.detection_count_label)
        
        layout.addWidget(stats_group)
        
        layout.addStretch()
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """创建右侧显示面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # 视频显示标签
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1a1a1a; border: 2px solid #555;")
        self.video_label.setText("等待开始推理...")
        layout.addWidget(self.video_label)
        
        # 检测结果Tab
        self.result_tabs = QTabWidget()
        
        # 检测列表Tab
        self.detection_list = QTableWidget()
        self.detection_list.setColumnCount(4)
        self.detection_list.setHorizontalHeaderLabels(["类别", "置信度", "位置", "操作"])
        self.detection_list.horizontalHeader().setStretchLastSection(True)
        self.result_tabs.addTab(self.detection_list, "检测结果")
        
        # 日志Tab
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.result_tabs.addTab(self.log_text, "日志")
        
        layout.addWidget(self.result_tabs)
        
        return panel
    
    def _load_config(self):
        """加载配置"""
        self.model_path_input.setText(self.config.inference.model_path)
        
        model_type_index = {
            "detection": 0, "segmentation": 1, "pose": 2, "classification": 3
        }.get(self.config.inference.model_type, 0)
        self.model_type_combo.setCurrentIndex(model_type_index)
        
        source_type_index = {
            "camera": 0, "file": 1, "http": 2, "mqtt": 3
        }.get(self.config.source.source_type, 0)
        self.source_type_combo.setCurrentIndex(source_type_index)
        
        self.camera_id_spin.setValue(self.config.source.camera_id)
        self.file_path_input.setText(self.config.source.file_path)
        self.http_url_input.setText(self.config.source.http_url)
        self.mqtt_topic_input.setText(self.config.source.mqtt_topic)
        
        self.conf_threshold_spin.setValue(self.config.inference.conf_threshold)
        self.iou_threshold_spin.setValue(self.config.inference.iou_threshold)
        self.img_size_spin.setValue(self.config.inference.img_size)
        self.half_precision_check.setChecked(self.config.inference.half_precision)
    
    def _on_source_type_changed(self, index: int):
        """源类型改变"""
        self.camera_config.setVisible(index == 0)
        self.file_config.setVisible(index == 1)
        self.http_config.setVisible(index == 2)
        self.mqtt_config.setVisible(index == 3)
    
    def _on_model_source_changed(self, index: int):
        """模型来源改变"""
        is_official = (index == 0)
        self.official_model_widget.setVisible(is_official)
        self.custom_model_widget.setVisible(not is_official)
        if is_official:
            self._check_official_model_status()
    
    def _on_task_type_changed(self):
        """任务类型改变"""
        self._update_model_combo_items()
        self._check_official_model_status()
    
    def _update_model_combo_items(self):
        """更新模型下拉框选项"""
        task_type = self.task_type_combo.currentIndex()
        suffixes = ["", "-seg", "-pose", "-obb", "-cls"]
        suffix = suffixes[task_type] if task_type < len(suffixes) else ""
        
        models = [
            (f"yolo26n{suffix}", "Nano - 快速"),
            (f"yolo26s{suffix}", "Small - 平衡"),
            (f"yolo26m{suffix}", "Medium - 精确"),
            (f"yolo26l{suffix}", "Large - 高精度"),
            (f"yolo26x{suffix}", "XLarge - 最高精度")
        ]
        
        current_index = self.model_size_combo.currentIndex()
        self.model_size_combo.clear()
        for model_name, desc in models:
            self.model_size_combo.addItem(f"{model_name} ({desc})")
        
        if current_index >= 0 and current_index < self.model_size_combo.count():
            self.model_size_combo.setCurrentIndex(current_index)
    
    def _sync_model_type(self):
        """同步任务类型和模型类型"""
        task_index = self.task_type_combo.currentIndex()
        if task_index >= 0 and task_index < self.model_type_combo.count():
            self.model_type_combo.setCurrentIndex(task_index)
    
    def _get_current_model_name(self) -> str:
        """获取当前选中的完整模型名称"""
        full_text = self.model_size_combo.currentText()
        model_name = full_text.split()[0] if full_text else "yolo26n"
        return model_name
    
    def _check_official_model_status(self):
        """检查官方模型状态"""
        model_name = self._get_current_model_name()
        model_path = self._get_official_model_path(model_name)
        
        if model_path.exists():
            self.download_model_btn.setText("已下载")
            self.download_model_btn.setStyleSheet("background-color: #4caf50;")
            self.download_model_btn.setEnabled(False)
            self.model_path_input.setText(str(model_path))
        else:
            self.download_model_btn.setText("下载模型")
            self.download_model_btn.setStyleSheet("background-color: #ff9800;")
            self.download_model_btn.setEnabled(True)
    
    def _get_official_model_path(self, model_name: str) -> Path:
        """获取官方模型保存路径"""
        models_dir = Path(__file__).parent.parent.parent / "models"
        models_dir.mkdir(exist_ok=True)
        return models_dir / f"{model_name}.pt"
    
    def _download_official_model(self):
        """下载官方模型"""
        # 简化版，实际实现需要下载逻辑
        QMessageBox.information(self, "提示", "模型下载功能需要网络连接")
    
    def _browse_model(self):
        """浏览模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.pt *.pth *.onnx);;所有文件 (*.*)"
        )
        if file_path:
            self.model_path_input.setText(file_path)
    
    def _browse_file(self):
        """浏览文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频或图像", "", "视频/图像 (*.mp4 *.avi *.jpg *.jpeg *.png);;所有文件 (*.*)"
        )
        if file_path:
            self.file_path_input.setText(file_path)
    
    def _load_model(self):
        """加载模型"""
        if self.model_source_combo.currentIndex() == 0:
            model_name = self._get_current_model_name()
            model_path = str(self._get_official_model_path(model_name))
            task_type = self.task_type_combo.currentIndex()
            
            if not Path(model_path).exists():
                reply = QMessageBox.question(
                    self, "模型未下载", f"官方模型 {model_name} 尚未下载。\n\n是否立即下载?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._download_official_model()
                return
        else:
            model_path = self.model_path_input.text().strip()
            if not model_path:
                QMessageBox.warning(self, "警告", "请选择模型文件")
                return
            task_type = self.custom_task_type_combo.currentIndex()
        
        self.model_load_requested.emit(model_path, task_type)
    
    def on_model_loaded(self, success: bool):
        """模型加载回调"""
        if success:
            self.model_status_label.setText("模型状态: 已加载")
            self.model_status_label.setStyleSheet("color: #4caf50;")
            self._log("模型加载成功")
        else:
            self.model_status_label.setText("模型状态: 加载失败")
            self.model_status_label.setStyleSheet("color: #f44336;")
            self._log("模型加载失败")
    
    def set_inference_engine(self, engine: YOLOInference):
        """设置推理引擎"""
        self.inference_engine = engine
    
    def _start_inference(self):
        """开始推理"""
        if self.inference_engine is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        self._save_config()
        
        source_type = ["camera", "file", "http", "mqtt"][self.source_type_combo.currentIndex()]
        source_config = {}
        
        if source_type == "camera":
            source_config['camera_id'] = self.camera_id_spin.value()
        elif source_type == "file":
            source_config['file_path'] = self.file_path_input.text()
        elif source_type == "http":
            source_config['http_url'] = self.http_url_input.text()
        elif source_type == "mqtt":
            source_config['mqtt_topic'] = self.mqtt_topic_input.text()
        
        self.inference_thread = InferenceThread(
            self.inference_engine, source_type, source_config
        )
        self.inference_thread.frame_ready.connect(self._on_frame_ready)
        self.inference_thread.fps_updated.connect(self._on_fps_updated)
        self.inference_thread.error_occurred.connect(self._on_inference_error)
        self.inference_thread.start()
        
        self.is_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._log(f"开始推理: {source_type}")
    
    def stop_inference(self):
        """公共方法：停止推理"""
        self._stop_inference()
    
    def _stop_inference(self):
        """停止推理"""
        if self.inference_thread:
            self.inference_thread.stop()
            self.inference_thread = None
        
        self.is_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.video_label.setText("推理已停止")
        self._log("推理已停止")
    
    def _on_frame_ready(self, frame: np.ndarray, result: InferenceResult):
        """帧就绪回调"""
        try:
            if frame is None or frame.size == 0:
                return
            
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if not rgb_image.flags['C_CONTIGUOUS']:
                rgb_image = np.ascontiguousarray(rgb_image)
            
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            qt_image = QImage(rgb_image.copy().data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.video_label.setPixmap(scaled_pixmap)
            self._update_detection_list(result)
            
        except Exception as e:
            logger.exception(f"处理帧时出错: {e}")
    
    def _on_fps_updated(self, fps: float):
        """FPS更新回调"""
        self.current_fps = fps
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    def _on_inference_error(self, error: str):
        """推理错误回调"""
        self._log(f"推理错误: {error}")
        QMessageBox.critical(self, "推理错误", error)
        self._stop_inference()
    
    def _update_detection_list(self, result: InferenceResult):
        """更新检测列表"""
        if result.classifications:
            self.detection_list.setRowCount(len(result.classifications))
            for i, cls in enumerate(result.classifications):
                self.detection_list.setItem(i, 0, QTableWidgetItem(cls.class_name))
                self.detection_list.setItem(i, 1, QTableWidgetItem(f"{cls.confidence:.2f}"))
                top5_text = ", ".join([f"{name}:{conf:.2f}" for name, conf in cls.top5[:3]])
                self.detection_list.setItem(i, 2, QTableWidgetItem(top5_text))
            self.detection_count_label.setText(f"分类结果: {len(result.classifications)}")
        elif result.segmentations:
            self.detection_list.setRowCount(len(result.segmentations))
            for i, seg in enumerate(result.segmentations):
                self.detection_list.setItem(i, 0, QTableWidgetItem(seg.class_name))
                self.detection_list.setItem(i, 1, QTableWidgetItem(f"{seg.confidence:.2f}"))
                bbox_text = f"[{seg.bbox[0]:.0f}, {seg.bbox[1]:.0f}, {seg.bbox[2]:.0f}, {seg.bbox[3]:.0f}]"
                self.detection_list.setItem(i, 2, QTableWidgetItem(bbox_text))
            self.detection_count_label.setText(f"分割数量: {len(result.segmentations)}")
        elif result.keypoints:
            self.detection_list.setRowCount(len(result.keypoints))
            for i, kpt in enumerate(result.keypoints):
                self.detection_list.setItem(i, 0, QTableWidgetItem(f"姿态 {i+1}"))
                self.detection_list.setItem(i, 1, QTableWidgetItem(f"{kpt.confidence:.2f}"))
                bbox_text = f"[{kpt.bbox[0]:.0f}, {kpt.bbox[1]:.0f}, {kpt.bbox[2]:.0f}, {kpt.bbox[3]:.0f}]"
                self.detection_list.setItem(i, 2, QTableWidgetItem(bbox_text))
            self.detection_count_label.setText(f"姿态数量: {len(result.keypoints)}")
        else:
            self.detection_list.setRowCount(len(result.detections))
            for i, det in enumerate(result.detections):
                self.detection_list.setItem(i, 0, QTableWidgetItem(det.class_name))
                self.detection_list.setItem(i, 1, QTableWidgetItem(f"{det.confidence:.2f}"))
                bbox_text = f"[{det.bbox[0]:.0f}, {det.bbox[1]:.0f}, {det.bbox[2]:.0f}, {det.bbox[3]:.0f}]"
                self.detection_list.setItem(i, 2, QTableWidgetItem(bbox_text))
            self.detection_count_label.setText(f"检测数量: {len(result.detections)}")
        
        self.inference_time_label.setText(f"推理时间: {result.inference_time*1000:.1f} ms")
    
    def _save_config(self):
        """保存配置"""
        self.config.inference.model_path = self.model_path_input.text()
        self.config.inference.model_type = ["detection", "segmentation", "pose", "obb", "classification"][
            self.task_type_combo.currentIndex()
        ]
        self.config.source.source_type = ["camera", "file", "http", "mqtt"][
            self.source_type_combo.currentIndex()
        ]
        self.config.source.camera_id = self.camera_id_spin.value()
        self.config.source.file_path = self.file_path_input.text()
        self.config.source.http_url = self.http_url_input.text()
        self.config.source.mqtt_topic = self.mqtt_topic_input.text()
        self.config.inference.conf_threshold = self.conf_threshold_spin.value()
        self.config.inference.iou_threshold = self.iou_threshold_spin.value()
        self.config.inference.img_size = self.img_size_spin.value()
        self.config.inference.half_precision = self.half_precision_check.isChecked()
        self.config_manager.save()
    
    def take_screenshot(self):
        """截图"""
        if self.video_label.pixmap():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            screenshot_dir = Path("screenshots")
            screenshot_dir.mkdir(exist_ok=True)
            filepath = screenshot_dir / filename
            self.video_label.pixmap().save(str(filepath))
            self._log(f"截图已保存: {filepath}")
            QMessageBox.information(self, "截图", f"截图已保存到:\n{filepath}")
    
    def _log(self, message: str):
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def apply_theme(self, theme: str):
        """应用主题"""
        pass


# ============== MediaPipe 专栏 ==============
class MediaPipePanel(QWidget):
    """MediaPipe关键点检测专栏"""
    
    # 信号
    inference_start_requested = pyqtSignal()
    inference_stop_requested = pyqtSignal()
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        
        self.mediapipe_thread: Optional[MediaPipeThread] = None
        self.is_running = False
        self.current_fps = 0.0
        self.detection_results: List[MediaPipeResult] = []
        
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # 标题
        title_label = QLabel("🎭 MediaPipe 关键点检测")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2196f3;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # 分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # 左侧控制面板
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # 右侧视频显示面板
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        # 设置分割比例
        splitter.setSizes([350, 850])
    
    def _create_left_panel(self) -> QWidget:
        """创建左侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # 模型设置组
        model_group = QGroupBox("检测模型设置")
        model_layout = QVBoxLayout(model_group)
        
        # 模型状态标签
        self.model_status_label = QLabel("模型状态: 未加载")
        self.model_status_label.setStyleSheet("color: #ff9800;")
        model_layout.addWidget(self.model_status_label)
        
        # 启用姿态检测
        self.enable_pose_check = QCheckBox("启用姿态检测 (Pose)")
        self.enable_pose_check.setChecked(True)
        model_layout.addWidget(self.enable_pose_check)
        
        # 启用手部检测
        self.enable_hands_check = QCheckBox("启用手部检测 (Hands)")
        self.enable_hands_check.stateChanged.connect(self._on_hands_check_changed)
        model_layout.addWidget(self.enable_hands_check)

        # 启用手势识别（仅在手部检测启用时可用）
        self.enable_gesture_check = QCheckBox("  └─ 同时识别手势 (Gesture)")
        self.enable_gesture_check.setEnabled(False)
        self.enable_gesture_check.setToolTip("识别手势类别：👍 👎 ✌️ ☝️ ✊ 👋")
        model_layout.addWidget(self.enable_gesture_check)

        # 启用面部检测
        self.enable_face_check = QCheckBox("启用面部检测 (Face Mesh)")
        model_layout.addWidget(self.enable_face_check)
        
        # 模型复杂度
        complexity_layout = QHBoxLayout()
        complexity_layout.addWidget(QLabel("模型复杂度:"))
        self.complexity_combo = QComboBox()
        self.complexity_combo.addItems(["轻量 (0)", "标准 (1)", "重型 (2)"])
        self.complexity_combo.setCurrentIndex(1)
        complexity_layout.addWidget(self.complexity_combo)
        model_layout.addLayout(complexity_layout)
        
        # 置信度阈值
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("置信度阈值:"))
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setSingleStep(0.1)
        self.confidence_spin.setValue(0.5)
        confidence_layout.addWidget(self.confidence_spin)
        model_layout.addLayout(confidence_layout)
        
        layout.addWidget(model_group)
        
        # 推理源设置组
        source_group = QGroupBox("推理源设置")
        source_layout = QVBoxLayout(source_group)
        
        self.source_type_combo = QComboBox()
        self.source_type_combo.addItems(["摄像头", "本地文件"])
        self.source_type_combo.currentIndexChanged.connect(self._on_source_type_changed)
        source_layout.addWidget(self.source_type_combo)
        
        self.source_config_widget = QWidget()
        self.source_config_layout = QVBoxLayout(self.source_config_widget)
        self.source_config_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.addWidget(self.source_config_widget)
        
        # 摄像头配置
        self.camera_config = QWidget()
        camera_layout = QHBoxLayout(self.camera_config)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        camera_layout.addWidget(QLabel("摄像头ID:"))
        self.camera_id_spin = QSpinBox()
        self.camera_id_spin.setRange(0, 10)
        camera_layout.addWidget(self.camera_id_spin)
        camera_layout.addStretch()
        self.source_config_layout.addWidget(self.camera_config)
        
        # 文件配置
        self.file_config = QWidget()
        file_layout = QHBoxLayout(self.file_config)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.file_path_input = QLineEdit()
        self.file_path_input.setPlaceholderText("选择视频或图像文件...")
        file_layout.addWidget(self.file_path_input)
        self.browse_file_btn = QPushButton("浏览")
        self.browse_file_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(self.browse_file_btn)
        self.source_config_layout.addWidget(self.file_config)
        self.file_config.setVisible(False)
        
        layout.addWidget(source_group)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("开始检测")
        self.start_btn.setStyleSheet("background-color: #4caf50; font-size: 14px; padding: 10px;")
        self.start_btn.clicked.connect(self._start_inference)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setStyleSheet("background-color: #f44336; font-size: 14px; padding: 10px;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_inference)
        btn_layout.addWidget(self.stop_btn)
        
        layout.addLayout(btn_layout)
        
        # 数据导出按钮
        export_layout = QHBoxLayout()
        
        self.export_json_btn = QPushButton("导出JSON")
        self.export_json_btn.clicked.connect(self._export_json)
        export_layout.addWidget(self.export_json_btn)
        
        self.export_csv_btn = QPushButton("导出CSV")
        self.export_csv_btn.clicked.connect(self._export_csv)
        export_layout.addWidget(self.export_csv_btn)
        
        layout.addLayout(export_layout)
        
        # 推理统计
        stats_group = QGroupBox("检测统计")
        stats_layout = QVBoxLayout(stats_group)
        
        self.fps_label = QLabel("FPS: 0")
        stats_layout.addWidget(self.fps_label)
        
        self.pose_count_label = QLabel("姿态数量: 0")
        stats_layout.addWidget(self.pose_count_label)
        
        self.hand_count_label = QLabel("手部数量: 0")
        stats_layout.addWidget(self.hand_count_label)
        
        self.face_count_label = QLabel("面部数量: 0")
        stats_layout.addWidget(self.face_count_label)
        
        layout.addWidget(stats_group)
        
        layout.addStretch()
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """创建右侧显示面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # 视频显示标签
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1a1a1a; border: 2px solid #555;")
        self.video_label.setText("等待开始检测...")
        layout.addWidget(self.video_label)
        
        # 结果Tab
        self.result_tabs = QTabWidget()
        
        # 关键点列表Tab
        self.keypoint_list = QTableWidget()
        self.keypoint_list.setColumnCount(5)
        self.keypoint_list.setHorizontalHeaderLabels(["类型", "ID", "关键点数", "置信度", "位置"])
        self.keypoint_list.horizontalHeader().setStretchLastSection(True)
        self.result_tabs.addTab(self.keypoint_list, "关键点列表")
        
        # 日志Tab
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.result_tabs.addTab(self.log_text, "日志")
        
        layout.addWidget(self.result_tabs)
        
        return panel
    
    def _on_source_type_changed(self, index: int):
        """源类型改变"""
        self.camera_config.setVisible(index == 0)
        self.file_config.setVisible(index == 1)

    def _on_hands_check_changed(self, state: int):
        """手部检测复选框状态改变"""
        self.enable_gesture_check.setEnabled(state == Qt.CheckState.Checked.value)
        if state != Qt.CheckState.Checked.value:
            self.enable_gesture_check.setChecked(False)

    def _browse_file(self):
        """浏览文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频或图像", "", "视频/图像 (*.mp4 *.avi *.jpg *.jpeg *.png);;所有文件 (*.*)"
        )
        if file_path:
            self.file_path_input.setText(file_path)
    
    def _start_inference(self):
        """开始推理"""
        source_type = ["camera", "file"][self.source_type_combo.currentIndex()]
        source_config = {}
        
        if source_type == "camera":
            source_config['camera_id'] = self.camera_id_spin.value()
        elif source_type == "file":
            source_config['file_path'] = self.file_path_input.text()
        
        # 获取模型复杂度 (0=轻量, 1=标准, 2=重型)
        model_complexity = self.complexity_combo.currentIndex()

        # 启用手势识别（仅在手部检测启用时）
        enable_gesture = self.enable_hands_check.isChecked() and self.enable_gesture_check.isChecked()

        self.mediapipe_thread = MediaPipeThread(
            source_type=source_type,
            source_config=source_config,
            enable_pose=self.enable_pose_check.isChecked(),
            enable_hands=self.enable_hands_check.isChecked() and not enable_gesture,  # 如果启用手势识别，则不单独使用手部检测
            enable_face=self.enable_face_check.isChecked(),
            model_complexity=model_complexity,
            enable_gesture=enable_gesture
        )
        self.mediapipe_thread.frame_ready.connect(self._on_frame_ready)
        self.mediapipe_thread.fps_updated.connect(self._on_fps_updated)
        self.mediapipe_thread.error_occurred.connect(self._on_inference_error)
        self.mediapipe_thread.start()
        
        self.is_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.detection_results.clear()
        self._log(f"开始MediaPipe检测: {source_type}")
    
    def stop_inference(self):
        """公共方法：停止推理"""
        self._stop_inference()
    
    def _stop_inference(self):
        """停止推理"""
        if self.mediapipe_thread:
            self.mediapipe_thread.stop()
            self.mediapipe_thread = None
        
        self.is_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.video_label.setText("检测已停止")
        self._log("检测已停止")
    
    def _on_frame_ready(self, frame: np.ndarray, result: MediaPipeResult):
        """帧就绪回调"""
        try:
            if frame is None or frame.size == 0:
                return
            
            # 转换OpenCV图像为QPixmap
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.video_label.setPixmap(scaled_pixmap)
            
            # 保存结果用于导出
            self.detection_results.append(result)
            
            # 更新统计
            self._update_stats(result)
            
        except Exception as e:
            logger.exception(f"处理帧时出错: {e}")
    
    def _on_fps_updated(self, fps: float):
        """FPS更新回调"""
        self.current_fps = fps
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    def _on_inference_error(self, error: str):
        """推理错误回调"""
        self._log(f"检测错误: {error}")
        QMessageBox.critical(self, "检测错误", error)
        self._stop_inference()
    
    def _on_fallback_mode(self, is_fallback: bool):
        """备用模式切换回调"""
        if is_fallback:
            self._log("MediaPipe加载失败，已切换到OpenCV备用模式")
            if hasattr(self, 'model_status_label'):
                self.model_status_label.setText("模型状态: OpenCV备用模式")
                self.model_status_label.setStyleSheet("color: #ff9800;")
    
    def _update_stats(self, result: MediaPipeResult):
        """更新统计信息"""
        self.pose_count_label.setText(f"姿态数量: {len(result.poses)}")
        self.hand_count_label.setText(f"手部数量: {len(result.hands)}")
        self.face_count_label.setText(f"面部数量: {len(result.faces)}")
        
        # 更新关键点列表
        total_items = len(result.poses) + len(result.hands) + len(result.faces)
        self.keypoint_list.setRowCount(total_items)
        
        row = 0
        for i, pose in enumerate(result.poses):
            self.keypoint_list.setItem(row, 0, QTableWidgetItem("姿态"))
            self.keypoint_list.setItem(row, 1, QTableWidgetItem(str(i)))
            self.keypoint_list.setItem(row, 2, QTableWidgetItem(str(len(pose.keypoints))))
            self.keypoint_list.setItem(row, 3, QTableWidgetItem(f"{pose.confidence:.2f}"))
            if pose.bbox:
                bbox_text = f"[{pose.bbox[0]:.0f}, {pose.bbox[1]:.0f}, {pose.bbox[2]:.0f}, {pose.bbox[3]:.0f}]"
                self.keypoint_list.setItem(row, 4, QTableWidgetItem(bbox_text))
            row += 1
        
        for i, hand in enumerate(result.hands):
            self.keypoint_list.setItem(row, 0, QTableWidgetItem("手部"))
            self.keypoint_list.setItem(row, 1, QTableWidgetItem(str(i)))
            self.keypoint_list.setItem(row, 2, QTableWidgetItem(str(len(hand.keypoints))))
            # 如果有手势，显示手势信息
            if hand.gesture:
                self.keypoint_list.setItem(row, 3, QTableWidgetItem(f"{hand.gesture} ({hand.gesture_score:.2f})"))
            else:
                self.keypoint_list.setItem(row, 3, QTableWidgetItem("-"))
            row += 1
        
        for i, face in enumerate(result.faces):
            self.keypoint_list.setItem(row, 0, QTableWidgetItem("面部"))
            self.keypoint_list.setItem(row, 1, QTableWidgetItem(str(i)))
            self.keypoint_list.setItem(row, 2, QTableWidgetItem(str(len(face))))
            self.keypoint_list.setItem(row, 3, QTableWidgetItem("-"))
            row += 1
    
    def _export_json(self):
        """导出为JSON格式"""
        if not self.detection_results:
            QMessageBox.warning(self, "警告", "没有检测结果可导出")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出JSON", "mediapipe_results.json", "JSON文件 (*.json)"
        )
        if not file_path:
            return
        
        try:
            export_data = []
            for result in self.detection_results:
                frame_data = {
                    "fps": result.fps,
                    "inference_time": result.inference_time,
                    "poses": [],
                    "hands": [],
                    "faces": []
                }
                
                for pose in result.poses:
                    frame_data["poses"].append({
                        "confidence": pose.confidence,
                        "keypoints": [{"x": kp.x, "y": kp.y, "z": kp.z, "visibility": kp.visibility} 
                                     for kp in pose.keypoints]
                    })
                
                for hand in result.hands:
                    frame_data["hands"].append([
                        {"x": kp.x, "y": kp.y, "z": kp.z} for kp in hand
                    ])
                
                for face in result.faces:
                    frame_data["faces"].append([
                        {"x": kp.x, "y": kp.y, "z": kp.z} for kp in face
                    ])
                
                export_data.append(frame_data)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self._log(f"结果已导出到: {file_path}")
            QMessageBox.information(self, "导出成功", f"结果已导出到:\n{file_path}")
            
        except Exception as e:
            logger.exception(f"导出JSON失败: {e}")
            QMessageBox.critical(self, "导出失败", f"导出失败:\n{e}")
    
    def _export_csv(self):
        """导出为CSV格式"""
        if not self.detection_results:
            QMessageBox.warning(self, "警告", "没有检测结果可导出")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出CSV", "mediapipe_results.csv", "CSV文件 (*.csv)"
        )
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["frame", "type", "id", "kp_id", "x", "y", "z", "visibility"])
                
                for frame_idx, result in enumerate(self.detection_results):
                    for pose_idx, pose in enumerate(result.poses):
                        for kp_idx, kp in enumerate(pose.keypoints):
                            writer.writerow([
                                frame_idx, "pose", pose_idx, kp_idx,
                                kp.x, kp.y, kp.z, kp.visibility
                            ])
                    
                    for hand_idx, hand in enumerate(result.hands):
                        for kp_idx, kp in enumerate(hand):
                            writer.writerow([
                                frame_idx, "hand", hand_idx, kp_idx,
                                kp.x, kp.y, kp.z, ""
                            ])
                    
                    for face_idx, face in enumerate(result.faces):
                        for kp_idx, kp in enumerate(face):
                            writer.writerow([
                                frame_idx, "face", face_idx, kp_idx,
                                kp.x, kp.y, kp.z, ""
                            ])
            
            self._log(f"结果已导出到: {file_path}")
            QMessageBox.information(self, "导出成功", f"结果已导出到:\n{file_path}")
            
        except Exception as e:
            logger.exception(f"导出CSV失败: {e}")
            QMessageBox.critical(self, "导出失败", f"导出失败:\n{e}")
    
    def _log(self, message: str):
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def apply_theme(self, theme: str):
        """应用主题"""
        pass


# ============== 主推理面板 ==============
class InferenceWidget(QWidget):
    """推理面板 - 包含YOLO和MediaPipe两个专栏"""
    
    # 信号
    inference_start_requested = pyqtSignal()
    inference_stop_requested = pyqtSignal()
    model_load_requested = pyqtSignal(str, int)  # 模型路径, 任务类型
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 创建专栏切换Tab
        self.column_tabs = QTabWidget()
        self.column_tabs.setDocumentMode(True)
        self.column_tabs.setTabPosition(QTabWidget.TabPosition.North)
        
        # YOLO专栏
        self.yolo_panel = YOLOPanel(self.config_manager)
        self.yolo_panel.model_load_requested.connect(self.model_load_requested)
        self.yolo_panel.inference_start_requested.connect(self.inference_start_requested)
        self.yolo_panel.inference_stop_requested.connect(self.inference_stop_requested)
        self.column_tabs.addTab(self.yolo_panel, "🎯 YOLO目标检测")
        
        # MediaPipe专栏
        self.mediapipe_panel = MediaPipePanel(self.config_manager)
        self.mediapipe_panel.inference_start_requested.connect(self.inference_start_requested)
        self.mediapipe_panel.inference_stop_requested.connect(self.inference_stop_requested)
        self.column_tabs.addTab(self.mediapipe_panel, "🎭 MediaPipe关键点检测")
        
        layout.addWidget(self.column_tabs)
    
    def on_model_loaded(self, success: bool):
        """模型加载回调 - 转发给YOLO面板"""
        self.yolo_panel.on_model_loaded(success)
    
    def set_inference_engine(self, engine: YOLOInference):
        """设置推理引擎 - 转发给YOLO面板"""
        self.yolo_panel.set_inference_engine(engine)
    
    def apply_theme(self, theme: str):
        """应用主题"""
        self.yolo_panel.apply_theme(theme)
        self.mediapipe_panel.apply_theme(theme)
    
    def start_inference(self):
        """外部调用开始推理"""
        current_tab = self.column_tabs.currentIndex()
        if current_tab == 0:
            self.yolo_panel.start_inference()
        else:
            self.mediapipe_panel.start_inference()
    
    def stop_inference(self):
        """外部调用停止推理"""
        self.yolo_panel.stop_inference()
        self.mediapipe_panel.stop_inference()
